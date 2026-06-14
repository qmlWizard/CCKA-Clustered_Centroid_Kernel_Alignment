"""
Kernel Target Alignment (KTA) optimizers for quantum kernel methods.

This module provides five strategies for aligning a parameterized quantum kernel
to a classification task via gradient-based or analytical KTA maximization:

    FullKTA           – gradient computed on the entire training set each epoch
    RandomKTA         – stochastic mini-batch sampling each epoch
    GreedyKTA         – active-learning selection of the most uncertain samples
    CentroidBasedKTA  – alternating gradient-based optimization of kernel weights and centroids
    QuackKTA          – QUACK strategy: uses full training data instead of sub-centroids
                        with gradient-based optimization of kernel weights and main centroids

All strategies share a common abstract base (BaseKTA) that houses kernel matrix
construction, SVM evaluation, centering, and the main training loop.

Alignment with PyTorch TrainModel (train_method='ccka'):
  -------------------------------------------------------
  PyTorch uses a SINGLE Adam optimizer for the KAO step that jointly updates
  both kernel weights and sub-centroids:

      self._kernel_optimizer = optim.Adam([
          {'params': self._kernel.parameters(), 'lr': self._lr},
          {'params': self._class_centroids,     'lr': self._cclr},
      ])

  The CO step uses a *separate* per-class optimizer that updates ONLY the
  main centroid for the selected class:

      self._optimizers[_class] = optim.Adam([{'params': main_centroid, 'lr': self._mclr}])

  This module replicates that behaviour exactly:
    - _kao_weight_optimizer  (lr=learning_rate)   ← kernel weights
    - _kao_sub_optimizer     (lr=sub_centroid_lr) ← sub-centroids (jointly w/ KAO)
    - _co_main_optimizer     (lr=centroid_lr)     ← main centroids only (CO step)

  CO box constraint uses relu(c-1)+relu(-c) matching PyTorch (not per-feature bounds).
  CO step uses raw sub-centroid labels, not ±1 conversion.

Backward-compatible lowercase aliases are exported at module level.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pprint import pformat
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax as ox
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Display utilities
# ─────────────────────────────────────────────────────────────────────────────

def _print_box(title: str, lines: list[str], width: int = 78) -> None:
    bar = "─" * (width - 2)
    print(f"┌{bar}┐")
    print(f"│ {title.center(width - 4)} │")
    print(f"├{bar}┤")
    for line in lines:
        for segment in pformat(line, width=width - 6).splitlines():
            print(f"│ {segment.ljust(width - 4)} │")
    print(f"└{bar}┘\n")


def print_training_summary(history: dict[str, Any], width: int = 78) -> None:
    """Pretty-print a training history dictionary returned by ``align()``."""
    _print_box(
        "TRAINING SUMMARY",
        [
            f"Epochs run          : {len(history['loss_history'])}",
            f"Total training time : {history['time']:.2f} s",
        ],
        width,
    )
    _print_box(
        "ACCURACY METRICS",
        [
            f"Initial train accuracy : {history['init_train_accuracy']:.4f}",
            f"Final   train accuracy : {history['train_accuracy_history'][-1]:.4f}",
            f"Initial test  accuracy : {history['init_test_accuracy']:.4f}",
            f"Final   test  accuracy : {history['test_accuracy_history'][-1]:.4f}",
        ],
        width,
    )
    _print_box(
        "CLASSIFICATION METRICS (FINAL EPOCH)",
        [
            f"F1 score  : {history['f1_score_history'][-1]:.4f}",
            f"Precision : {history['precision_score_history'][-1]:.4f}",
            f"Recall    : {history['recall_score_history'][-1]:.4f}",
        ],
        width,
    )
    _print_box(
        "ALIGNMENT & OPTIMIZATION",
        [
            f"Initial alignment    : {history['alignment_history'][0]:.6f}",
            f"Final alignment    : {history['alignment_history'][-1]:.6f}",
            f"Circuit executions : {history['circuit_executions']}",
        ],
        width,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BaseKTA(ABC):
    """
    Abstract base class for KTA-based quantum kernel optimizers.

    Concrete subclasses implement :meth:`_get_batch` to control *which* data
    subset drives the gradient at each training step.  Everything else —
    kernel matrix construction, centering, SVM evaluation, and the training
    loop — lives here.

    Parameters
    ----------
    kernel_model :
        Object exposing ``kernel_model.circuit.init_weights()`` and
        ``kernel_model.forward(x1, x2, weights)``.
    data : jnp.ndarray, shape (N, D)
    labels : jnp.ndarray, shape (N,)
    split_size : float
        Fraction of data used for training (default 0.8).
    matrix_type : {'regular', 'nystrom'}
        How to build the kernel matrix.
    landmark_points : int
        Number of Nyström landmarks; required when *matrix_type='nystrom'*.
    centering : bool
        Apply kernel centering (H K H) before use.
    epochs : int
    learning_rate : float
    optimizer : {'adam', 'sgd'}
    """

    _OPTIMIZERS: dict[str, Any] = {"adam": ox.adam, "sgd": ox.sgd}
    _MATRIX_TYPES: frozenset[str] = frozenset({"regular", "nystrom"})

    def __init__(
        self,
        kernel_model,
        data: jnp.ndarray,
        labels: jnp.ndarray,
        *,
        split_size: float = 0.8,
        matrix_type: str = "regular",
        landmark_points: int = 0,
        centering: bool = False,
        epochs: int = 100,
        learning_rate: float = 0.01,
        optimizer: str = "adam",
        **_ignored: Any,
    ) -> None:
        # ── Validation ────────────────────────────────────────────────────
        if matrix_type not in self._MATRIX_TYPES:
            raise ValueError(
                f"matrix_type must be one of {self._MATRIX_TYPES!r}, got {matrix_type!r}"
            )
        if not (0.0 < split_size < 1.0):
            raise ValueError(f"split_size must be in (0, 1), got {split_size}")
        if matrix_type == "nystrom" and landmark_points <= 0:
            raise ValueError(
                "landmark_points must be > 0 when matrix_type='nystrom'"
            )

        # ── Store hyperparameters ─────────────────────────────────────────
        self.kernel_model = kernel_model
        self.matrix_type = matrix_type
        self.landmark_points = landmark_points
        self.centering = centering
        self.epochs = epochs
        self.split_size = split_size
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer.lower()

        # ── Data split ────────────────────────────────────────────────────
        self.xtrain, self.xtest, self.ytrain, self.ytest = self._split_data(
            data, labels, seed=42
        )

        # ── Weights & optimizer ───────────────────────────────────────────
        self.weights = kernel_model.circuit.init_weights()
        self._optimizer = self._build_optimizer(self.learning_rate)
        self.opt_state = self._optimizer.init(self.weights)

        # ── JIT-compiled functions ────────────────────────────────────────
        self._loss_fn = jax.jit(self._loss_kta)
        self._grad_fn = jax.jit(jax.grad(self._loss_kta))

    # ── Optimizer factory ──────────────────────────────────────────────────

    def _build_optimizer(self, lr: float) -> ox.GradientTransformation:
        if self.optimizer_name not in self._OPTIMIZERS:
            raise ValueError(
                f"Optimizer {self.optimizer_name!r} not supported. "
                f"Choose from: {list(self._OPTIMIZERS)}"
            )
        return self._OPTIMIZERS[self.optimizer_name](lr)

    # ── Data splitting ─────────────────────────────────────────────────────

    def _split_data(
        self,
        data: jnp.ndarray,
        labels: jnp.ndarray,
        seed: int = 42,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        n = len(data)
        perm = jax.random.permutation(jax.random.PRNGKey(seed), n)
        split = int(n * self.split_size)
        tr, te = perm[:split], perm[split:]
        return data[tr], data[te], labels[tr], labels[te]

    # ── Kernel matrix helpers ──────────────────────────────────────────────

    @staticmethod
    def _pairwise(
        A: jnp.ndarray, B: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Expand (A, B) so that row ``i * M + j`` of the returned arrays
        equals ``(A[i], B[j])`` — covering all N×M ordered pairs.
        Mirrors PyTorch's  x_0 = a.repeat(M, 1)  /  x_1 = b.repeat_interleave(N, dim=0).
        """
        N, M = A.shape[0], B.shape[0]
        return jnp.repeat(A, M, axis=0), jnp.tile(B, (N, 1))

    def regular_kernel_matrix(
        self, weights, X: jnp.ndarray
    ) -> jnp.ndarray:
        """Full NxN kernel matrix using upper-triangular computation."""
        N = X.shape[0]

        # Get indices for upper triangular (including diagonal)
        iu, ju = jnp.triu_indices(N)

        # Gather pairs
        x1 = X[iu]
        x2 = X[ju]

        # Compute kernel values only for upper triangle
        k_vals = self.kernel_model.forward(x1, x2, weights)

        # Initialize full matrix
        K = jnp.zeros((N, N))

        # Fill upper triangle
        K = K.at[iu, ju].set(k_vals)

        # Mirror to lower triangle
        K = K.at[ju, iu].set(k_vals)

        return K

    def nystrom_kernel_matrix(
        self, weights, X: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Nyström approximation: K ≈ K_NM · K_MM⁻¹ · K_NM^T.

        The first ``landmark_points`` rows of X are used as landmarks.
        """
        M = self.landmark_points
        if not (0 < M <= len(X)):
            raise ValueError(
                f"landmark_points={M} is invalid for data of length {len(X)}"
            )
        N = X.shape[0]
        landmarks = X[:M]

        x1, x2 = self._pairwise(X, landmarks)
        KNM = self.kernel_model.forward(x1, x2, weights).reshape(N, M)

        x1, x2 = self._pairwise(landmarks, landmarks)
        KMM = self.kernel_model.forward(x1, x2, weights).reshape(M, M)
        KMM_inv = jnp.linalg.inv(KMM + 1e-8 * jnp.eye(M))

        return KNM @ KMM_inv @ KNM.T

    def test_kernel_matrix(
        self, weights, X_train: jnp.ndarray, X_test: jnp.ndarray
    ) -> jnp.ndarray:
        """M×N cross-kernel matrix between X_test (rows) and X_train (cols).

        Mirrors PyTorch:
            x_0 = test_data.repeat_interleave(N_train, dim=0)
            x_1 = train_data.repeat(N_test, 1)
            K   = kernel(x_0, x_1).reshape(N_test, N_train)
        """
        N, M = X_train.shape[0], X_test.shape[0]
        x1 = jnp.repeat(X_test, N, axis=0)
        x2 = jnp.tile(X_train, (M, 1))
        return self.kernel_model.forward(x1, x2, weights).reshape(M, N)

    def _kernel_matrix(self, weights, X: jnp.ndarray) -> jnp.ndarray:
        """Dispatch to the configured matrix type."""
        if self.matrix_type == "regular":
            return self.regular_kernel_matrix(weights, X)
        return self.nystrom_kernel_matrix(weights, X)

    def _apply_centering(self, K: jnp.ndarray) -> jnp.ndarray:
        """Apply kernel centering H·K·H if enabled, otherwise pass through."""
        if not self.centering:
            return K
        n = K.shape[0]
        H = jnp.eye(n) - jnp.ones((n, n)) / n
        return H @ K @ H

    # ── KTA (full-matrix variant, used by FullKTA / RandomKTA / GreedyKTA) ─

    def alignment(
        self, weights, X: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Kernel–Target Alignment between the (centered) kernel matrix and the
        label outer product.

        Matches PyTorch _loss_ta:
            yTKy / (sqrt(trace(K²)) * N)
        which equals  <K, T>_F / (||K||_F · ||T||_F)  for ±1 labels since
        ||T||_F = ||y||² = N.
        """
        K = self._apply_centering(self._kernel_matrix(weights, X))
        T = y[:, None] * y[None, :]          # label outer product — target kernel
        norm = jnp.linalg.norm(K, ord="fro") * jnp.linalg.norm(T, ord="fro")
        return jnp.sum(K * T) / (norm + 1e-10)

    def _loss_kta(
        self, weights, X: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        return 1.0 - self.alignment(weights, X, y)

    # ── SVM evaluation ─────────────────────────────────────────────────────

    def svm_training(
        self, X: jnp.ndarray, y: jnp.ndarray
    ) -> dict[str, Any]:
        """
        Fit an SVM with the current kernel and evaluate on train + test sets.

        Returns a dict with keys: svm, train_accuracy, test_accuracy,
        f1_score, precision_score, recall_score.
        """
        K_train = np.asarray(
            self._apply_centering(self._kernel_matrix(self.weights, X))
        )
        y_train_np = np.asarray(y)
        y_test_np  = np.asarray(self.ytest)

        svm = SVC(kernel="precomputed", C=1.0, probability=True, max_iter=10_000)
        svm.fit(K_train, y_train_np)

        K_test_raw = np.asarray(
            self.test_kernel_matrix(self.weights, self.xtrain, self.xtest)
        )
        if self.centering:
            n_train = K_train.shape[0]
            train_col_means = K_train.mean(axis=0, keepdims=True)
            train_mean      = K_train.mean()
            K_test = (
                K_test_raw
                - K_test_raw.mean(axis=1, keepdims=True)
                - train_col_means
                + train_mean
            )
        else:
            K_test = K_test_raw

        y_pred_train = svm.predict(K_train)
        y_pred_test  = svm.predict(K_test)

        return {
            "svm": svm,
            "train_accuracy":   float(accuracy_score(y_train_np, y_pred_train)),
            "test_accuracy":    float(accuracy_score(y_test_np,  y_pred_test)),
            "f1_score":         float(f1_score(y_test_np,        y_pred_test, average="macro")),
            "precision_score":  float(precision_score(y_test_np, y_pred_test, average="macro")),
            "recall_score":     float(recall_score(y_test_np,    y_pred_test, average="macro")),
        }

    # ── Abstract interface ─────────────────────────────────────────────────

    @abstractmethod
    def _get_batch(
        self, epoch: int
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return *(X_batch, y_batch)* for the current gradient step."""

    # ── Main training loop ─────────────────────────────────────────────────

    def align(self) -> dict[str, Any]:
        """
        Run KTA optimization and return a training history dictionary.

        History keys
        ------------
        weights, init_train_accuracy, init_test_accuracy,
        alignment_history, loss_history,
        train_accuracy_history, test_accuracy_history,
        f1_score_history, precision_score_history, recall_score_history,
        time, circuit_executions
        """
        init = self.svm_training(self.xtrain, self.ytrain)

        alignment_hist: list[float] = []
        loss_hist:      list[float] = []
        train_acc, test_acc, f1s, precs, recs = [], [], [], [], []

        start = time.perf_counter()
        desc  = f"[{type(self).__name__}] KTA alignment"

        for epoch in range(self.epochs):
            X_b, y_b = self._get_batch(epoch)

            loss_hist.append(float(self._loss_fn(self.weights, X_b, y_b)))
            alignment_hist.append(
                float(self.alignment(self.weights, self.xtrain, self.ytrain))
            )

            grads = self._grad_fn(self.weights, X_b, y_b)
            updates, self.opt_state = self._optimizer.update(grads, self.opt_state)
            self.weights = ox.apply_updates(self.weights, updates)

            result = self.svm_training(self.xtrain, self.ytrain)
            train_acc.append(result["train_accuracy"])
            test_acc.append(result["test_accuracy"])
            f1s.append(result["f1_score"])
            precs.append(result["precision_score"])
            recs.append(result["recall_score"])

        history: dict[str, Any] = {
            "weights":                  self.weights,
            "init_train_accuracy":      init["train_accuracy"],
            "init_test_accuracy":       init["test_accuracy"],
            "alignment_history":        alignment_hist,
            "loss_history":             loss_hist,
            "train_accuracy_history":   train_acc,
            "test_accuracy_history":    test_acc,
            "f1_score_history":         f1s,
            "precision_score_history":  precs,
            "recall_score_history":     recs,
            "time":                     time.perf_counter() - start,
            "circuit_executions":       self.kernel_model.circuit_executions,
        }
        return history


# ─────────────────────────────────────────────────────────────────────────────
# Concrete KTA strategies — gradient-based
# ─────────────────────────────────────────────────────────────────────────────

class FullKTA(BaseKTA):
    """
    Full-batch KTA: gradient is computed over the entire training set each epoch.
    Mirrors PyTorch  train_method='full'.
    """

    def _get_batch(self, epoch: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.xtrain, self.ytrain


class RandomKTA(BaseKTA):
    """
    Stochastic KTA: draw a random mini-batch each epoch.
    Mirrors PyTorch  train_method='random'.

    Parameters
    ----------
    random_samples : int
        Mini-batch size (default 4).
    """

    def __init__(self, *args, random_samples: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if random_samples <= 0:
            raise ValueError("random_samples must be > 0")
        self.random_samples = random_samples
        self._rng = jax.random.PRNGKey(0)
        self._perm: jnp.ndarray | None = None
        self._ptr: int = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self._rng, subkey = jax.random.split(self._rng)
        self._perm = jax.random.permutation(subkey, len(self.xtrain))
        self._ptr = 0

    def _get_batch(self, epoch: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self._ptr + self.random_samples > len(self.xtrain):
            self._reshuffle()
        idx = self._perm[self._ptr : self._ptr + self.random_samples]
        self._ptr += self.random_samples
        return self.xtrain[idx], self.ytrain[idx]


class GreedyKTA(BaseKTA):
    """
    Active-learning KTA: select the *k* most uncertain training points per epoch.

    Parameters
    ----------
    greedy_samples : int
        Number of high-uncertainty samples to select (default 4).
    """

    def __init__(self, *args, greedy_samples: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if greedy_samples <= 0:
            raise ValueError("greedy_samples must be > 0")
        self.greedy_samples = greedy_samples

    def _get_batch(self, epoch: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        K = self.nystrom_kernel_matrix(self.weights, self.xtrain)
        svm = SVC(kernel="precomputed", C=1.0, probability=True, max_iter=10000)
        svm.fit(K, np.asarray(self.ytrain))
        probs       = svm.predict_proba(K)[:, 1]
        uncertainty = 1.0 - np.abs(2.0 * probs - 1.0)
        k    = min(self.greedy_samples, len(self.xtrain))
        topk = np.argsort(-uncertainty)[:k]
        return self.xtrain[topk], self.ytrain[topk]


# ─────────────────────────────────────────────────────────────────────────────
# Centroid-based alternating optimizer — matches PyTorch train_method='ccka'
# ─────────────────────────────────────────────────────────────────────────────

class CentroidBasedKTA(BaseKTA):
    """
    Alternating centroid–kernel optimization (CCKA).
    Mirrors PyTorch  train_method='ccka' exactly.

    PyTorch optimizer structure (replicated here):
    ──────────────────────────────────────────────
    KAO step — ONE joint Adam across kernel weights + sub-centroids:

        optim.Adam([
            {'params': kernel.parameters(), 'lr': lr},
            {'params': class_centroids,     'lr': cclr},
        ])

    CO step — ONE Adam per main centroid (only that centroid updated):

        optim.Adam([{'params': main_centroids[cl], 'lr': mclr}])

    This is replicated using three separate optax optimizers:
      - _kao_weight_optimizer  (learning_rate)   ← kernel weights  (KAO only)
      - _kao_sub_optimizer     (sub_centroid_lr) ← sub-centroids   (KAO only)
      - _co_main_optimizer     (centroid_lr)     ← main centroids  (CO only)

    KAO loss: 1 − TA(K, Y_raw, l) + λ_kao · L2(weights)
      where K = kernel(main_centroid_cl, all_sub_centroids),
            Y_raw = raw sub-centroid labels,
            l = +current_label  (matches PyTorch centroid_target_alignment)

    CO loss:  1 − TA(K, Y_raw, l) + λ_co · Σ relu(c−1) + relu(−c)
      where l = −current_label  (flipped, matching PyTorch _loss_co call),
            box constraint is [0, 1] matching PyTorch relu(c−1)+relu(−c).

    Parameters
    ----------
    centroids : int
        Number of sub-centroids per class (default 4).
    clustering : {'regular', 'kmeans'}
        How to initialise sub-centroids.
    lambda_co : float
        Weight of the box-constraint regulariser in the CO loss.
    lambda_kao : float
        Weight of the L2 regulariser in the KAO loss.
    learning_rate : float
        Learning rate for kernel weight updates (maps to PyTorch lr).
    centroid_lr : float
        Learning rate for main centroid CO updates (maps to PyTorch mclr).
        Defaults to learning_rate.
    sub_centroid_lr : float
        Learning rate for sub-centroid KAO updates (maps to PyTorch cclr).
        Defaults to centroid_lr.
    """

    def __init__(
        self,
        kernel_model,
        data: jnp.ndarray,
        labels: jnp.ndarray,
        *,
        centroids: int = 4,
        clustering: str = "regular",
        lambda_co: float = 0.001,
        lambda_kao: float = 0.001,
        learning_rate: float = 0.01,
        centroid_lr: float | None = None,
        sub_centroid_lr: float | None = None,   # maps to PyTorch cclr
        **kwargs: Any,
    ) -> None:
        super().__init__(kernel_model, data, labels, learning_rate=learning_rate, **kwargs)

        self.n_centroids     = centroids
        self.use_kmeans      = clustering.lower() == "kmeans"
        self.lambda_co       = lambda_co
        self.lambda_kao      = lambda_kao
        self.n_classes       = int(jnp.unique(self.ytrain).shape[0])

        # Learning rates: kernel_lr | main_centroid_lr (mclr) | sub_centroid_lr (cclr)
        self._centroid_lr    = centroid_lr if centroid_lr is not None else learning_rate
        self._sub_centroid_lr = (
            sub_centroid_lr if sub_centroid_lr is not None else self._centroid_lr
        )

        (
            self.main_centroids,
            self.main_centroid_labels,
            self.sub_centroids,
            self.sub_centroid_labels,
        ) = self._compute_centroids(self.xtrain, self.ytrain)

        # ── Three separate optimizers matching PyTorch's optimizer structure ─
        #
        #   PyTorch KAO: optim.Adam([kernel_params@lr, class_centroids@cclr])
        #   → replicated with two separate optax optimizers that both step
        #     during the KAO update (joint gradient, separate states).
        #
        #   PyTorch CO: optim.Adam([main_centroid@mclr])
        #   → replicated with one optax optimizer on main_centroids only.

        # KAO — kernel weights
        self._kao_weight_optimizer = self._build_optimizer(self.learning_rate)
        self._kao_weight_opt_state = self._kao_weight_optimizer.init(self.weights)

        # KAO — sub-centroids (jointly updated with weights in KAO step)
        self._kao_sub_optimizer    = self._build_optimizer(self._sub_centroid_lr)
        self._kao_sub_opt_state    = self._kao_sub_optimizer.init(self.sub_centroids)

        # CO — main centroids only
        self._co_main_optimizer    = self._build_optimizer(self._centroid_lr)
        self._co_main_opt_state    = self._co_main_optimizer.init(self.main_centroids)

    # ── Centroid initialisation ────────────────────────────────────────────

    def _compute_centroids(
        self, X: jnp.ndarray, y: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Matches PyTorch _get_centroids:
          - main centroid  = class mean
          - sub-centroids  = KMeans cluster centres or equal-chunk means
        """
        X_np = np.asarray(X)
        y_np = np.asarray(y)

        unique_labels = np.unique(y_np)
        n_cls         = len(unique_labels)
        D             = X_np.shape[1]

        main_cents  = jnp.zeros((n_cls, D),                    dtype=jnp.float32)
        main_labels = jnp.zeros((n_cls,),                      dtype=jnp.float32)
        sub_cents   = jnp.zeros((n_cls * self.n_centroids, D), dtype=jnp.float32)
        sub_labels  = jnp.zeros((n_cls * self.n_centroids,),   dtype=jnp.float32)

        for ci, label in enumerate(unique_labels):
            class_data = jnp.array(X_np[y_np == label], dtype=jnp.float32)

            main_cents  = main_cents.at[ci].set(jnp.mean(class_data, axis=0))
            main_labels = main_labels.at[ci].set(float(label))

            if self.use_kmeans and class_data.shape[0] >= self.n_centroids:
                km = KMeans(
                    n_clusters=self.n_centroids, n_init="auto", random_state=42
                ).fit(np.asarray(class_data))
                sc = jnp.array(km.cluster_centers_, dtype=jnp.float32)
            else:
                chunks = jnp.array_split(class_data, self.n_centroids)
                sc = jnp.stack(
                    [jnp.mean(chunk, axis=0) for chunk in chunks]
                ).astype(jnp.float32)

            for si in range(self.n_centroids):
                idx = ci * self.n_centroids + si
                sub_cents  = sub_cents.at[idx].set(sc[si])
                sub_labels = sub_labels.at[idx].set(float(label))

        return main_cents, main_labels, sub_cents, sub_labels

    # ── Centroid-specific kernel (vector) ─────────────────────────────────

    def _centroid_kernel_vec(
        self,
        weights,
        main_centroid: jnp.ndarray,
        X: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute  K[i] = kernel(main_centroid, X[i])  for all i.

        Mirrors PyTorch:
            x_0 = main_centroid.repeat(X.shape[0], 1)
            x_1 = X
            K   = kernel(x_0, x_1)   # shape (N,)
        """
        N  = X.shape[0]
        x0 = jnp.repeat(main_centroid[None, :], N, axis=0)
        return self.kernel_model.forward(x0, X, weights).reshape(-1)

    # ── KAO loss — differentiable in both weights AND sub_centroids ────────

    def _loss_kao_cl(
        self,
        weights,
        sub_centroids: jnp.ndarray,   # explicit arg so JAX can diff w.r.t. it
        main_centroid: jnp.ndarray,
        y_raw: jnp.ndarray,           # raw sub-centroid labels (not ±1 converted)
        l: float = 1.0,
    ) -> jnp.ndarray:
        """
        KAO loss for the selected class:

            loss = 1 − TA(K, Y_raw, l) + λ_kao · L2(weights)

        K is the vector kernel(main_centroid, sub_centroids).
        Y_raw are the raw class labels of each sub-centroid.
        l = +current_label (matches PyTorch centroid_target_alignment).

        Both ``weights`` (argnums=0) and ``sub_centroids`` (argnums=1) are
        differentiable — matching PyTorch's joint Adam optimizer that updates
        kernel params and class_centroids together.
        """
        K = self._centroid_kernel_vec(weights, main_centroid, sub_centroids)
        Y = y_raw.astype(jnp.float32)

        numerator   = l * jnp.dot(K, Y)
        denominator = jnp.linalg.norm(K) * jnp.linalg.norm(Y)
        kta         = numerator / (denominator + 1e-10)

        leaves   = jax.tree_util.tree_leaves(weights)
        n_params = max(sum(leaf.size for leaf in leaves), 1)
        l2       = sum(jnp.sum(leaf ** 2) for leaf in leaves) / n_params

        return 1.0 - kta + self.lambda_kao * l2

    # ── CO loss — differentiable in main_centroids only ───────────────────

    def _loss_co_cl(
        self,
        weights,
        main_centroids: jnp.ndarray,
        cl: float,
        y_raw: jnp.ndarray,           # raw sub-centroid labels
        l: float = -1.0,              # −current_label (PyTorch convention)
    ) -> jnp.ndarray:
        """
        CO loss for the selected class:

            loss = 1 − TA(K, Y_raw, l) + λ_co · Σ relu(c−1) + relu(−c)

        K = kernel(main_centroid_cl, all_sub_centroids).
        Box constraint uses relu(c−1)+relu(−c) matching PyTorch [0,1] penalty.
        l = −current_label (flipped vs KAO, matching PyTorch _loss_co call).

        Only ``main_centroids`` (argnums=1) is differentiated — matching
        PyTorch's per-class Adam that updates only the relevant main centroid.
        sub_centroids are read from self.sub_centroids (treated as constants).
        """
        main_idx      = jnp.argmax(self.main_centroid_labels == cl)
        main_centroid = main_centroids[main_idx]

        # self.sub_centroids is a constant here — not in argnums
        K = self._centroid_kernel_vec(weights, main_centroid, self.sub_centroids)
        Y = y_raw.astype(jnp.float32)

        numerator   = l * jnp.dot(K, Y)
        denominator = jnp.linalg.norm(K) * jnp.linalg.norm(Y)
        kta         = numerator / (denominator + 1e-10)

        # [0,1] box constraint — matches PyTorch:
        #   relu(centroid - 1.0) + relu(-centroid)
        penalty = jnp.sum(
            jax.nn.relu(main_centroid - 1.0) + jax.nn.relu(-main_centroid)
        )

        return 1.0 - kta + self.lambda_co * penalty

    # ── Joint KAO update: weights + sub_centroids simultaneously ──────────

    def _kao_joint_update(
        self,
        main_centroid: jnp.ndarray,
        y_raw: jnp.ndarray,
        l: float = 1.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        One gradient step jointly updating kernel *weights* and *sub_centroids*.

        Matches PyTorch:
            optimizer = Adam([kernel_params@lr, class_centroids@cclr])
            loss_kao.backward()
            optimizer.step()

        Gradients are computed simultaneously w.r.t. both (argnums 0 and 1),
        then applied via their respective optax optimizers (separate lr support).

        Returns
        -------
        new_weights, new_sub_centroids
        """
        grad_fn = jax.grad(self._loss_kao_cl, argnums=(0, 1))
        grads_w, grads_sub = grad_fn(
            self.weights, self.sub_centroids, main_centroid, y_raw, l
        )

        # Update kernel weights (lr = learning_rate)
        updates_w, self._kao_weight_opt_state = self._kao_weight_optimizer.update(
            grads_w, self._kao_weight_opt_state
        )
        new_weights = ox.apply_updates(self.weights, updates_w)

        # Update sub-centroids (lr = sub_centroid_lr / cclr)
        updates_sub, self._kao_sub_opt_state = self._kao_sub_optimizer.update(
            grads_sub, self._kao_sub_opt_state
        )
        new_sub = ox.apply_updates(self.sub_centroids, updates_sub)

        return new_weights, new_sub

    # ── CO update: main centroid only ─────────────────────────────────────

    def _co_main_update(
        self,
        cl: float,
        y_raw: jnp.ndarray,
        l: float = -1.0,
    ) -> jnp.ndarray:
        """
        One gradient step updating *main_centroids* only for class ``cl``.

        Matches PyTorch:
            self._optimizers[_class].step()  # only main_centroid[cl]

        sub_centroids are NOT updated here (they are constants in _loss_co_cl).

        Returns
        -------
        new_main_centroids   (only [cl_idx] row actually moves)
        """
        grad_fn = jax.grad(self._loss_co_cl, argnums=1)
        grads_main = grad_fn(
            self.weights, self.main_centroids, cl, y_raw, l
        )

        updates_main, self._co_main_opt_state = self._co_main_optimizer.update(
            grads_main, self._co_main_opt_state
        )
        new_main = ox.apply_updates(self.main_centroids, updates_main)

        # Clip to [0, 1] — enforces the box constraint used in the CO loss penalty
        new_main = jnp.clip(new_main, 0.0, 1.0)
        return new_main

    # ── BaseKTA abstract method (fallback; align() is fully overridden) ────

    def _get_batch(self, epoch: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.xtrain, self.ytrain

    def kernel_pca(self,
        K,
        n_components=2,
        eps=1e-12
    ):

        eigvals, eigvecs = jnp.linalg.eigh(K)

        # Sort descending
        idx = jnp.argsort(eigvals)[::-1]

        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Keep top components
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]

        # Numerical stability
        eigvals = jnp.maximum(eigvals, eps)

        # ---------------------------------------
        # Construct embedding
        # ---------------------------------------

        X_embedded = eigvecs * jnp.sqrt(eigvals)

        return X_embedded

    # ── Custom training loop ───────────────────────────────────────────────

    def align(self) -> dict[str, Any]:
        """
        Alternating KAO / CO optimisation loop matching PyTorch fit_kernel
        with  train_method='ccka'.

        For each outer epoch:
        ├─ (1) Select class cl (cycles through unique labels)
        ├─ (2) 10× KAO steps — joint gradient on kernel *weights* AND
        │      *sub_centroids* simultaneously (matching PyTorch's single Adam
        │      with both param groups).  l = +current_label.
        ├─ (3) 10× CO steps — gradient on *main_centroids* only.
        │      sub_centroids are held fixed.  l = −current_label.
        └─ (4) SVM evaluation and history logging.
        """
        init = self.svm_training(self.xtrain, self.ytrain)
        alignment_hist: list[float] = []
        loss_hist:      list[float] = []
        train_acc, test_acc, f1s, precs, recs = [], [], [], [], []
        main_cent_hist: list[jnp.ndarray] = []
        sub_cent_hist:  list[jnp.ndarray] = []
        coords:  list[jnp.ndarray] = []

        unique_labels_np = np.unique(np.asarray(self.ytrain))
        n_cls            = len(unique_labels_np)

        best_test_acc       = -jnp.inf
        best_weights        = self.weights
        best_main_centroids = self.main_centroids
        best_sub_centroids  = self.sub_centroids

        # Raw sub-centroid labels (unchanged throughout training)
        y_raw = self.sub_centroid_labels

        start = time.perf_counter()
        for epoch in range(self.epochs):

            # ── (1) Select class ───────────────────────────────────────────
            cl_kao = unique_labels_np[epoch % n_cls]
            l_kao  = float(cl_kao)       # KAO: +current_label
            l_co   = -float(cl_kao)      # CO:  -current_label (PyTorch)

            main_idx      = int(jnp.argmax(self.main_centroid_labels == cl_kao))
            main_centroid = self.main_centroids[main_idx]


            self.weights, self.sub_centroids = self._kao_joint_update(
                    main_centroid, y_raw, l=l_kao
                )
                
            main_centroid = self.main_centroids[main_idx]

            # ── (3) 10× CO steps: update main_centroid only ────────────────
            #   Mirrors PyTorch:
            #     for nco in range(10):
            #         loss_co.backward(); self._optimizers[_class].step()

            self.main_centroids = self._co_main_update(
                    cl=cl_kao, y_raw=y_raw, l=l_co
                )

            # ── (4) Evaluation & history ───────────────────────────────────
            alignment_hist.append(
                float(self.alignment(self.weights, self.xtrain, self.ytrain))
            )
            main_cent_hist.append(self.main_centroids)
            sub_cent_hist.append(self.sub_centroids)

            result = self.svm_training(self.xtrain, self.ytrain)
            train_acc.append(result["train_accuracy"])
            test_acc.append(result["test_accuracy"])
            f1s.append(result["f1_score"])
            precs.append(result["precision_score"])
            recs.append(result["recall_score"])

            if result["test_accuracy"] > best_test_acc:
                best_test_acc       = result["test_accuracy"]
                best_weights        = self.weights
                best_main_centroids = self.main_centroids
                best_sub_centroids  = self.sub_centroids


            all_points = jnp.concatenate(
                [
                    self.xtrain,
                    self.main_centroids,
                    self.sub_centroids,
                ],
                axis=0,
            )
            all_labels = jnp.concatenate(
                [
                    self.ytrain,
                    self.main_centroid_labels,
                    self.sub_centroid_labels,
                ],
                axis=0,
            )
            K = np.asarray(self._apply_centering(self._kernel_matrix(self.weights, all_points)))
            coords.append(self.kernel_pca(K))

        # Restore best checkpoint
        self.weights        = best_weights
        self.main_centroids = best_main_centroids
        self.sub_centroids  = best_sub_centroids

        final_result = self.svm_training(self.xtrain, self.ytrain)
        train_acc.append(final_result["train_accuracy"])
        test_acc.append(final_result["test_accuracy"])
        f1s.append(final_result["f1_score"])
        precs.append(final_result["precision_score"])
        recs.append(final_result["recall_score"])

        history: dict[str, Any] = {
            "weights":                  self.weights,
            "main_centroids":           main_cent_hist,
            "sub_centroids":            sub_cent_hist,
            "init_train_accuracy":      init["train_accuracy"],
            "init_test_accuracy":       init["test_accuracy"],
            "alignment_history":        alignment_hist,
            "loss_history":             loss_hist,
            "train_accuracy_history":   train_acc,
            "test_accuracy_history":    test_acc,
            "best_test_accuracy":       float(best_test_acc),
            "final_svm_metrics":        final_result,
            "f1_score_history":         f1s,
            "precision_score_history":  precs,
            "recall_score_history":     recs,
            "time":                     time.perf_counter() - start,
            "circuit_executions":       self.kernel_model.circuit_executions,
            "best_main_centroids":      best_main_centroids,
            "best_sub_centroids":       best_sub_centroids,
            "coords":                   coords,
            "coords_labels":            all_labels,
            "xtrain":                   self.xtrain,
            "ytrain":                   self.ytrain,
        }
        return history


# ─────────────────────────────────────────────────────────────────────────────
# QUACK — full-data variant of CentroidBasedKTA, optax gradient descent
# ─────────────────────────────────────────────────────────────────────────────

class QuackKTA(BaseKTA):
    """
    QUACK: Quantum Alignment via Class Kernel optimisation.
    Direct JAX port of PyTorch  train_method='quack'.

    Identical in structure to CentroidBasedKTA **except** that the full
    training set is used in place of sub-centroids:

        x_0 = main_centroid.repeat(N_train, 1)
        x_1 = X_train                               ← all training data
        K   = kernel(x_0, x_1)                      ← shape (N_train,)

    This means:
    * **No sub-centroids** are allocated or optimised.
    * **KAO** uses optax gradient descent on kernel *weights* using the full
      training-set kernel vector (no joint sub-centroid update — matches
      PyTorch QUACK which only puts kernel.parameters() in kernel_optimizer).
    * **CO**  uses optax gradient descent on the *main centroid* for the
      selected class (training data is immutable).

    Parameters
    ----------
    lambda_co : float
        Weight of the box-constraint regulariser in the CO loss.
    lambda_kao : float
        Weight of the L2 regulariser in the KAO loss.
    centroid_lr : float
        Learning rate for centroid (CO) updates. Defaults to learning_rate.
    """

    def __init__(
        self,
        kernel_model,
        data: jnp.ndarray,
        labels: jnp.ndarray,
        *,
        lambda_co: float  = 0.01,
        lambda_kao: float = 0.001,
        centroid_lr: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(kernel_model, data, labels, **kwargs)

        self.lambda_co  = lambda_co
        self.lambda_kao = lambda_kao
        self.n_classes  = int(jnp.unique(self.ytrain).shape[0])
        self._centroid_lr = centroid_lr if centroid_lr is not None else self.learning_rate

        X_np = np.asarray(self.xtrain)
        self._feat_min = jnp.array(X_np.min(axis=0), dtype=jnp.float32)
        self._feat_max = jnp.array(X_np.max(axis=0), dtype=jnp.float32)

        self.main_centroids, self.main_centroid_labels = self._init_main_centroids()

        # KAO: kernel weights only (no sub-centroids in QUACK)
        self._kao_optimizer = self._build_optimizer(self.learning_rate)
        self._kao_opt_state = self._kao_optimizer.init(self.weights)

        # CO: main centroids only
        self._co_optimizer  = self._build_optimizer(self._centroid_lr)
        self._co_opt_state  = self._co_optimizer.init(self.main_centroids)

    # ── Centroid initialisation ────────────────────────────────────────────

    def _init_main_centroids(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """One main centroid per class = class mean."""
        X_np = np.asarray(self.xtrain)
        y_np = np.asarray(self.ytrain)
        unique_labels = np.unique(y_np)
        D = X_np.shape[1]

        main_cents  = jnp.zeros((len(unique_labels), D), dtype=jnp.float32)
        main_labels = jnp.zeros((len(unique_labels),),   dtype=jnp.float32)

        for ci, label in enumerate(unique_labels):
            class_data  = jnp.array(X_np[y_np == label], dtype=jnp.float32)
            main_cents  = main_cents.at[ci].set(jnp.mean(class_data, axis=0))
            main_labels = main_labels.at[ci].set(float(label))

        return main_cents, main_labels

    # ── Vector kernel (main centroid × full training set) ─────────────────

    def _quack_kernel_vec(
        self,
        weights,
        main_centroid: jnp.ndarray,
        X_train: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        K[i] = kernel(main_centroid, X_train[i])  for all i.

        Mirrors PyTorch QUACK:
            x_0 = main_centroid.repeat(N_train, 1)
            x_1 = training_data
            K   = kernel(x_0, x_1)
        """
        N  = X_train.shape[0]
        x0 = jnp.repeat(main_centroid[None, :], N, axis=0)
        return self.kernel_model.forward(x0, X_train, weights).reshape(-1)

    # ── KAO loss ───────────────────────────────────────────────────────────

    def _loss_kao_quack(
        self,
        weights,
        main_centroid: jnp.ndarray,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        l: float = 1.0,
    ) -> jnp.ndarray:
        """
        KAO loss for QUACK:

            loss = 1 − TA(K, Y, l) + λ_kao · L2(weights)

        K is computed over the full training set.
        """
        K = self._quack_kernel_vec(weights, main_centroid, X_train)
        Y = y_train.astype(jnp.float32)

        numerator   = l * jnp.dot(K, Y)
        denominator = jnp.linalg.norm(K) * jnp.linalg.norm(Y)
        kta         = numerator / (denominator + 1e-10)

        leaves   = jax.tree_util.tree_leaves(weights)
        n_params = max(sum(leaf.size for leaf in leaves), 1)
        l2       = sum(jnp.sum(leaf ** 2) for leaf in leaves) / n_params

        return 1.0 - kta + self.lambda_kao * l2

    # ── CO loss ────────────────────────────────────────────────────────────

    def _loss_co_quack(
        self,
        weights,
        main_centroids: jnp.ndarray,
        main_centroid_idx: int,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        l: float = -1.0,
    ) -> jnp.ndarray:
        """
        CO loss for QUACK:

            loss = 1 − TA(K, Y, l) + λ_co · box_penalty(main_centroid)

        Only the main centroid is penalised (no sub-centroids in QUACK).
        """
        main_c = main_centroids[main_centroid_idx]
        K      = self._quack_kernel_vec(weights, main_c, X_train)
        Y      = y_train.astype(jnp.float32)

        numerator   = l * jnp.dot(K, Y)
        denominator = jnp.linalg.norm(K) * jnp.linalg.norm(Y)
        kta         = numerator / (denominator + 1e-10)

        penalty = jnp.sum(
            jax.nn.relu(main_c - self._feat_max)
            + jax.nn.relu(self._feat_min - main_c)
        )
        return 1.0 - kta + self.lambda_co * penalty

    # ── Gradient-descent weight update (KAO) ──────────────────────────────

    def _kao_weight_update_quack(
        self,
        main_centroid: jnp.ndarray,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        l: float = 1.0,
    ) -> jnp.ndarray:
        grad_fn = jax.grad(self._loss_kao_quack, argnums=0)
        grads = grad_fn(self.weights, main_centroid, X_train, y_train, l)
        updates, self._kao_opt_state = self._kao_optimizer.update(
            grads, self._kao_opt_state
        )
        return ox.apply_updates(self.weights, updates)

    # ── Gradient-descent centroid update (CO) ─────────────────────────────

    def _main_centroid_gradient_update_quack(
        self,
        main_centroid_idx: int,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        l: float = -1.0,
    ) -> jnp.ndarray:
        grad_fn = jax.grad(self._loss_co_quack, argnums=1)
        grads = grad_fn(
            self.weights, self.main_centroids, main_centroid_idx, X_train, y_train, l
        )
        updates, self._co_opt_state = self._co_optimizer.update(
            grads, self._co_opt_state
        )
        new_main = ox.apply_updates(self.main_centroids, updates)
        return jnp.clip(new_main, self._feat_min, self._feat_max)

    # ── BaseKTA abstract method ────────────────────────────────────────────

    def _get_batch(self, epoch: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.xtrain, self.ytrain

    # ── Custom training loop ───────────────────────────────────────────────

    def align(self) -> dict[str, Any]:
        """
        QUACK alternating KAO / CO optimisation loop.
        Mirrors PyTorch fit_kernel with  train_method='quack'.
        """
        alignment_hist: list[float] = []
        loss_hist:      list[float] = []
        train_acc, test_acc, f1s, precs, recs = [], [], [], [], []
        main_cent_hist: list[jnp.ndarray] = []

        unique_labels_np = np.unique(np.asarray(self.ytrain))
        n_cls            = len(unique_labels_np)

        best_test_acc       = -jnp.inf
        best_weights        = self.weights
        best_main_centroids = self.main_centroids

        init = self.svm_training(self.xtrain, self.ytrain)
        start = time.perf_counter()
        for epoch in range(self.epochs):

            cl_kao = unique_labels_np[epoch % n_cls]
            l_kao  = float(cl_kao)
            l_co   = -float(cl_kao)

            main_idx      = int(jnp.argmax(self.main_centroid_labels == cl_kao))
            main_centroid = self.main_centroids[main_idx]

            y_kao = jnp.where(self.ytrain == cl_kao, 1.0, -1.0)

            if epoch % 2 == 0:
                for _ in range(10):
                    self.weights = self._kao_weight_update_quack(
                        main_centroid, self.xtrain, y_kao, l=l_kao
                    )

                    alignment_hist.append(
                        float(self.alignment(self.weights, self.xtrain, self.ytrain))
                    )
                    loss_hist.append(
                        float(self._loss_kao_quack(
                            self.weights,
                            self.main_centroids[main_idx],
                            self.xtrain,
                            y_kao,
                            l_kao,
                        ))
                    )
                    main_cent_hist.append(self.main_centroids)

                    result = self.svm_training(self.xtrain, self.ytrain)
                    train_acc.append(result["train_accuracy"])
                    test_acc.append(result["test_accuracy"])
                    f1s.append(result["f1_score"])
                    precs.append(result["precision_score"])
                    recs.append(result["recall_score"])

                    if result["test_accuracy"] > best_test_acc:
                        best_test_acc       = result["test_accuracy"]
                        best_weights        = self.weights
                        best_main_centroids = self.main_centroids

            else:
                for _ in range(10):
                    self.main_centroids = self._main_centroid_gradient_update_quack(
                        main_idx, self.xtrain, y_kao, l=l_co
                    )

                    alignment_hist.append(
                        float(self.alignment(self.weights, self.xtrain, self.ytrain))
                    )
                    loss_hist.append(
                        float(self._loss_kao_quack(
                            self.weights,
                            self.main_centroids[main_idx],
                            self.xtrain,
                            y_kao,
                            l_kao,
                        ))
                    )
                    main_cent_hist.append(self.main_centroids)

                    result = self.svm_training(self.xtrain, self.ytrain)
                    train_acc.append(result["train_accuracy"])
                    test_acc.append(result["test_accuracy"])
                    f1s.append(result["f1_score"])
                    precs.append(result["precision_score"])
                    recs.append(result["recall_score"])

                    if result["test_accuracy"] > best_test_acc:
                        best_test_acc       = result["test_accuracy"]
                        best_weights        = self.weights
                        best_main_centroids = self.main_centroids

            alignment_hist.append(
                float(self.alignment(self.weights, self.xtrain, self.ytrain))
            )
            loss_hist.append(
                float(self._loss_kao_quack(
                    self.weights,
                    self.main_centroids[main_idx],
                    self.xtrain,
                    y_kao,
                    l_kao,
                ))
            )
            main_cent_hist.append(self.main_centroids)

            result = self.svm_training(self.xtrain, self.ytrain)
            train_acc.append(result["train_accuracy"])
            test_acc.append(result["test_accuracy"])
            f1s.append(result["f1_score"])
            precs.append(result["precision_score"])
            recs.append(result["recall_score"])

            if result["test_accuracy"] > best_test_acc:
                best_test_acc       = result["test_accuracy"]
                best_weights        = self.weights
                best_main_centroids = self.main_centroids

        # Restore best checkpoint
        self.weights        = best_weights
        self.main_centroids = best_main_centroids

        final_result = self.svm_training(self.xtrain, self.ytrain)

        history: dict[str, Any] = {
            "weights":                  self.weights,
            "main_centroids":           main_cent_hist,
            "init_train_accuracy":      init["train_accuracy"],
            "init_test_accuracy":       init["test_accuracy"],
            "alignment_history":        alignment_hist,
            "loss_history":             loss_hist,
            "train_accuracy_history":   train_acc,
            "test_accuracy_history":    test_acc,
            "best_test_accuracy":       float(best_test_acc),
            "final_svm_metrics":        final_result,
            "f1_score_history":         f1s,
            "precision_score_history":  precs,
            "recall_score_history":     recs,
            "time":                     time.perf_counter() - start,
            "circuit_executions":       self.kernel_model.circuit_executions,
        }
        return history


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible lowercase aliases
# ─────────────────────────────────────────────────────────────────────────────

fullKTA          = FullKTA
randomKTA        = RandomKTA
greedyKTA        = GreedyKTA
centroidBasedKTA = CentroidBasedKTA
quackKTA         = QuackKTA

__all__ = [
    "BaseKTA",
    "FullKTA",
    "RandomKTA",
    "GreedyKTA",
    "CentroidBasedKTA",
    "QuackKTA",
    "print_training_summary",
    # aliases
    "fullKTA",
    "randomKTA",
    "greedyKTA",
    "centroidBasedKTA",
    "quackKTA",
]