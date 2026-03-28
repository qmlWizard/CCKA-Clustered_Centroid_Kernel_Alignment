"""
Kernel Target Alignment (KTA) optimizers for quantum kernel methods.

This module provides five strategies for aligning a parameterized quantum kernel
to a classification task via gradient-based or analytical KTA maximization:

    FullKTA           – gradient computed on the entire training set each epoch
    RandomKTA         – stochastic mini-batch sampling each epoch
    GreedyKTA         – active-learning selection of the most uncertain samples
    CentroidBasedKTA  – alternating analytical optimization of kernel weights and centroids
    QuackKTA          – QUACK strategy: uses full training data instead of sub-centroids
                        (direct port of the PyTorch 'quack' method)

All strategies share a common abstract base (BaseKTA) that houses kernel matrix
construction, SVM evaluation, centering, and the main training loop.

Kernel computation alignment with PyTorch TrainModel:
  - KAO loss uses  kernel(main_centroid, sub_centroids)  [vector], not a self-matrix
  - Centroid alignment includes the class-label multiplier  l  matching
    PyTorch's  centroid_target_alignment(K, Y, l)
  - CO loss penalises only the relevant main centroid (matching PyTorch)
  - CO step uses the flipped label  –l  (matching  -current_label  in PyTorch)
  - CentroidBasedKTA and QuackKTA retain the analytical (arctan2 / finite-diff)
    parameter update rules — these are NOT changed.

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
        """Full N×N kernel matrix for a single dataset X."""
        N = X.shape[0]
        x1, x2 = self._pairwise(X, X)
        return self.kernel_model.forward(x1, x2, weights).reshape(N, N)

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

        # Centre the test kernel relative to the training kernel's column means —
        # statistically correct for kernel centering at inference time.
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

        for epoch in tqdm(range(self.epochs), desc=desc):
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
        print_training_summary(history)
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

    Uses a shuffled permutation that resets when exhausted, ensuring every
    sample is eventually seen without replacement within each pass.

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

    Uncertainty is measured by SVM margin — points closest to the decision
    boundary (``|2p − 1|`` small) are the most informative.

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
        K = np.asarray(
            self._apply_centering(self._kernel_matrix(self.weights, self.xtrain))
        )
        svm = SVC(kernel="precomputed", C=1.0, probability=True, max_iter=10_000)
        svm.fit(K, np.asarray(self.ytrain))

        probs       = svm.predict_proba(K)[:, 1]
        uncertainty = 1.0 - np.abs(2.0 * probs - 1.0)
        k    = min(self.greedy_samples, len(self.xtrain))
        topk = np.argsort(-uncertainty)[:k]
        return self.xtrain[topk], self.ytrain[topk]


# ─────────────────────────────────────────────────────────────────────────────
# Centroid-based alternating optimizer — analytical updates
# ─────────────────────────────────────────────────────────────────────────────

class CentroidBasedKTA(BaseKTA):
    """
    Alternating centroid–kernel optimization (CCKA).
    Mirrors PyTorch  train_method='ccka'.

    Each outer epoch alternates between:

    * **KAO inner step** – analytically update kernel *weights* to maximise
      KTA computed as  kernel(main_centroid_cl, sub_centroids)  (a vector,
      matching PyTorch's  x_0 = main_centroid.repeat(N,1);  x_1 = centroids).
      L2 regularisation is applied to the weights.

    * **CO inner step** – analytically update *centroid positions* (main and
      sub) to maximise the same centroid-KTA, with the class label *flipped*
      (``-cl_kao``) matching PyTorch's  ``-current_label`` in the CO loss.
      A box-constraint regulariser keeps centroid values within the per-feature
      min/max of the training data.

    The update rules themselves (arctan2 for weights, finite-diff Newton for
    centroids) are the analytical approach and are preserved exactly.

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
    eps : float
        Finite-difference step size for centroid gradient estimation.
    alpha : float
        Newton step damping factor for centroid updates.
    """

    def __init__(
        self,
        kernel_model,
        data: jnp.ndarray,
        labels: jnp.ndarray,
        *,
        centroids: int = 4,
        clustering: str = "regular",
        lambda_co: float = 0.01,
        lambda_kao: float = 0.001,
        eps: float = 0.01,
        alpha: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(kernel_model, data, labels, **kwargs)

        self.n_centroids  = centroids
        self.use_kmeans   = clustering.lower() == "kmeans"
        self.lambda_co    = lambda_co
        self.lambda_kao   = lambda_kao
        self.n_classes    = int(jnp.unique(self.ytrain).shape[0])
        self.eps          = eps
        self.alpha        = alpha

        # Per-feature min/max of training data — used in the CO box constraint
        # so centroids are kept within the actual data range, not a hard [0, 1].
        X_np = np.asarray(self.xtrain)
        self._feat_min = jnp.array(X_np.min(axis=0), dtype=jnp.float32)
        self._feat_max = jnp.array(X_np.max(axis=0), dtype=jnp.float32)

        (
            self.main_centroids,
            self.main_centroid_labels,
            self.sub_centroids,
            self.sub_centroid_labels,
        ) = self._compute_centroids(self.xtrain, self.ytrain)

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

        main_cents  = jnp.zeros((n_cls, D),                   dtype=jnp.float32)
        main_labels = jnp.zeros((n_cls,),                     dtype=jnp.float32)
        sub_cents   = jnp.zeros((n_cls * self.n_centroids, D), dtype=jnp.float32)
        sub_labels  = jnp.zeros((n_cls * self.n_centroids,),  dtype=jnp.float32)

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

    # ── Centroid-alignment (vector KTA) ───────────────────────────────────

    def _centroid_alignment(
        self,
        weights,
        main_centroids: jnp.ndarray,
        sub_centroids: jnp.ndarray,
        cl: float,
        l: float = 1.0,
    ) -> jnp.ndarray:
        """
        Centroid-level KTA matching PyTorch centroid_target_alignment(K, Y, l):

            TA = l · dot(K, Y) / (||K|| · ||Y||)

        where
            K[i] = kernel(main_centroid_cl, sub_centroid_i)
            Y[i] = +1  if  sub_centroid_label[i] == cl,  else  −1

        Parameters
        ----------
        cl : float
            The class whose main centroid is used as the reference point.
        l : float
            Class-label multiplier (matches PyTorch's ``l`` parameter).
            Pass  +current_label  for KAO,  -current_label  for CO.
        """
        main_idx       = jnp.where(self.main_centroid_labels == cl)[0][0]
        main_centroid  = main_centroids[main_idx]

        K = self._centroid_kernel_vec(weights, main_centroid, sub_centroids)
        Y = jnp.where(self.sub_centroid_labels == cl, 1.0, -1.0)

        numerator   = l * jnp.dot(K, Y)
        denominator = jnp.linalg.norm(K) * jnp.linalg.norm(Y)
        return numerator / (denominator + 1e-10)

    # ── KAO loss (kernel weights) ──────────────────────────────────────────

    def _loss_kao_cl(
        self,
        weights,
        main_centroid: jnp.ndarray,
        X_cl: jnp.ndarray,
        y_cl: jnp.ndarray,
        l: float = 1.0,
    ) -> jnp.ndarray:
        """
        KAO loss for the selected class — matches PyTorch _loss_kao(K, Y, cl):

            loss = 1 − TA + λ_kao · L2(weights)

        Kernel is computed as a vector:
            K[i] = kernel(main_centroid, X_cl[i])
        matching PyTorch's  x_0 = main_centroid.repeat(N,1);  x_1 = class_centroids.

        Parameters
        ----------
        main_centroid : jnp.ndarray, shape (D,)
            Main centroid of the current class.
        X_cl : jnp.ndarray, shape (M, D)
            Sub-centroids of the current class.
        y_cl : jnp.ndarray, shape (M,)
            ±1 labels for X_cl.
        l : float
            Class-label multiplier (matches PyTorch's  current_label  argument).
        """
        K = self._centroid_kernel_vec(weights, main_centroid, X_cl)
        Y = y_cl.astype(jnp.float32)

        numerator   = l * jnp.dot(K, Y)
        denominator = jnp.linalg.norm(K) * jnp.linalg.norm(Y)
        kta         = numerator / (denominator + 1e-10)

        leaves   = jax.tree_util.tree_leaves(weights)
        n_params = max(sum(leaf.size for leaf in leaves), 1)
        l2       = sum(jnp.sum(leaf ** 2) for leaf in leaves) / n_params

        return 1.0 - kta + self.lambda_kao * l2

    # ── CO loss (centroid positions) ───────────────────────────────────────

    def _loss_co(
        self,
        weights,
        main_centroids: jnp.ndarray,
        sub_centroids: jnp.ndarray,
        cl: float,
        l: float = 1.0,
    ) -> jnp.ndarray:
        """
        CO loss — matches PyTorch _loss_co(K, Y, centroid, cl=-current_label):

            loss = 1 − TA + λ_co · box_penalty(main_centroid_cl)

        Only the main centroid for class ``cl`` is regularised, matching
        PyTorch which penalises  self._main_centroids[_class]  only.

        Box constraint uses per-feature training-data min/max rather than the
        hard [0, 1] in PyTorch — keeps centroids within the actual data range.

        Parameters
        ----------
        cl : float
            Class label for which the main centroid is selected.
        l : float
            Class-label multiplier; pass  -current_label  for the CO step to
            match PyTorch's flipped label.
        """
        kta = self._centroid_alignment(weights, main_centroids, sub_centroids, cl, l)

        # Penalise only the relevant main centroid (matches PyTorch behaviour)
        main_idx = jnp.where(self.main_centroid_labels == cl)[0][0]
        main_c   = main_centroids[main_idx]
        penalty  = jnp.sum(
            jax.nn.relu(main_c - self._feat_max)
            + jax.nn.relu(self._feat_min - main_c)
        )
        return 1.0 - kta + self.lambda_co * penalty

    # ── Analytical weight update (KAO) ────────────────────────────────────
    # Analytical approach — do NOT modify the update rule.

    def _kao_parameter_update(
        self,
        main_centroid: jnp.ndarray,
        X_cl: jnp.ndarray,
        y_cl: jnp.ndarray,
        l: float = 1.0,
    ) -> jnp.ndarray:
        """
        Analytical kernel-weight update using an arctan2-based rule.

        Evaluates the KAO loss at three parameter offsets (0, π/2, π) per
        parameter index and solves for the optimal shift analytically.
        Matches the original analytical approach — update rule is unchanged.

        Parameters
        ----------
        main_centroid : jnp.ndarray, shape (D,)
            Main centroid for the current class (passed through to _loss_kao_cl).
        X_cl : jnp.ndarray
            Sub-centroids for the current class.
        y_cl : jnp.ndarray
            ±1 labels for X_cl.
        l : float
            Class-label multiplier forwarded to _loss_kao_cl.
        """
        param_shape = self.weights.shape

        for idx in np.ndindex(param_shape):
            param_plus_delta  = self.weights.at[idx].add(jnp.pi)
            param_minus_delta = self.weights.at[idx].add(jnp.pi / 2)

            loss_pi        = self._loss_kao_cl(param_plus_delta,  main_centroid, X_cl, y_cl, l)
            loss_pi_over_2 = self._loss_kao_cl(param_minus_delta, main_centroid, X_cl, y_cl, l)
            loss_zero      = self._loss_kao_cl(self.weights,       main_centroid, X_cl, y_cl, l)

            numer = 2 * loss_pi_over_2 - loss_pi - loss_zero
            denom = loss_pi - loss_zero

            delta   = jnp.arctan2(numer, denom + 1e-10)
            new_val = self.weights[idx] - delta
            self.weights = self.weights.at[idx].set(new_val)

        return self.weights

    # ── Analytical centroid update (CO) ───────────────────────────────────
    # Analytical approach — do NOT modify the update rule.

    def _centroid_update(
        self,
        main_centroids: jnp.ndarray,
        sub_centroids: jnp.ndarray,
        y_sub: jnp.ndarray,
        eps: float = 0.01,
        cl_kao: float | None = None,
        mask_kao: np.ndarray | None = None,
        l: float = -1.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Analytical centroid update via finite-difference Newton steps.

        Passes ``l`` (flipped class label) to ``_loss_co``, matching PyTorch's
        ``-current_label`` in the CO step.
        Analytical update rule is unchanged.

        Parameters
        ----------
        l : float
            Class-label multiplier for _loss_co; should be ``-cl_kao`` to match
            PyTorch's  loss_co(..., -current_label).
        """
        main_centroids = self.main_centroids
        sub_centroids  = self.sub_centroids

        main_idx    = np.where(self.main_centroid_labels == cl_kao)[0][0]
        sub_indices = np.where(mask_kao)[0]

        # ── Update main centroid ───────────────────────────────────────────
        for idx in np.ndindex(self.main_centroids[main_idx].shape):
            feat_idx = idx[0]

            def loss_fn_main(val):
                mc_updated = self.main_centroids.at[main_idx, feat_idx].set(val)
                return -self._loss_co(self.weights, mc_updated, self.sub_centroids, cl_kao, l)

            current_val = self.main_centroids[main_idx, feat_idx]

            L_plus  = loss_fn_main(current_val + eps)
            L_minus = loss_fn_main(current_val - eps)
            L0      = loss_fn_main(current_val)

            grad = (L_plus - L_minus) / (2 * eps)
            hess = (L_plus - 2 * L0 + L_minus) / (eps ** 2 + 1e-10)

            if jnp.abs(hess) < 1e-6 or jnp.isnan(hess) or jnp.isnan(grad):
                step = -self.alpha * grad
            else:
                step = -self.alpha * grad / (jnp.abs(hess) + 1e-10)

            new_val = current_val + step
            new_val = jnp.clip(new_val, self._feat_min[feat_idx], self._feat_max[feat_idx])

            global_L0 = -self._loss_co(
                self.weights, self.main_centroids, self.sub_centroids, cl_kao, l
            )
            if loss_fn_main(new_val) < global_L0:
                self.main_centroids = self.main_centroids.at[main_idx, feat_idx].set(new_val)

        # ── Update sub-centroids ───────────────────────────────────────────
        for si in sub_indices:
            for idx in np.ndindex(self.sub_centroids[si].shape):
                feat_idx = idx[0]

                def loss_fn_sub(val):
                    sc_updated = self.sub_centroids.at[si, feat_idx].set(val)
                    return self._loss_co(
                        self.weights, self.main_centroids, sc_updated, cl_kao, l
                    )

                current_val = self.sub_centroids[si, feat_idx]

                L_plus  = loss_fn_sub(current_val + eps)
                L_minus = loss_fn_sub(current_val - eps)
                L0      = loss_fn_sub(current_val)

                grad = (L_plus - L_minus) / (2 * eps)
                hess = (L_plus - 2 * L0 + L_minus) / (eps ** 2 + 1e-10)

                if jnp.abs(hess) < 1e-6 or jnp.isnan(hess) or jnp.isnan(grad):
                    step = self.alpha * grad
                else:
                    step = self.alpha * grad / (jnp.abs(hess) + 1e-10)

                new_val = current_val + step
                new_val = jnp.clip(new_val, self._feat_min[feat_idx], self._feat_max[feat_idx])

                global_L0 = self._loss_co(
                    self.weights, self.main_centroids, self.sub_centroids, cl_kao, l
                )
                if loss_fn_sub(new_val) < global_L0:
                    self.sub_centroids = self.sub_centroids.at[si, feat_idx].set(new_val)

        return self.main_centroids, self.sub_centroids

    def get_circuit_executions(self) -> int:
        self.kernel_model.circuit_executions = 0
        return self.kernel_model.circuit_executions

    # ── BaseKTA abstract method (fallback; align() is fully overridden) ────

    def _get_batch(self, epoch: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.xtrain, self.ytrain

    # ── Custom training loop ───────────────────────────────────────────────

    def align(self) -> dict[str, Any]:
        """
        Alternating KAO / CO optimisation loop matching PyTorch fit_kernel
        with  train_method='ccka':

        For each outer epoch:
        ├─ (1) Select class  cl  (cycles through unique labels)
        ├─ (2) KAO step  — analytically update kernel *weights* using
        │      kernel(main_centroid_cl, sub_centroids)  with  l = +cl_kao.
        ├─ (3) CO step (every 2 epochs)  — analytically update centroid
        │      positions using the FLIPPED label  l = −cl_kao, matching
        │      PyTorch's  loss_co(..., −current_label).
        └─ (4) SVM evaluation and history logging.

        In addition to the standard history keys the returned dict also
        contains ``main_centroids`` and ``sub_centroids`` (per-epoch snapshots).
        """
        init = self.svm_training(self.xtrain, self.ytrain)
        alignment_hist: list[float] = []
        loss_hist:      list[float] = []
        train_acc, test_acc, f1s, precs, recs = [], [], [], [], []
        main_cent_hist: list[jnp.ndarray] = []
        sub_cent_hist:  list[jnp.ndarray] = []

        sc_labels_np       = np.asarray(self.sub_centroid_labels)
        unique_labels_np   = np.unique(np.asarray(self.ytrain))
        n_cls              = len(unique_labels_np)

        best_test_acc      = -jnp.inf
        best_weights       = self.weights
        best_main_centroids = self.main_centroids
        best_sub_centroids  = self.sub_centroids

        start = time.perf_counter()
        for epoch in tqdm(range(self.epochs), desc="[CentroidBasedKTA] KTA alignment"):

            # ── Select class for this epoch ────────────────────────────────
            cl_kao = unique_labels_np[epoch % n_cls]
            l_kao  = float(cl_kao)          # KAO uses +current_label
            l_co   = -float(cl_kao)         # CO  uses -current_label (PyTorch convention)

            mask_kao    = sc_labels_np == cl_kao
            X_cl        = jnp.array(np.asarray(self.sub_centroids)[mask_kao])
            y_cl        = jnp.where(sc_labels_np[mask_kao] == cl_kao, 1.0, -1.0)

            # Main centroid for current class
            main_idx      = int(jnp.where(self.main_centroid_labels == cl_kao)[0][0])
            main_centroid = self.main_centroids[main_idx]

            # ── KAO step: update kernel weights ───────────────────────────
            self.weights = self._kao_parameter_update(main_centroid, X_cl, y_cl, l=l_kao)

            # ── CO step: update centroid positions (every 2 epochs) ───────
            if epoch % 2 == 0:
                y_sub = jnp.where(self.sub_centroid_labels == cl_kao, 1.0, -1.0)
                self.main_centroids, self.sub_centroids = self._centroid_update(
                    self.main_centroids,
                    self.sub_centroids,
                    y_sub,
                    eps=self.eps,
                    cl_kao=cl_kao,
                    mask_kao=mask_kao,
                    l=l_co,         # flipped label for CO — matches PyTorch
                )

            alignment_hist.append(
                float(self.alignment(self.weights, self.xtrain, self.ytrain))
            )
            loss_hist.append(
                float(self._loss_kao_cl(self.weights, main_centroid, X_cl, y_cl, l_kao))
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
        }
        print_training_summary(history)
        return history


# ─────────────────────────────────────────────────────────────────────────────
# QUACK — full-data variant of CentroidBasedKTA
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
    * **KAO** analytically updates kernel *weights* using the full training
      set kernel vector.
    * **CO**  analytically updates the *main centroid* for the selected class
      (sub-centroids = training data, immutable).

    Both the arctan2 weight update and finite-diff centroid update rules are
    the same analytical approach as CentroidBasedKTA.

    Parameters
    ----------
    lambda_co : float
        Weight of the box-constraint regulariser in the CO loss.
    lambda_kao : float
        Weight of the L2 regulariser in the KAO loss.
    eps : float
        Finite-difference step size for centroid gradient estimation.
    alpha : float
        Newton step damping factor for centroid updates.
    """

    def __init__(
        self,
        kernel_model,
        data: jnp.ndarray,
        labels: jnp.ndarray,
        *,
        lambda_co: float  = 0.01,
        lambda_kao: float = 0.001,
        eps: float        = 0.01,
        alpha: float      = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(kernel_model, data, labels, **kwargs)

        self.lambda_co  = lambda_co
        self.lambda_kao = lambda_kao
        self.eps        = eps
        self.alpha      = alpha
        self.n_classes  = int(jnp.unique(self.ytrain).shape[0])

        X_np = np.asarray(self.xtrain)
        self._feat_min = jnp.array(X_np.min(axis=0), dtype=jnp.float32)
        self._feat_max = jnp.array(X_np.max(axis=0), dtype=jnp.float32)

        self.main_centroids, self.main_centroid_labels = self._init_main_centroids()

    # ── Centroid initialisation ────────────────────────────────────────────

    def _init_main_centroids(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        One main centroid per class = class mean.
        Matches PyTorch _get_centroids (main centroid part only).
        """
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
        KAO loss for QUACK — matches PyTorch _loss_kao(K, training_labels, current_label):

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
        CO loss for QUACK — matches PyTorch _loss_co(K, training_labels, main_centroid, -current_label):

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

    # ── Analytical weight update (KAO) ────────────────────────────────────
    # Analytical approach — do NOT modify the update rule.

    def _kao_parameter_update_quack(
        self,
        main_centroid: jnp.ndarray,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        l: float = 1.0,
    ) -> jnp.ndarray:
        """
        Analytical kernel-weight update (arctan2 rule) using the full training set.
        Same analytical approach as CentroidBasedKTA._kao_parameter_update.
        """
        param_shape = self.weights.shape

        for idx in np.ndindex(param_shape):
            param_plus_delta  = self.weights.at[idx].add(jnp.pi)
            param_minus_delta = self.weights.at[idx].add(jnp.pi / 2)

            loss_pi        = self._loss_kao_quack(param_plus_delta,  main_centroid, X_train, y_train, l)
            loss_pi_over_2 = self._loss_kao_quack(param_minus_delta, main_centroid, X_train, y_train, l)
            loss_zero      = self._loss_kao_quack(self.weights,       main_centroid, X_train, y_train, l)

            numer = 2 * loss_pi_over_2 - loss_pi - loss_zero
            denom = loss_pi - loss_zero

            delta   = jnp.arctan2(numer, denom + 1e-10)
            new_val = self.weights[idx] - delta
            self.weights = self.weights.at[idx].set(new_val)

        return self.weights

    # ── Analytical centroid update (CO) ───────────────────────────────────
    # Analytical approach — do NOT modify the update rule.

    def _main_centroid_update_quack(
        self,
        main_centroid_idx: int,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        l: float = -1.0,
    ) -> jnp.ndarray:
        """
        Analytical main-centroid update (finite-diff Newton) for QUACK.
        Only main centroids are updated — training data is immutable.
        Same analytical approach as CentroidBasedKTA._centroid_update (main part).

        Parameters
        ----------
        l : float
            Flipped class-label multiplier for CO (matches PyTorch's -current_label).
        """
        for idx in np.ndindex(self.main_centroids[main_centroid_idx].shape):
            feat_idx = idx[0]

            def loss_fn(val):
                mc_updated = self.main_centroids.at[main_centroid_idx, feat_idx].set(val)
                return self._loss_co_quack(
                    self.weights, mc_updated, main_centroid_idx, X_train, y_train, l
                )

            current_val = self.main_centroids[main_centroid_idx, feat_idx]

            L_plus  = loss_fn(current_val + self.eps)
            L_minus = loss_fn(current_val - self.eps)
            L0      = loss_fn(current_val)

            grad = (L_plus - L_minus) / (2 * self.eps)
            hess = (L_plus - 2 * L0 + L_minus) / (self.eps ** 2 + 1e-10)

            if jnp.abs(hess) < 1e-6 or jnp.isnan(hess) or jnp.isnan(grad):
                step = -self.alpha * grad
            else:
                step = -self.alpha * grad / (jnp.abs(hess) + 1e-10)

            new_val = current_val + step
            new_val = jnp.clip(new_val, self._feat_min[feat_idx], self._feat_max[feat_idx])

            if loss_fn(new_val) < L0:
                self.main_centroids = self.main_centroids.at[
                    main_centroid_idx, feat_idx
                ].set(new_val)

        return self.main_centroids

    # ── BaseKTA abstract method ────────────────────────────────────────────

    def _get_batch(self, epoch: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.xtrain, self.ytrain

    # ── Custom training loop ───────────────────────────────────────────────

    def align(self) -> dict[str, Any]:
        """
        QUACK alternating KAO / CO optimisation loop.
        Mirrors PyTorch fit_kernel with  train_method='quack':

        For each outer epoch:
        ├─ (1) Select class  cl  (cycles through unique labels)
        ├─ (2) KAO step — analytically update kernel *weights* using
        │      kernel(main_centroid_cl, X_train)  with  l = +current_label.
        ├─ (3) CO step (every 2 epochs) — analytically update *main_centroid_cl*
        │      using  l = -current_label  (flipped, matching PyTorch).
        └─ (4) SVM evaluation and history logging.
        """
        init = self.svm_training(self.xtrain, self.ytrain)
        alignment_hist: list[float] = []
        loss_hist:      list[float] = []
        train_acc, test_acc, f1s, precs, recs = [], [], [], [], []
        main_cent_hist: list[jnp.ndarray] = []

        unique_labels_np = np.unique(np.asarray(self.ytrain))
        n_cls            = len(unique_labels_np)

        best_test_acc       = -jnp.inf
        best_weights        = self.weights
        best_main_centroids = self.main_centroids

        start = time.perf_counter()
        for epoch in tqdm(range(self.epochs), desc="[QuackKTA] KTA alignment"):

            # ── Select class and compute labels ───────────────────────────
            cl_kao = unique_labels_np[epoch % n_cls]
            l_kao  = float(cl_kao)          # KAO: +current_label
            l_co   = -float(cl_kao)         # CO:  -current_label (PyTorch)

            main_idx      = int(jnp.where(self.main_centroid_labels == cl_kao)[0][0])
            main_centroid = self.main_centroids[main_idx]

            # y for this class: +1 for same class, -1 for others
            y_kao = jnp.where(self.ytrain == cl_kao, 1.0, -1.0)

            # ── KAO step: update kernel weights ───────────────────────────
            self.weights = self._kao_parameter_update_quack(
                main_centroid, self.xtrain, y_kao, l=l_kao
            )

            # ── CO step: update main centroid (every 2 epochs) ────────────
            if epoch % 2 == 0:
                self.main_centroids = self._main_centroid_update_quack(
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
        print_training_summary(history)
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