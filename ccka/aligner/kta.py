"""
Kernel Target Alignment (KTA) optimizers for quantum kernel methods.

This module provides four strategies for aligning a parameterized quantum kernel
to a classification task via gradient-based KTA maximization:

    FullKTA           – gradient computed on the entire training set each epoch
    RandomKTA         – stochastic mini-batch sampling each epoch
    GreedyKTA         – active-learning selection of the most uncertain samples
    CentroidBasedKTA  – alternating optimization of kernel weights and centroids

All strategies share a common abstract base (BaseKTA) that houses kernel matrix
construction, SVM evaluation, centering, and the main training loop.

Backward-compatible lowercase aliases (fullKTA, randomKTA, greedyKTA,
centroidBasedKTA) are exported at module level.
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
            # FIX 3: loss_history no longer pre-seeded, so len() == epochs run
            f"Epochs run          : ",
            f"Total training time : {history['time']:.2f} s",
        ],
        width,
    )
    _print_box(
        "ACCURACY METRICS",
        [
            f"Initial train accuracy : {history['init_train_accuracy']:.4f}",
            f"Final   train accuracy : {history['train_accuracy_history'][-1]:.4f}",
            f"Best    train accuracy : {max(history['train_accuracy_history']):.4f}",
            f"Initial test  accuracy : {history['init_test_accuracy']:.4f}",
            f"Final   test  accuracy : {history['test_accuracy_history'][-1]:.4f}",
            f"Best    test  accuracy : {max(history['test_accuracy_history']):.4f}",
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
            f"Final alignment    : {history['alignment_history'][-1]:.6f}",
            f"Best  alignment    : {max(history['alignment_history']):.6f}",
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
        # Absorb any extra kwargs so subclasses can forward **kwargs freely
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
        self._grad_fn = jax.jit(jax.grad(self._loss_kta))  # grad w.r.t. weights

    # ── Optimizer factory ──────────────────────────────────────────────────

    def _build_optimizer(self, lr: float) -> ox.GradientTransformation:
        if self.optimizer_name not in self._OPTIMIZERS:
            raise ValueError(
                f"Optimizer {self.optimizer_name!r} not supported. "
                f"Choose from: {list(self._OPTIMIZERS)}"
            )
        return self._OPTIMIZERS[self.optimizer_name](lr)

    # ── Data splitting ─────────────────────────────────────────────────────

    # FIX 1: Removed the dead @staticmethod placeholder that was silently
    # overridden and returned (data, labels) unchanged — a footgun if ever called.
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
        """M×N cross-kernel matrix between X_test (rows) and X_train (cols)."""
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

    # ── KTA ───────────────────────────────────────────────────────────────

    def alignment(
        self, weights, X: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        """Kernel–Target Alignment between kernel matrix and label outer product."""
        K = self._apply_centering(self._kernel_matrix(weights, X))
        T = y[:, None] * y[None, :]  # label outer product — target kernel
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
        # FIX 2: was hardcoded to self.ytrain, ignoring the passed-in y argument.
        y_train_np = np.asarray(y)
        y_test_np = np.asarray(self.ytest)

        svm = SVC(kernel="precomputed", C=1.0, probability=True, max_iter=10_000)
        svm.fit(K_train, y_train_np)

        # FIX 5: Centre the test kernel relative to the training kernel's column
        # means rather than the test matrix's own means, which is the statistically
        # correct procedure for kernel centering at inference time.
        K_test_raw = np.asarray(
            self.test_kernel_matrix(self.weights, self.xtrain, self.xtest)
        )
        if self.centering:
            n_train = K_train.shape[0]
            # Column means of the training kernel (1 × N_train)
            train_col_means = K_train.mean(axis=0, keepdims=True)
            # Overall mean of the training kernel (scalar)
            train_mean = K_train.mean()
            # Test kernel centered w.r.t. training distribution
            K_test = (
                K_test_raw
                - K_test_raw.mean(axis=1, keepdims=True)
                - train_col_means
                + train_mean
            )
        else:
            K_test = K_test_raw

        y_pred_train = svm.predict(K_train)
        y_pred_test = svm.predict(K_test)

        return {
            "svm": svm,
            "train_accuracy": float(accuracy_score(y_train_np, y_pred_train)),
            "test_accuracy": float(accuracy_score(y_test_np, y_pred_test)),
            "f1_score": float(f1_score(y_test_np, y_pred_test)),
            "precision_score": float(precision_score(y_test_np, y_pred_test)),
            "recall_score": float(recall_score(y_test_np, y_pred_test)),
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

        # FIX 3: Do NOT pre-seed loss_hist / alignment_hist with an epoch-0
        # value.  Pre-seeding caused len(loss_history) == epochs + 1, making
        # print_training_summary report the wrong epoch count.  The pre-training
        # state is already captured in init_{train,test}_accuracy.
        alignment_hist: list[float] = []
        loss_hist: list[float] = []
        train_acc, test_acc, f1s, precs, recs = [], [], [], [], []

        start = time.perf_counter()
        desc = f"[{type(self).__name__}] KTA alignment"

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
            "weights": self.weights,
            "init_train_accuracy": init["train_accuracy"],
            "init_test_accuracy": init["test_accuracy"],
            "alignment_history": alignment_hist,
            "loss_history": loss_hist,
            "train_accuracy_history": train_acc,
            "test_accuracy_history": test_acc,
            "f1_score_history": f1s,
            "precision_score_history": precs,
            "recall_score_history": recs,
            "time": time.perf_counter() - start,
            "circuit_executions": self.kernel_model.circuit_executions,
        }
        print_training_summary(history)
        return history


# ─────────────────────────────────────────────────────────────────────────────
# Concrete KTA strategies
# ─────────────────────────────────────────────────────────────────────────────

class FullKTA(BaseKTA):
    """
    Full-batch KTA: gradient is computed over the entire training set each epoch.

    This is the most accurate but most expensive strategy.
    """

    def _get_batch(self, epoch: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.xtrain, self.ytrain


class RandomKTA(BaseKTA):
    """
    Stochastic KTA: draw a random mini-batch each epoch.

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

        probs = svm.predict_proba(K)[:, 1]          # P(y = 1)
        uncertainty = 1.0 - np.abs(2.0 * probs - 1.0)   # 0 = certain, 1 = max uncertain
        k = min(self.greedy_samples, len(self.xtrain))
        topk = np.argsort(-uncertainty)[:k]
        return self.xtrain[topk], self.ytrain[topk]


# ─────────────────────────────────────────────────────────────────────────────
# Centroid-based alternating optimizer
# ─────────────────────────────────────────────────────────────────────────────

class CentroidBasedKTA(BaseKTA):
    """
    Alternating centroid–kernel optimization.

    Each outer epoch alternates between:

    * **KAO inner loop** – update kernel *weights* to maximize KTA computed
      on sub-centroids (with L2 regularization on the weights).
    * **CO inner loop**  – update *centroid positions* to maximize the same
      KTA (with a box-constraint regularizer keeping values in [0, 1]).

    Parameters
    ----------
    centroids : int
        Number of sub-centroids per class (default 4).
    clustering : {'regular', 'kmeans'}
        How to initialize sub-centroids.  'regular' splits class data into
        equal chunks; 'kmeans' runs k-means.
    inner_loop : int
        Number of gradient steps in each of the KAO and CO inner loops.
    lambda_co : float
        Weight of the box-constraint regularizer in the CO loss.
    lambda_kao : float
        Weight of the L2 regularizer in the KAO loss.
    centroid_learning_rate : float
        Learning rate used exclusively for centroid optimizers.
    """

    def __init__(
        self,
        kernel_model,
        data: jnp.ndarray,
        labels: jnp.ndarray,
        *,
        centroids: int = 4,
        clustering: str = "regular",
        lambda_co: float = 0.01,   # FIX B/C: was 1.0 — that magnitude dominates the KTA
        lambda_kao: float = 0.001, # term and drives weights/centroids to collapse.
        eps = 0.01,
        alpha = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(kernel_model, data, labels, **kwargs)

        self.n_centroids = centroids
        self.use_kmeans = clustering.lower() == "kmeans"
        self.inner_loop = inner_loop
        self.lambda_co = lambda_co
        self.lambda_kao = lambda_kao
        self.centroid_lr = centroid_learning_rate
        self.n_classes = int(jnp.unique(self.ytrain).shape[0])
        self.eps = eps
        self.alpha = alpha
        # Per-feature min/max of training data — used in the CO box constraint
        # so centroids are kept within the actual data range, not a hard [0, 1].
        X_np = np.asarray(self.xtrain)
        self._feat_min = jnp.array(X_np.min(axis=0), dtype=jnp.float32)
        self._feat_max = jnp.array(X_np.max(axis=0), dtype=jnp.float32)

        # Initialize centroid arrays
        (
            self.main_centroids,
            self.main_centroid_labels,
            self.sub_centroids,
            self.sub_centroid_labels,
        ) = self._compute_centroids(self.xtrain, self.ytrain)

    # ── Centroid initialization ────────────────────────────────────────────

    def _compute_centroids(
        self, X: jnp.ndarray, y: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # FIX 4: Convert to NumPy before boolean indexing. JAX boolean indexing
        # produces dynamically-shaped arrays, which are fragile outside jit and
        # trigger shape-inference errors on some JAX backends.
        X_np = np.asarray(X)
        y_np = np.asarray(y)

        unique_labels = np.unique(y_np)
        n_cls = len(unique_labels)
        D = X_np.shape[1]

        main_cents = jnp.zeros((n_cls, D), dtype=jnp.float32)
        main_labels = jnp.zeros((n_cls,), dtype=jnp.float32)
        sub_cents = jnp.zeros((n_cls * self.n_centroids, D), dtype=jnp.float32)
        sub_labels = jnp.zeros((n_cls * self.n_centroids,), dtype=jnp.float32)

        for ci, label in enumerate(unique_labels):
            class_data = jnp.array(X_np[y_np == label], dtype=jnp.float32)

            # Main centroid = class mean
            main_cents = main_cents.at[ci].set(jnp.mean(class_data, axis=0))
            main_labels = main_labels.at[ci].set(float(label))

            # Sub-centroids via k-means or equal splits
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
                sub_cents = sub_cents.at[idx].set(sc[si])
                sub_labels = sub_labels.at[idx].set(float(label))

        return main_cents, main_labels, sub_cents, sub_labels

    # ── Centroid-specific kernel & alignment ──────────────────────────────

    def _square_kernel(
        self, weights, X: jnp.ndarray
    ) -> jnp.ndarray:
        """N×N kernel matrix for an arbitrary point set X."""
        N = X.shape[0]
        x1, x2 = self._pairwise(X, X)
        return self.kernel_model.forward(x1, x2, weights).reshape(N, N)

    def _centroid_alignment(
        self,
        weights,
        main_centroids: jnp.ndarray,
        sub_centroids: jnp.ndarray,
        y_sub: jnp.ndarray,
    ) -> jnp.ndarray:
        """KTA computed over ALL centroids (main + sub) combined.

        Previously only sub_centroids entered the kernel, leaving grad w.r.t.
        main_centroids identically zero and making the main-centroid optimizer
        a no-op.  Concatenating both sets gives the CO loop a meaningful signal
        for main_centroids and a richer kernel matrix for KAO.
        """
        # self.main_centroid_labels is a compile-time constant inside JIT — fine.
        all_X = jnp.concatenate([main_centroids, sub_centroids], axis=0)
        all_y = jnp.concatenate([self.main_centroid_labels, y_sub], axis=0)
        K = self._square_kernel(weights, all_X)
        T = all_y[:, None] * all_y[None, :]
        norm = jnp.linalg.norm(K, ord="fro") * jnp.linalg.norm(T, ord="fro")
        return jnp.sum(K * T) / (norm + 1e-10)

    # ── KAO & CO losses ───────────────────────────────────────────────────

    def _loss_kao_cl(
        self,
        weights,
        X_cl: jnp.ndarray,
        y_cl: jnp.ndarray,
    ) -> jnp.ndarray:
        """KAO loss for a SINGLE selected class (matching the flowchart).

        Uses only the sub-centroids of class ``cl`` so the kernel is pushed to
        maximise intra-class compactness for that class at each epoch.
        λ_kao regularises the weights (mean-normalised to be circuit-size-invariant).
        """
        K = self._square_kernel(weights, X_cl)
        T = y_cl[:, None] * y_cl[None, :]
        norm = jnp.linalg.norm(K, ord="fro") * jnp.linalg.norm(T, ord="fro")
        kta = jnp.sum(K * T) / (norm + 1e-10)
        leaves = jax.tree_util.tree_leaves(weights)
        n_params = max(sum(leaf.size for leaf in leaves), 1)
        l2 = sum(jnp.sum(leaf ** 2) for leaf in leaves) / n_params
        return 1.0 - kta + self.lambda_kao * l2

    def _loss_co(
        self,
        weights,
        main_centroids: jnp.ndarray,
        sub_centroids: jnp.ndarray,
        y_sub: jnp.ndarray,
    ) -> jnp.ndarray:
        """1 − KTA + λ_co · box-constraint penalty on centroids.

        The box constraint uses the per-feature min/max of the TRAINING DATA
        (self._feat_min / _feat_max), not a hardcoded [0, 1].  Using [0, 1]
        when features are not normalised into that range forces centroids into
        a completely different distribution from the data, destroying the
        training signal.  Both main and sub centroids are penalised.
        """
        kta = self._centroid_alignment(weights, main_centroids, sub_centroids, y_sub)
        all_cents = jnp.concatenate([main_centroids, sub_centroids], axis=0)
        penalty = jnp.sum(
            jax.nn.relu(all_cents - self._feat_max)
            + jax.nn.relu(self._feat_min - all_cents)
        )
        return 1.0 - kta + self.lambda_co * penalty

    # -- Analytical approach to update centroids and kernel parameters

    def _kao_parameter_update(self, X_cl, y_cl):

        param_shape  = self.weights.shape
        
        for idx in np.ndindex(param_shape):
            
            param_plus_delta = self.weights.at[idx].add(jnp.pi)
            param_minus_delta = self.weights.at[idx].add(jnp.pi / 2)

            loss_pi = self._loss_kao_cl(param_plus_delta, X_cl, y_cl)
            loss_pi_over_2 = self._loss_kao_cl(param_minus_delta, X_cl, y_cl)
            loss_zero = self._loss_kao_cl(self.weights, X_cl, y_cl)

            numer = 2 * loss_pi_over_2 -loss_pi - loss_zero
            denom = loss_pi - loss_zero


            delta = jnp.arctan2(numer, denom + 1e-10) 
            new_val =  self.weights[idx] - delta  # Add small term to avoid division by zero
            self.weights = self.weights.at[idx].set(new_val)

        return self.weights

    def _centroid_update(self, main_centroids, sub_centroids, y_sub, eps = 0.01, cl_kao = None, mask_kao = None):

        main_centroids = self.main_centroids
        sub_centroids  = self.sub_centroids

        main_idx = np.where(self.main_centroid_labels == cl_kao)[0][0]
        sub_indices = np.where(mask_kao)[0]

        main_centroid_shape = main_centroids.shape
        sub_centroid_shape = sub_centroids.shape

        # Update main centroids
        for idx in np.ndindex(self.main_centroids[main_idx].shape):

            def loss_fn(shift):
                mc_shifted = self.main_centroids.at[main_idx, idx[0]].add(shift)
                return self._loss_co(self.weights, mc_shifted, self.sub_centroids, y_sub)

            L_plus  = loss_fn(+eps)
            L_minus = loss_fn(-eps)
            L0      = self._loss_co(self.weights, self.main_centroids, self.sub_centroids, y_sub)

            grad = (L_plus - L_minus) / (2 * eps)
            hess = (L_plus - 2 * L0 + L_minus) / (eps**2 + 1e-10)

            if jnp.abs(hess) < 1e-6:
                continue

            delta = self.alpha * grad / (hess + 1e-10)

            feat_idx = idx[0]

            new_val = self.main_centroids[main_idx, feat_idx] - delta
            new_val = jnp.clip(new_val, self._feat_min[feat_idx], self._feat_max[feat_idx])

            self.main_centroids = self.main_centroids.at[main_idx, idx[0]].set(new_val)
        
        # Update sub-centroids
        for si in sub_indices:
            for idx in np.ndindex(self.sub_centroids[si].shape):

                def loss_fn(shift):
                    sc_shifted = self.sub_centroids.at[si, idx[0]].add(shift)
                    return self._loss_co(self.weights, self.main_centroids, sc_shifted, y_sub)

                L_plus  = loss_fn(+eps)
                L_minus = loss_fn(-eps)
                L0      = self._loss_co(self.weights, self.main_centroids, self.sub_centroids, y_sub)

                grad = (L_plus - L_minus) / (2 * eps)
                hess = (L_plus - 2 * L0 + L_minus) / (eps**2 + 1e-10)

                if jnp.abs(hess) < 1e-6 or jnp.isnan(hess) or jnp.isnan(grad):
                    continue

                delta = self.alpha * grad / (hess + 1e-10)

                feat_idx = idx[0]

                new_val = self.sub_centroids[si, feat_idx] - delta
                new_val = jnp.clip(new_val, self._feat_min[feat_idx], self._feat_max[feat_idx])

                self.sub_centroids = self.sub_centroids.at[si, idx[0]].set(new_val)
    
        return self.main_centroids, self.sub_centroids



    # ── BaseKTA abstract method (unused — align() is fully overridden) ─────

    def _get_batch(self, epoch: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.xtrain, self.ytrain  # fallback; align() is overridden

    # ── Custom training loop ───────────────────────────────────────────────

    def align(self) -> dict[str, Any]:  # type: ignore[override]
        """
        Alternating KAO / CO optimization loop — matches the reference flowchart:

        For each outer epoch
        ├─ (1) Select centroid class ``cl`` (cycles through unique labels)
        ├─ (2) KAO inner loop  — update kernel *weights* using sub-centroids of
        │       class ``cl`` only.
        ├─ (3) Flip: ``cl = −cl``  (for binary labels; next class for multi-class)
        ├─ (4) CO inner loop   — update *main_centroids* only, using flipped ``cl``
        └─ (5) Single gradient step to update *sub_centroids* (outside the loop)

        In addition to the standard history keys the returned dict also
        contains ``main_centroids`` and ``sub_centroids`` (per-epoch snapshots).
        """
        init = self.svm_training(self.xtrain, self.ytrain)
        alignment_hist: list[float] = []
        loss_hist: list[float] = []
        train_acc, test_acc, f1s, precs, recs = [], [], [], [], []
        main_cent_hist: list[jnp.ndarray] = []
        sub_cent_hist: list[jnp.ndarray] = []

        # Stable NumPy view of sub-centroid labels for mask operations
        sc_labels_np = np.asarray(self.sub_centroid_labels)
        unique_labels_np = np.unique(np.asarray(self.ytrain))
        n_cls = len(unique_labels_np)
        y_sub = self.sub_centroid_labels  # full label vector used by CO

        start = time.perf_counter()
        for epoch in tqdm(range(self.epochs), desc="[CentroidBasedKTA] KTA alignment"):

            cl_kao = unique_labels_np[epoch % n_cls]

            mask_kao = sc_labels_np == cl_kao
            X_cl = jnp.array(np.asarray(self.sub_centroids)[mask_kao])
            y_cl = jnp.array(sc_labels_np[mask_kao], dtype=jnp.float32)

            # KAO inner loop: update weights using sub-centroids of class cl_kao
            self.weights = self._kao_parameter_update(X_cl, y_cl)

            if epoch % 2 == 0:   # update every 2 epochs
                self.main_centroids, self.sub_centroids = self._centroid_update(
                    self.main_centroids,
                    self.sub_centroids,
                    y_sub,
                    eps=self.eps,
                    cl_kao=cl_kao,
                    mask_kao=mask_kao,
                )

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

        history: dict[str, Any] = {
            "weights": self.weights,
            "main_centroids": main_cent_hist,
            "sub_centroids": sub_cent_hist,
            "init_train_accuracy": init["train_accuracy"],
            "init_test_accuracy": init["test_accuracy"],
            "alignment_history": alignment_hist,
            "train_accuracy_history": train_acc,
            "test_accuracy_history": test_acc,
            "f1_score_history": f1s,
            "precision_score_history": precs,
            "recall_score_history": recs,
            "time": time.perf_counter() - start,
            "circuit_executions": self.kernel_model.circuit_executions,
        }
        #print_training_summary(history)
        return history


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible lowercase aliases
# ─────────────────────────────────────────────────────────────────────────────

fullKTA = FullKTA
randomKTA = RandomKTA
greedyKTA = GreedyKTA
centroidBasedKTA = CentroidBasedKTA

__all__ = [
    "BaseKTA",
    "FullKTA",
    "RandomKTA",
    "GreedyKTA",
    "CentroidBasedKTA",
    "print_training_summary",
    # aliases
    "fullKTA",
    "randomKTA",
    "greedyKTA",
    "centroidBasedKTA",
]