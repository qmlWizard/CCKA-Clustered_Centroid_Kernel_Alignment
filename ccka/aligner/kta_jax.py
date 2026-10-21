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


# ---------------------------------------------------------------------------
# Display utilities
# ---------------------------------------------------------------------------

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
            f"Initial validation accuracy : {history['init_validation_accuracy']:.4f}",
            f"Final   validation accuracy : {history['final_svm_metrics']['validation_accuracy']:.4f}",
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
            f"Final alignment      : {history['alignment_history'][-1]:.6f}",
            f"Circuit executions   : {history['circuit_executions']}",
        ],
        width,
    )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseKTA(ABC):
    """
    Abstract base class for KTA-based quantum kernel optimizers.

    Concrete subclasses implement :meth:`_get_batch` to control *which* data
    subset drives the gradient at each training step.  Everything else --
    kernel matrix construction, centering, SVM evaluation, and the training
    loop -- lives here.

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
        Number of Nystrom landmarks; required when *matrix_type='nystrom'*.
    centering : bool
        Must remain false; kernel centering is not supported.
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
        validation_fraction: float = 0.5,
        validation_data: jnp.ndarray | None = None,
        validation_labels: jnp.ndarray | None = None,
        test_data: jnp.ndarray | None = None,
        test_labels: jnp.ndarray | None = None,
        matrix_type: str = "regular",
        landmark_points: int = 0,
        inference_landmarks: int = 32,
        centering: bool = False,
        collect_diagnostics: bool = True,
        svm_c: float = 1.0,
        epochs: int = 100,
        learning_rate: float = 0.01,
        optimizer: str = "adam",
        seed: int = 42,
        **_ignored: Any,
    ) -> None:

        self.seed = seed
        self.split_size = split_size
        
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

        self.kernel_model = kernel_model
        self.matrix_type = matrix_type
        self.landmark_points = landmark_points
        if centering:
            raise ValueError("Kernel centering is not supported.")
        self.collect_diagnostics = collect_diagnostics
        if inference_landmarks <= 0:
            raise ValueError("inference_landmarks must be > 0")
        if svm_c <= 0:
            raise ValueError("svm_c must be > 0")
        supplied_splits = (
            validation_data is not None,
            validation_labels is not None,
            test_data is not None,
            test_labels is not None,
        )
        if any(supplied_splits) and not all(supplied_splits):
            raise ValueError(
                "validation_data, validation_labels, test_data, and test_labels "
                "must be provided together."
            )
        if all(supplied_splits):
            if len(data) != len(labels):
                raise ValueError("data and labels must have the same length.")
            self.xtrain, self.ytrain = data, labels
            self.xvalidation, self.yvalidation = validation_data, validation_labels
            self.xtest, self.ytest = test_data, test_labels
        else:
            if not (0.0 < validation_fraction < 1.0):
                raise ValueError(
                    "validation_fraction must be in (0, 1), "
                    f"got {validation_fraction}"
                )
            (
                self.xtrain,
                self.xvalidation,
                self.xtest,
                self.ytrain,
                self.yvalidation,
                self.ytest,
            ) = self._split_data(
                data,
                labels,
                seed=self.seed,
                validation_fraction=validation_fraction,
            )
        self.inference_landmarks = inference_landmarks
        self.svm_c = svm_c
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer.lower()

        self.weights = kernel_model.circuit.init_weights(seed= self.seed)
        self._optimizer = self._build_optimizer(self.learning_rate)
        self.opt_state = self._optimizer.init(self.weights)

        self._loss_fn = jax.jit(self._loss_kta)
        self._grad_fn = jax.jit(jax.grad(self._loss_kta))

    # -- Optimizer factory --------------------------------------------------

    def _build_optimizer(self, lr: float) -> ox.GradientTransformation:
        if self.optimizer_name not in self._OPTIMIZERS:
            raise ValueError(
                f"Optimizer {self.optimizer_name!r} not supported. "
                f"Choose from: {list(self._OPTIMIZERS)}"
            )
        return self._OPTIMIZERS[self.optimizer_name](lr)

    # -- Data splitting -----------------------------------------------------

    def _split_data(
        self,
        data: jnp.ndarray,
        labels: jnp.ndarray,
        seed: int = 42,
        validation_fraction: float = 0.5,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        n = len(data)
        perm = jax.random.permutation(jax.random.PRNGKey(seed), n)
        train_end = int(n * self.split_size)
        validation_count = int((n - train_end) * validation_fraction)
        validation_end = train_end + validation_count
        if train_end == 0 or validation_count == 0 or validation_end >= n:
            raise ValueError(
                "The split must leave non-empty training, validation, and test sets."
            )
        train_idx = perm[:train_end]
        validation_idx = perm[train_end:validation_end]
        test_idx = perm[validation_end:]
        return (
            data[train_idx],
            data[validation_idx],
            data[test_idx],
            labels[train_idx],
            labels[validation_idx],
            labels[test_idx],
        )

    # -- Kernel matrix helpers ----------------------------------------------

    @staticmethod
    def _pairwise(
        A: jnp.ndarray, B: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        N, M = A.shape[0], B.shape[0]
        return jnp.repeat(A, M, axis=0), jnp.tile(B, (N, 1))

    def regular_kernel_matrix(self, weights, X: jnp.ndarray) -> jnp.ndarray:
        """Full NxN kernel matrix using upper-triangular computation."""
        N = X.shape[0]
        iu, ju = jnp.triu_indices(N)
        x1 = X[iu]
        x2 = X[ju]
        k_vals = self.kernel_model.forward(x1, x2, weights)
        K = jnp.zeros((N, N))
        K = K.at[iu, ju].set(k_vals)
        K = K.at[ju, iu].set(k_vals)
        return K

    def nystrom_kernel_matrix(self, weights, X: jnp.ndarray) -> jnp.ndarray:
        """Nystrom approximation: K ~= K_NM * K_MM^-1 * K_NM^T."""
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
        KMM = 0.5 * (KMM + KMM.T)
        KMM_inv = jnp.linalg.pinv(KMM)

        return KNM @ KMM_inv @ KNM.T

    def test_kernel_matrix(
        self, weights, X_train: jnp.ndarray, X_test: jnp.ndarray
    ) -> jnp.ndarray:
        """M x N cross-kernel matrix between X_test (rows) and X_train (cols)."""
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
        """Compatibility helper; kernels are never centered."""
        return K

    # -- KTA (full-matrix variant) ------------------------------------------

    def alignment(
        self, weights, X: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Kernel-Target Alignment between the (centered) kernel matrix and the
        label outer product.  <K, T>_F / (||K||_F * ||T||_F).
        """
        K = self._kernel_matrix(weights, X)
        T = y[:, None] * y[None, :]
        norm = jnp.linalg.norm(K, ord="fro") * jnp.linalg.norm(T, ord="fro")
        return jnp.sum(K * T) / (norm + 1e-10)

    def _loss_kta(
        self, weights, X: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        return 1.0 - self.alignment(weights, X, y)

    # =========================================================================
    # Experiment 1: Centroid-Space KTA
    # =========================================================================

    def centroid_kta(
        self, weights, X: jnp.ndarray, y: jnp.ndarray
    ) -> float:
        """
        Centroid-space KTA for Experiment 1.

        For non-centroid methods (FullKTA, RandomKTA, GreedyKTA) there is no
        separate centroid space so this falls back to global KTA.

        CentroidBasedKTA and QuackKTA override this to return the TA that
        their KAO step actually maximizes, making the two quantities
        directly comparable in the Experiment 1 diagnostic plots.

        Returns
        -------
        float
        """
        return float(self.alignment(weights, X, y))

    # =========================================================================
    # Experiment 2: Kernel Matrix Block-Diagonal Ratio
    # =========================================================================

    def block_diagonal_ratio(
        self, weights, X: jnp.ndarray, y: jnp.ndarray
    ) -> float:
        """
        Measures the class-separability structure of the kernel matrix.

            ratio = mean K[i,j]  (same class, i != j)
                  / mean K[i,j]  (different class)

        A ratio >> 1 means the kernel has learned strong within-class
        cohesion relative to between-class overlap -- i.e. a good
        block-diagonal structure -- regardless of what global KTA reports.

        Returns
        -------
        float  (0.0 if denominator is negligible)
        """
        K = np.asarray(self._kernel_matrix(weights, X))
        y_np = np.asarray(y)

        same_mask = (y_np[:, None] == y_np[None, :])
        # Exclude diagonal: self-similarity is always 1 and inflates the ratio
        np.fill_diagonal(same_mask, False)
        diff_mask = ~same_mask
        np.fill_diagonal(diff_mask, False)

        within  = float(K[same_mask].mean()) if same_mask.any()  else 0.0
        between = float(K[diff_mask].mean()) if diff_mask.any() else 1e-10

        return within / (abs(between) + 1e-10)

    # =========================================================================
    # Experiment 4: Kernel PCA (moved to base so ALL methods share it)
    # =========================================================================

    def kernel_pca(
        self, K: np.ndarray, n_components: int = 2, eps: float = 1e-12
    ) -> np.ndarray:
        """
        2-D embedding via eigendecomposition of a kernel matrix.

        Parameters
        ----------
        K : array (N, N) -- pre-centered if desired
        n_components : int
        eps : float -- numerical floor for eigenvalues

        Returns
        -------
        X_embedded : array (N, n_components)
        """
        K_j = jnp.asarray(K)
        eigvals, eigvecs = jnp.linalg.eigh(K_j)

        # Sort descending
        idx     = jnp.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Keep top components and apply numerical floor
        eigvals = jnp.maximum(eigvals[:n_components], eps)
        eigvecs = eigvecs[:, :n_components]

        return np.asarray(eigvecs * jnp.sqrt(eigvals))

    # -- SVM evaluation -----------------------------------------------------

    def svm_training(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        *,
        evaluate_test: bool = False,
    ) -> dict[str, Any]:
        """
        Fit an SVM with the current kernel and evaluate on validation data.

        Returns a dict with keys: svm, train_accuracy, test_accuracy,
        validation_accuracy, test_accuracy, f1_score, precision_score,
        recall_score, margin. Test metrics are populated only when
        ``evaluate_test=True``.
        """
        K_train, K_validation, K_test = self._evaluation_kernel_matrices(
            self.weights, X, self.xvalidation, self.xtest if evaluate_test else None
        )
        K_train = np.asarray(K_train)
        K_validation = np.asarray(K_validation)
        y_train_np = np.asarray(y)

        svm = SVC(
            kernel="precomputed",
            C=self.svm_c,
            probability=True,
            max_iter=10_000,
            random_state=self.seed,
        )
        svm.fit(K_train, y_train_np)

        y_pred_train = svm.predict(K_train)
        y_pred_validation = svm.predict(K_validation)
        dual_coefs   = svm.dual_coef_
        w_norm_sq    = np.sum(dual_coefs * dual_coefs)
        margin       = 1.0 / np.sqrt(w_norm_sq)

        metrics = {
            "svm":            svm,
            "train_accuracy": float(accuracy_score(y_train_np, y_pred_train)),
            "validation_accuracy": float(
                accuracy_score(np.asarray(self.yvalidation), y_pred_validation)
            ),
            "margin":         float(margin),
        }
        if evaluate_test:
            K_test = np.asarray(K_test)
            y_test = np.asarray(self.ytest)
            y_pred_test = svm.predict(K_test)
            metrics.update(
                {
                    "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
                    "f1_score": float(
                        f1_score(y_test, y_pred_test, average="macro")
                    ),
                    "precision_score": float(
                        precision_score(y_test, y_pred_test, average="macro")
                    ),
                    "recall_score": float(
                        recall_score(y_test, y_pred_test, average="macro")
                    ),
                }
            )
        return metrics

    def _evaluation_kernel_matrices(
        self,
        weights,
        X_train: jnp.ndarray,
        X_validation: jnp.ndarray,
        X_test: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        K_train = self._kernel_matrix(weights, X_train)
        K_validation = self.test_kernel_matrix(weights, X_train, X_validation)
        K_test = (
            self.test_kernel_matrix(weights, X_train, X_test)
            if X_test is not None
            else None
        )
        return K_train, K_validation, K_test

    def _nystrom_evaluation_kernel_matrices(
        self,
        weights,
        X_train: jnp.ndarray,
        X_validation: jnp.ndarray,
        X_test: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        landmark_count = min(self.inference_landmarks, len(X_train))
        indices = jax.random.permutation(
            jax.random.PRNGKey(self.seed), len(X_train)
        )[:landmark_count]
        landmarks = X_train[indices]

        x_train_pairs, x_landmark_pairs = self._pairwise(X_train, landmarks)
        K_train_landmarks = self.kernel_model.forward(
            x_train_pairs, x_landmark_pairs, weights
        ).reshape(len(X_train), landmark_count)

        x_left, x_right = self._pairwise(landmarks, landmarks)
        K_landmarks = self.kernel_model.forward(
            x_left, x_right, weights
        ).reshape(landmark_count, landmark_count)
        K_landmarks = 0.5 * (K_landmarks + K_landmarks.T)
        K_landmarks_inv = jnp.linalg.pinv(K_landmarks)

        K_train = (
            K_train_landmarks @ K_landmarks_inv @ K_train_landmarks.T
        )

        def cross_kernel(query: jnp.ndarray) -> jnp.ndarray:
            x_query_pairs, x_landmark_pairs = self._pairwise(query, landmarks)
            K_query_landmarks = self.kernel_model.forward(
                x_query_pairs, x_landmark_pairs, weights
            ).reshape(len(query), landmark_count)
            return K_query_landmarks @ K_landmarks_inv @ K_train_landmarks.T

        K_validation = cross_kernel(X_validation)
        K_test = cross_kernel(X_test) if X_test is not None else None
        return K_train, K_validation, K_test

    # -- Abstract interface -------------------------------------------------

    @abstractmethod
    def _get_batch(self, epoch: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return *(X_batch, y_batch)* for the current gradient step."""

    # -- Main training loop -------------------------------------------------

    def align(self) -> dict[str, Any]:
        """
        Run KTA optimization and return a training history dictionary.

        History keys
        ------------
        weights, init_train_accuracy, init_test_accuracy,
        alignment_history         -- global KTA per epoch
        centroid_kta_history      -- Experiment 1: KTA in the space the method optimizes
        block_ratio_history       -- Experiment 2: within/between kernel ratio
        loss_history,
        train_accuracy_history, test_accuracy_history,
        f1_score_history, precision_score_history, recall_score_history,
        margin_history,
        coords                    -- Experiment 4: list of (N, 2) kPCA arrays per epoch
        coords_labels             -- Experiment 4: label array (ytrain for base methods)
        time, circuit_executions
        """
        init = self.svm_training(self.xtrain, self.ytrain)

        alignment_hist:     list[float]      = []
        centroid_kta_hist:  list[float]      = []   # Experiment 1
        block_ratio_hist:   list[float]      = []   # Experiment 2
        loss_hist:          list[float]      = []
        train_acc, validation_acc, svm_margin = [], [], []
        best_validation_acc = -np.inf
        best_weights = self.weights
        coords_hist:        list[np.ndarray] = []   # Experiment 4

        start = time.perf_counter()

        for epoch in range(self.epochs):
            X_b, y_b = self._get_batch(epoch)

            if self.collect_diagnostics:
                loss_hist.append(float(self._loss_fn(self.weights, X_b, y_b)))
                alignment_hist.append(
                    float(self.alignment(self.weights, self.xtrain, self.ytrain))
                )
                centroid_kta_hist.append(
                    self.centroid_kta(self.weights, self.xtrain, self.ytrain)
                )
                block_ratio_hist.append(
                    self.block_diagonal_ratio(
                        self.weights, self.xtrain, self.ytrain
                    )
                )

            # Gradient step
            grads = self._grad_fn(self.weights, X_b, y_b)
            updates, self.opt_state = self._optimizer.update(grads, self.opt_state)
            self.weights = ox.apply_updates(self.weights, updates)

            result = self.svm_training(self.xtrain, self.ytrain)
            train_acc.append(result["train_accuracy"])
            validation_acc.append(result["validation_accuracy"])
            svm_margin.append(result["margin"])
            if result["validation_accuracy"] > best_validation_acc:
                best_validation_acc = result["validation_accuracy"]
                best_weights = self.weights

            if self.collect_diagnostics:
                K_train = np.asarray(
                    self._kernel_matrix(self.weights, self.xtrain)
                )
                coords_hist.append(self.kernel_pca(K_train))

        self.weights = best_weights
        final_result = self.svm_training(
            self.xtrain, self.ytrain, evaluate_test=True
        )
        history: dict[str, Any] = {
            "weights":                self.weights,
            "init_train_accuracy":    init["train_accuracy"],
            "init_validation_accuracy": init["validation_accuracy"],
            "init_test_accuracy":     float("nan"),
            "alignment_history":      alignment_hist,
            "centroid_kta_history":   centroid_kta_hist,   # Experiment 1
            "block_ratio_history":    block_ratio_hist,    # Experiment 2
            "loss_history":           loss_hist,
            "train_accuracy_history": train_acc,
            "validation_accuracy_history": validation_acc,
            "test_accuracy_history":  [final_result["test_accuracy"]],
            "best_validation_accuracy": float(best_validation_acc),
            "final_svm_metrics":       final_result,
            "f1_score_history":       [final_result["f1_score"]],
            "precision_score_history":[final_result["precision_score"]],
            "recall_score_history":   [final_result["recall_score"]],
            "margin_history":         svm_margin,
            "coords":                 coords_hist,         # Experiment 4
            "coords_labels":          self.ytrain,         # Experiment 4
            "time":                   time.perf_counter() - start,
            "circuit_executions":     self.kernel_model.circuit_executions,
        }
        return history


# ---------------------------------------------------------------------------
# Concrete KTA strategies -- gradient-based
# ---------------------------------------------------------------------------

class FullKTA(BaseKTA):
    """
    Full-batch KTA: gradient computed over the entire training set each epoch.
    Mirrors PyTorch train_method='full'.

    centroid_kta() falls back to global KTA (no separate centroid space).
    block_diagonal_ratio() and kernel_pca() use xtrain only.
    """

    def _get_batch(self, epoch: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.xtrain, self.ytrain


class RandomKTA(BaseKTA):
    """
    Stochastic KTA: draw a random mini-batch each epoch.
    Mirrors PyTorch train_method='random'.

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
        self._rng = jax.random.PRNGKey(self.seed)
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
        svm = SVC(kernel="precomputed", C=1.0, probability=True, max_iter=10000, random_state=self.seed)
        svm.fit(K, np.asarray(self.ytrain))
        probs       = svm.predict_proba(K)[:, 1]
        uncertainty = 1.0 - np.abs(2.0 * probs - 1.0)
        k    = min(self.greedy_samples, len(self.xtrain))
        topk = np.argsort(-uncertainty)[:k]
        return self.xtrain[topk], self.ytrain[topk]


# ---------------------------------------------------------------------------
# Centroid-based alternating optimizer -- matches PyTorch train_method='ccka'
# ---------------------------------------------------------------------------

class CentroidBasedKTA(BaseKTA):
    """
    Alternating centroid-kernel optimization (CCKA).
    Mirrors PyTorch train_method='ccka' exactly.

    PyTorch optimizer structure (replicated here):
    ----------------------------------------------
    KAO step -- ONE joint Adam across kernel weights + sub-centroids:
        optim.Adam([
            {'params': kernel.parameters(), 'lr': lr},
            {'params': class_centroids,     'lr': cclr},
        ])
    CO step -- ONE Adam per main centroid (only that centroid updated):
        optim.Adam([{'params': main_centroids[cl], 'lr': mclr}])

    Replicated with three separate optax optimizers:
      - _kao_weight_optimizer  (learning_rate)   <- kernel weights  (KAO only)
      - _kao_sub_optimizer     (sub_centroid_lr) <- sub-centroids   (KAO only)
      - _co_main_optimizer     (centroid_lr)     <- main centroids  (CO only)

    Experiment 1 override:
    ----------------------
    centroid_kta() returns the mean TA(K_vec, y_sub, l) over all classes
    where K_vec = kernel(main_centroid, sub_centroids). This is exactly
    what the KAO step maximizes, making the gap vs global KTA interpretable.

    Parameters
    ----------
    centroids : int
        Number of sub-centroids per class (default 4).
    clustering : {'regular', 'kmeans'}
        How to initialise sub-centroids.
    lambda_co : float
    lambda_kao : float
    learning_rate : float  (maps to PyTorch lr)
    centroid_lr : float    (maps to PyTorch mclr, defaults to learning_rate)
    sub_centroid_lr : float (maps to PyTorch cclr, defaults to centroid_lr)
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
        sub_centroid_lr: float | None = None,
        update_kao: bool = True,
        update_co: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(kernel_model, data, labels, learning_rate=learning_rate, **kwargs)

        self.n_centroids      = centroids
        if clustering.lower() not in {"regular", "random", "kmeans"}:
            raise ValueError("clustering must be 'regular', 'random', or 'kmeans'")
        self.clustering       = clustering.lower()
        self.use_kmeans       = self.clustering == "kmeans"
        self.update_kao       = update_kao
        self.update_co        = update_co
        self.lambda_co        = lambda_co
        self.lambda_kao       = lambda_kao
        self.n_classes        = int(jnp.unique(self.ytrain).shape[0])
        self._centroid_lr     = centroid_lr if centroid_lr is not None else learning_rate
        self._sub_centroid_lr = (
            sub_centroid_lr if sub_centroid_lr is not None else self._centroid_lr
        )

        (
            self.main_centroids,
            self.main_centroid_labels,
            self.sub_centroids,
            self.sub_centroid_labels,
        ) = self._compute_centroids(self.xtrain, self.ytrain)

        # Three separate optimizers matching PyTorch's optimizer structure
        self._kao_weight_optimizer = self._build_optimizer(self.learning_rate)
        self._kao_weight_opt_state = self._kao_weight_optimizer.init(self.weights)

        self._kao_sub_optimizer    = self._build_optimizer(self._sub_centroid_lr)
        self._kao_sub_opt_state    = self._kao_sub_optimizer.init(self.sub_centroids)

        self._co_main_optimizer    = self._build_optimizer(self._centroid_lr)
        self._co_main_opt_state    = self._co_main_optimizer.init(self.main_centroids)

    # -- Centroid initialisation -------------------------------------------

    def _compute_centroids(
        self, X: jnp.ndarray, y: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Matches PyTorch _get_centroids."""
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
            class_data  = jnp.array(X_np[y_np == label], dtype=jnp.float32)
            main_cents  = main_cents.at[ci].set(jnp.mean(class_data, axis=0))
            main_labels = main_labels.at[ci].set(float(label))

            if self.use_kmeans and class_data.shape[0] >= self.n_centroids:
                km = KMeans(
                    n_clusters=self.n_centroids, n_init="auto", random_state=self.seed
                ).fit(np.asarray(class_data))
                sc = jnp.array(km.cluster_centers_, dtype=jnp.float32)
            elif self.clustering == "random":
                rng = np.random.default_rng(self.seed + ci)
                indices = rng.choice(
                    class_data.shape[0],
                    size=self.n_centroids,
                    replace=class_data.shape[0] < self.n_centroids,
                )
                sc = class_data[indices]
            else:
                perm = np.random.default_rng(self.seed).permutation(class_data.shape[0])
                shuffled = class_data[perm]
                chunks = jnp.array_split(shuffled, self.n_centroids)
                sc = jnp.stack(
                    [
                        jnp.mean(chunk, axis=0)
                        if len(chunk)
                        else shuffled[si % len(shuffled)]
                        for si, chunk in enumerate(chunks)
                    ]
                ).astype(jnp.float32)

            for si in range(self.n_centroids):
                idx = ci * self.n_centroids + si
                sub_cents  = sub_cents.at[idx].set(sc[si])
                sub_labels = sub_labels.at[idx].set(float(label))

        return main_cents, main_labels, sub_cents, sub_labels

    # -- Centroid-specific kernel vector -----------------------------------

    def _centroid_kernel_vec(
        self,
        weights,
        main_centroid: jnp.ndarray,
        X: jnp.ndarray,
    ) -> jnp.ndarray:
        """K[i] = kernel(main_centroid, X[i]) for all i."""
        N  = X.shape[0]
        x0 = jnp.repeat(main_centroid[None, :], N, axis=0)
        return self.kernel_model.forward(x0, X, weights).reshape(-1)

    def _evaluation_kernel_matrices(
        self,
        weights,
        X_train: jnp.ndarray,
        X_validation: jnp.ndarray,
        X_test: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        return self._nystrom_evaluation_kernel_matrices(
            weights, X_train, X_validation, X_test
        )

    # =========================================================================
    # Experiment 1 override: centroid-space KTA
    # =========================================================================

    def centroid_kta(
        self, weights, X: jnp.ndarray, y: jnp.ndarray
    ) -> float:
        """
        Returns mean TA(K_vec, y_sub, l) over all classes -- exactly the
        quantity that CCKA's KAO step maximizes.

        K_vec = kernel(main_centroid_cl, all_sub_centroids)  shape (N_sub,)
        y_sub = raw sub-centroid labels
        l     = +class_label  (matching KAO sign convention)

        Averaging over classes gives one scalar to track per epoch.
        The gap between this and global KTA reveals WHY global KTA stays
        low: CCKA optimizes a coarser, more class-representative target.
        """
        unique_labels = np.unique(np.asarray(self.main_centroid_labels))
        y_raw = self.sub_centroid_labels.astype(jnp.float32)
        tas = []
        for cl in unique_labels:
            main_idx      = int(jnp.argmax(self.main_centroid_labels == cl))
            main_centroid = self.main_centroids[main_idx]
            K = self._centroid_kernel_vec(weights, main_centroid, self.sub_centroids)
            l = float(cl)
            numerator   = l * jnp.dot(K, y_raw)
            denominator = jnp.linalg.norm(K) * jnp.linalg.norm(y_raw)
            tas.append(float(numerator / (denominator + 1e-10)))
        return float(np.mean(tas))

    # -- KAO loss ----------------------------------------------------------

    def _loss_kao_cl(
        self,
        weights,
        sub_centroids: jnp.ndarray,
        main_centroid: jnp.ndarray,
        y_raw: jnp.ndarray,
        l: float = 1.0,
    ) -> jnp.ndarray:
        """KAO loss: 1 - TA(K_vec, y_raw, l) + lambda_kao * L2(weights)."""
        K = self._centroid_kernel_vec(weights, main_centroid, sub_centroids)
        Y = y_raw.astype(jnp.float32)
        numerator   = l * jnp.dot(K, Y)
        denominator = jnp.linalg.norm(K) * jnp.linalg.norm(Y)
        kta         = numerator / (denominator + 1e-10)
        leaves   = jax.tree_util.tree_leaves(weights)
        n_params = max(sum(leaf.size for leaf in leaves), 1)
        l2       = sum(jnp.sum(leaf ** 2) for leaf in leaves) / n_params
        return 1.0 - kta + self.lambda_kao * l2

    # -- CO loss -----------------------------------------------------------

    def _loss_co_cl(
        self,
        weights,
        main_centroids: jnp.ndarray,
        cl: float,
        y_raw: jnp.ndarray,
        l: float = -1.0,
    ) -> jnp.ndarray:
        """CO loss: 1 - TA(K_vec, y_raw, l) + lambda_co * box_penalty."""
        main_idx      = jnp.argmax(self.main_centroid_labels == cl)
        main_centroid = main_centroids[main_idx]
        K = self._centroid_kernel_vec(weights, main_centroid, self.sub_centroids)
        Y = y_raw.astype(jnp.float32)
        numerator   = l * jnp.dot(K, Y)
        denominator = jnp.linalg.norm(K) * jnp.linalg.norm(Y)
        kta         = numerator / (denominator + 1e-10)
        penalty = jnp.sum(
            jax.nn.relu(main_centroid - 1.0) + jax.nn.relu(-main_centroid)
        )
        return 1.0 - kta + self.lambda_co * penalty

    # -- Joint KAO update --------------------------------------------------

    def _kao_joint_update(
        self,
        main_centroid: jnp.ndarray,
        y_raw: jnp.ndarray,
        l: float = 1.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Jointly update kernel weights and sub-centroids (one Adam step each)."""
        grad_fn = jax.grad(self._loss_kao_cl, argnums=(0, 1))
        grads_w, grads_sub = grad_fn(
            self.weights, self.sub_centroids, main_centroid, y_raw, l
        )
        updates_w, self._kao_weight_opt_state = self._kao_weight_optimizer.update(
            grads_w, self._kao_weight_opt_state
        )
        new_weights = ox.apply_updates(self.weights, updates_w)

        updates_sub, self._kao_sub_opt_state = self._kao_sub_optimizer.update(
            grads_sub, self._kao_sub_opt_state
        )
        new_sub = ox.apply_updates(self.sub_centroids, updates_sub)
        return new_weights, new_sub

    # -- CO update ---------------------------------------------------------

    def _co_main_update(
        self,
        cl: float,
        y_raw: jnp.ndarray,
        l: float = -1.0,
    ) -> jnp.ndarray:
        """Update main centroid for class cl only."""
        grad_fn = jax.grad(self._loss_co_cl, argnums=1)
        grads_main = grad_fn(self.weights, self.main_centroids, cl, y_raw, l)
        updates_main, self._co_main_opt_state = self._co_main_optimizer.update(
            grads_main, self._co_main_opt_state
        )
        new_main = ox.apply_updates(self.main_centroids, updates_main)
        return jnp.clip(new_main, 0.0, 1.0)

    # -- BaseKTA abstract method fallback ----------------------------------

    def _get_batch(self, epoch: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.xtrain, self.ytrain

    # -- Custom training loop ----------------------------------------------

    def align(self) -> dict[str, Any]:
        """
        Alternating KAO / CO optimisation loop matching PyTorch fit_kernel
        with train_method='ccka'.

        For each outer epoch:
        |-- (1) Select class cl (cycles through unique labels)
        |-- (2) 10x KAO steps -- joint gradient on kernel *weights* AND *sub_centroids*
        |-- (3) 10x CO steps  -- gradient on *main_centroids* only
        +-- (4) SVM evaluation + Experiments 1 / 2 / 4 diagnostics

        New history keys (Experiments 1, 2, 4):
          centroid_kta_history  -- TA in centroid space (CCKA's actual objective)
          block_ratio_history   -- within/between class kernel ratio
          coords                -- list of (N_train + N_main + N_sub, 2) kPCA per epoch
          coords_labels         -- matching labels (constant across epochs)
        """
        init = self.svm_training(self.xtrain, self.ytrain)

        alignment_hist:    list[float]       = []
        centroid_kta_hist: list[float]       = []   # Experiment 1
        block_ratio_hist:  list[float]       = []   # Experiment 2
        loss_hist:         list[float]       = []
        train_acc, validation_acc, svm_margin = [], [], []
        main_cent_hist:    list[jnp.ndarray] = []
        sub_cent_hist:     list[jnp.ndarray] = []
        coords_hist:       list[np.ndarray]  = []   # Experiment 4

        unique_labels_np = np.unique(np.asarray(self.ytrain))
        n_cls            = len(unique_labels_np)

        best_validation_acc = -np.inf
        best_weights        = self.weights
        best_main_centroids = self.main_centroids
        best_sub_centroids  = self.sub_centroids

        y_raw = self.sub_centroid_labels

        # Labels for Experiment 4 kPCA (includes centroids for visualizing motion)
        coords_labels = jnp.concatenate(
            [self.ytrain, self.main_centroid_labels, self.sub_centroid_labels], axis=0
        )

        start = time.perf_counter()

        for epoch in range(self.epochs):

            # (1) Select class
            cl_kao = unique_labels_np[epoch % n_cls]
            l_kao  = float(cl_kao)
            l_co   = -float(cl_kao)

            main_idx      = int(jnp.argmax(self.main_centroid_labels == cl_kao))
            main_centroid = self.main_centroids[main_idx]

            # (2) KAO: 10x joint weight + sub-centroid updates
            if self.update_kao:
                self.weights, self.sub_centroids = self._kao_joint_update(
                    main_centroid, y_raw, l=l_kao
                )

            # (3) CO update
            if self.update_co:
                self.main_centroids = self._co_main_update(
                    cl=cl_kao, y_raw=y_raw, l=l_co
                )

            # (4) Diagnostics --------------------------------------------------

            if self.collect_diagnostics:
                alignment_hist.append(
                    float(self.alignment(self.weights, self.xtrain, self.ytrain))
                )
                centroid_kta_hist.append(
                    self.centroid_kta(self.weights, self.xtrain, self.ytrain)
                )
                block_ratio_hist.append(
                    self.block_diagonal_ratio(
                        self.weights, self.xtrain, self.ytrain
                    )
                )

            main_cent_hist.append(self.main_centroids)
            sub_cent_hist.append(self.sub_centroids)

            result = self.svm_training(self.xtrain, self.ytrain)
            train_acc.append(result["train_accuracy"])
            validation_acc.append(result["validation_accuracy"])
            svm_margin.append(result["margin"])

            if result["validation_accuracy"] > best_validation_acc:
                best_validation_acc = result["validation_accuracy"]
                best_weights        = self.weights
                best_main_centroids = self.main_centroids
                best_sub_centroids  = self.sub_centroids

            if self.collect_diagnostics:
                all_points = jnp.concatenate(
                    [self.xtrain, self.main_centroids, self.sub_centroids], axis=0
                )
                K_all = np.asarray(self._kernel_matrix(self.weights, all_points))
                coords_hist.append(self.kernel_pca(K_all))

        # Restore best checkpoint
        self.weights        = best_weights
        self.main_centroids = best_main_centroids
        self.sub_centroids  = best_sub_centroids

        final_result = self.svm_training(
            self.xtrain, self.ytrain, evaluate_test=True
        )

        return {
            "weights":                self.weights,
            "main_centroids":         main_cent_hist,
            "sub_centroids":          sub_cent_hist,
            "init_train_accuracy":    init["train_accuracy"],
            "init_validation_accuracy": init["validation_accuracy"],
            "init_test_accuracy":     float("nan"),
            "alignment_history":      alignment_hist,
            "centroid_kta_history":   centroid_kta_hist,   # Experiment 1
            "block_ratio_history":    block_ratio_hist,    # Experiment 2
            "loss_history":           loss_hist,
            "train_accuracy_history": train_acc,
            "validation_accuracy_history": validation_acc,
            "test_accuracy_history":  [final_result["test_accuracy"]],
            "best_validation_accuracy": float(best_validation_acc),
            "final_svm_metrics":      final_result,
            "f1_score_history":       [final_result["f1_score"]],
            "precision_score_history":[final_result["precision_score"]],
            "recall_score_history":   [final_result["recall_score"]],
            "margin_history":         svm_margin,
            "time":                   time.perf_counter() - start,
            "circuit_executions":     self.kernel_model.circuit_executions,
            "best_main_centroids":    best_main_centroids,
            "best_sub_centroids":     best_sub_centroids,
            "coords":                 coords_hist,          # Experiment 4
            "coords_labels":          coords_labels,        # Experiment 4
            "xtrain":                 self.xtrain,
            "ytrain":                 self.ytrain,
        }


# ---------------------------------------------------------------------------
# QUACK -- full-data variant of CentroidBasedKTA
# ---------------------------------------------------------------------------

class QuackKTA(BaseKTA):
    """
    QUACK: Quantum Alignment via Class Kernel optimisation.
    Direct JAX port of PyTorch train_method='quack'.

    Identical in structure to CentroidBasedKTA except that the full training
    set is used in place of sub-centroids:
        K_vec = kernel(main_centroid, X_train)   shape (N_train,)

    No sub-centroids are allocated or optimised.

    Experiment 1 override:
    ----------------------
    centroid_kta() returns mean TA(K_vec, y_binary, l) over all classes,
    where K_vec = kernel(main_centroid, X_train) and y_binary = +/-1 per
    class vs rest. This is exactly what QUACK's KAO step maximizes.

    Parameters
    ----------
    lambda_co : float
    lambda_kao : float
    centroid_lr : float  (defaults to learning_rate)
    inner_steps : int  (defaults to 10)
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
        inner_steps: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(kernel_model, data, labels, **kwargs)

        if inner_steps <= 0:
            raise ValueError("inner_steps must be > 0")
        self.lambda_co    = lambda_co
        self.lambda_kao   = lambda_kao
        self.n_classes    = int(jnp.unique(self.ytrain).shape[0])
        self._centroid_lr = centroid_lr if centroid_lr is not None else self.learning_rate
        self.inner_steps  = inner_steps

        X_np = np.asarray(self.xtrain)
        self._feat_min = jnp.array(X_np.min(axis=0), dtype=jnp.float32)
        self._feat_max = jnp.array(X_np.max(axis=0), dtype=jnp.float32)

        self.main_centroids, self.main_centroid_labels = self._init_main_centroids()

        self._kao_optimizer = self._build_optimizer(self.learning_rate)
        self._kao_opt_state = self._kao_optimizer.init(self.weights)

        self._co_optimizer  = self._build_optimizer(self._centroid_lr)
        self._co_opt_state  = self._co_optimizer.init(self.main_centroids)

    def _init_main_centroids(self) -> tuple[jnp.ndarray, jnp.ndarray]:
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

    def _quack_kernel_vec(
        self,
        weights,
        main_centroid: jnp.ndarray,
        X_train: jnp.ndarray,
    ) -> jnp.ndarray:
        """K[i] = kernel(main_centroid, X_train[i]) for all i."""
        N  = X_train.shape[0]
        x0 = jnp.repeat(main_centroid[None, :], N, axis=0)
        return self.kernel_model.forward(x0, X_train, weights).reshape(-1)

    # =========================================================================
    # Experiment 1 override: centroid-space KTA for QUACK
    # =========================================================================

    def centroid_kta(
        self, weights, X: jnp.ndarray, y: jnp.ndarray
    ) -> float:
        """
        Returns mean TA(K_vec, y_binary, l) over all classes.
        K_vec    = kernel(main_centroid, X_train)  -- QUACK's actual KAO target
        y_binary = +1 for class, -1 for rest
        l        = +class_label  (matching KAO sign convention)

        This is directly comparable to CCKA's centroid_kta() and to global KTA,
        isolating where the two methods differ in what they optimize.
        """
        unique_labels = np.unique(np.asarray(self.main_centroid_labels))
        tas = []
        for cl in unique_labels:
            main_idx = int(jnp.argmax(self.main_centroid_labels == cl))
            main_centroid = self.main_centroids[main_idx]
            K = self._quack_kernel_vec(weights, main_centroid, self.xtrain)
            y_binary = jnp.where(self.ytrain == cl, 1.0, -1.0).astype(jnp.float32)
            l = float(cl)
            numerator   = l * jnp.dot(K, y_binary)
            denominator = jnp.linalg.norm(K) * jnp.linalg.norm(y_binary)
            tas.append(float(numerator / (denominator + 1e-10)))
        return float(np.mean(tas))

    def _loss_kao_quack(
        self,
        weights,
        main_centroid: jnp.ndarray,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        l: float = 1.0,
    ) -> jnp.ndarray:
        K = self._quack_kernel_vec(weights, main_centroid, X_train)
        Y = y_train.astype(jnp.float32)
        numerator   = l * jnp.dot(K, Y)
        denominator = jnp.linalg.norm(K) * jnp.linalg.norm(Y)
        kta         = numerator / (denominator + 1e-10)
        leaves   = jax.tree_util.tree_leaves(weights)
        n_params = max(sum(leaf.size for leaf in leaves), 1)
        l2       = sum(jnp.sum(leaf ** 2) for leaf in leaves) / n_params
        return 1.0 - kta + self.lambda_kao * l2

    def _loss_co_quack(
        self,
        weights,
        main_centroids: jnp.ndarray,
        main_centroid_idx: int,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        l: float = -1.0,
    ) -> jnp.ndarray:
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

    def _kao_weight_update_quack(
        self,
        main_centroid: jnp.ndarray,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        l: float = 1.0,
    ) -> jnp.ndarray:
        grad_fn = jax.grad(self._loss_kao_quack, argnums=0)
        grads = grad_fn(self.weights, main_centroid, X_train, y_train, l)
        updates, self._kao_opt_state = self._kao_optimizer.update(grads, self._kao_opt_state)
        return ox.apply_updates(self.weights, updates)

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
        updates, self._co_opt_state = self._co_optimizer.update(grads, self._co_opt_state)
        new_main = ox.apply_updates(self.main_centroids, updates)
        return jnp.clip(new_main, self._feat_min, self._feat_max)

    def _get_batch(self, epoch: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.xtrain, self.ytrain

    def align(self) -> dict[str, Any]:
        """
        QUACK alternating KAO / CO optimisation loop.
        Mirrors PyTorch fit_kernel with train_method='quack'.

        New history keys (Experiments 1, 2, 4):
          centroid_kta_history  -- TA in centroid space (per outer epoch)
          block_ratio_history   -- within/between class kernel ratio
          coords                -- list of (N_train, 2) kPCA projections
          coords_labels         -- ytrain (constant)
        """
        alignment_hist:    list[float]       = []
        centroid_kta_hist: list[float]       = []   # Experiment 1
        block_ratio_hist:  list[float]       = []   # Experiment 2
        loss_hist:         list[float]       = []
        train_acc, validation_acc, svm_margin = [], [], []
        main_cent_hist:    list[jnp.ndarray] = []
        coords_hist:       list[np.ndarray]  = []   # Experiment 4

        unique_labels_np = np.unique(np.asarray(self.ytrain))
        n_cls            = len(unique_labels_np)

        best_validation_acc = -np.inf
        best_weights        = self.weights
        best_main_centroids = self.main_centroids

        init  = self.svm_training(self.xtrain, self.ytrain)
        start = time.perf_counter()

        for epoch in range(self.epochs):
            cl_kao = unique_labels_np[epoch % n_cls]
            l_kao  = float(cl_kao)
            l_co   = -float(cl_kao)

            main_idx      = int(jnp.argmax(self.main_centroid_labels == cl_kao))
            main_centroid = self.main_centroids[main_idx]
            y_kao = jnp.where(self.ytrain == cl_kao, 1.0, -1.0)

            # Inner loop: alternate KAO / CO every other outer epoch
            if epoch % 2 == 0:
                for _ in range(self.inner_steps):
                    self.weights = self._kao_weight_update_quack(
                        main_centroid, self.xtrain, y_kao, l=l_kao
                    )
                    result = self.svm_training(self.xtrain, self.ytrain)
                    train_acc.append(result["train_accuracy"])
                    validation_acc.append(result["validation_accuracy"])
                    if result["validation_accuracy"] > best_validation_acc:
                        best_validation_acc = result["validation_accuracy"]
                        best_weights        = self.weights
                        best_main_centroids = self.main_centroids
            else:
                for _ in range(self.inner_steps):
                    self.main_centroids = self._main_centroid_gradient_update_quack(
                        main_idx, self.xtrain, y_kao, l=l_co
                    )
                    result = self.svm_training(self.xtrain, self.ytrain)
                    train_acc.append(result["train_accuracy"])
                    validation_acc.append(result["validation_accuracy"])
                    if result["validation_accuracy"] > best_validation_acc:
                        best_validation_acc = result["validation_accuracy"]
                        best_weights        = self.weights
                        best_main_centroids = self.main_centroids

            if self.collect_diagnostics:
                alignment_hist.append(
                    float(self.alignment(self.weights, self.xtrain, self.ytrain))
                )
                loss_hist.append(
                    float(self._loss_kao_quack(
                        self.weights,
                        self.main_centroids[main_idx],
                        self.xtrain, y_kao, l_kao,
                    ))
                )
                centroid_kta_hist.append(
                    self.centroid_kta(self.weights, self.xtrain, self.ytrain)
                )
                block_ratio_hist.append(
                    self.block_diagonal_ratio(
                        self.weights, self.xtrain, self.ytrain
                    )
                )

            main_cent_hist.append(self.main_centroids)

            result = self.svm_training(self.xtrain, self.ytrain)
            svm_margin.append(result["margin"])
            if result["validation_accuracy"] > best_validation_acc:
                best_validation_acc = result["validation_accuracy"]
                best_weights        = self.weights
                best_main_centroids = self.main_centroids

            if self.collect_diagnostics:
                K_train = np.asarray(
                    self._kernel_matrix(self.weights, self.xtrain)
                )
                coords_hist.append(self.kernel_pca(K_train))

        # Restore best checkpoint
        self.weights        = best_weights
        self.main_centroids = best_main_centroids

        final_result = self.svm_training(
            self.xtrain, self.ytrain, evaluate_test=True
        )

        return {
            "weights":                self.weights,
            "main_centroids":         main_cent_hist,
            "init_train_accuracy":    init["train_accuracy"],
            "init_validation_accuracy": init["validation_accuracy"],
            "init_test_accuracy":     float("nan"),
            "alignment_history":      alignment_hist,
            "centroid_kta_history":   centroid_kta_hist,   # Experiment 1
            "block_ratio_history":    block_ratio_hist,    # Experiment 2
            "loss_history":           loss_hist,
            "train_accuracy_history": train_acc,
            "validation_accuracy_history": validation_acc,
            "test_accuracy_history":  [final_result["test_accuracy"]],
            "best_validation_accuracy": float(best_validation_acc),
            "final_svm_metrics":      final_result,
            "f1_score_history":       [final_result["f1_score"]],
            "precision_score_history":[final_result["precision_score"]],
            "recall_score_history":   [final_result["recall_score"]],
            "margin_history":         svm_margin,
            "time":                   time.perf_counter() - start,
            "circuit_executions":     self.kernel_model.circuit_executions,
            "coords":                 coords_hist,          # Experiment 4
            "coords_labels":          self.ytrain,          # Experiment 4
        }


# ---------------------------------------------------------------------------
# Backward-compatible lowercase aliases
# ---------------------------------------------------------------------------

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