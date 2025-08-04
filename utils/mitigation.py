import numpy as np
from utils.helper import to_numpy_kernel
import torch
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

class Mitigation:
    """
    Implements noise mitigation strategies for quantum kernel matrices.
    Supports:
        - M-SINGLE : Use first diagonal to estimate global survival probability
        - M-MEAN   : Use mean of diagonals
        - M-SPLIT  : Use element-wise survival probability (per datapoint)
        - ML-GLOBAL: (placeholder for ML global decay)
        - ML-PAIRWISE: (placeholder for ML pairwise decay)
    """

    def __init__(self, method, ml_model=None, degree=None, alpha=None, circuit_depth=None, n_qubits=None):
        self._method = method
        self._ml_model = ml_model
        self._degree = degree
        self._alpha = alpha
        self._circuit_depth = circuit_depth
        self._n_qubits = n_qubits

        self._trained_model = None

        _method_func_dict = {
            "M-MEAN": self._m_mean,
            "M-SINGLE": self._m_single,
            "M-SPLIT": self._m_split,
            "ML-GLOBAL": self._ml_global_decay,
            "ML-PAIRWISE": self._ml_pairwise_decay,
        }

        if self._method not in _method_func_dict:
            raise ValueError(f"Unknown mitigation method: {self._method}")

        self._method_func = _method_func_dict[self._method]
        self._yi = None  # survival probabilities λ_i

        if self._ml_model == 'POLY_REG':
            self._poly = PolynomialFeatures(degree = self._degree)
            self._trained_model = Ridge(alpha=alpha)

    # ------------------- Mitigation Strategies -------------------

    def _m_single(self, diag_vals):
        """
        Use the first diagonal entry to compute a single global survival probability λ.
        """
        diag_val = diag_vals[0]
        return np.full(len(diag_vals), np.sqrt(max(diag_val, 1e-12)))

    def _m_mean(self, diag_vals):
        """
        Use the mean of diagonal entries to compute a global survival probability λ.
        """
        mean_val = np.mean(diag_vals)
        return np.full(len(diag_vals), np.sqrt(max(mean_val, 1e-12)))

    def _m_split(self, diag_vals):
        """
        Compute survival probability λ_i for each data point individually
        using the square root of its diagonal entry.
        """
        return np.sqrt(np.clip(diag_vals, 1e-12, None))

    # ------------------- ML-Based Placeholders -------------------

    def train_global_decay(self, diag_vals):
        """
        Train polynomial ridge regression model for ML-GLOBAL decay.
        Includes circuit depth and number of qubits as features.
        """
        n_samples = len(diag_vals)

        # Feature: [index, circuit_depth, n_qubits]
        X = np.column_stack([
            np.arange(n_samples),
            np.full(n_samples, self._circuit_depth),
            np.full(n_samples, self._n_qubits)
        ])

        X_poly = self._poly.fit_transform(X)
        y = np.sqrt(np.clip(diag_vals, 1e-12, None))  # initial λ_i

        self._trained_model.fit(X_poly, y)
        print("[INFO] ML-GLOBAL model trained with depth and qubits features.")

    def train_pairwise_decay(self):
        """Placeholder for ML model to learn pairwise decay factors λ_ij."""
        pass

    def _ml_global_decay(self, diag_vals):
        """
        Predict λ_i using trained polynomial ridge regression model.
        """
        if self._trained_model is None:
            raise RuntimeError("Train the ML-GLOBAL model first using train_global_decay().")

        n_samples = len(diag_vals)
        X = np.column_stack([
            np.arange(n_samples),
            np.full(n_samples, self._circuit_depth if self._circuit_depth is not None else self._circuit_depth),
            np.full(n_samples, self._n_qubits if self._n_qubits is not None else self._n_qubits)
        ])

        X_poly = self._poly.transform(X)
        return self._trained_model.predict(X_poly)

    def _ml_pairwise_decay(self, diag_vals):
        """
        Placeholder for ML-based pairwise decay estimation.
        Should output per-point λ_i as in M-SPLIT for now.
        """
        return self._m_split(diag_vals)  # fallback

    # ------------------- Utility -------------------

    def get_survival_probability(self, diag_vals):
        """
        Compute λ_i for the chosen mitigation method using diagonal entries only.
        Handles PyTorch tensors that require grad.
        """
        # Convert to NumPy safely
        if hasattr(diag_vals, "detach"):  # PyTorch tensor
            diag_vals = diag_vals.detach().cpu().numpy()
        elif not isinstance(diag_vals, np.ndarray):
            diag_vals = np.array(diag_vals, dtype=np.float64)

        self._yi = self._method_func(diag_vals.astype(np.float64))
        print("Survival probability λ_i:", self._yi)

    def mitigate(self, kernel_matrix):
        """
        Apply the selected mitigation strategy to the given kernel matrix.
        """
        kernel_matrix = to_numpy_kernel(kernel_matrix)

        if self._yi is None:
            raise ValueError("Survival probabilities not computed. "
                             "Call get_survival_probability() first.")

        kernel_matrix_adjusted = np.zeros_like(kernel_matrix)

        for i in range(kernel_matrix.shape[0]):
            for j in range(kernel_matrix.shape[1]):
                if self._yi[i] * self._yi[j] > 0.0:
                    kernel_matrix_adjusted[i, j] = kernel_matrix[i, j] / (self._yi[i] * self._yi[j])
                else:
                    kernel_matrix_adjusted[i, j] = 0.0

        if kernel_matrix_adjusted.shape[0] == kernel_matrix_adjusted.shape[1]:
            np.fill_diagonal(kernel_matrix_adjusted, 1.0)

        kernel_matrix_tensor = torch.tensor(kernel_matrix_adjusted)

        return kernel_matrix_tensor
