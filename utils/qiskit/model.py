import torch
import torch.nn as nn
import math
import numpy as np
from pennylane.measurements import shots

from utils.qiskit.ansatz import (
    qk_he_noisy_single,
    qk_embedding_paper_noisy_single,
    qk_he,
    qk_embedding_paper, get_circuit_depth
)

from qiskit import QuantumRegister

from utils.qiskit.grad import parameter_shift_rule, spsa_optimizer
from utils.qiskit.mitigation import Mitigation

class Qkernel(nn.Module):
    def __init__(
        self,
        n_qubits,
        trainable,
        input_scaling,
        data_reuploading,
        ansatz,
        ansatz_layers,
        use_noisy_backend=False,
        simulator=None,
        shots=8192,
    ):
        super().__init__()

        self._n_qubits = n_qubits
        self._trainable = trainable
        self._input_scaling = input_scaling
        self._data_reuploading = data_reuploading
        self._ansatz = ansatz
        self._layers = ansatz_layers
        self._wires = list(range(self._n_qubits))
        self._use_noisy_backend = use_noisy_backend
        self._simulator = simulator
        self._circuit_executions = 0

        if not self._use_noisy_backend:
            self._projector = torch.zeros((2**self._n_qubits, 2**self._n_qubits))
            self._projector[0, 0] = 1.0

        if self._ansatz == 'he':
            self.input_scaling = nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=self._trainable)
            self.variational = nn.Parameter(
                (torch.rand(self._layers, self._n_qubits * 2) * 2 * math.pi) - math.pi,
                requires_grad=self._trainable
            )

        elif self._ansatz == 'embedding_paper':
            self.input_scaling = nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=self._trainable)
            self.variational = nn.Parameter(
                (torch.rand(self._layers, self._n_qubits) * 2 * math.pi) - math.pi,
                requires_grad=self._trainable
            )
            self.rotational = nn.Parameter(
                (torch.rand(self._layers, self._n_qubits) * 2 * math.pi) - math.pi,
                requires_grad=self._trainable
            )

        elif self._ansatz == 'covariant':
            self.input_scaling = nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=self._trainable)
            self.variational = nn.Parameter(
                (torch.rand(self._layers, self._n_qubits * 2) * 2 * math.pi) - math.pi,
                requires_grad=self._trainable
            )

        else:
            raise ValueError(f"Unsupported ansatz: {self._ansatz}")

        self._shots = shots

    def get_depth(self, x1, x2):
        x1 = x1.detach().cpu().numpy().flatten()
        x2 = x2.detach().cpu().numpy().flatten()

        weights = {
            "input_scaling": self.input_scaling.detach().cpu().numpy(),
            "variational": self.variational.detach().cpu().numpy()
        }

        if self._ansatz == 'embedding_paper':
            weights["rotational"] = self.rotational.detach().cpu().numpy()

            circuit_depth = get_circuit_depth(
                    self._ansatz, x1, x2, weights, self._wires, self._layers, self._simulator, self._data_reuploading
            )

        return circuit_depth

    def forward(self, x1, x2):
        x1 = x1.detach().cpu().numpy().flatten()
        x2 = x2.detach().cpu().numpy().flatten()

        weights = {
            "input_scaling": self.input_scaling.detach().cpu().numpy(),
            "variational": self.variational.detach().cpu().numpy()
        }

        if self._ansatz == 'embedding_paper':
            weights["rotational"] = self.rotational.detach().cpu().numpy()

        # Select kernel evaluation function
        if self._ansatz == 'he':
            if self._use_noisy_backend:
                kernel_value = qk_he_noisy_single(
                    x1, x2, weights, self._wires, self._layers, self._simulator, self._data_reuploading, shots= self._shots
                )
            else:
                kernel_value = qk_he(
                    x1[np.newaxis, :], x2[np.newaxis, :], weights, self._wires, self._layers,
                    self._projector.numpy(), self._data_reuploading
                )

        elif self._ansatz == 'embedding_paper':
            if self._use_noisy_backend:
                kernel_value = qk_embedding_paper_noisy_single(
                    x1, x2, weights, self._wires, self._layers, self._simulator, self._data_reuploading, shots= self._shots
                )
            else:
                kernel_value = qk_embedding_paper(
                    x1[np.newaxis, :], x2[np.newaxis, :], weights, self._wires, self._layers,
                    self._projector.numpy(), self._data_reuploading
                )

        else:
            raise ValueError("Unsupported ansatz!")

        self._circuit_executions += 1
        return torch.tensor(kernel_value, dtype=torch.float32)

    def kernel_matrix(self, x1, x2):
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        kernel_matrix = torch.zeros((n1, n2), dtype=torch.float32)

        for i in range(n1):
            for j in range(n2):
                kernel_matrix[i, j] = k.forward(x1[i], x2[j])

        return kernel_matrix

    def fit(self, training_data, training_labels, testing_data, testing_labels):

