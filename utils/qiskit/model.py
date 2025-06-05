import torch
import torch.nn as nn
import math
import numpy as np

from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from utils.qiskit.ansatz import qk_he, qk_covariant, qk_embedding_paper 

class Qkernel(nn.Module):
    def __init__(self, device, n_qubits, trainable, input_scaling, data_reuploading, ansatz, ansatz_layers):
        super().__init__()

        self._device = device
        self._n_qubits = n_qubits
        self._trainable = trainable
        self._input_scaling = input_scaling
        self._data_reuploading = data_reuploading
        self._ansatz = ansatz
        self._layers = ansatz_layers
        self._wires = list(range(self._n_qubits))
        self._projector = torch.zeros((2**self._n_qubits, 2**self._n_qubits))
        self._projector[0, 0] = 1.0
        self._circuit_executions = 0

        if self._ansatz == 'he':
            self.input_scaling = nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=self._trainable)
            self.variational = nn.Parameter((torch.rand(self._layers, self._n_qubits * 2) * 2 * math.pi) - math.pi, requires_grad=self._trainable)

        elif self._ansatz == 'embedding_paper':
            self.input_scaling = nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=self._trainable)
            self.variational = nn.Parameter((torch.rand(self._layers, self._n_qubits) * 2 * math.pi) - math.pi, requires_grad=self._trainable)
            self.rotational = nn.Parameter((torch.rand(self._layers, self._n_qubits) * 2 * math.pi) - math.pi, requires_grad=self._trainable)

        elif self._ansatz == 'covariant':
            self.input_scaling = nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=self._trainable)
            self.variational = nn.Parameter((torch.rand(self._layers, self._n_qubits * 2) * 2 * math.pi) - math.pi, requires_grad=self._trainable)

    def forward(self, x1, x2):
        x1 = x1.detach().cpu().numpy()
        x2 = x2.detach().cpu().numpy()
        weights = {}

        if self._ansatz == 'he':
            weights["input_scaling"] = self.input_scaling.detach().cpu().numpy()
            weights["variational"] = self.variational.detach().cpu().numpy()
            kernel_value = qk_he(x1, x2, weights, self._wires, self._layers, self._projector.numpy(), self._data_reuploading)

        elif self._ansatz == 'embedding_paper':
            weights["input_scaling"] = self.input_scaling.detach().cpu().numpy()
            weights["variational"] = self.variational.detach().cpu().numpy()
            weights["rotational"] = self.rotational.detach().cpu().numpy()
            kernel_value = qk_embedding_paper(x1, x2, weights, self._wires, self._layers, self._projector.numpy(), self._data_reuploading)

        elif self._ansatz == 'covariant':
            weights["input_scaling"] = self.input_scaling.detach().cpu().numpy()
            weights["variational"] = self.variational.detach().cpu().numpy()
            kernel_value = qk_covariant(x1, x2, weights, self._wires, self._layers, self._projector.numpy(), self._data_reuploading)

        else:
            raise ValueError("Unsupported ansatz!")

        self._circuit_executions += 1
        return torch.tensor(kernel_value, dtype=torch.float32)
