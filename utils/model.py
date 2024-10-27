import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.ansatz import qkhe, qkcovariant, qkembedding_paper
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import json
import os

torch.manual_seed(42)
np.random.seed(42)

class qkernel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self._device = config['qkernel']['device']
        self._n_qubits = config['qkernel']['n_qubits']
        self._trainable = config['qkernel']['trainable']
        self._input_scaling = config['qkernel']['input_scaling']
        self._data_reuploading = config['qkernel']['data_reuploading']
        self._ansatz = config['qkernel']['ansatz']
        self._layers = config['qkernel']['ansatz_layers']
        self._wires = range(self._n_qubits)
        self._projector = torch.zeros((2**self._n_qubits,2**self._n_qubits))
        self._projector[0,0] = 1
        self._circuit_executions = 0

        if self._ansatz == 'he':
            if self._input_scaling:
                self.register_parameter(name="input_scaling", param= nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=True))
            else:
                self.register_parameter(name="input_scaling", param= nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=True))
            self.register_parameter(name="variational", param= nn.Parameter(torch.ones(self._layers, self._n_qubits * 2) * 2 * torch.pi, requires_grad=True))

        elif self._ansatz == 'embedding_paper':
            if self._input_scaling:
                self.register_parameter(name="input_scaling", param= nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=True))
            else:
                self.register_parameter(name="input_scaling", param= nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=True))
            self.register_parameter(name="variational", param= nn.Parameter(torch.ones(self._layers, self._n_qubits) * 2 * torch.pi, requires_grad=True))
            self.register_parameter(name="rotational", param= nn.Parameter(torch.ones(self._layers, self._n_qubits) * 2 * torch.pi, requires_grad=True))

        dev = qml.device(self._device, wires = range(self._n_qubits))
        if self._ansatz == 'he':
            self._kernel = qml.QNode(qkhe, dev, diff_method='adjoint', interface='torch')
        if self._ansatz == 'embedding_paper':
            self._kernel = qml.QNode(qkembedding_paper, dev, diff_method='adjoint', interface='torch')
        if self._ansatz == 'covariant':
            self._kernel = qml.QNode(qkhe, dev, diff_method='adjoint', interface='torch')
        
    def forward(self, x1, x2):
        all_zero_state = self._kernel(x1, x2, self._parameters, self._wires, self._layers, self._projector, self._data_reuploading)
        self._circuit_executions += 1
        return all_zero_state