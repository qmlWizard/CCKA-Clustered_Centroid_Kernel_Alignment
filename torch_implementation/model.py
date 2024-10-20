import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn
import yaml
from torch_implementation.ansatz import *


torch.manual_seed(42)
np.random.seed(42)



class QuantumKernelLayer(nn.Module):
    def __init__(self):
        super(QuantumKernelLayer, self).__init__()
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

        n_qubits = config['model']['n_qubits']
        device = config['quantum_kernel']['device']
        dev = qml.device(device, wires=n_qubits)

        ansatz = config['ansatz']['type']
        layers = config['ansatz']['layers']


        
    @qml.qnode(dev, interface = 'torch')
    def kernel(self, x1, x2):
        embedding_circuit(x1)
        adjoint_embedding_circuit(x2)

        return qml.probs(wires = range(n_qubits))

    def forward(self, x1, x2):
        # Apply the embedding and its adjoint
        embedding1 = embedding_circuit(x1)
        adjoint_embedding2 = adjoint_embedding_circuit(x2)
        
        # Kernel as the overlap of embeddings (i.e., dot product or a chosen expectation value)
        kernel_value = self.kernel(self, x1, x2)[0]
        return kernel_value
