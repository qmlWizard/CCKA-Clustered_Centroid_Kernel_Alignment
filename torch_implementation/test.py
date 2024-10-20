import pennylane as qml
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_gaussian_quantiles
import torch
import pennylane as qml
from torch import nn

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"MPS is available. Using device: {device}")
else:
    print("MPS is not available. Using CPU.")

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

X, y = make_gaussian_quantiles(
    n_samples=1000, n_features=2, n_classes=2, random_state=0
)
y_ = torch.unsqueeze(torch.tensor(y), 1)  # used for one-hot encoded labels
y_hot = torch.scatter(torch.zeros((1000, 2)), 1, y_, 1)

c = ["#1f77b4" if y_ == 0 else "#ff7f0e" for y_ in y]  # colors for each class
plt.axis("off")
plt.scatter(X[:, 0], X[:, 1], c=c)
plt.show()

# Define the number of qubits and device
n_qubits = len(X[0])
dev = qml.device("default.qubit", wires=n_qubits)

# Define a simple embedding circuit
def embedding_circuit(x):
    # Data embedding into quantum circuit
    for i in range(n_qubits):
        qml.Hadamard(wires=i)  # Example of state preparation
        qml.RX(x[i], wires=i)  # Example of data embedding with RX rotation

# Define the adjoint embedding circuit
def adjoint_embedding_circuit(x):
    # Apply the adjoint (conjugate transpose) of the embedding
    for i in range(n_qubits):
        qml.Hadamard(wires=i)  # Inverse of state preparation
        qml.RX(-x[i], wires=i)  # Inverse RX rotation

# Define the Quantum Kernel Layer
class QuantumKernelLayer(nn.Module):
    def __init__(self):
        super(QuantumKernelLayer, self).__init__()

        
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

# Example usage
quantum_kernel_layer = QuantumKernelLayer()

kernel_output = quantum_kernel_layer(X[0], X[1])
print(f"Quantum Kernel Output: {kernel_output}")

def create_kernel_matrix(X, quantum_kernel_layer):
    n_samples = len(X)
    kernel_matrix = torch.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            # Compute the kernel value for each pair (X[i], X[j])
            kernel_value = quantum_kernel_layer(X[i], X[j])
            kernel_matrix[i, j] = kernel_value
            kernel_matrix[j, i] = kernel_value  # Kernel matrix is symmetric
        print(i)

    return kernel_matrix

# Compute the kernel matrix
kernel_matrix = create_kernel_matrix(X, quantum_kernel_layer)

# Display the kernel matrix
print(f"Quantum Kernel Matrix:\n{kernel_matrix}")
