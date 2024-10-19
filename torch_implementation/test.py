import pennylane as qml
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_gaussian_quantiles

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Generate dataset
X, y = make_gaussian_quantiles(
    n_samples=1000, n_features=2, n_classes=2, random_state=0
)
y_ = torch.unsqueeze(torch.tensor(y), 1)  # used for one-hot encoded labels
y_hot = torch.scatter(torch.zeros((1000, 2)), 1, y_, 1)

# Plotting (optional)
c = ["#1f77b4" if y_ == 0 else "#ff7f0e" for y_ in y]  # colors for each class
plt.axis("off")
plt.scatter(X[:, 0], X[:, 1], c=c)
plt.show()

# Quantum device and parameters
n_qubits = 2
n_layers = 6
dev = qml.device("default.qubit", wires=n_qubits)

def embedding(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

@qml.qnode(dev, interface='torch', diff_method='backprop')
def qnode(input_pair, weights):
    x1 = input_pair[0]  # Shape: (n_qubits,)
    x2 = input_pair[1]  # Shape: (n_qubits,)

    # Apply embedding to x1
    embedding(x1, weights[0])

    # Apply adjoint embedding to x2
    qml.adjoint(embedding)(x2, weights[1])

    return qml.expval(qml.PauliZ(wires=0))

# Wrap the qnode with vmap
from torch.func import vmap

batched_qnode = vmap(qnode, in_dims=(0, None), out_dims=0)

class BatchedQNode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(2, n_layers, n_qubits))

    def forward(self, inputs):
        return batched_qnode(inputs, self.weights)

model = BatchedQNode()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

X = torch.tensor(X).float()
y_hot = y_hot.float()

# Prepare inputs and labels
inputs = torch.stack([torch.stack([X[i], X[i]]) for i in range(len(X))])  # Shape: (1000, 2, 2)
labels = y_hot[:, 0]  # Use one class label for simplicity

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(inputs).squeeze()  # Shape: (1000,)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
