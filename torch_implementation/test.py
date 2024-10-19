import pennylane as qml
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_gaussian_quantiles

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

n_qubits = 2

# Adjusted: Remove batch_size since it's not needed without vectorization
dev = qml.device("default.qubit", wires=n_qubits)

def embedding(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(2))
    qml.BasicEntanglerLayers(weights, wires=range(2))

@qml.qnode(dev, interface='torch', diff_method='backprop')
def qnode(inputs, weights):
    x1 = inputs[ :len(X[0])]
    x2 = inputs[len(X[0]): ]

    # Apply embedding to x1
    embedding(x1, weights[0])

    # Apply adjoint embedding to x2
    qml.adjoint(embedding)(x2, weights[1])

    return qml.expval(qml.PauliZ(wires=0))

n_layers = 6
weight_shapes = {"weights": (2, n_layers, n_qubits)}

# Corrected: Remove input_dims
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

model = torch.nn.Sequential(qlayer)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.L1Loss()

X = torch.tensor(X).float()
y_hot = y_hot.float()

# Prepare inputs: since batch processing isn't set up, use individual inputs
#inputs = torch.stack(X[0] + X[1])

# Run the model
output = model(X[0] + X[1])
print(output)   