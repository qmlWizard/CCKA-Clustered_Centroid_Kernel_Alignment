import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from utils.kernel import initialize_kernel, kernel, get_circuit_executions
from utils.classification_data import checkerboard_data, linear_data, hidden_manifold_data, power_line_data, microgrid_data
from utils.train_kernel import target_alignment
from utils.sampling import subset_sampling
from utils.sampling import approx_greedy_sampling
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


data = microgrid_data()
print("Done..!")
print("Sample: ", data.head(1))
print("Data Size: ", data.shape)

features = np.asarray(data[[col for col in data.columns if col != 'target']].values.tolist())
target = np.asarray(data['target'].values.tolist())

print("Configuring Quantum Circuit")

n_qubits = len(features[0])
layers = 1

print("Number of Qubits: ", n_qubits)
print("Number of Variational Layers: ", layers)

wires, shape = initialize_kernel(n_qubits, 'strong_entangled', 1)
param_shape = (2,) + shape
params = np.random.uniform(0, 2 * np.pi, size=param_shape, requires_grad=True)

print("Shape for params: ", param_shape)
print("Dividing Testing and Training dataset")

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print("Train Size: ", len(x_train))
print("Test Size: ", len(x_test))

f_kernel = lambda x1, x2: kernel(x1, x2, params)
get_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, f_kernel)

km = get_kernel_matrix(x_test, x_test)
np.set_printoptions(precision=2, suppress=True)
print(km)

drawer = qml.draw(kernel)
print(drawer(x_train[0], x_train[105], params))
print(kernel(x_train[0], x_train[105], params))
