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


data = linear_data(3, 10)
print("Done..!")
print("Sample: ", data.head(1))
print("Data Size: ", data.shape)

features = np.asarray(data[[col for col in data.columns if col != 'target']].values.tolist())
target = np.asarray(data['target'].values.tolist())

print("Configuring Quantum Circuit")

n_qubits = len(features[0])
layers = 6

print("Number of Qubits: ", n_qubits)
print("Number of Variational Layers: ", layers)

wires, shape = initialize_kernel(n_qubits, 'tutorial_ansatz', layers)
param_shape = (2,) + shape
params = np.random.uniform(0, 2 * np.pi, size=param_shape, requires_grad=True)

print("Shape for params: ", param_shape)
print("Dividing Testing and Training dataset")

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print("Train Size: ", len(x_train))
print("Test Size: ", len(x_test))

f_kernel = lambda x1, x2: kernel(x1, x2, params)
get_kernel_matrix = qml.kernels.square_kernel_matrix(x_train, f_kernel, assume_normalized_kernel = True)


#km = get_kernel_matrix(x_train, x_train)
np.set_printoptions(precision=2, suppress=True)
print(get_kernel_matrix)

drawer = qml.draw(kernel)
print(drawer(x_train[0], x_train[5], params))
print(kernel(x_train[0], x_train[5], params))
print("Classical Dot Product Kernel: ", np.dot(x_train[0], x_train[5])/ x_train[0] @ x_train[5] )
