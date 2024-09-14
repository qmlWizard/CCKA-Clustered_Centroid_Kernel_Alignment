import pennylane as qml
from pennylane import numpy as np
from utils.kernel import initialize_kernel, kernel

wires, shape = initialize_kernel(3, 'strong_entangled', 2)

param_shape = (2,) + shape

params = np.random.random(size = param_shape)

print(param_shape)

x1 = [1, 2, 3]
x2 = [1, 2, 3]

distance = kernel(x1, x2, params)

print(distance)
