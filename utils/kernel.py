import pennylane as qml
from pennylane import numpy as np
from utils.variational_circuits import strong_entangled
np.random.seed(1359)

dev = qml.device("default.qubit", wires=5, shots=None)
wires = dev.wires.tolist()
layers = 1

def initialize_kernel(num_qubits, variational_circuit, variational_layers):
    global dev
    global wires
    global layers

    dev = qml.device("default.qubit", 
                      wires= num_qubits, 
                      shots=None
                    )
    wires = dev.wires.tolist()
    
    if variational_circuit == 'strong_entangled':
        shape = qml.StronglyEntanglingLayers.shape(n_layers= variational_layers, n_wires= num_qubits)
    
    layers = variational_layers

    return wires, shape

def encoding(x1, params):
    qml.AngleEmbedding(features = x1, wires = wires, rotation='X')
    for l in range(layers):
        p = params[l].reshape(1, 3, 3)
        strong_entangled(p, wires)
    
@qml.qnode(dev)
def kernel_circuit(x1, x2, params):
    
    encoding(x1, params[0])
    adjoint_circuit = qml.adjoint(encoding)
    adjoint_circuit(x2, params[1])

    return qml.probs(wires = wires)

def kernel(x1, x2, params):
    return kernel_circuit(x1, x2, params)[0]
