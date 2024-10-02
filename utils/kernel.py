import pennylane as qml
from pennylane import numpy as np
from utils.variational_circuits import strong_entangled, basic_entangled, tutorial_ansatz, efficientSU2

np.random.seed(1359)

# Initialize with 6 wires if needed
num_qubits = 6  # Adjust the number as needed
dev = qml.device("default.qubit", wires=num_qubits, shots=None)
wires = dev.wires.tolist()
layers = 1
circuit_executions = 0
ansatz = ''

def initialize_kernel(num_qubits, variational_circuit, variational_layers):
    global dev
    global wires
    global layers
    global ansatz
    
    dev = qml.device("default.qubit", wires=num_qubits, shots=None)
    wires = dev.wires.tolist()

    if variational_circuit == 'strong_entangled':
        shape = qml.StronglyEntanglingLayers.shape(n_layers=variational_layers, n_wires=num_qubits)
    elif variational_circuit == 'basic_entangled':
        shape = qml.BasicEntanglerLayers.shape(n_layers=variational_layers, n_wires=num_qubits)
    elif variational_circuit == 'tutorial_ansatz':
        shape = (variational_layers, 2, num_qubits)
    elif variational_circuit == 'efficientsu2':
        shape = (variational_layers, num_qubits, 4)
    layers = variational_layers
    ansatz = variational_circuit

    return wires, shape

def encoding(x1, params):
    if ansatz == 'tutorial_ansatz':
        tutorial_ansatz(x1, params, range(len(x1)))
    else:
        for l in range(layers):
            if ansatz == 'strong_entangled':
                qml.AngleEmbedding(features=x1, wires=wires, rotation='Z')
                req_shape = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=len(x1))
                p = params[l].reshape(req_shape)
                strong_entangled(p, wires)
            elif ansatz == 'basic_entangled':
                qml.AngleEmbedding(features=x1, wires=wires, rotation='Z')
                req_shape = qml.BasicEntanglerLayers.shape(n_layers=1, n_wires=len(x1))
                p = params[l].reshape(req_shape)
                basic_entangled(p, wires)
            elif ansatz == 'efficientsu2':
                qml.AngleEmbedding(features=x1, wires=wires, rotation='Z')
                req_shape = (1, len(x1), 4)
                p = params[l].reshape(req_shape)
                efficientSU2(p, wires)

@qml.qnode(dev)
def kernel_circuit(x1, x2, params):
    encoding(x1, np.asarray(params[0]))
    adjoint_circuit = qml.adjoint(encoding)
    adjoint_circuit(x2, np.asarray(params[0]))
    
    return qml.probs(wires=wires)

def kernel(x1, x2, params):
    global circuit_executions
    circuit_executions += 1
    return kernel_circuit(x1, x2, params)[0]

def get_circuit_executions():
    global circuit_executions
    return circuit_executions

