from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit_aer.noise import NoiseModel
from qiskit.test.mock import FakeLima
import numpy as np
import torch

############################################
# Feature Map Circuit Construction Methods #
############################################

def he_layer(qc, x, scaling, variational, wires, embed, reupload):
    if embed or reupload:
        for i, wire in enumerate(wires):
            qc.rx(scaling[i] * x[i], wire)
    for i, wire in enumerate(wires):
        qc.ry(variational[i], wire)
    for i, wire in enumerate(wires):
        qc.rz(variational[i + len(wires)], wire)
    if len(wires) == 2:
        qc.cz(wires[0], wires[1])
    else:
        for i in range(len(wires)):
            qc.cz(wires[i], wires[(i + 1) % len(wires)])

def embedding_paper_layer(qc, x, scaling, variational, rotational, wires, embed, reupload):
    for wire in wires:
        qc.h(wire)
    if embed or reupload:
        for i, wire in enumerate(wires):
            qc.rz(scaling[i] * x[i], wire)
    for i, wire in enumerate(wires):
        qc.ry(variational[i], wire)
    for i in range(len(wires)):
        qc.crz(rotational[i], wires[i], wires[(i + 1) % len(wires)])

def build_he_circuit(x, weights, wires, layers, reupload):
    qc = QuantumCircuit(len(wires))
    for l in range(layers):
        he_layer(qc, x, weights["input_scaling"][l], weights["variational"][l], wires, l == 0, reupload)
    return qc

def build_embedding_paper_circuit(x, weights, wires, layers, reupload):
    qc = QuantumCircuit(len(wires))
    for l in range(layers):
        embedding_paper_layer(
            qc, x,
            weights["input_scaling"][l],
            weights["variational"][l],
            weights["rotational"][l],
            wires, l == 0, reupload
        )
    return qc

####################################
# Statevector Kernel Computation   #
####################################

def compute_kernel_statevector(circ1, circ2):
    state1 = Statevector(circ1)
    state2 = Statevector(circ2)
    fidelity = np.abs(state1.inner(state2)) ** 2
    return fidelity

def qk_he(x1, x2, weights, wires, layers, projector=None, reupload=True):
    x1 = np.resize(x1, len(wires))
    x2 = np.resize(x2, len(wires))

    circ1 = build_he_circuit(x1, weights, wires, layers, reupload)
    circ2 = build_he_circuit(x2, weights, wires, layers, reupload)

    kernel_val = compute_kernel_statevector(circ1, circ2)
    return torch.tensor(kernel_val, dtype=torch.float32)

def qk_embedding_paper(x1, x2, weights, wires, layers, projector=None, reupload=True):
    x1 = np.resize(x1, len(wires))
    x2 = np.resize(x2, len(wires))

    circ1 = build_embedding_paper_circuit(x1, weights, wires, layers, reupload)
    circ2 = build_embedding_paper_circuit(x2, weights, wires, layers, reupload)

    kernel_val = compute_kernel_statevector(circ1, circ2)
    return torch.tensor(kernel_val, dtype=torch.float32)


##################################
# Quantum Circuit Layer Builders #
##################################

def he_layer(qc, x, scaling, variational, wires, embed, reupload):
    if embed or reupload:
        for i, wire in enumerate(wires):
            qc.rx(scaling[i] * x[i], wire)
    for i, wire in enumerate(wires):
        qc.ry(variational[i], wire)
    for i, wire in enumerate(wires):
        qc.rz(variational[i + len(wires)], wire)
    if len(wires) == 2:
        qc.cz(wires[0], wires[1])
    else:
        for i in range(len(wires)):
            qc.cz(wires[i], wires[(i + 1) % len(wires)])

def embedding_paper_layer(qc, x, scaling, variational, rotational, wires, embed, reupload):
    for wire in wires:
        qc.h(wire)
    if embed or reupload:
        for i, wire in enumerate(wires):
            qc.rz(scaling[i] * x[i], wire)
    for i, wire in enumerate(wires):
        qc.ry(variational[i], wire)
    for i in range(len(wires)):
        qc.crz(rotational[i], wires[i], wires[(i + 1) % len(wires)])

def build_he_circuit(x, weights, wires, layers, reupload):
    qc = QuantumCircuit(len(wires))
    for l in range(layers):
        he_layer(qc, x, weights["input_scaling"][l], weights["variational"][l], wires, l == 0, reupload)
    return qc

def build_embedding_paper_circuit(x, weights, wires, layers, reupload):
    qc = QuantumCircuit(len(wires))
    for l in range(layers):
        embedding_paper_layer(
            qc, x,
            weights["input_scaling"][l],
            weights["variational"][l],
            weights["rotational"][l],
            wires, l == 0, reupload
        )
    return qc

###################################
# Noisy Backend and Kernel Helper #
###################################

# Fidelity estimation using adjoint method
def compute_noisy_kernel(circ1, circ2, wires, simulator, shots=8192):
    full_circ = circ2.inverse().compose(circ1)
    full_circ.measure_all()
    transpiled = transpile(full_circ, simulator)
    result = simulator.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    zero_state = '0' * len(wires)
    p0 = counts.get(zero_state, 0) / shots
    return p0

##################################
# Noisy Kernel Evaluation Method #
##################################

def qk_he_noisy_single(x1, x2, weights, wires, layers, simulator, reupload=True, shots=8192):
    x1 = np.resize(x1, len(wires))
    x2 = np.resize(x2, len(wires))

    circ1 = build_he_circuit(x1, weights, wires, layers, reupload)
    circ2 = build_he_circuit(x2, weights, wires, layers, reupload)

    kernel_val = compute_noisy_kernel(circ1, circ2, wires, simulator, shots)
    return torch.tensor(kernel_val, dtype=torch.float32)

def qk_embedding_paper_noisy_single(x1, x2, weights, wires, layers, simulator, reupload=True, shots=8192):
    x1 = np.resize(x1, len(wires))
    x2 = np.resize(x2, len(wires))

    circ1 = build_embedding_paper_circuit(x1, weights, wires, layers, reupload)
    circ2 = build_embedding_paper_circuit(x2, weights, wires, layers, reupload)

    kernel_val = compute_noisy_kernel(circ1, circ2, wires, simulator, shots)
    return torch.tensor(kernel_val, dtype=torch.float32)