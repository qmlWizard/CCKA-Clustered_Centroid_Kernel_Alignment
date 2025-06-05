from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import RXGate, RYGate, RZGate, CRZGate
import numpy as np
import torch

# Helper to apply HE layer
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

def covariant_layer(qc, x, scaling, variational, wires, embed, reupload, entanglement=None):
    if entanglement is None:
        entanglement = [[i, i + 1] for i in range(len(wires) - 1)]
    for i, wire in enumerate(wires):
        qc.ry(variational[i + len(wires)], wire)
    for source, target in entanglement:
        qc.cz(source, target)
    if embed or reupload:
        for i, wire in enumerate(wires):
            qc.rz(scaling[i] * x[2 * i + 1], wire)
            qc.rx(scaling[i] * x[2 * i], wire)

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

# Higher-level wrappers
def build_he_circuit(x, weights, wires, layers, reupload):
    qc = QuantumCircuit(len(wires))
    for l in range(layers):
        he_layer(qc, x, weights["input_scaling"][l], weights["variational"][l], wires, l == 0, reupload)
    return qc

def build_covariant_circuit(x, weights, wires, layers, reupload, entanglement=None):
    qc = QuantumCircuit(len(wires))
    for l in range(layers):
        covariant_layer(qc, x, weights["input_scaling"][l], weights["variational"][l], wires, l == 0, reupload, entanglement)
    return qc

def build_embedding_paper_circuit(x, weights, wires, layers, reupload):
    qc = QuantumCircuit(len(wires))
    for l in range(layers):
        embedding_paper_layer(qc, x, weights["input_scaling"][l], weights["variational"][l], weights["rotational"][l], wires, l == 0, reupload)
    return qc

# Quantum kernel computation using statevector
def compute_kernel(circ1, circ2, projector, wires):
    full_circ = circ1.compose(circ2.inverse())
    state = Statevector.from_instruction(full_circ)
    proj_op = Operator(projector)
    return np.real(state.expectation_value(proj_op))

def qk_he(x1, x2, weights, wires, layers, projector, reupload):
    """
    Batched kernel evaluation using the HE ansatz.
    Expects x1 and x2 to be torch tensors of shape [B, D]
    """
    if isinstance(x1, torch.Tensor): x1 = x1.detach().cpu().numpy()
    if isinstance(x2, torch.Tensor): x2 = x2.detach().cpu().numpy()
    
    results = []
    for i in range(x1.shape[0]):
        _x1 = np.resize(x1[i], len(wires))
        _x2 = np.resize(x2[i], len(wires))
        circ1 = build_he_circuit(_x1, weights, wires, layers, reupload)
        circ2 = build_he_circuit(_x2, weights, wires, layers, reupload)
        kernel_val = compute_kernel(circ1, circ2, projector, wires)
        results.append(kernel_val)
    return torch.tensor(results, dtype=torch.float32)

def qk_covariant(x1, x2, weights, wires, layers, projector, reupload, entanglement=None):
    if isinstance(x1, torch.Tensor): x1 = x1.detach().cpu().numpy()
    if isinstance(x2, torch.Tensor): x2 = x2.detach().cpu().numpy()
    
    results = []
    for i in range(x1.shape[0]):
        _x1 = np.resize(x1[i], len(wires))
        _x2 = np.resize(x2[i], len(wires))
        circ1 = build_covariant_circuit(_x1, weights, wires, layers, reupload, entanglement)
        circ2 = build_covariant_circuit(_x2, weights, wires, layers, reupload, entanglement)
        kernel_val = compute_kernel(circ1, circ2, projector, wires)
        results.append(kernel_val)
    return torch.tensor(results, dtype=torch.float32)

def qk_embedding_paper(x1, x2, weights, wires, layers, projector, reupload):
    if isinstance(x1, torch.Tensor): x1 = x1.detach().cpu().numpy()
    if isinstance(x2, torch.Tensor): x2 = x2.detach().cpu().numpy()
    
    results = []
    for i in range(x1.shape[0]):
        _x1 = np.resize(x1[i], len(wires))
        _x2 = np.resize(x2[i], len(wires))
        circ1 = build_embedding_paper_circuit(_x1, weights, wires, layers, reupload)
        circ2 = build_embedding_paper_circuit(_x2, weights, wires, layers, reupload)
        kernel_val = compute_kernel(circ1, circ2, projector, wires)
        results.append(kernel_val)
    return torch.tensor(results, dtype=torch.float32)