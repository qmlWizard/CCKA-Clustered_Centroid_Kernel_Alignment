from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.test.mock import FakeLima
import numpy as np
import torch

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

# Setup fake backend with noise model
fake_backend = FakeLima()
noise_model = NoiseModel.from_backend(fake_backend)
simulator = AerSimulator(noise_model=noise_model,
                         basis_gates=noise_model.basis_gates,
                         coupling_map=fake_backend.configuration().coupling_map)

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

##################
# Example Usage  #
##################

if __name__ == "__main__":
    x1 = np.array([0.1, 0.2])
    x2 = np.array([0.3, 0.4])
    wires = [0, 1]
    layers = 2
    reupload = True

    # HE weights
    weights_he = {
        "input_scaling": [np.ones(len(wires)) for _ in range(layers)],
        "variational": [np.random.randn(2 * len(wires)) for _ in range(layers)]
    }

    # Embedding paper weights
    weights_ep = {
        "input_scaling": [np.ones(len(wires)) for _ in range(layers)],
        "variational": [np.random.randn(len(wires)) for _ in range(layers)],
        "rotational": [np.random.randn(len(wires)) for _ in range(layers)]
    }

    # Run HE kernel
    val_he = qk_he_noisy_single(x1, x2, weights_he, wires, layers, simulator, reupload)
    print(f"Noisy Kernel Value (HE Ansatz): {val_he.item():.6f}")

    # Run Embedding Paper kernel
    val_ep = qk_embedding_paper_noisy_single(x1, x2, weights_ep, wires, layers, simulator, reupload)
    print(f"Noisy Kernel Value (Embedding Paper Ansatz): {val_ep.item():.6f}")
