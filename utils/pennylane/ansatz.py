import pennylane as qml
from pennylane import numpy as np
import torch

# --------------------------
# Unitary layers (unchanged)
# --------------------------
def _he_layer(x, _scaling_params, _variational_params, _wires, _embedding, _data_reuploading, entanglement=None):
    if _embedding or _data_reuploading:
        for i, wire in enumerate(_wires):
            qml.RX(_scaling_params[i] * x[:, i], wires=[wire])
    for i, wire in enumerate(_wires):
        qml.RY(_variational_params[i], wires=[wire])
    for i, wire in enumerate(_wires):
        qml.RZ(_variational_params[i + len(_wires)], wires=[wire])
    if len(_wires) == 2:
        qml.CZ(wires=[_wires[0], _wires[1]])
    else:
        num_wires = len(_wires)
        for i in range(num_wires):
            qml.CZ(wires=[_wires[i], _wires[(i + 1) % num_wires]])

def _covariant_layer(x, _scaling_params, _variational_params, _wires, _embedding, _data_reuploading, entanglement=None):
    if entanglement is None:
        entanglement = [[i, i + 1] for i in range(len(_wires) - 1)]
    for i, wire in enumerate(_wires):
        qml.RY(_variational_params[i + len(_wires)], wires=[wire])
    for source, target in entanglement:
        qml.CZ(wires=[source, target])
    if _embedding or _data_reuploading:
        for i, wire in enumerate(_wires):
            qml.RZ(_scaling_params[i] * x[:, 2 * i + 1], wires=[wire])
            qml.RX(_scaling_params[i] * x[:, 2 * i], wires=[wire])

def _embedding_paper_layer(x, _scaling_params, _variational_params, _rotational_params, _wires, _embedding, _data_reuploading, entanglement=None):
    for i, wire in enumerate(_wires):
        qml.Hadamard(wires=wire)
    if _embedding or _data_reuploading:
        for i, wire in enumerate(_wires):
            qml.RZ(_scaling_params[i] * x[:, i], wires=[wire])
    for i, wire in enumerate(_wires):
        qml.RY(_variational_params[i], wires=[wire])
    num_wires = len(_wires)
    for i in range(num_wires):
        qml.CRZ(_rotational_params[i], wires=[_wires[i], _wires[(i + 1) % num_wires]])

# --------------------------
# Noise helper (outer blocks)
# --------------------------
def _spray_depolarizing(noise_p, wires):
    """Apply depolarizing noise independently on each wire (no-op if p<=0)."""
    if noise_p and float(noise_p) > 0.0:
        for w in wires:
            qml.DepolarizingChannel(noise_p, wires=w)

# --------------------------
# Unitary stacks (layer by layer)
# --------------------------
def _he(x, weights, wires, layers, use_data_reuploading):
    first_layer = True
    for layer in range(layers):
        _he_layer(x, weights["input_scaling"][layer], weights["variational"][layer],
                  wires, first_layer, use_data_reuploading)
        first_layer = False

def _covariant(x, weights, wires, layers, use_data_reuploading, entanglement=None):
    first_layer = True
    for layer in range(layers):
        _covariant_layer(x, weights["input_scaling"][layer], weights["variational"][layer],
                         wires, first_layer, use_data_reuploading, entanglement)
        first_layer = False

def _embedding_paper(x, weights, wires, layers, use_data_reuploading):
    first_layer = True
    for layer in range(layers):
        _embedding_paper_layer(x, weights["input_scaling"][layer], weights["variational"][layer],
                               weights["rotational"][layer], wires, first_layer, use_data_reuploading)
        first_layer = False

# ----------------------------------------------------------
# Noisy kernel builders: interleave noise *between* layers
# ----------------------------------------------------------
def qkhe(x1, x2, weights, wires, layers, projector, data_reuploading, entanglement=None, noise_p: float = 0.0):
    """
    Implements:  [U_he(x1, layer1); N; U_he(x1, layer2); N; ...]  ·  [ ... ; U_he(x2,layer2); N; U_he(x2,layer1)]^†
    Noise blocks N are *not* adjointed; they are inserted explicitly between unitary layers on both sides.
    """
    # Repeat features to match number of wires
    x1 = x1.repeat(1, len(wires) // len(x1[0]) + 1)[:, :len(wires)]
    x2 = x2.repeat(1, len(wires) // len(x2[0]) + 1)[:, :len(wires)]

    # Forward (x1): layer-by-layer with noise in between
    first_layer = True
    for layer in range(layers):
        _he_layer(x1, weights["input_scaling"][layer], weights["variational"][layer],
                  wires, first_layer, data_reuploading)
        first_layer = False
        _spray_depolarizing(noise_p, wires)

    # Backward (x2): apply adjoint of each layer in reverse order; add noise between those as well
    for layer in reversed(range(layers)):
        first_layer_back = (layer == 0)  # embedding active only for the "first" logical layer
        qml.adjoint(_he_layer)(
            x2,
            weights["input_scaling"][layer],
            weights["variational"][layer],
            wires,
            first_layer_back,
            data_reuploading
        )
        _spray_depolarizing(noise_p, wires)

    return qml.expval(qml.Hermitian(projector, wires=wires))

def qkcovariant(x1, x2, weights, wires, layers, projector, data_reuploading, entanglement=None, noise_p: float = 0.0):
    x1 = x1.repeat(1, len(wires) // len(x1[0]) + 1)[:, :len(wires)]
    x2 = x2.repeat(1, len(wires) // len(x2[0]) + 1)[:, :len(wires)]

    first_layer = True
    for layer in range(layers):
        _covariant_layer(x1, weights["input_scaling"][layer], weights["variational"][layer],
                         wires, first_layer, data_reuploading, entanglement)
        first_layer = False
        _spray_depolarizing(noise_p, wires)

    for layer in reversed(range(layers)):
        first_layer_back = (layer == 0)
        qml.adjoint(_covariant_layer)(
            x2,
            weights["input_scaling"][layer],
            weights["variational"][layer],
            wires,
            first_layer_back,
            data_reuploading,
            entanglement
        )
        _spray_depolarizing(noise_p, wires)

    return qml.expval(qml.Hermitian(projector, wires=wires))

def qkembedding_paper(x1, x2, weights, wires, layers, projector, data_reuploading, entanglement=None, noise_p: float = 0.0):
    x1 = x1.repeat(1, len(wires) // len(x1[0]) + 1)[:, :len(wires)]
    x2 = x2.repeat(1, len(wires) // len(x2[0]) + 1)[:, :len(wires)]

    first_layer = True
    for layer in range(layers):
        _embedding_paper_layer(x1, weights["input_scaling"][layer], weights["variational"][layer],
                               weights["rotational"][layer], wires, first_layer, data_reuploading)
        first_layer = False
        _spray_depolarizing(noise_p, wires)

    for layer in reversed(range(layers)):
        first_layer_back = (layer == 0)
        qml.adjoint(_embedding_paper_layer)(
            x2,
            weights["input_scaling"][layer],
            weights["variational"][layer],
            weights["rotational"][layer],
            wires,
            first_layer_back,
            data_reuploading
        )
        _spray_depolarizing(noise_p, wires)

    return qml.expval(qml.Hermitian(projector, wires=wires))
