import pennylane as qml
from pennylane import numpy as np
import torch

# --------------------------
# Unitary layers (unchanged)
# --------------------------
def _he_layer(x, _scaling_params, _variational_params, _wires, _embedding, _data_reuploading, entanglement=None, noise_p: float = 0.0):
    if _embedding or _data_reuploading:
        for i, wire in enumerate(_wires):
            qml.RX(_scaling_params[i,:] * x[:, i], wires=[wire])
    for i, wire in enumerate(_wires):
        qml.RY(_variational_params[i,:], wires=[wire])
    for i, wire in enumerate(_wires):
        qml.RZ(_variational_params[i + len(_wires)], wires=[wire])
    if len(_wires) == 2:
        qml.CZ(wires=[_wires[0], _wires[1]])
    else:
        num_wires = len(_wires)
        for i in range(num_wires):
            qml.CZ(wires=[_wires[i], _wires[(i + 1) % num_wires]])
    for i, wire in enumerate(_wires):
        qml.DepolarizingChannel(noise_p, wires=wire)

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

def _embedding_paper_layer(x, _scaling_params, _variational_params, _rotational_params, _wires, _embedding, _data_reuploading, entanglement=None, noise_p: float = 0.0):
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
    for i, wire in enumerate(_wires):
        qml.DepolarizingChannel(noise_p, wires=wire)

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
def _he(x, weights, wires, layers, use_data_reuploading, noise_p):
    first_layer = True
    for layer in range(layers):
        _he_layer(x, weights["input_scaling"][layer], weights["variational"][layer],
                  wires, first_layer, use_data_reuploading, noise_p=noise_p)
        first_layer = False

def _covariant(x, weights, wires, layers, use_data_reuploading, entanglement=None):
    first_layer = True
    for layer in range(layers):
        _covariant_layer(x, weights["input_scaling"][layer], weights["variational"][layer],
                         wires, first_layer, use_data_reuploading, entanglement)
        first_layer = False

def _embedding_paper(x, weights, wires, layers, use_data_reuploading, noise_p):
    first_layer = True
    for layer in range(layers):
        _embedding_paper_layer(x, weights["input_scaling"][layer], weights["variational"][layer],
                               weights["rotational"][layer], wires, first_layer, use_data_reuploading, noise_p=noise_p)
        first_layer = False

# ----------------------------------------------------------
# Noisy kernel builders: interleave noise *between* layers
# ----------------------------------------------------------
def qkhe(x1, x2, weights, wires, layers, projector, data_reuploading, entanglement=None, noise_p: float = 0.0):
    """
    Implements:  [U_he(x1, layer1); N; U_he(x1, layer2); N; ...]  ·  [ ... ; U_he(x2,layer2); N; U_he(x2,layer1)]^†
    Noise blocks N are not adjointed; they are inserted explicitly between unitary layers on both sides.
    """
    # Repeat features to match number of wires
    x1 = x1.repeat(1, len(wires) // len(x1[0]) + 1)[:, :len(wires)]
    x2 = x2.repeat(1, len(wires) // len(x2[0]) + 1)[:, :len(wires)]

    # Forward (x1): layer-by-layer with noise in between (inside _he_layer)
    first_layer = True
    for layer in range(layers):
        _he_layer(
            x1,
            weights["input_scaling"][layer],
            weights["variational"][layer],
            wires,
            first_layer,
            data_reuploading,
            noise_p=noise_p
        )
        first_layer = False

    # Backward (x2): manual inverse of unitary parts in reverse order; inject noise BETWEEN inverse layers
    for layer in reversed(range(layers)):
        # Insert noise between inverse layers to mirror forward interleaving
        _spray_depolarizing(noise_p, wires)

        # Determine whether to apply the embedding inverse for this logical "first" layer
        first_layer_back = (layer == 0)

        # Inverse entanglement (CZ is self-adjoint)
        if len(wires) == 2:
            qml.CZ(wires=[wires[0], wires[1]])
        else:
            num_wires = len(wires)
            for i in range(num_wires):
                qml.CZ(wires=[wires[i], wires[(i + 1) % num_wires]])

        # Inverse of RZ then RY (reverse order compared to forward within the layer)
        for i, wire in enumerate(wires):
            qml.RZ(-weights["variational"][layer][i + len(wires)], wires=[wire])
        for i, wire in enumerate(wires):
            qml.RY(-weights["variational"][layer][i], wires=[wire])

        # Inverse of input embedding RX, but only for the logical first layer
        if first_layer_back:
            for i, wire in enumerate(wires):
                qml.RX(-(weights["input_scaling"][layer][i] * x2[:, i]), wires=[wire])

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
        # Manually invert covariant layer (no in-layer noise to worry about)
        # Inverse entanglement (CZ self-adjoint)
        if entanglement is None:
            ent = [[i, i + 1] for i in range(len(wires) - 1)]
        else:
            ent = entanglement

        for source, target in ent:
            qml.CZ(wires=[source, target])

        # Inverse RY block
        for i, wire in enumerate(wires):
            qml.RY(-weights["variational"][layer][i + len(wires)], wires=[wire])

        # Inverse embedding RX/RZ only for the logical first layer
        if first_layer_back:
            for i, wire in enumerate(wires):
                qml.RX(-(weights["input_scaling"][layer][i] * x2[:, 2 * i]), wires=[wire])
                qml.RZ(-(weights["input_scaling"][layer][i] * x2[:, 2 * i + 1]), wires=[wire])

        # Optional: mirror forward interleaving on the right arm as well
        _spray_depolarizing(noise_p, wires)

    return qml.expval(qml.Hermitian(projector, wires=wires))

def qkembedding_paper(x1, x2, weights, wires, layers, projector, data_reuploading, entanglement=None, noise_p: float = 0.0):
    x1 = x1.repeat(1, len(wires) // len(x1[0]) + 1)[:, :len(wires)]
    x2 = x2.repeat(1, len(wires) // len(x2[0]) + 1)[:, :len(wires)]

    first_layer = True
    for layer in range(layers):
        _embedding_paper_layer(
            x1,
            weights["input_scaling"][layer],
            weights["variational"][layer],
            weights["rotational"][layer],
            wires,
            first_layer,
            data_reuploading,
            noise_p=noise_p
        )
        first_layer = False

    # Backward: manual inverse of unitary parts; add noise between inverse layers
    for layer in reversed(range(layers)):
        _spray_depolarizing(noise_p, wires)

        first_layer_back = (layer == 0)

        # Inverse CRZs in reverse-ring order (CRZ is its own inverse up to angle sign)
        num_wires = len(wires)
        for i in reversed(range(num_wires)):
            qml.CRZ(-weights["rotational"][layer][i], wires=[wires[i], wires[(i + 1) % num_wires]])

        # Inverse RY block
        for i, wire in enumerate(wires):
            qml.RY(-weights["variational"][layer][i], wires=[wire])

        # Inverse embedding RZ only for logical first layer
        if first_layer_back:
            for i, wire in enumerate(wires):
                qml.RZ(-(weights["input_scaling"][layer][i] * x2[:, i]), wires=[wire])

        # Inverse Hadamard (self-adjoint)
        for i, wire in enumerate(wires):
            qml.Hadamard(wires=wire)

    return qml.expval(qml.Hermitian(projector, wires=wires))

    # returns the full density matrix after your noisy feature map
def he_state(x, weights, wires, layers, data_reuploading, noise_p):
    x = x.repeat(1, len(wires) // len(x[0]) + 1)[:, :len(wires)]
    first = True
    for L in range(layers):
        _he_layer(x, weights["input_scaling"][L], weights["variational"][L],
                  wires, first, data_reuploading, noise_p=noise_p)
        first = False
    return qml.density_matrix(wires=wires)

def covariant_state(x, weights, wires, layers, data_reuploading, entanglement, noise_p):
    x = x.repeat(1, len(wires) // len(x[0]) + 1)[:, :len(wires)]
    first = True
    for L in range(layers):
        _covariant_layer(x, weights["input_scaling"][L], weights["variational"][L],
                         wires, first, data_reuploading, entanglement)
        first = False
        # keep your between-layer noise if desired:
        for w in wires: 
            if noise_p > 0: qml.DepolarizingChannel(noise_p, wires=w)
    return qml.density_matrix(wires=wires)

def embedding_paper_state(x, weights, wires, layers, data_reuploading, noise_p):
    x = x.repeat(1, len(wires) // len(x[0]) + 1)[:, :len(wires)]
    first = True
    for L in range(layers):
        _embedding_paper_layer(x, weights["input_scaling"][L], weights["variational"][L],
                               weights["rotational"][L], wires, first, data_reuploading,
                               noise_p=noise_p)
        first = False
    return qml.density_matrix(wires=wires)

