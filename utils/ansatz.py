import pennylane as qml
from pennylane import numpy as np
import torch

def _he_layer(x, _scaling_params, _variational_params, _wires, _embedding, _data_reuploading, entanglement = None):
    if _embedding or _data_reuploading:
        for i, wire in enumerate(_wires):
            qml.RX(_scaling_params[i] * x[:,i], wires = [wire])
    for i, wire in enumerate(_wires):
        qml.RY(_variational_params[i], wires = [wire])
    for i, wire in enumerate(_wires):
        qml.RZ(_variational_params[i+len(_wires)], wires = [wire])
    if len(_wires) == 2:
        qml.broadcast(unitary=qml.CZ, pattern = "chain", wires = _wires)
    else:
        qml.broadcast(unitary=qml.CZ, pattern = "ring", wires = _wires)

def _covariant_layer(x, _scaling_params, _variational_params, _wires, _embedding, _data_reuploading, entanglement = None):
    if entanglement == None:
        entanglement = [[i, i + 1] for i in range(len(_wires) - 1)]
    for i, wire in enumerate(_wires):
        qml.RY(_variational_params[i+len(_wires)], wires = [wire])
    for source, target in entanglement:
        qml.CZ(wires=[source, target])
    if _embedding or _data_reuploading:
        for i, wire in enumerate(_wires):
            qml.RZ(_scaling_params[i] * x[:,2 * i + 1], wires = [wire])
            qml.RX(_scaling_params[i] * x[:,2 * i], wires = [wire])

def _embedding_paper_layer(x, _scaling_params, _variational_params, _rotational_params, _wires, _embedding, _data_reuploading, entanglement = None):
    for i,wire in enumerate(_wires):
        qml.Hadamard(wires = wire)
    if _embedding or _data_reuploading:
        for i, wire in enumerate(_wires):
            qml.RZ(_scaling_params[i] * x[:, i], wires = [wire])
    for i, wire in enumerate(_wires):
        qml.RY(_variational_params[i], wires = [wire])
    qml.broadcast(unitary = qml.CRZ, pattern = "ring", wires = _wires, parameters=_rotational_params)

def _he(x, weights, wires, layers, use_data_reuploading, entanglement = None):
    first_layer = True
    for layer in range(layers):
        _he_layer(x, weights["input_scaling"][layer], weights["variational"][layer], wires, first_layer, use_data_reuploading)
        first_layer = False

def _covariant(x, weights, wires, layers, use_data_reuploading, entanglement = None):
    first_layer = True
    for layer in range(layers):
        _covariant_layer(x, weights["input_scaling"][layer], weights["variational"][layer], wires, first_layer, use_data_reuploading, entanglement)
        first_layer = False

def _embedding_paper(x, weights, wires, layers, use_data_reuploading, entanglement = None):
    first_layer = True
    for layer in range(layers):
        _embedding_paper_layer(x, weights["input_scaling"][layer], weights["variational"][layer], weights["rotational"][layer], wires, first_layer, use_data_reuploading)
        first_layer = False

def qkhe(x1 , x2, weights, wires, layers, projector, data_reuploading, entanglement = None):
    x1 = x1.repeat(1, len(wires) // len(x1[0]) + 1)[:, :len(wires)]
    x2 = x2.repeat(1, len(wires) // len(x2[0]) + 1)[:, :len(wires)]
    _he(x1,weights,wires,layers,data_reuploading)
    qml.adjoint(_he)(x2,weights,wires,layers,data_reuploading)
    return qml.expval(qml.Hermitian(projector, wires = wires))

def qkcovariant(x1 , x2, weights, wires, layers, projector, data_reuploading, entanglement = None):
    x1 = x1.repeat(1, len(wires) // len(x1[0]) + 1)[:, :len(wires)]
    x2 = x2.repeat(1, len(wires) // len(x2[0]) + 1)[:, :len(wires)]
    _covariant(x1,weights,wires,layers,data_reuploading)
    qml.adjoint(_covariant)(x2,weights,wires,layers,data_reuploading, entanglement)
    return qml.expval(qml.Hermitian(projector, wires = wires))

def qkembedding_paper(x1 , x2, weights, wires, layers, projector, data_reuploading, entanglement = None):
    x1 = x1.repeat(1, len(wires) // len(x1[0]) + 1)[:, :len(wires)]
    x2 = x2.repeat(1, len(wires) // len(x2[0]) + 1)[:, :len(wires)]
    _embedding_paper(x1,weights,wires,layers,data_reuploading)
    qml.adjoint(_embedding_paper)(x2,weights,wires,layers,data_reuploading)
    return qml.expval(qml.Hermitian(projector, wires = wires))