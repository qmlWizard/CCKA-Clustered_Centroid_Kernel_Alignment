import pennylane as qml


def _he_layer(x, _scaling_params, _variational_params, _wires, _embedding, _data_reuploading):
    if _embedding or _data_reuploading:
        for i, wire in enumerate(_wires):
            qml.RX(_scaling_params[i] * x[:, i], wires = [wire])
    for i, wire in enumerate(_wires):
        qml.RY(_variational_params[i], wires = [wire])
    for i, wire in enumerate(_wires):
        qml.RZ(_variational_params[i+len(_wires)], wires = [wire])
    qml.broadcast(unitary=qml.CZ, pattern = "ring", wires = _wires)

def ra():
    pass

def _covariant_layer(x, _scaling_params, _variational_params, _rotational_params, _wires, _embedding, _data_reuploading):
    for i,wire in enumerate(_wires):
        qml.Hadamard(wires = wire)
    if _embedding or _data_reuploading:
        for i, wire in enumerate(_wires):
            qml.RZ(_scaling_params[i] * x[:,i], wires = [wire])
    for i, wire in enumerate(_wires):
        qml.RY(_variational_params[i+len(_wires)], wires = [wire])
    qml.broadcast(unitary = qml.CRZ, pattern = "ring", wires = _wires, parameters=_rotational_params)

def he(x, weights, wires, layers, use_data_reuploading):
    first_layer = True
    for layer in range(layers):
        _he_layer(x, weights["input_scaling"][layer], weights["variational"][layer], wires, first_layer, use_data_reuploading)
        first_layer = False

def covariant(x, weights, wires, layers, use_data_reuploading):
    first_layer = True
    for layer in range(layers):
        _covariant_layer(x, weights["input_scaling"][layer], weights["variational"][layer], weights["rotational"][layer], wires, first_layer, use_data_reuploading)
        first_layer = False

def qkhe(x1 , x2, weights, wires, layers, projector, data_reuploading):
    x1 = x1.repeat(1, len(wires) // len(x1[0]) + 1)[:, :len(wires)]
    x2 = x2.repeat(1, len(wires) // len(x2[0]) + 1)[:, :len(wires)]
    he(x1,weights,wires,layers,data_reuploading)
    qml.adjoint(he)(x2,weights,wires,layers,data_reuploading)
    return qml.expval(qml.Hermitian(projector, wires = wires))

def qkra():
    pass

def qkcovariant(x1 , x2, weights, wires, layers, projector, data_reuploading):
    x1 = x1.repeat(1, len(wires) // len(x1[0]) + 1)[:, :len(wires)]
    x2 = x2.repeat(1, len(wires) // len(x2[0]) + 1)[:, :len(wires)]
    covariant(x1,weights,wires,layers,data_reuploading)
    qml.adjoint(covariant)(x2,weights,wires,layers,data_reuploading)
    return qml.expval(qml.Hermitian(projector, wires = wires))