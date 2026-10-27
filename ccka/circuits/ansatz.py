"""Gate-level definitions of the trainable kernel ansätze.

Every ansatz is described by two functions that return *operation lists*
``(gate, angles, wires)``. A list can be applied directly, or inverted by
walking it backwards (see :func:`invert_op`), which keeps the forward and the
adjoint pass of the kernel circuit exactly consistent for every ansatz.

All ansätze use the same weight layout per repetition (``scale``, ``var``,
``rot``, each of length ``n_qubits``), so the optimisers and the weight
initialisation do not depend on the chosen ansatz.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

Op = tuple[str, tuple, tuple]

SELF_INVERSE_GATES = frozenset({"H", "CNOT", "CZ"})
ROTATION_GATES = frozenset({"RZ", "RY", "CRZ", "RZZ"})


def invert_op(op: Op) -> Op:
    gate, angles, wires = op
    if gate in SELF_INVERSE_GATES:
        return op
    if gate in ROTATION_GATES:
        return gate, tuple(-angle for angle in angles), wires
    raise ValueError(f"Cannot invert unknown gate {gate!r}.")


def _ring_pairs(n: int) -> list[tuple[int, int]]:
    if n == 2:
        return [(0, 1)]
    return [(i, (i + 1) % n) for i in range(n)]


def _hadamard_rz_map(x, scale, n):
    ops: list[Op] = []
    for i in range(n):
        ops.append(("H", (), (i,)))
        ops.append(("RZ", (scale[i] * x[i],), (i,)))
    return ops


def _ry_map(x, scale, n):
    return [("RY", (scale[i] * x[i],), (i,)) for i in range(n)]


def _iqp_map(x, scale, n):
    ops = _hadamard_rz_map(x, scale, n)
    for i, j in _ring_pairs(n):
        ops.append(("RZZ", (scale[i] * scale[j] * x[i] * x[j],), (i, j)))
    return ops


def _zz_map(x, scale, n):
    ops: list[Op] = []
    for i in range(n):
        ops.append(("H", (), (i,)))
        ops.append(("RZ", (2.0 * scale[i] * x[i],), (i,)))
    for i, j in _ring_pairs(n):
        angle = (
            2.0 * scale[i] * scale[j] * (math.pi - x[i]) * (math.pi - x[j])
        )
        ops.append(("RZZ", (angle,), (i, j)))
    return ops


def _crz_ring_ansatz(var, rot, n, rep):
    ops: list[Op] = [("RY", (var[i],), (i,)) for i in range(n)]
    ops.extend(("CRZ", (rot[i],), (i, (i + 1) % n)) for i in range(n))
    return ops


def _hardware_efficient_ansatz(var, rot, n, rep):
    ops: list[Op] = []
    for i in range(n):
        ops.append(("RZ", (rot[i],), (i,)))
        ops.append(("RY", (var[i],), (i,)))
    ops.extend(("CNOT", (), (i, i + 1)) for i in range(n - 1))
    return ops


def _iqp_ansatz(var, rot, n, rep):
    ops: list[Op] = [("RZ", (var[i],), (i,)) for i in range(n)]
    ops.extend(("RZZ", (rot[k],), pair) for k, pair in enumerate(_ring_pairs(n)))
    return ops


def _local_ansatz(var, rot, n, rep):
    ops: list[Op] = []
    for i in range(n):
        ops.append(("RY", (var[i],), (i,)))
        ops.append(("RZ", (rot[i],), (i,)))
    return ops


def _strongly_entangling_ansatz(var, rot, n, rep):
    ops = _local_ansatz(var, rot, n, rep)
    reach = (rep % (n - 1)) + 1 if n > 1 else 1
    ops.extend(("CNOT", (), (i, (i + reach) % n)) for i in range(n))
    return ops


ANSATZE: dict[str, tuple[Callable, Callable, bool]] = {
    # name: (feature map, trainable block, entangles)
    "crz_ring": (_hadamard_rz_map, _crz_ring_ansatz, True),
    "hardware_efficient": (_ry_map, _hardware_efficient_ansatz, True),
    "iqp": (_iqp_map, _iqp_ansatz, True),
    "zz_feature_map": (_zz_map, _local_ansatz, True),
    "strongly_entangling": (
        _hadamard_rz_map,
        _strongly_entangling_ansatz,
        True,
    ),
    "separable": (_hadamard_rz_map, _local_ansatz, False),
}

DEFAULT_ANSATZ = "crz_ring"


def ansatz_names() -> Sequence[str]:
    return tuple(ANSATZE)


def get_ansatz(name: str) -> tuple[Callable, Callable, bool]:
    try:
        return ANSATZE[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown ansatz {name!r}. Choose from {sorted(ANSATZE)}."
        ) from exc
