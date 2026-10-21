"""PyTorch-backed PennyLane kernel model used when JAX is unavailable."""

from __future__ import annotations

import pennylane as qml
import torch


class TorchKernelModel:
    def __init__(
        self,
        circuit,
        *,
        device_name: str = "default.qubit",
        diff_method: str = "backprop",
        noisy: bool = False,
        shots: int | None = None,
    ) -> None:
        if circuit is None:
            raise ValueError("A circuit must be provided to the kernel model.")
        if noisy and shots is not None and shots <= 0:
            raise ValueError("shots must be > 0 when noisy=True")
        self.circuit = circuit
        self.noisy = noisy
        self.circuit_executions = 0
        self.device_name = "default.mixed" if noisy else device_name
        self.shots = (1024 if shots is None else shots) if noisy else None
        device = qml.device(
            self.device_name, wires=circuit.num_qubits, shots=self.shots
        )
        self.circuit_instance = qml.QNode(
            circuit.kernel_circuit,
            device,
            interface="torch",
            diff_method=diff_method,
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        if len(x1) != len(x2):
            raise ValueError("x1 and x2 must contain the same number of points.")
        self.circuit_executions += len(x1)
        return torch.stack(
            [
                self.circuit_instance(first, second, weights)
                for first, second in zip(x1, x2)
            ]
        )
