from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from ccka_experiments.multiclass import MulticlassAligner


torch = pytest.importorskip("torch")


class _TinyCircuit:
    @staticmethod
    def init_weights(seed=0):
        del seed
        return np.asarray([1.0], dtype=np.float32)


class _TinyKernel:
    circuit = _TinyCircuit()

    def __init__(self):
        self.circuit_executions = 0

    def forward(self, x1, x2, weights):
        self.circuit_executions += len(x1)
        distance = torch.sum((x1 - x2) ** 2, dim=1)
        return torch.exp(-weights[0].square() * distance)


def _multiclass_data():
    return (
        np.asarray([[-2.2], [-1.8], [-0.2], [0.2], [1.8], [2.2]], dtype=np.float32),
        np.asarray([[-2.0], [0.0], [2.0]], dtype=np.float32),
        np.asarray([[-2.1], [0.1], [2.1]], dtype=np.float32),
        np.asarray([0, 0, 1, 1, 2, 2]),
        np.asarray([0, 1, 2]),
        np.asarray([0, 1, 2]),
    )


@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
def test_multiclass_alignment_works_with_torch_kernel(strategy):
    from ccka.aligner.torch_kta import TorchExperimentKTA

    data = _multiclass_data()

    def factory(binary_data):
        model = _TinyKernel()
        aligner = TorchExperimentKTA(
            kernel_model=model,
            data=binary_data[0],
            labels=binary_data[3],
            validation_data=binary_data[1],
            validation_labels=binary_data[4],
            test_data=binary_data[2],
            test_labels=binary_data[5],
            method="fullKTA",
            epochs=1,
            inference_method="svm",
            inference_landmarks="all",
            seed=0,
        )
        return aligner, model

    result = MulticlassAligner(
        factory, data, strategy=strategy, seed=0
    ).align()

    assert result["n_binary_problems"] == 3
    assert result["final_svm_metrics"]["validation_accuracy"] == pytest.approx(1.0)
    assert result["final_svm_metrics"]["test_accuracy"] == pytest.approx(1.0)


@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
def test_run_one_uses_torch_backend_for_multiclass(monkeypatch, strategy):
    from ccka_experiments import experiments

    monkeypatch.setattr(experiments, "BACKEND", "torch")
    monkeypatch.setattr(
        experiments, "_build_model", Mock(return_value=_TinyKernel())
    )

    result = experiments._run_one(
        dataset_name="synthetic",
        method_name="fullKTA",
        data=_multiclass_data(),
        settings={
            "epochs": 1,
            "reps": 1,
            "inference_landmarks": "all",
            "multiclass_strategy": strategy,
        },
        seed=0,
        noise={"type": "none", "level": 0.0},
        shots=0,
        centroids=1,
    )

    assert result["backend"] == "torch"
    assert result["multiclass_strategy"] == strategy
    assert result["test_accuracy"] == pytest.approx(1.0)
