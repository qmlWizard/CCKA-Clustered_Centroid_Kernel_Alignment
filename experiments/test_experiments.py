import json

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from ccka.aligner.kta import (
    CentroidBasedKTA,
    FullKTA,
    QuackKTA,
    RandomKTA,
)
from ccka.circuits.angleEmbeddingKernel import quackEmbeddingCircuit
from ccka.models.kernel import KernelModel
from experiments.experiments import (
    METHOD_LABELS,
    METHODS,
    _experiment_specs,
    _make_aligner,
    _prepare_data,
    _run_one,
)


class _TinyCircuit:
    num_qubits = 2

    @staticmethod
    def init_weights(seed=0):
        return jnp.asarray([0.7], dtype=jnp.float32)


class _TinyKernel:
    def __init__(self):
        self.circuit = _TinyCircuit()
        self.circuit_executions = 0

    def forward(self, x1, x2, weights):
        self.circuit_executions += len(x1)
        distance = jnp.sum((x1 - x2) ** 2, axis=1)
        return jnp.exp(-jnp.abs(weights[0]) * distance)


def _splits():
    X = np.asarray(
        [[i / 10, (i % 3) / 3] for i in range(24)], dtype=np.float32
    )
    y = np.asarray([-1] * 12 + [1] * 12, dtype=np.int32)
    return (
        np.concatenate((X[:6], X[12:18])),
        np.concatenate((X[6:9], X[18:21])),
        np.concatenate((X[9:12], X[21:24])),
        np.concatenate((y[:6], y[12:18])),
        np.concatenate((y[6:9], y[18:21])),
        np.concatenate((y[9:12], y[21:24])),
    )


def _settings():
    return {
        "epochs": 1,
        "min_n_qubits": 5,
        "reps": 1,
        "random_samples": 2,
        "inference_landmarks": 3,
        "learning_rate": 0.01,
        "centroid_lr": 0.005,
        "sub_centroid_lr": 0.005,
    }


def test_ccka_alignment_evaluates_only_centroid_to_subcentroid_pairs():
    model = _TinyKernel()
    aligner = CentroidBasedKTA(
        model,
        jnp.asarray(_splits()[0]),
        jnp.asarray(_splits()[3]),
        validation_data=jnp.asarray(_splits()[1]),
        validation_labels=jnp.asarray(_splits()[4]),
        test_data=jnp.asarray(_splits()[2]),
        test_labels=jnp.asarray(_splits()[5]),
        centroids=2,
        epochs=1,
        seed=9,
    )
    before = model.circuit_executions
    aligner._loss_kao_cl(
        aligner.weights,
        aligner.sub_centroids,
        aligner.main_centroids[0],
        aligner.sub_centroid_labels,
        l=1.0,
    )
    aligner._loss_co_cl(
        aligner.weights,
        aligner.main_centroids,
        float(aligner.main_centroid_labels[0]),
        aligner.sub_centroid_labels,
        l=-1.0,
    )
    assert model.circuit_executions - before == 2 * 2 * 2


def test_ccka_svm_kernel_matrices_use_nystrom_for_validation_and_test():
    model = _TinyKernel()
    data = _splits()
    aligner = CentroidBasedKTA(
        model,
        jnp.asarray(data[0]),
        jnp.asarray(data[3]),
        validation_data=jnp.asarray(data[1]),
        validation_labels=jnp.asarray(data[4]),
        test_data=jnp.asarray(data[2]),
        test_labels=jnp.asarray(data[5]),
        centroids=2,
        inference_landmarks=3,
        seed=9,
    )
    model.circuit_executions = 0
    K_train, K_validation, K_test = aligner._evaluation_kernel_matrices(
        aligner.weights,
        aligner.xtrain,
        aligner.xvalidation,
        aligner.xtest,
    )
    n, v, t, m = len(data[0]), len(data[1]), len(data[2]), 3
    assert K_train.shape == (n, n)
    assert K_validation.shape == (v, n)
    assert K_test.shape == (t, n)
    assert model.circuit_executions == n * m + m * m + v * m + t * m
    np.testing.assert_allclose(K_train, K_train.T, atol=1e-5)


def test_svm_uses_validation_until_final_test_evaluation():
    model = _TinyKernel()
    data = _splits()
    aligner = FullKTA(
        model,
        jnp.asarray(data[0]),
        jnp.asarray(data[3]),
        validation_data=jnp.asarray(data[1]),
        validation_labels=jnp.asarray(data[4]),
        test_data=jnp.asarray(data[2]),
        test_labels=jnp.asarray(data[5]),
        epochs=1,
    )
    validation_metrics = aligner.svm_training(aligner.xtrain, aligner.ytrain)
    assert "validation_accuracy" in validation_metrics
    assert "test_accuracy" not in validation_metrics
    final_metrics = aligner.svm_training(
        aligner.xtrain, aligner.ytrain, evaluate_test=True
    )
    assert "test_accuracy" in final_metrics


def test_noisy_kernel_uses_serial_pairs_with_parameter_shift():
    circuit = quackEmbeddingCircuit(
        2, reps=1, noisy=True, noise_type="coherent", noise_level=0.01
    )
    model = KernelModel(
        circuit=circuit,
        device_name="default.mixed",
        interface="jax",
        diff_method="parameter-shift",
        noisy=True,
        shots=8,
    )
    x1 = jnp.asarray([[0.1, 0.2], [0.3, 0.4]])
    x2 = jnp.asarray([[0.2, 0.3], [0.4, 0.5]])
    weights = jnp.asarray(circuit.init_weights(seed=1))

    values = model.forward(x1, x2, weights)
    gradient = jax.grad(
        lambda current_weights: jnp.sum(
            model.forward(x1[:1], x2[:1], current_weights)
        )
    )(weights)

    assert values.shape == (2,)
    assert model.circuit_executions == 3
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_depolarizing_spelling_applies_noise_channel():
    circuit = quackEmbeddingCircuit(
        num_qubits=1,
        reps=1,
        noisy=True,
        noise_type="depolarizing",
        noise_level=0.1,
    )
    with qml.tape.QuantumTape() as tape:
        circuit.feature_map(
            jnp.asarray([0.2]),
            jnp.asarray([1.0]),
            apply_noise=True,
        )
    assert (
        sum(
            operation.name == "DepolarizingChannel"
            for operation in tape.operations
        )
        == 2
    )


def test_experiment_methods_exclude_greedy_and_have_paper_labels():
    assert "greedyKTA" not in METHODS
    assert METHOD_LABELS == {
        "ccka": "CCKA",
        "quackKTA": "QUACK",
        "fullKTA": "All-Pairwise KTA (Full KTA)",
        "randomKTA": "Random-Pairwise KTA",
    }


def test_torch_ccka_fallback_runs_if_torch_is_installed():
    torch = pytest.importorskip("torch")
    from ccka.aligner.torch_kta import TorchExperimentKTA

    class TorchTinyKernel:
        def __init__(self):
            self.circuit = _TinyCircuit()
            self.circuit_executions = 0

        def forward(self, x1, x2, weights):
            self.circuit_executions += len(x1)
            return torch.exp(-torch.abs(weights[0]) * ((x1 - x2) ** 2).sum(dim=1))

    data = _splits()
    aligner = TorchExperimentKTA(
        TorchTinyKernel(),
        data[0],
        data[3],
        method="ccka",
        validation_data=data[1],
        validation_labels=data[4],
        test_data=data[2],
        test_labels=data[5],
        centroids=2,
        epochs=1,
        inference_landmarks=3,
        seed=9,
    )
    result = aligner.align()
    assert len(result["test_accuracy_history"]) == 1
    assert np.isfinite(result["best_validation_accuracy"])


@pytest.mark.parametrize(
    ("method", "initialization"),
    [
        ("fullKTA", "not_applicable"),
        ("randomKTA", "not_applicable"),
        ("quackKTA", "not_applicable"),
        ("ccka", "regular"),
        ("ccka", "random"),
        ("ccka", "kmeans"),
    ],
)
def test_method_and_initialization_smoke(method, initialization):
    model = _TinyKernel()
    aligner = _make_aligner(
        method,
        model,
        _splits(),
        _settings(),
        seed=3,
        centroids=2,
        initialization=initialization,
    )
    result = aligner.align()
    assert result["test_accuracy_history"]
    expected_validation_steps = 10 if method == "quackKTA" else 1
    assert len(result["validation_accuracy_history"]) == expected_validation_steps
    assert "best_validation_accuracy" in result
    assert np.isfinite(result["test_accuracy_history"][-1])


def test_smoke_configs_cover_both_datasets_and_noise_modes():
    config_path = __import__("pathlib").Path(__file__).parent / "configs" / "smoke.yaml"
    import yaml

    config = yaml.safe_load(config_path.read_text())
    assert config["datasets"] == ["checkerboard", "corners"]
    assert set(config["noise"]["types"]) == {
        "none",
        "depolarizing",
        "coherent",
    }
    assert config["noise"]["shots"] == [8]
    assert config["training"]["quack_inner_steps"] == 1
    assert config["training"]["collect_diagnostics"] is False
    settings = dict(config["training"])
    settings.pop("max_samples")
    for dataset in config["datasets"]:
        data = _prepare_data(dataset, seed=0, settings=config["training"])
        assert len(data) == 6
        assert data[0].shape[1] == 5


def test_prepare_data_uses_all_features_above_minimum_qubit_count():
    data = _prepare_data(
        "qsar_biodegradation",
        seed=0,
        settings={"min_n_qubits": 5, "validation_fraction": 0.2},
    )
    assert data[0].shape[1] == 10


def test_ccka_specs_include_matching_k_one_baseline_and_unique_ablations():
    config = {
        "training": {"centroids": [2, 4]},
        "experiments": {
            "methods": ["ccka"],
            "ccka_initializations": ["regular"],
            "ccka_ablations": {
                "main_only": {"centroids": 1, "update_kao": False},
                "sub_only": {"centroids": 1, "update_co": False},
            },
        },
    }
    specs = list(_experiment_specs(config))
    assert sum(spec[3] == "full" for spec in specs) == 3
    assert sum(spec[3] == "main_only" for spec in specs) == 1
    assert sum(spec[3] == "sub_only" for spec in specs) == 1
    assert any(spec[1] == 1 and spec[3] == "full" for spec in specs)


def test_experiment_result_serializes_per_epoch_validation_history(monkeypatch):
    class _FakeAligner:
        def align(self):
            return {
                "validation_accuracy_history": [0.5, 0.75],
                "best_validation_accuracy": 0.75,
                "final_svm_metrics": {
                    "validation_accuracy": 0.75,
                    "test_accuracy": 0.7,
                    "f1_score": 0.7,
                    "precision_score": 0.7,
                    "recall_score": 0.7,
                },
                "circuit_executions": 12,
            }

    monkeypatch.setattr(
        "experiments.experiments._make_aligner",
        lambda *args, **kwargs: _FakeAligner(),
    )
    result = _run_one(
        "checkerboard",
        "fullKTA",
        _splits(),
        _settings(),
        0,
        {"type": "none", "level": 0.0},
        0,
        0,
    )
    assert json.loads(result["validation_accuracy_history"]) == [0.5, 0.75]
