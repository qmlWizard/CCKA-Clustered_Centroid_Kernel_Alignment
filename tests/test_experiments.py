import json
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from ccka.aligner.kta import (
    CentroidBasedKTA,
    FullKTA,
    GreedyKTA,
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
    _read_dataset,
    _run_one,
)
from experiments.accuracy_vs_executions import _build_aligner, _run_specs
from experiments import accuracy_vs_executions as execution_module


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
        "ccka_inner_steps": 1,
        "learning_rate": 0.01,
        "centroid_lr": 0.005,
        "sub_centroid_lr": 0.005,
    }


def test_execution_experiment_runs_all_ccka_initializations():
    assert _run_specs() == (
        ("fullKTA", "not_applicable"),
        ("randomKTA", "not_applicable"),
        ("quackKTA", "not_applicable"),
        ("ccka", "regular"),
        ("ccka", "kmeans"),
        ("ccka", "random"),
    )


def test_execution_experiment_can_select_quack_and_ccka():
    assert _run_specs(["quackKTA", "ccka"]) == (
        ("quackKTA", "not_applicable"),
        ("ccka", "regular"),
        ("ccka", "kmeans"),
        ("ccka", "random"),
    )


def test_execution_experiment_can_select_one_ccka_initialization():
    assert _run_specs(["ccka"], "kmeans") == (("ccka", "kmeans"),)


def test_inference_methods_can_be_selected_independently():
    data = _splits()
    settings = {
        **_settings(),
        "inference_method": "svm",
        "fullKTA_inference_method": "kernel_distance",
        "randomKTA_inference_method": "svm",
        "quackKTA_inference_method": "kernel_distance",
        "ccka_inference_method": "kernel_distance",
    }
    for method, expected in (
        ("fullKTA", "kernel_distance"),
        ("randomKTA", "svm"),
        ("quackKTA", "kernel_distance"),
        ("ccka", "kernel_distance"),
    ):
        aligner = _build_aligner(
            method, data, settings, seed=0, centroids=2
        )
        assert aligner.inference_method == expected


def test_ccka_learning_rate_can_override_global_rate():
    data = _splits()
    settings = {
        **_settings(),
        "learning_rate": 0.01,
        "ccka_learning_rate": 0.02,
    }
    ccka = _build_aligner(
        "ccka", data, settings, seed=0, centroids=2
    )
    quack = _build_aligner(
        "quackKTA", data, settings, seed=0, centroids=2
    )
    assert ccka.learning_rate == 0.02
    assert quack.learning_rate == 0.01


def test_global_learning_rate_is_shared_with_centroid_optimizers():
    data = _splits()
    settings = {
        **_settings(),
        "learning_rate": 1.5,
        "fullKTA_learning_rate": 1.5,
        "randomKTA_learning_rate": 1.5,
        "quackKTA_learning_rate": 1.5,
        "ccka_learning_rate": 1.5,
        "centroid_lr": 1.5,
        "sub_centroid_lr": 1.5,
    }
    for method in ("fullKTA", "randomKTA", "quackKTA", "ccka"):
        aligner = _build_aligner(
            method, data, settings, seed=0, centroids=6, initialization="kmeans"
        )
        assert aligner.learning_rate == 1.5
    assert aligner._centroid_lr == 1.5
    assert aligner._sub_centroid_lr == 1.5


def test_run_method_can_continue_after_reaching_target(monkeypatch):
    class _InitialModel:
        inner_steps = 2
        inference_landmarks = 4
        inference_method = "svm"
        xtrain = np.zeros((2, 1))
        ytrain = np.asarray([-1, 1])

        @staticmethod
        def svm_training(*args, **kwargs):
            return {"validation_accuracy": 1.0, "test_accuracy": 1.0}

    steps = []

    def fake_step(
        method, aligner, epoch, target, rows, seed, limit, init, stop, checkpoint
    ):
        steps.append((epoch, stop))
        return 0, stop

    monkeypatch.setattr(
        execution_module, "_build_aligner", lambda *args: _InitialModel()
    )
    monkeypatch.setattr(execution_module, "BACKEND", "jax")
    monkeypatch.setattr(execution_module, "_jax_step", fake_step)

    execution_module.run_method(
        "ccka",
        (),
        {},
        seed=0,
        centroids=2,
        target_accuracy=0.9,
        max_iterations=5,
        stop_on_target=False,
    )

    assert steps == [(0, False), (1, False), (2, False)]


def test_execution_comparison_uses_configured_ccka_inference_landmarks():
    settings = {**_settings(), "inference_landmarks": 32}
    aligner = _build_aligner(
        "ccka",
        _splits(),
        settings,
        seed=0,
        centroids=2,
        initialization="kmeans",
    )
    assert aligner.inference_landmarks == 32


@pytest.mark.parametrize("method", ["fullKTA", "randomKTA", "quackKTA", "ccka"])
def test_experiment_inference_rejects_methods_other_than_svm_and_distance(method):
    with pytest.raises(ValueError, match="choose 'svm' or 'kernel_distance'"):
        _build_aligner(
            method,
            _splits(),
            {**_settings(), f"{method}_inference_method": "centroid"},
            seed=0,
            centroids=2,
        )


def test_kernel_distance_inference_uses_configured_neighbor_count():
    data = _splits()
    settings = {
        **_settings(),
        "inference_method": "kernel_distance",
        "inference_neighbors": 3,
    }
    aligner = _build_aligner(
        "ccka", data, settings, seed=0, centroids=2, initialization="kmeans"
    )
    metrics = aligner.svm_training(aligner.xtrain, aligner.ytrain)
    assert aligner.inference_method == "kernel_distance"
    assert aligner.inference_neighbors == 3
    assert 0.0 <= metrics["validation_accuracy"] <= 1.0


def test_ccka_co_margin_loss_rewards_same_class_similarity():
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
        co_margin=0.2,
        co_temperature=0.1,
        lambda_co=0.0,
        seed=9,
    )
    aligned_kernel = jnp.asarray([0.9, 0.8, 0.2, 0.1])
    crossed_kernel = jnp.asarray([0.2, 0.1, 0.9, 0.8])
    aligner._centroid_kernel_vec = lambda weights, main, points: aligned_kernel
    aligned_loss = aligner._loss_co_cl(
        aligner.weights,
        aligner.main_centroids,
        float(aligner.main_centroid_labels[0]),
        aligner.sub_centroid_labels,
    )
    aligner._centroid_kernel_vec = lambda weights, main, points: crossed_kernel
    crossed_loss = aligner._loss_co_cl(
        aligner.weights,
        aligner.main_centroids,
        float(aligner.main_centroid_labels[0]),
        aligner.sub_centroid_labels,
    )
    assert aligned_loss < crossed_loss


def test_torch_ccka_co_margin_loss_rewards_same_class_similarity():
    torch = pytest.importorskip("torch")
    from ccka.aligner.torch_kta import TorchExperimentKTA

    class TorchTinyKernel:
        circuit = _TinyCircuit()
        circuit_executions = 0

        def forward(self, x1, x2, weights):
            self.circuit_executions += len(x1)
            return torch.exp(-((x1 - x2) ** 2).sum(dim=1))

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
        inference_landmarks="all",
        co_margin=0.2,
        co_temperature=0.1,
        lambda_co=0.0,
        seed=9,
    )
    labels = aligner.sub_centroid_labels
    aligned_kernel = torch.tensor([0.9, 0.8, 0.2, 0.1])
    crossed_kernel = torch.tensor([0.2, 0.1, 0.9, 0.8])
    aligner._kernel_vector = lambda left, right: aligned_kernel
    aligned_loss = aligner._centroid_loss(
        aligner.main_centroids[0],
        aligner.sub_centroids,
        labels,
        -1.0,
        0.0,
        class_label=float(aligner.main_centroid_labels[0]),
    )
    aligner._kernel_vector = lambda left, right: crossed_kernel
    crossed_loss = aligner._centroid_loss(
        aligner.main_centroids[0],
        aligner.sub_centroids,
        labels,
        -1.0,
        0.0,
        class_label=float(aligner.main_centroid_labels[0]),
    )
    assert aligned_loss < crossed_loss


def test_ccka_co_update_changes_main_centroid_only():
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
        seed=9,
    )
    original_main = aligner.main_centroids
    original_sub = aligner.sub_centroids
    updated_main = aligner._co_main_update(
        float(aligner.main_centroid_labels[0]),
        aligner.sub_centroid_labels,
    )
    assert not np.array_equal(np.asarray(updated_main), np.asarray(original_main))
    np.testing.assert_array_equal(np.asarray(aligner.sub_centroids), np.asarray(original_sub))


def test_torch_ccka_co_update_changes_main_centroid_only():
    torch = pytest.importorskip("torch")
    from ccka.aligner.torch_kta import TorchExperimentKTA

    class TorchTinyKernel:
        circuit = _TinyCircuit()
        circuit_executions = 0

        def forward(self, x1, x2, weights):
            self.circuit_executions += len(x1)
            return torch.exp(-((x1 - x2) ** 2).sum(dim=1))

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
        inference_landmarks="all",
        update_kao=False,
        update_co=True,
        seed=9,
    )
    original_main = aligner.main_centroids.detach().clone()
    original_sub = aligner.sub_centroids.detach().clone()
    aligner._step(1, inner_steps=1)
    assert not torch.equal(aligner.main_centroids.detach(), original_main)
    assert torch.equal(aligner.sub_centroids.detach(), original_sub)


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
        epochs=5,
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


def test_experiment_specs_can_disable_implicit_single_centroid_baseline():
    config = {
        "training": {"centroids": [5, 10, 20], "inference_landmarks": 32},
        "experiments": {
            "methods": ["ccka"],
            "ccka_initializations": ["regular"],
            "include_single_centroid_baseline": False,
            "ccka_ablations": {},
        },
    }

    specs = list(_experiment_specs(config))

    assert [spec[1] for spec in specs] == [5, 10, 20]
    assert all(spec[3] == "full" for spec in specs)


def test_ccka_treats_epochs_as_total_inner_updates():
    data = _splits()
    aligner = CentroidBasedKTA(
        _TinyKernel(),
        jnp.asarray(data[0]),
        jnp.asarray(data[3]),
        validation_data=jnp.asarray(data[1]),
        validation_labels=jnp.asarray(data[4]),
        test_data=jnp.asarray(data[2]),
        test_labels=jnp.asarray(data[5]),
        centroids=2,
        epochs=5,
        inner_steps=2,
        seed=9,
    )
    update_order = []

    def kao_update(main_centroid, labels, l=1.0):
        update_order.append("kao")
        return aligner.weights, aligner.sub_centroids

    def co_update(cl, y_raw, l=-1.0):
        update_order.append("co")
        return aligner.main_centroids

    aligner._kao_joint_update = kao_update
    aligner._co_main_update = co_update
    result = aligner.align()

    assert update_order == ["kao", "co"] * 5
    assert len(result["validation_accuracy_history"]) == 3


def test_quack_treats_epochs_as_total_inner_updates():
    data = _splits()
    aligner = QuackKTA(
        _TinyKernel(),
        jnp.asarray(data[0]),
        jnp.asarray(data[3]),
        validation_data=jnp.asarray(data[1]),
        validation_labels=jnp.asarray(data[4]),
        test_data=jnp.asarray(data[2]),
        test_labels=jnp.asarray(data[5]),
        epochs=5,
        inner_steps=2,
        seed=9,
    )
    update_order = []

    def kao_update(*args, **kwargs):
        update_order.append("kao")
        return aligner.weights

    def co_update(*args, **kwargs):
        update_order.append("co")
        return aligner.main_centroids

    aligner._kao_weight_update_quack = kao_update
    aligner._main_centroid_gradient_update_quack = co_update
    result = aligner.align()

    assert update_order == ["kao", "kao", "co", "co", "kao"]
    assert len(result["validation_accuracy_history"]) == 3


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


@pytest.mark.parametrize("method", ["fullKTA", "randomKTA"])
def test_jax_global_methods_run_once_per_outer_epoch(method):
    data = _splits()
    aligner = _build_aligner(
        method,
        data,
        {**_settings(), "epochs": 5, "inner_steps": 2},
        seed=9,
        centroids=2,
    )

    result = aligner.align()

    assert aligner.epochs == 5
    assert aligner.inner_steps == 2
    assert len(result["validation_accuracy_history"]) == 3


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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        validation_metrics = aligner.svm_training(
            aligner.xtrain, aligner.ytrain
        )
    assert not any(
        "probability" in str(warning.message) and "deprecated" in str(warning.message)
        for warning in caught
    )
    assert "validation_accuracy" in validation_metrics
    assert "test_accuracy" not in validation_metrics
    final_metrics = aligner.svm_training(
        aligner.xtrain, aligner.ytrain, evaluate_test=True
    )
    assert "test_accuracy" in final_metrics


def test_greedy_kta_calibrates_probabilities_without_deprecation_warning():
    data = _splits()
    aligner = GreedyKTA(
        _TinyKernel(),
        jnp.asarray(data[0]),
        jnp.asarray(data[3]),
        validation_data=jnp.asarray(data[1]),
        validation_labels=jnp.asarray(data[4]),
        test_data=jnp.asarray(data[2]),
        test_labels=jnp.asarray(data[5]),
        landmark_points=3,
        greedy_samples=2,
        seed=9,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        batch, labels = aligner._get_batch(epoch=0)

    assert batch.shape == (2, data[0].shape[1])
    assert labels.shape == (2,)
    assert not any(
        "probability" in str(warning.message) and "deprecated" in str(warning.message)
        for warning in caught
    )


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
    with pytest.raises(ValueError, match="Unknown configured method 'greedyKTA'"):
        list(
            _experiment_specs(
                {"training": {}, "experiments": {"methods": ["greedyKTA"]}}
            )
        )


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


def test_torch_full_inference_kernel_uses_all_training_points():
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
    model = TorchTinyKernel()
    aligner = TorchExperimentKTA(
        model,
        data[0],
        data[3],
        method="ccka",
        validation_data=data[1],
        validation_labels=data[4],
        test_data=data[2],
        test_labels=data[5],
        centroids=2,
        inference_landmarks="all",
        seed=9,
    )
    model.circuit_executions = 0
    K_train, K_validation = aligner._nystrom_matrices(aligner.xvalidation)
    K_test_train, K_test = aligner._nystrom_matrices(aligner.xtest)
    n, v, t = len(data[0]), len(data[1]), len(data[2])

    assert K_train.shape == (n, n)
    assert K_validation.shape == (v, n)
    assert K_test_train.shape == (n, n)
    assert K_test.shape == (t, n)
    assert model.circuit_executions == (
        2 * (n * (n + 1) // 2) + (v + t) * n
    )


@pytest.mark.parametrize(
    "method", ["fullKTA", "randomKTA", "quackKTA", "ccka"]
)
def test_torch_methods_scale_outer_epochs_from_update_budget(method):
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
        method=method,
        validation_data=data[1],
        validation_labels=data[4],
        test_data=data[2],
        test_labels=data[5],
        centroids=2,
        epochs=5,
        inner_steps=2,
        seed=9,
    )
    operations = []
    centroid_loss = aligner._centroid_loss

    def record_operation(*args, **kwargs):
        operations.append("kao" if kwargs.get("weight_penalty") else "co")
        return centroid_loss(*args, **kwargs)

    aligner._centroid_loss = record_operation
    result = aligner.align()

    if method == "ccka":
        assert operations == ["kao", "co"] * 5
    elif method == "quackKTA":
        assert operations == ["kao", "kao", "co", "co", "kao"]
    assert len(result["validation_accuracy_history"]) == 3


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
    expected_validation_steps = 1
    assert len(result["validation_accuracy_history"]) == expected_validation_steps
    assert "best_validation_accuracy" in result
    assert np.isfinite(result["test_accuracy_history"][-1])


def test_smoke_configs_cover_both_datasets_and_noise_modes():
    config_path = __import__("pathlib").Path(__file__).parent.parent / "configs" / "smoke.yaml"
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
    assert config["training"]["inference_method"] == "kernel_distance"
    assert config["training"]["quackKTA_inference_method"] == "svm"
    assert config["training"]["ccka_inference_method"] == "kernel_distance"
    assert config["training"]["test_fraction"] == 0.5
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
        settings={"min_n_qubits": 5},
    )
    assert data[0].shape[1] == 10


def test_prepare_data_uses_separate_stratified_validation_and_half_test_split():
    source_train, source_test, _, _ = _read_dataset(
        "checkerboard", seed=3, max_samples=None
    )
    train, validation, test, train_labels, validation_labels, test_labels = (
        _prepare_data(
            "checkerboard",
            seed=3,
            settings={"min_n_qubits": 5},
        )
    )

    total_samples = len(source_train) + len(source_test)
    assert len(test) == int(np.ceil(total_samples * 0.5))
    assert len(validation) == int(np.ceil(total_samples * 0.5 * 0.2))
    assert len(train) + len(validation) + len(test) == total_samples
    assert not np.array_equal(validation, test[: len(validation)])
    assert len(train_labels) == len(train)
    assert len(validation_labels) == len(validation)
    assert len(test_labels) == len(test)


def test_prepare_data_can_explicitly_reuse_test_as_validation():
    data = _prepare_data(
        "checkerboard",
        seed=3,
        settings={"min_n_qubits": 5, "test_fraction": 0.5, "test_as_validation": True},
    )
    train, validation, test, train_labels, validation_labels, test_labels = data

    assert len(train) == 30
    assert len(validation) == 30
    np.testing.assert_array_equal(validation, test)
    np.testing.assert_array_equal(validation_labels, test_labels)
    assert len(train_labels) == 30


def test_run_method_restores_best_validation_checkpoint(monkeypatch):
    class _FakeAligner:
        inner_steps = 2
        inference_landmarks = 3
        inference_method = "svm"
        xtrain = np.zeros((2, 1))
        ytrain = np.asarray([-1, 1])

        def __init__(self):
            self.weights = 0

        def svm_training(self, *args, evaluate_test=False):
            if evaluate_test:
                return {"test_accuracy": float(self.weights) / 10}
            return {"validation_accuracy": 0.5}

    def fake_step(
        method, aligner, epoch, target, rows, seed, limit, init, stop, checkpoint
    ):
        for iteration, (weight, accuracy) in enumerate(((1, 0.8), (2, 0.6)), 1):
            aligner.weights = weight
            row = {
                "method": method,
                "method_label": "CCKA",
                "initialization": init,
                "centroids": 2,
                "seed": seed,
                "inference_landmarks": 3,
                "inference_method": "svm",
                "outer_epoch": epoch + 1,
                "inner_step": iteration,
                "optimization_updates": iteration * 2,
                "optimization_iterations": iteration,
                "training_circuit_executions": iteration * 5,
                "validation_accuracy": accuracy,
                "target_reached": accuracy >= target,
                "best_checkpoint_iteration": None,
                "restored_test_accuracy": None,
            }
            rows.append(row)
            execution_module._update_best_checkpoint(
                aligner, row, checkpoint
            )
        return 4, False

    aligner = _FakeAligner()
    monkeypatch.setattr(execution_module, "_build_aligner", lambda *args: aligner)
    monkeypatch.setattr(execution_module, "BACKEND", "jax")
    monkeypatch.setattr(execution_module, "_jax_step", fake_step)
    rows = execution_module.run_method(
        "ccka",
        (),
        {},
        seed=0,
        centroids=2,
        target_accuracy=0.9,
        max_iterations=2,
        stop_on_target=False,
    )

    assert aligner.weights == 1
    assert rows[-1]["best_checkpoint_iteration"] == 1
    assert rows[-1]["restored_test_accuracy"] == 0.1


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


def test_ccka_specs_sweep_configured_inference_landmarks():
    config = {
        "datasets": ["checkerboard"],
        "training": {"centroids": [2], "inference_landmarks": 5},
        "experiments": {
            "methods": ["ccka"],
            "ccka_initializations": ["regular"],
            "ccka_inference_landmarks": [2, 4, "all"],
            "ccka_ablations": {},
        },
        "seeds": [0],
        "noise": {"types": ["none"], "levels": [0.01], "shots": [8]},
    }
    specs = list(_experiment_specs(config))
    assert [spec[-1] for spec in specs] == [2, 4, "all", 2, 4, "all"]


def test_ccka_full_inference_kernel_uses_all_training_points():
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
        inference_landmarks="all",
        seed=9,
    )
    model.circuit_executions = 0
    K_train, K_validation, K_test = aligner._evaluation_kernel_matrices(
        aligner.weights,
        aligner.xtrain,
        aligner.xvalidation,
        aligner.xtest,
    )
    n, v, t = len(data[0]), len(data[1]), len(data[2])
    assert K_train.shape == (n, n)
    assert K_validation.shape == (v, n)
    assert K_test.shape == (t, n)
    assert model.circuit_executions == n * (n + 1) // 2 + (v + t) * n


def test_experiment_result_serializes_per_epoch_validation_history(monkeypatch):
    class _FakeAligner:
        inference_method = "svm"

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
