from __future__ import annotations

import argparse
import csv
import itertools
import functools
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ccka.backend import BACKEND
from ccka.circuits.angleEmbeddingKernel import quackEmbeddingCircuit
from ccka.circuits.ansatz import DEFAULT_ANSATZ, ansatz_names
from experiments.datasets import GENERATED_DATASETS, generate_dataset
from experiments.multiclass import MulticlassAligner, normalise_strategy

if BACKEND == "jax":
    import jax.numpy as jnp

    from ccka.aligner.kta import (
        CentroidBasedKTA,
        FullKTA,
        QuackKTA,
        RandomKTA,
    )
    from ccka.models.kernel import KernelModel

    METHODS = {
        "fullKTA": FullKTA,
        "randomKTA": RandomKTA,
        "quackKTA": QuackKTA,
        "ccka": CentroidBasedKTA,
    }
else:
    import torch

    from ccka.aligner.torch_kta import TorchExperimentKTA
    from ccka.models.torch_kernel import TorchKernelModel

    METHODS = {
        "fullKTA": TorchExperimentKTA,
        "randomKTA": TorchExperimentKTA,
        "quackKTA": TorchExperimentKTA,
        "ccka": TorchExperimentKTA,
    }


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

DATASETS = {
    "adult": DATA_DIR / "adult.npy",
    "banknote_authentication": DATA_DIR / "banknote_authentication.npy",
    "breast_cancer_wisconsin": DATA_DIR / "breast_cancer_wisconsin.npy",
    "checkerboard": DATA_DIR / "checkerboard_dataset.npy",
    "corners": DATA_DIR / "corners.npy",
    "corners_reduced": DATA_DIR / "corners_reduced.npy",
    "covtype": DATA_DIR / "covtype.npy",
    "donuts": DATA_DIR / "donuts.npy",
    "german_credit": DATA_DIR / "german_credit.npy",
    "one_vs_nonone": DATA_DIR / "one_vs_nonone.npy",
    "pfault": DATA_DIR / "pfault.npy",
    "pima_diabetes": DATA_DIR / "pima_diabetes.npy",
    "qsar_biodegradation": DATA_DIR / "qsar_biodegradation.npy",
    "zero_vs_nonzero": DATA_DIR / "zero_vs_nonzero.npy",
    **{name: None for name in GENERATED_DATASETS},
}

METHOD_LABELS = {
    "ccka": "CCKA",
    "quackKTA": "QUACK",
    "fullKTA": "All-Pairwise KTA (Full KTA)",
    "randomKTA": "Random-Pairwise KTA",
}

LOGGER = logging.getLogger("ccka.experiments")


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config; an optional ``base:`` key names a parent config
    (relative to the config directory) that is deep-merged underneath."""
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = config.pop("base", None)
    if base is None:
        return config
    parent = Path(base)
    if not parent.is_absolute():
        parent = path.parent / parent
    return _merge(load_config(parent), config)


def _parse_inference_landmarks(value: int | str) -> int | str:
    if value == "all":
        return value
    if isinstance(value, bool):
        raise ValueError("inference_landmarks must be a positive integer or 'all'.")
    try:
        landmarks = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "inference_landmarks must be a positive integer or 'all'."
        ) from exc
    if landmarks <= 0:
        raise ValueError("inference_landmarks must be a positive integer or 'all'.")
    return landmarks


def _read_dataset(
    name: str, seed: int, max_samples: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {sorted(DATASETS)}.")
    path = DATASETS[name]
    if path is None:
        X, y = generate_dataset(name, seed, max_samples or 300)
        return train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    dataset = np.load(path, allow_pickle=True).item()
    required = {"x_train", "y_train", "x_test", "y_test"}
    if not required.issubset(dataset):
        raise ValueError(
            f"{path} must contain {sorted(required)}; found {sorted(dataset)}."
        )
    X_train = np.asarray(dataset["x_train"], dtype=np.float32)
    y_train = np.asarray(dataset["y_train"]).reshape(-1)
    X_test = np.asarray(dataset["x_test"], dtype=np.float32)
    y_test = np.asarray(dataset["y_test"]).reshape(-1)

    if max_samples is not None and len(X_train) + len(X_test) > max_samples:
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=max_samples,
            random_state=seed,
            stratify=y,
        )
    return X_train, X_test, y_train, y_test


def _imbalance(
    X: np.ndarray, y: np.ndarray, minority_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample the smaller class so it makes up ``minority_fraction`` of the data."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("class_imbalance is only defined for binary datasets.")
    if not 0.0 < minority_fraction <= 0.5:
        raise ValueError("class_imbalance must be in (0, 0.5].")
    minority = classes[np.argmin(counts)]
    majority_count = int(counts.max())
    keep = int(round(majority_count * minority_fraction / (1 - minority_fraction)))
    keep = max(2, min(keep, int(counts.min())))
    rng = np.random.default_rng(seed)
    minority_index = rng.choice(np.flatnonzero(y == minority), keep, replace=False)
    selected = np.sort(np.concatenate((np.flatnonzero(y != minority), minority_index)))
    return X[selected], y[selected]


def _flip_labels(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("label_noise must be in [0, 1).")
    rng = np.random.default_rng(seed + 10_007)
    classes = np.unique(y)
    flipped = np.array(y, copy=True)
    for index in rng.choice(len(y), int(round(fraction * len(y))), replace=False):
        others = classes[classes != y[index]]
        flipped[index] = rng.choice(others)
    return flipped


def _prepare_data(
    name: str,
    seed: int,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_train, source_test, source_y_train, source_y_test = _read_dataset(
        name, seed, settings.get("max_samples")
    )
    X_all = np.concatenate((source_train, source_test), axis=0)
    y_all = np.concatenate((source_y_train, source_y_test), axis=0)
    imbalance = settings.get("class_imbalance")
    if imbalance is not None:
        X_all, y_all = _imbalance(X_all, y_all, float(imbalance), seed)
    test_fraction = float(settings.get("test_fraction", 0.5))
    validation_fraction = float(settings.get("validation_fraction", 0.2))
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1).")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_fraction,
        random_state=seed,
        stratify=y_all,
    )
    if settings.get("test_as_validation", False):
        X_train = X_train_validation
        y_train = y_train_validation
        X_validation = X_test
        y_validation = y_test
    else:
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train_validation,
            y_train_validation,
            test_size=validation_fraction,
            random_state=seed,
            stratify=y_train_validation,
        )

    n_train = settings.get("n_train")
    if n_train is not None and int(n_train) < len(X_train):
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=int(n_train),
            random_state=seed,
            stratify=y_train,
        )
    label_noise = float(settings.get("label_noise", 0.0))
    if label_noise > 0.0:
        y_train = _flip_labels(y_train, label_noise, seed)

    if settings.get("n_qubits") is not None:
        n_qubits = int(settings["n_qubits"])
    else:
        n_qubits = max(
            5,
            int(settings.get("min_n_qubits", 5)),
            X_train.shape[1],
        )
    if X_train.shape[1] > n_qubits:
        n_components = min(n_qubits, X_train.shape[1], len(X_train))
        pca = PCA(n_components=n_components, random_state=seed)
        X_train = pca.fit_transform(X_train)
        X_validation = pca.transform(X_validation)
        X_test = pca.transform(X_test)

    feature_range = tuple(settings.get("feature_range", [0.0, 1.0]))
    scaler = MinMaxScaler(feature_range=feature_range)
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)

    if X_train.shape[1] < n_qubits:
        pad = n_qubits - X_train.shape[1]
        X_train = np.pad(X_train, ((0, 0), (0, pad)))
        X_validation = np.pad(X_validation, ((0, 0), (0, pad)))
        X_test = np.pad(X_test, ((0, 0), (0, pad)))

    classes = np.unique(y_train)
    if len(classes) == 2:
        label_map = {classes[0]: -1, classes[1]: 1}
    else:
        label_map = {label: index + 1 for index, label in enumerate(classes)}
    y_train = np.asarray([label_map[label] for label in y_train], dtype=np.int32)
    y_validation = np.asarray(
        [label_map[label] for label in y_validation], dtype=np.int32
    )
    y_test = np.asarray([label_map.get(label, label) for label in y_test], dtype=np.int32)

    return (
        X_train.astype(np.float32),
        X_validation,
        X_test.astype(np.float32),
        y_train,
        y_validation,
        y_test,
    )


def _make_aligner(
    method: str,
    model: KernelModel,
    data: tuple[np.ndarray, ...],
    settings: dict[str, Any],
    seed: int,
    *,
    centroids: int,
    initialization: str = "regular",
    update_kao: bool = True,
    update_co: bool = True,
) -> Any:
    X_train, X_validation, X_test, y_train, y_validation, y_test = data
    inference_method = settings.get(
        f"{method}_inference_method",
        settings.get("inference_method", "svm"),
    )
    if inference_method not in {"svm", "kernel_distance"}:
        raise ValueError(
            f"Unsupported inference method {inference_method!r} for {method}; "
            "choose 'svm' or 'kernel_distance'."
        )
    inference_neighbors = int(settings.get("inference_neighbors", 1))
    learning_rate = float(
        settings.get(
            f"{method}_learning_rate",
            settings.get("learning_rate", 0.05),
        )
    )
    if method == "ccka":
        inner_steps = int(
            settings.get(
                "ccka_inner_steps",
                settings.get("quack_inner_steps", 10),
            )
        )
    elif method == "quackKTA":
        inner_steps = int(settings.get("quack_inner_steps", 10))
    else:
        inner_steps = int(
            settings.get(
                "inner_steps",
                settings.get(
                    "quack_inner_steps",
                    settings.get("ccka_inner_steps", 10),
                ),
            )
        )
    if BACKEND == "torch":
        return TorchExperimentKTA(
            kernel_model=model,
            data=X_train,
            labels=y_train,
            validation_data=X_validation,
            validation_labels=y_validation,
            test_data=X_test,
            test_labels=y_test,
            method=method,
            centroids=centroids,
            clustering=initialization,
            random_samples=centroids,
            epochs=int(settings.get("epochs", 1)),
            learning_rate=learning_rate,
            centroid_lr=float(settings.get("centroid_lr", 0.01)),
            sub_centroid_lr=float(settings.get("sub_centroid_lr", 0.01)),
            lambda_co=float(settings.get("lambda_co", 0.001)),
            lambda_kao=float(settings.get("lambda_kao", 0.001)),
            co_margin=float(settings.get("co_margin", 0.1)),
            co_temperature=float(settings.get("co_temperature", 0.1)),
            svm_c=float(settings.get("svm_c", 1.0)),
            inference_method=inference_method,
            inference_neighbors=inference_neighbors,
            inference_landmarks=_parse_inference_landmarks(
                settings.get("inference_landmarks", 32)
            ),
            inner_steps=inner_steps,
            update_kao=update_kao,
            update_co=update_co,
            seed=seed,
        )
    common = {
        "kernel_model": model,
        "data": jnp.asarray(X_train),
        "labels": jnp.asarray(y_train),
        "validation_data": jnp.asarray(X_validation),
        "validation_labels": jnp.asarray(y_validation),
        "test_data": jnp.asarray(X_test),
        "test_labels": jnp.asarray(y_test),
        "matrix_type": "regular",
        "centering": False,
        "collect_diagnostics": bool(settings.get("collect_diagnostics", False)),
        "inference_landmarks": _parse_inference_landmarks(
            settings.get("inference_landmarks", 32)
        ),
        "svm_c": float(settings.get("svm_c", 1.0)),
        "inference_method": inference_method,
        "inference_neighbors": inference_neighbors,
        "epochs": int(settings.get("epochs", 1)),
        "inner_steps": inner_steps,
        "learning_rate": learning_rate,
        "optimizer": settings.get("optimizer", "adam"),
        "seed": seed,
    }
    if method == "fullKTA":
        return FullKTA(**common)
    if method == "randomKTA":
        return RandomKTA(
            **common,
            random_samples=centroids,
        )
    if method == "quackKTA":
        return QuackKTA(
            **common,
            lambda_co=float(settings.get("lambda_co", 0.001)),
            lambda_kao=float(settings.get("lambda_kao", 0.001)),
            centroid_lr=float(settings.get("centroid_lr", 0.01)),
        )
    if method == "ccka":
        return CentroidBasedKTA(
            **common,
            centroids=centroids,
            clustering=initialization,
            centroid_lr=float(settings.get("centroid_lr", 0.01)),
            sub_centroid_lr=float(settings.get("sub_centroid_lr", 0.01)),
            lambda_co=float(settings.get("lambda_co", 0.001)),
            lambda_kao=float(settings.get("lambda_kao", 0.001)),
            co_margin=float(settings.get("co_margin", 0.1)),
            co_temperature=float(settings.get("co_temperature", 0.1)),

            update_kao=update_kao,
            update_co=update_co,
        )
    raise ValueError(f"Unknown method {method!r}.")


def _scaled_init(base_init, scale: float, seed: int = 0, **kwargs):
    # scale=0 gives all-zero weights; small scales give near-identity circuits.
    return base_init(seed=seed, minval=-np.pi * scale, maxval=np.pi * scale)


def _build_model(
    settings: dict[str, Any],
    num_qubits: int,
    noise: dict[str, Any],
    shots: int,
    seed: int,
    ansatz: str,
):
    noisy = noise["type"] != "none"
    circuit = quackEmbeddingCircuit(
        num_qubits=num_qubits,
        reps=int(settings.get("reps", 2)),
        reupload=True,
        noisy=noisy,
        noise_type=noise["type"],
        noise_level=float(noise["level"]),
        seed=seed,
        ansatz=ansatz,
    )
    init_scale = float(settings.get("init_scale", 1.0))
    if init_scale != 1.0:
        circuit.init_weights = functools.partial(
            _scaled_init, circuit.init_weights, init_scale
        )
    if BACKEND == "jax":
        return KernelModel(
            circuit=circuit,
            device_name="default.mixed" if noisy else "default.qubit",
            interface="jax",
            diff_method="parameter-shift" if noisy else "backprop",
            noisy=noisy,
            noise_type=noise["type"] != "none",
            shots=shots if noisy else None,
        )
    return TorchKernelModel(
        circuit=circuit,
        device_name="default.mixed" if noisy else "default.qubit",
        diff_method="parameter-shift" if noisy else "backprop",
        noisy=noisy,
        shots=shots if noisy else None,
    )


def _jsonable(value: Any) -> Any:
    return json.dumps(value) if isinstance(value, (list, tuple, dict)) else value


def _run_one(
    dataset_name: str,
    method_name: str,
    data: tuple[np.ndarray, ...],
    settings: dict[str, Any],
    seed: int,
    noise: dict[str, Any],
    shots: int,
    centroids: int,
    *,
    initialization: str = "regular",
    ablation: str = "none",
    update_kao: bool = True,
    update_co: bool = True,
    ansatz: str = DEFAULT_ANSATZ,
    extra_columns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    noisy = noise["type"] != "none"
    n_classes = int(len(np.unique(data[3])))
    strategy = None
    if n_classes > 2:
        if BACKEND != "jax":
            raise NotImplementedError(
                "Multiclass training requires the JAX backend."
            )
        strategy = normalise_strategy(settings.get("multiclass_strategy", "ovr"))
        settings = {
            **settings,
            "inference_method": "svm",
            f"{method_name}_inference_method": "svm",
        }

    def build(sub_data: tuple[np.ndarray, ...]):
        model = _build_model(
            settings, sub_data[0].shape[1], noise, shots, seed, ansatz
        )
        aligner = _make_aligner(
            method_name,
            model,
            sub_data,
            settings,
            seed,
            centroids=centroids,
            initialization=initialization,
            update_kao=update_kao,
            update_co=update_co,
        )
        return aligner, model

    if strategy is None:
        aligner, _ = build(data)
    else:
        aligner = MulticlassAligner(
            build,
            data,
            strategy=strategy,
            svm_c=float(settings.get("svm_c", 1.0)),
            seed=seed,
        )
    started = time.perf_counter()
    history = aligner.align()
    final_metrics = history["final_svm_metrics"]
    reps = int(settings.get("reps", 2))
    row = {
        "dataset": dataset_name,
        "method": method_name,
        "method_label": METHOD_LABELS[method_name],
        "backend": BACKEND,
        "ansatz": ansatz,
        "reps": reps,
        "n_qubits": int(data[0].shape[1]),
        "n_params": 3 * reps * int(data[0].shape[1]),
        "n_classes": n_classes,
        "multiclass_strategy": strategy,
        "n_train": int(len(data[0])),
        "n_test": int(len(data[2])),
        "class_imbalance": settings.get("class_imbalance"),
        "label_noise": float(settings.get("label_noise", 0.0)),
        "optimizer": settings.get("optimizer", "adam"),
        "learning_rate": float(
            settings.get(
                f"{method_name}_learning_rate",
                settings.get("learning_rate", 0.05),
            )
        ),
        "centroid_lr": float(settings.get("centroid_lr", 0.01)),
        "sub_centroid_lr": float(settings.get("sub_centroid_lr", 0.01)),
        "lambda_kao": float(settings.get("lambda_kao", 0.001)),
        "lambda_co": float(settings.get("lambda_co", 0.001)),
        "ablation": ablation,
        "initialization": initialization,
        "centroids_per_class": centroids if method_name == "ccka" else None,
        "inference_landmarks": _parse_inference_landmarks(
            settings.get("inference_landmarks", 32)
        ),
        "inference_method": aligner.inference_method,
        "update_kao": update_kao if method_name == "ccka" else None,
        "update_co": update_co if method_name == "ccka" else None,
        "noise_type": noise["type"],
        "noise_level": noise["level"],
        "shots": shots if noisy else None,
        "seed": seed,
        "epochs": int(settings.get("epochs", 1)),
        "validation_accuracy_history": json.dumps(
            [
                float(value)
                for value in history.get("validation_accuracy_history", [])
            ],
            separators=(",", ":"),
        ),
        "best_validation_accuracy": float(history.get("best_validation_accuracy", max(
            history.get("validation_accuracy_history", [float("nan")])
        ))),
        "validation_accuracy": float(final_metrics["validation_accuracy"]),
        "test_accuracy": float(final_metrics["test_accuracy"]),
        "test_f1_macro": float(final_metrics["f1_score"]),
        "test_precision_macro": float(final_metrics["precision_score"]),
        "test_recall_macro": float(final_metrics["recall_score"]),
        "circuit_executions": int(history["circuit_executions"]),
        "wall_time_seconds": time.perf_counter() - started,
    }
    for name, value in (extra_columns or {}).items():
        row.setdefault(name, _jsonable(value))
    return row


def _experiment_specs(config: dict[str, Any], *, include_ablations: bool = True):
    settings = config["training"]
    experiment_settings = config["experiments"]
    K_values = [int(value) for value in settings.get("centroids", [4])]
    ccka_landmarks = [
        _parse_inference_landmarks(value)
        for value in experiment_settings.get(
            "ccka_inference_landmarks",
            [settings.get("inference_landmarks", 32)],
        )
    ]
    methods = experiment_settings.get("methods", list(METHODS))
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown configured method {method!r}.")
        if method == "ccka":
            for K in K_values:
                for initialization in experiment_settings.get(
                    "ccka_initializations", ["regular"]
                ):
                    for landmarks in ccka_landmarks:
                        yield (
                            method,
                            K,
                            initialization,
                            "full",
                            True,
                            True,
                            {},
                            landmarks,
                        )
            initializations = experiment_settings.get(
                "ccka_initializations", ["regular"]
            )
            if (
                experiment_settings.get("include_single_centroid_baseline", True)
                and 1 not in K_values
                and "regular" in initializations
            ):
                for landmarks in ccka_landmarks:
                    yield (
                        method, 1, "regular", "full", True, True, {}, landmarks
                    )
            for name, variant in (
                config["experiments"].get("ccka_ablations", {}).items()
                if include_ablations
                else ()
            ):
                for landmarks in ccka_landmarks:
                    yield (
                        method,
                        int(variant.get("centroids", 1)),
                        str(variant.get("initialization", "regular")),
                        name,
                        bool(variant.get("update_kao", True)),
                        bool(variant.get("update_co", True)),
                        dict(variant.get("settings", {})),
                        landmarks,
                    )
        else:
            K_iter = K_values if method == "randomKTA" else [0]
            for K in K_iter:
                yield (
                    method,
                    K,
                    "not_applicable",
                    "none",
                    True,
                    True,
                    {},
                    _parse_inference_landmarks(
                        settings.get("inference_landmarks", 32)
                    ),
                )


# Ablations are only meaningful (and only affordable) on these two datasets.
ABLATION_DATASETS = ("checkerboard", "corners")


def _grid_combinations(grid: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def iter_experiment_jobs(config: dict[str, Any], *, smoke: bool = False):
    datasets = config["datasets"]
    if datasets == "all":
        dataset_names = list(DATASETS)
    else:
        dataset_names = list(datasets)
    unknown = sorted(set(dataset_names) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown configured datasets: {unknown}")

    training = dict(config["training"])
    if smoke:
        training.update(config.get("smoke_overrides", {}))
        dataset_names = list(config.get("smoke_datasets", ["checkerboard", "corners"]))
    ablation_datasets = set(
        config["experiments"].get("ablation_datasets", ABLATION_DATASETS)
    )
    ansatze = list(config.get("ansatze", [DEFAULT_ANSATZ]))
    unknown_ansatze = sorted(set(ansatze) - set(ansatz_names()))
    if unknown_ansatze:
        raise ValueError(f"Unknown configured ansatze: {unknown_ansatze}")
    noise_settings = config["noise"]
    noises = []
    for kind, level in itertools.product(
        noise_settings.get("types", ["none"]),
        noise_settings.get("levels", [0.01]),
    ):
        level = 0.0 if kind in {"none", "shots_only"} else float(level)
        noise = {"type": kind, "level": level}
        if noise not in noises:
            noises.append(noise)
    seeds = [int(seed) for seed in config.get("seeds", [0])]
    shot_counts = [int(value) for value in noise_settings.get("shots", [1024])]
    grid = _grid_combinations(config.get("training_grid"))
    for dataset_name in dataset_names:
        for overrides in grid:
            grid_training = {**training, **overrides}
            for seed in seeds:
                data = _prepare_data(dataset_name, seed, grid_training)
                for ansatz in ansatze:
                    for noise in noises:
                        current_shots = (
                            shot_counts if noise["type"] != "none" else [0]
                        )
                        for shots in current_shots:
                            for (
                                method,
                                K,
                                initialization,
                                ablation,
                                update_kao,
                                update_co,
                                variant_settings,
                                inference_landmarks,
                            ) in _experiment_specs(
                                config,
                                include_ablations=dataset_name
                                in ablation_datasets,
                            ):
                                job_settings = {
                                    **grid_training,
                                    **variant_settings,
                                }
                                if method == "ccka":
                                    job_settings["inference_landmarks"] = (
                                        inference_landmarks
                                    )
                                yield {
                                    "dataset": dataset_name,
                                    "method": method,
                                    "settings": job_settings,
                                    "seed": seed,
                                    "noise": noise,
                                    "shots": shots,
                                    "centroids": K,
                                    "initialization": initialization,
                                    "ablation": ablation,
                                    "update_kao": update_kao,
                                    "update_co": update_co,
                                    "inference_landmarks": inference_landmarks,
                                    "ansatz": ansatz,
                                    "extra_columns": {
                                        **overrides,
                                        **variant_settings,
                                    },
                                    "data": data,
                                }


def run_experiment_job(job: dict[str, Any], data=None) -> dict[str, Any]:
    if data is None:
        data = job["data"]
    try:
        result = _run_one(
            job["dataset"],
            job["method"],
            data,
            job["settings"],
            job["seed"],
            job["noise"],
            job["shots"],
            job["centroids"],
            initialization=job["initialization"],
            ablation=job["ablation"],
            update_kao=job["update_kao"],
            update_co=job["update_co"],
            ansatz=job.get("ansatz", DEFAULT_ANSATZ),
            extra_columns=job.get("extra_columns"),
        )
    except Exception:
        LOGGER.exception(
            "Experiment failed: dataset=%s method=%s noise=%s seed=%s "
            "K=%s init=%s ablation=%s",
            job["dataset"],
            job["method"],
            job["noise"]["type"],
            job["seed"],
            job["centroids"],
            job["initialization"],
            job["ablation"],
        )
        raise
    LOGGER.info(
        "%s | %s | M=%s | validation=%.4f test=%.4f circuits=%d",
        job["dataset"],
        job["method"],
        result["inference_landmarks"],
        result["validation_accuracy"],
        result["test_accuracy"],
        result["circuit_executions"],
    )
    return result


def run(config: dict[str, Any], *, smoke: bool = False) -> list[dict[str, Any]]:
    output_path = ROOT / config.get("output_csv", "results/experiments.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for job in iter_experiment_jobs(config, smoke=smoke):
        results.append(run_experiment_job(job))

    if results:
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        LOGGER.info("Wrote %d results to %s", len(results), output_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CCKA method comparisons and ablation experiments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "default.yaml",
        help="YAML experiment configuration.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the tiny checkerboard/corners smoke-test configuration.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_config(args.config)
    run(config, smoke=args.smoke)


if __name__ == "__main__":
    main()
