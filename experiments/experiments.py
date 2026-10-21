from __future__ import annotations

import argparse
import csv
import itertools
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
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

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
    "double_cake": None,
    "moons": None,
}

METHOD_LABELS = {
    "ccka": "CCKA",
    "quackKTA": "QUACK",
    "fullKTA": "All-Pairwise KTA (Full KTA)",
    "randomKTA": "Random-Pairwise KTA",
}

LOGGER = logging.getLogger("ccka.experiments")


def _generate_dataset(name: str, seed: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if name == "moons":
        from sklearn.datasets import make_moons

        return make_moons(n_samples=n_samples, noise=0.12, random_state=seed)
    if name == "double_cake":
        angles = rng.uniform(0.0, 2.0 * np.pi, n_samples)
        labels = rng.integers(0, 2, n_samples)
        radii = np.where(labels == 0, 0.35, 0.9) + rng.normal(0, 0.06, n_samples)
        X = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        return X, labels
    raise ValueError(f"No generator is defined for dataset {name!r}.")


def _read_dataset(
    name: str, seed: int, max_samples: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {sorted(DATASETS)}.")
    path = DATASETS[name]
    if path is None:
        X, y = _generate_dataset(name, seed, max_samples or 300)
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


def _prepare_data(
    name: str,
    seed: int,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = _read_dataset(
        name, seed, settings.get("max_samples")
    )
    validation_fraction = float(settings.get("validation_fraction", 0.2))
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y_train,
    )

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
        [label_map.get(label, label) for label in y_validation], dtype=np.int32
    )
    y_test = np.asarray([label_map.get(label, label) for label in y_test], dtype=np.int32)

    return (
        X_train.astype(np.float32),
        X_validation.astype(np.float32),
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
            random_samples=int(settings.get("random_samples", centroids)),
            epochs=int(settings.get("epochs", 1)),
            learning_rate=float(settings.get("learning_rate", 0.05)),
            centroid_lr=float(settings.get("centroid_lr", 0.01)),
            sub_centroid_lr=float(settings.get("sub_centroid_lr", 0.01)),
            lambda_co=float(settings.get("lambda_co", 0.001)),
            lambda_kao=float(settings.get("lambda_kao", 0.001)),
            svm_c=float(settings.get("svm_c", 1.0)),
            inference_landmarks=int(settings.get("inference_landmarks", 32)),
            inner_steps=int(settings.get("quack_inner_steps", 10)),
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
        "inference_landmarks": int(settings.get("inference_landmarks", 32)),
        "svm_c": float(settings.get("svm_c", 1.0)),
        "epochs": int(settings.get("epochs", 1)),
        "learning_rate": float(settings.get("learning_rate", 0.05)),
        "optimizer": settings.get("optimizer", "adam"),
        "seed": seed,
    }
    if method == "fullKTA":
        return FullKTA(**common)
    if method == "randomKTA":
        return RandomKTA(
            **common,
            random_samples=int(settings.get("random_samples", centroids)),
        )
    if method == "quackKTA":
        return QuackKTA(
            **common,
            lambda_co=float(settings.get("lambda_co", 0.001)),
            lambda_kao=float(settings.get("lambda_kao", 0.001)),
            centroid_lr=float(settings.get("centroid_lr", 0.01)),
            inner_steps=int(settings.get("quack_inner_steps", 10)),
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
            update_kao=update_kao,
            update_co=update_co,
        )
    raise ValueError(f"Unknown method {method!r}.")


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
) -> dict[str, Any]:
    noisy = noise["type"] != "none"
    circuit = quackEmbeddingCircuit(
        num_qubits=data[0].shape[1],
        reps=int(settings.get("reps", 2)),
        reupload=True,
        noisy=noisy,
        noise_type=noise["type"],
        noise_level=float(noise["level"]),
        seed=seed,
    )
    if BACKEND == "jax":
        model = KernelModel(
            circuit=circuit,
            device_name="default.mixed" if noisy else "default.qubit",
            interface="jax",
            diff_method="parameter-shift" if noisy else "backprop",
            noisy=noisy,
            noise_type=noise["type"] != "none",
            shots=shots if noisy else None,
        )
    else:
        model = TorchKernelModel(
            circuit=circuit,
            device_name="default.mixed" if noisy else "default.qubit",
            diff_method="parameter-shift" if noisy else "backprop",
            noisy=noisy,
            shots=shots if noisy else None,
        )
    aligner = _make_aligner(
        method_name,
        model,
        data,
        settings,
        seed,
        centroids=centroids,
        initialization=initialization,
        update_kao=update_kao,
        update_co=update_co,
    )
    started = time.perf_counter()
    history = aligner.align()
    final_metrics = history["final_svm_metrics"]
    return {
        "dataset": dataset_name,
        "method": method_name,
        "method_label": METHOD_LABELS[method_name],
        "backend": BACKEND,
        "n_qubits": int(data[0].shape[1]),
        "ablation": ablation,
        "initialization": initialization,
        "centroids_per_class": centroids if method_name == "ccka" else None,
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


def _experiment_specs(config: dict[str, Any]):
    settings = config["training"]
    K_values = [int(value) for value in settings.get("centroids", [4])]
    methods = config["experiments"].get("methods", list(METHODS))
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown configured method {method!r}.")
        if method == "ccka":
            for K in K_values:
                for initialization in config["experiments"].get(
                    "ccka_initializations", ["regular"]
                ):
                    yield method, K, initialization, "full", True, True
            initializations = config["experiments"].get(
                "ccka_initializations", ["regular"]
            )
            if 1 not in K_values and "regular" in initializations:
                yield method, 1, "regular", "full", True, True
            for name, variant in config["experiments"].get(
                "ccka_ablations", {}
            ).items():
                yield (
                    method,
                    int(variant.get("centroids", 1)),
                    str(variant.get("initialization", "regular")),
                    name,
                    bool(variant.get("update_kao", True)),
                    bool(variant.get("update_co", True)),
                )
        else:
            K_iter = K_values if method in {"randomKTA", "quackKTA"} else [0]
            for K in K_iter:
                yield method, K, "not_applicable", "none", True, True


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
    noise_settings = config["noise"]
    noises = [
        {"type": kind, "level": 0.0 if kind == "none" else float(level)}
        for kind, level in itertools.product(
            noise_settings.get("types", ["none"]),
            noise_settings.get("levels", [0.01]),
        )
    ]
    seeds = [int(seed) for seed in config.get("seeds", [0])]
    shot_counts = [int(value) for value in noise_settings.get("shots", [1024])]
    for dataset_name in dataset_names:
        for seed in seeds:
            data = _prepare_data(dataset_name, seed, training)
            for noise in noises:
                current_shots = shot_counts if noise["type"] != "none" else [0]
                for shots in current_shots:
                    for (
                        method,
                        K,
                        initialization,
                        ablation,
                        update_kao,
                        update_co,
                    ) in _experiment_specs(config):
                        yield {
                            "dataset": dataset_name,
                            "method": method,
                            "settings": training,
                            "seed": seed,
                            "noise": noise,
                            "shots": shots,
                            "centroids": K,
                            "initialization": initialization,
                            "ablation": ablation,
                            "update_kao": update_kao,
                            "update_co": update_co,
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
        "%s | %s | validation=%.4f test=%.4f circuits=%d",
        job["dataset"],
        job["method"],
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
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run(config, smoke=args.smoke)


if __name__ == "__main__":
    main()
