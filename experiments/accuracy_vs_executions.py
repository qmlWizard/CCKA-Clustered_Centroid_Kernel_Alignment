from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from ccka.backend import BACKEND
from ccka.circuits.angleEmbeddingKernel import quackEmbeddingCircuit
from experiments.experiments import (
    CONFIG_DIR,
    METHOD_LABELS,
    ROOT,
    _make_aligner,
    _parse_inference_landmarks,
    _prepare_data,
)

if BACKEND == "jax":
    import jax
    import jax.numpy as jnp
    import optax as ox

    from ccka.models.kernel import KernelModel
else:
    from ccka.models.torch_kernel import TorchKernelModel


LOGGER = logging.getLogger("ccka.accuracy_vs_executions")
CCKA_INITIALIZATIONS = ("regular", "kmeans", "random")
COLORS = {
    "fullKTA": "#1f77b4",
    "randomKTA": "#ff7f0e",
    "quackKTA": "#2ca02c",
    "ccka_regular": "#d62728",
    "ccka_kmeans": "#9467bd",
    "ccka_random": "#8c564b",
}
FIELDS = (
    "method",
    "method_label",
    "initialization",
    "centroids",
    "seed",
    "inference_landmarks",
    "inference_method",
    "outer_epoch",
    "inner_step",
    "optimization_iterations",
    "optimization_updates",
    "training_circuit_executions",
    "validation_accuracy",
    "target_reached",
    "best_checkpoint_iteration",
    "restored_test_accuracy",
)


def _run_specs(
    methods: list[str] | tuple[str, ...] | None = None,
    ccka_initialization: str | None = None,
) -> tuple[tuple[str, str], ...]:
    specs = (
        ("fullKTA", "not_applicable"),
        ("randomKTA", "not_applicable"),
        ("quackKTA", "not_applicable"),
        *(
            ("ccka", initialization)
            for initialization in (
                (ccka_initialization,)
                if ccka_initialization is not None
                else CCKA_INITIALIZATIONS
            )
        ),
    )
    if methods is None:
        return specs
    unknown = set(methods) - {"fullKTA", "randomKTA", "quackKTA", "ccka"}
    if unknown:
        raise ValueError(f"Unknown comparison methods: {sorted(unknown)}")
    return tuple(spec for spec in specs if spec[0] in methods)


def _snapshot_aligner(aligner: Any) -> dict[str, Any]:
    snapshot = {}
    for name in ("weights", "main_centroids", "sub_centroids"):
        value = getattr(aligner, name, None)
        if value is None:
            continue
        snapshot[name] = (
            value.detach().clone()
            if hasattr(value, "detach")
            else value
        )
    return snapshot


def _restore_aligner(aligner: Any, snapshot: dict[str, Any]) -> None:
    if BACKEND == "torch":
        import torch

        with torch.no_grad():
            for name, value in snapshot.items():
                current = getattr(aligner, name)
                current.copy_(value)
    else:
        for name, value in snapshot.items():
            setattr(aligner, name, value)


def _update_best_checkpoint(
    aligner: Any,
    row: dict[str, Any],
    best_checkpoint: dict[str, Any],
) -> None:
    if row["validation_accuracy"] > best_checkpoint["accuracy"]:
        best_checkpoint["accuracy"] = row["validation_accuracy"]
        best_checkpoint["iteration"] = row["optimization_iterations"]
        best_checkpoint["state"] = _snapshot_aligner(aligner)


def _method_key(method: str, initialization: str) -> str:
    if method == "ccka":
        return f"ccka_{initialization}"
    return method


def _method_label(method: str, initialization: str) -> str:
    label = METHOD_LABELS[method]
    return f"{label} ({initialization})" if method == "ccka" else label


def _build_aligner(
    method: str,
    data: tuple[np.ndarray, ...],
    settings: dict[str, Any],
    seed: int,
    centroids: int,
    initialization: str = "regular",
    noise_type: str = "none",
    noise_level: float = 0.0,
    shots: int = 0,
) -> Any:
    noisy = noise_type != "none"
    circuit = quackEmbeddingCircuit(
        num_qubits=data[0].shape[1],
        reps=int(settings.get("reps", 2)),
        reupload=True,
        noisy=noisy,
        noise_type=noise_type,
        noise_level=noise_level,
        seed=seed,
    )
    if BACKEND == "jax":
        model = KernelModel(
            circuit=circuit,
            device_name="default.mixed" if noisy else "default.qubit",
            interface="jax",
            diff_method="parameter-shift" if noisy else "backprop",
            noisy=noisy,
            noise_type=noisy,
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
    return _make_aligner(
        method,
        model,
        data,
        settings,
        seed,
        centroids=centroids,
        initialization=initialization,
    )


def _jax_step(
    method: str,
    aligner: Any,
    epoch: int,
    target_accuracy: float,
    rows: list[dict[str, Any]],
    seed: int,
    max_iterations: int,
    initialization: str,
    stop_on_target: bool,
    best_checkpoint: dict[str, Any],
) -> tuple[int, bool]:
    updates = 0

    def record(
        inner_step: int,
        metrics: dict[str, Any],
        updates_this_step: int,
        iterations_this_step: int,
        executions_this_step: int,
    ) -> bool:
        reached = metrics["validation_accuracy"] >= target_accuracy
        row = {
                "method": method,
                "method_label": _method_label(method, initialization),
                "initialization": initialization,
                "centroids": (
                    aligner.n_centroids if method == "ccka" else None
                ),
                "seed": seed,
                "inference_landmarks": aligner.inference_landmarks,
                "inference_method": aligner.inference_method,
                "outer_epoch": epoch + 1,
                "inner_step": inner_step,
                "optimization_updates": (
                    rows[-1]["optimization_updates"] + updates_this_step
                ),
                "optimization_iterations": (
                    rows[-1]["optimization_iterations"] + iterations_this_step
                ),
                "training_circuit_executions": (
                    rows[-1]["training_circuit_executions"]
                    + executions_this_step
                ),
                "validation_accuracy": float(metrics["validation_accuracy"]),
                "target_reached": reached,
                "best_checkpoint_iteration": None,
                "restored_test_accuracy": None,
            }
        rows.append(row)
        _update_best_checkpoint(aligner, row, best_checkpoint)
        return reached and stop_on_target

    if method in {"fullKTA", "randomKTA"}:
        X_batch, y_batch = aligner._get_batch(epoch)
        executions_before = aligner.kernel_model.circuit_executions
        gradients = jax.grad(aligner._loss_kta)(
            aligner.weights, X_batch, y_batch
        )
        executions_this_step = (
            aligner.kernel_model.circuit_executions - executions_before
        )
        weight_updates, aligner.opt_state = aligner._optimizer.update(
            gradients, aligner.opt_state
        )
        aligner.weights = ox.apply_updates(aligner.weights, weight_updates)
        metrics = aligner.svm_training(aligner.xtrain, aligner.ytrain)
        return 1, record(1, metrics, 1, 1, executions_this_step)

    if method == "ccka":
        class_labels = np.unique(np.asarray(aligner.ytrain))
        class_label = float(class_labels[epoch % len(class_labels)])
        main_index = int(
            jnp.argmax(aligner.main_centroid_labels == class_label)
        )
        enabled_update_count = int(aligner.update_kao) + int(aligner.update_co)
        steps_this_epoch = min(
            aligner.inner_steps,
            max_iterations - rows[-1]["optimization_iterations"],
        )
        for inner_step in range(steps_this_epoch):
            executions_before = aligner.kernel_model.circuit_executions
            if aligner.update_kao:
                main_centroid = aligner.main_centroids[main_index]
                aligner.weights, aligner.sub_centroids = (
                    aligner._kao_joint_update(
                        main_centroid,
                        aligner.sub_centroid_labels,
                        l=class_label,
                    )
                )
            if aligner.update_co:
                aligner.main_centroids = aligner._co_main_update(
                    cl=class_label,
                    y_raw=aligner.sub_centroid_labels,
                    l=-class_label,
                )
            executions_this_step = (
                aligner.kernel_model.circuit_executions - executions_before
            )
            updates += enabled_update_count
            metrics = aligner.svm_training(aligner.xtrain, aligner.ytrain)
            if record(
                inner_step + 1,
                metrics,
                enabled_update_count,
                1,
                executions_this_step,
            ):
                return updates, True
        return updates, False

    class_labels = np.unique(np.asarray(aligner.ytrain))
    class_label = float(class_labels[epoch % len(class_labels)])
    main_index = int(jnp.argmax(aligner.main_centroid_labels == class_label))
    main_centroid = aligner.main_centroids[main_index]
    binary_labels = jnp.where(aligner.ytrain == class_label, 1.0, -1.0)
    steps_this_epoch = min(
        aligner.inner_steps,
        max_iterations - rows[-1]["optimization_iterations"],
    )
    if epoch % 2 == 0:
        for inner_step in range(steps_this_epoch):
            executions_before = aligner.kernel_model.circuit_executions
            aligner.weights = aligner._kao_weight_update_quack(
                main_centroid,
                aligner.xtrain,
                binary_labels,
                l=class_label,
            )
            executions_this_step = (
                aligner.kernel_model.circuit_executions - executions_before
            )
            updates += 1
            metrics = aligner.svm_training(aligner.xtrain, aligner.ytrain)
            if record(
                inner_step + 1,
                metrics,
                1,
                1,
                executions_this_step,
            ):
                return updates, True
    else:
        for inner_step in range(steps_this_epoch):
            executions_before = aligner.kernel_model.circuit_executions
            aligner.main_centroids = (
                aligner._main_centroid_gradient_update_quack(
                    main_index,
                    aligner.xtrain,
                    binary_labels,
                    l=-class_label,
                )
            )
            executions_this_step = (
                aligner.kernel_model.circuit_executions - executions_before
            )
            updates += 1
            metrics = aligner.svm_training(aligner.xtrain, aligner.ytrain)
            if record(
                inner_step + 1,
                metrics,
                1,
                1,
                executions_this_step,
            ):
                return updates, True
    return updates, False


def _torch_step(
    method: str,
    aligner: Any,
    epoch: int,
    target_accuracy: float,
    rows: list[dict[str, Any]],
    seed: int,
    max_iterations: int,
    initialization: str,
    stop_on_target: bool,
    best_checkpoint: dict[str, Any],
) -> tuple[int, bool]:
    executions_before = aligner.kernel_model.circuit_executions
    if method in {"quackKTA", "ccka"}:
        updates = min(
            aligner.inner_steps,
            max_iterations - rows[-1]["optimization_iterations"],
        )
        aligner._step(epoch, updates)
        iterations = updates
    else:
        updates = 1
        iterations = 1
        aligner._step(epoch)
    executions_this_step = (
        aligner.kernel_model.circuit_executions - executions_before
    )
    metrics = aligner._svm_metrics()
    if method == "quackKTA":
        inner_step = updates
    elif method == "ccka":
        enabled_updates = int(aligner.update_kao) + int(aligner.update_co)
        updates *= enabled_updates
        inner_step = iterations
    else:
        inner_step = 1
    reached = metrics["validation_accuracy"] >= target_accuracy
    row = {
            "method": method,
            "method_label": _method_label(method, initialization),
            "initialization": initialization,
            "centroids": (
                aligner.main_centroids.shape[0] if method == "ccka" else None
            ),
            "seed": seed,
            "inference_landmarks": aligner.inference_landmarks,
            "inference_method": aligner.inference_method,
            "outer_epoch": epoch + 1,
            "inner_step": inner_step,
            "optimization_iterations": (
                rows[-1]["optimization_iterations"] + iterations
            ),
            "optimization_updates": rows[-1]["optimization_updates"] + updates,
            "training_circuit_executions": (
                rows[-1]["training_circuit_executions"]
                + executions_this_step
            ),
            "validation_accuracy": float(metrics["validation_accuracy"]),
            "target_reached": reached,
        }
    row["best_checkpoint_iteration"] = None
    row["restored_test_accuracy"] = None
    rows.append(row)
    _update_best_checkpoint(aligner, row, best_checkpoint)
    return updates, reached and stop_on_target


def run_method(
    method: str,
    data: tuple[np.ndarray, ...],
    settings: dict[str, Any],
    seed: int,
    centroids: int,
    target_accuracy: float,
    max_iterations: int,
    initialization: str = "regular",
    stop_on_target: bool = True,
    noise_type: str = "none",
    noise_level: float = 0.0,
    shots: int = 0,
) -> list[dict[str, Any]]:
    aligner = _build_aligner(
        method,
        data,
        settings,
        seed,
        centroids,
        initialization,
        noise_type,
        noise_level,
        shots,
    )
    inner_steps = (
        aligner.inner_steps if method in {"quackKTA", "ccka"} else 1
    )
    outer_epochs = (max_iterations + inner_steps - 1) // inner_steps
    if BACKEND == "jax":
        initial = aligner.svm_training(aligner.xtrain, aligner.ytrain)
    else:
        initial = aligner._svm_metrics()
    initial_reached = initial["validation_accuracy"] >= target_accuracy
    reached = stop_on_target and initial_reached
    best_checkpoint = {
        "accuracy": float(initial["validation_accuracy"]),
        "iteration": 0,
        "state": _snapshot_aligner(aligner),
    }
    rows = [
        {
            "method": method,
            "method_label": _method_label(method, initialization),
            "initialization": initialization,
            "centroids": centroids if method == "ccka" else None,
            "seed": seed,
            "inference_landmarks": aligner.inference_landmarks,
            "inference_method": aligner.inference_method,
            "outer_epoch": 0,
            "inner_step": 0,
            "optimization_iterations": 0,
            "optimization_updates": 0,
            "training_circuit_executions": 0,
            "validation_accuracy": float(initial["validation_accuracy"]),
            "target_reached": initial_reached,
            "best_checkpoint_iteration": 0,
            "restored_test_accuracy": None,
        }
    ]
    for epoch in range(outer_epochs):
        if reached:
            break
        if BACKEND == "jax":
            _, reached = _jax_step(
                method,
                aligner,
                epoch,
                target_accuracy,
                rows,
                seed,
                max_iterations,
                initialization,
                stop_on_target,
                best_checkpoint,
            )
        else:
            _, reached = _torch_step(
                method,
                aligner,
                epoch,
                target_accuracy,
                rows,
                seed,
                max_iterations,
                initialization,
                stop_on_target,
                best_checkpoint,
            )
    _restore_aligner(aligner, best_checkpoint["state"])
    if BACKEND == "jax":
        final = aligner.svm_training(
            aligner.xtrain, aligner.ytrain, evaluate_test=True
        )
    else:
        final = aligner._svm_metrics(evaluate_test=True)
    rows[-1]["best_checkpoint_iteration"] = best_checkpoint["iteration"]
    rows[-1]["restored_test_accuracy"] = final["test_accuracy"]
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]) if rows else FIELDS
        )
        writer.writeheader()
        writer.writerows(rows)


def _summarize(
    rows: list[dict[str, Any]],
    target_accuracy: float,
    run_specs: tuple[tuple[str, str], ...] | None = None,
    centroid_values: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    summaries = []
    for method, initialization in run_specs or _run_specs():
        values = (centroid_values or (None,)) if method == "ccka" else (None,)
        for centroids in values:
            method_rows = [
                row
                for row in rows
                if row["method"] == method
                and row["initialization"] == initialization
                and row["centroids"] == (centroids if method == "ccka" else None)
            ]
            if not method_rows:
                continue
            reached = next(
                (row for row in method_rows if row["target_reached"]), None
            )
            best = max(method_rows, key=lambda row: row["validation_accuracy"])
            final = method_rows[-1]
            summaries.append(
                {
                    "method": method,
                    "method_label": (
                        f"{_method_label(method, initialization)} (K={centroids})"
                        if method == "ccka" and centroids is not None
                        else _method_label(method, initialization)
                    ),
                    "initialization": initialization,
                    "centroids": centroids if method == "ccka" else None,
                    "target_accuracy": target_accuracy,
                    "inference_landmarks": method_rows[0]["inference_landmarks"],
                    "inference_method": method_rows[0]["inference_method"],
                    "target_reached": reached is not None,
                    "outer_epochs": final["outer_epoch"],
                    "optimization_updates": final["optimization_updates"],
                    "optimization_iterations": final["optimization_iterations"],
                    "training_circuit_executions": final[
                        "training_circuit_executions"
                    ],
                    "best_validation_accuracy": best["validation_accuracy"],
                    "best_checkpoint_iteration": final[
                        "best_checkpoint_iteration"
                    ],
                    "restored_test_accuracy": final["restored_test_accuracy"],
                }
            )
    return summaries


def _plot(
    path: Path,
    rows: list[dict[str, Any]],
    target_accuracy: float,
    dataset: str,
    run_specs: tuple[tuple[str, str], ...] | None = None,
    centroid_values: tuple[int, ...] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, initialization in run_specs or _run_specs():
        values = (centroid_values or (None,)) if method == "ccka" else (None,)
        for centroids in values:
            method_rows = [
                row
                for row in rows
                if row["method"] == method
                and row["initialization"] == initialization
                and row["centroids"] == (centroids if method == "ccka" else None)
            ]
            if not method_rows:
                continue
            ax.plot(
                [row["training_circuit_executions"] for row in method_rows],
                [row["validation_accuracy"] * 100 for row in method_rows],
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=(
                    f"{_method_label(method, initialization)} K={centroids}"
                    if method == "ccka"
                    else _method_label(method, initialization)
                ),
            )
    ax.axhline(target_accuracy * 100, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Cumulative training circuit-pair evaluations")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"{dataset.title()} accuracy vs. circuit executions")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare dataset accuracy against cumulative circuit "
            "executions until a target validation accuracy is reached."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("checkerboard", "corners"),
        default="checkerboard",
        help="Synthetic dataset to compare.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "default.yaml",
        help="YAML file supplying training hyperparameters and seeds.",
    )
    parser.add_argument("--target-accuracy", type=float, default=0.9)
    parser.add_argument(
        "--max-epochs",
        "--max-iterations",
        dest="max_iterations",
        type=int,
        default=None,
        help=(
            "Maximum inner iterations per method (legacy --max-epochs alias); "
            "defaults to training.epochs in the config."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--inference-landmarks",
        default=None,
        help=(
            "CCKA Nyström landmark count or 'all'; defaults to "
            "training.inference_landmarks in the config."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("fullKTA", "randomKTA", "quackKTA", "ccka"),
        default=None,
        help="Methods to compare; defaults to all supported methods.",
    )
    parser.add_argument(
        "--inference-method",
        choices=("svm", "kernel_distance"),
        default=None,
        help="Override the inference method for all methods in this run.",
    )
    parser.add_argument(
        "--full-kta-inference-method",
        choices=("svm", "kernel_distance"),
        default=None,
        help="Override the configured Full KTA inference method.",
    )
    parser.add_argument(
        "--random-kta-inference-method",
        choices=("svm", "kernel_distance"),
        default=None,
        help="Override the configured Random KTA inference method.",
    )
    parser.add_argument(
        "--quack-inference-method",
        choices=("svm", "kernel_distance"),
        default=None,
        help="Override the configured QUACK inference method.",
    )
    parser.add_argument(
        "--ccka-inference-method",
        choices=("svm", "kernel_distance"),
        default=None,
        help="Override the configured CCKA inference method.",
    )
    parser.add_argument(
        "--ccka-learning-rate",
        type=float,
        default=None,
        help="Override the CCKA kernel-weight learning rate for this run.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Set the optimizer learning rate equally for every method and centroid optimizer.",
    )
    parser.add_argument(
        "--run-full-iterations",
        action="store_true",
        help="Continue training to the iteration cap even after reaching target accuracy.",
    )
    parser.add_argument(
        "--use-test-as-validation",
        action="store_true",
        help="Use the held-out test partition for validation and final scoring.",
    )
    parser.add_argument(
        "--centroids",
        nargs="+",
        type=int,
        default=None,
        help="One or more centroid counts to compare.",
    )
    parser.add_argument(
        "--ccka-initialization",
        choices=CCKA_INITIALIZATIONS,
        default=None,
        help="Run only the selected CCKA initialization.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path prefix; defaults to a dataset-specific results path.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not 0.0 < args.target_accuracy <= 1.0:
        raise ValueError("--target-accuracy must be in (0, 1].")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = dict(config["training"])
    inference_landmarks = _parse_inference_landmarks(
        args.inference_landmarks
        if args.inference_landmarks is not None
        else settings.get("inference_landmarks", 32)
    )
    settings["inference_landmarks"] = inference_landmarks
    if args.inference_method is not None:
        settings["inference_method"] = args.inference_method
        for method in ("fullKTA", "randomKTA", "quackKTA", "ccka"):
            settings[f"{method}_inference_method"] = args.inference_method
    for method, inference_method in (
        ("fullKTA", args.full_kta_inference_method),
        ("randomKTA", args.random_kta_inference_method),
        ("quackKTA", args.quack_inference_method),
        ("ccka", args.ccka_inference_method),
    ):
        if inference_method is not None:
            settings[f"{method}_inference_method"] = inference_method
    if args.ccka_learning_rate is not None:
        if args.ccka_learning_rate <= 0:
            raise ValueError("--ccka-learning-rate must be positive.")
        settings["ccka_learning_rate"] = args.ccka_learning_rate
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError("--learning-rate must be positive.")
        settings["learning_rate"] = args.learning_rate
        settings["centroid_lr"] = args.learning_rate
        settings["sub_centroid_lr"] = args.learning_rate
        for method in ("fullKTA", "randomKTA", "quackKTA", "ccka"):
            settings[f"{method}_learning_rate"] = args.learning_rate
    max_iterations = (
        args.max_iterations
        if args.max_iterations is not None
        else int(settings.get("epochs", 100))
    )
    if max_iterations <= 0:
        raise ValueError("--max-iterations must be positive.")
    settings["epochs"] = max_iterations
    settings["collect_diagnostics"] = False
    settings["test_as_validation"] = args.use_test_as_validation
    seeds = config.get("seeds", [0])
    seed = args.seed if args.seed is not None else int(seeds[0] if seeds else 0)
    centroid_values = settings.get("centroids", [4])
    centroids_to_run = tuple(
        args.centroids
        if args.centroids is not None
        else [int(centroid_values[0])]
    )
    if any(value <= 0 for value in centroids_to_run):
        raise ValueError("--centroids must be positive.")

    dataset = args.dataset
    data = _prepare_data(dataset, seed, settings)
    rows: list[dict[str, Any]] = []
    run_specs = _run_specs(args.methods, args.ccka_initialization)
    for method, initialization in run_specs:
        method_centroids = (
            centroids_to_run if method == "ccka" else (centroids_to_run[0],)
        )
        for centroids in method_centroids:
            LOGGER.info(
            "Running %s on %s (seed=%d; K=%d; target=%.1f%%; "
            "M=%s; max iterations=%d)",
            _method_label(method, initialization),
            dataset,
            seed,
            centroids,
            args.target_accuracy * 100,
            inference_landmarks,
            max_iterations,
        )
            method_rows = run_method(
                method,
                data,
                settings,
                seed,
                centroids,
                args.target_accuracy,
                max_iterations,
                initialization,
                stop_on_target=not args.run_full_iterations,
            )
            rows.extend(method_rows)

    summaries = _summarize(
        rows, args.target_accuracy, run_specs, centroids_to_run
    )
    for summary in summaries:
        if summary["target_reached"]:
            LOGGER.info(
                "%s reached %.1f%% after %d training pair evaluations.",
                summary["method_label"],
                args.target_accuracy * 100,
                summary["training_circuit_executions"],
            )
        else:
            LOGGER.warning(
                "%s did not reach %.1f%% in %d optimizer updates; best "
                "validation accuracy was %.1f%% after %d training pair "
                "evaluations.",
                summary["method_label"],
                args.target_accuracy * 100,
                max_iterations,
                summary["best_validation_accuracy"] * 100,
                summary["training_circuit_executions"],
            )

    prefix = args.output_prefix or (
        ROOT / "results" / f"{dataset}_accuracy_vs_executions"
    )
    _write_csv(prefix.with_suffix(".csv"), rows)
    _write_csv(prefix.with_name(prefix.name + "_summary").with_suffix(".csv"), summaries)
    _plot(
        prefix.with_suffix(".png"),
        rows,
        args.target_accuracy,
        dataset,
        run_specs,
        centroids_to_run,
    )
    print(
        "method,initialization,centroids,inference_landmarks,target_reached,"
        "optimization_iterations,optimization_updates,training_circuit_executions,"
        "best_validation_accuracy,best_checkpoint_iteration,restored_test_accuracy"
    )
    for summary in summaries:
        print(
            f"{summary['method']}:{summary['initialization']},"
            f"{summary['centroids']},"
            f"{summary['inference_landmarks']},"
            f"{summary['target_reached']},"
            f"{summary['optimization_iterations']},"
            f"{summary['optimization_updates']},"
            f"{summary['training_circuit_executions']},"
            f"{summary['best_validation_accuracy']:.4f},"
            f"{summary['best_checkpoint_iteration']},"
            f"{summary['restored_test_accuracy']:.4f}"
        )
    LOGGER.info("Wrote curve data, summary, and plot using prefix %s", prefix)


if __name__ == "__main__":
    main()
