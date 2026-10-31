"""Decision-boundary comparison for RandomKTA, QUACK, and CCKA."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.svm import SVC

from ccka_experiments.experiments import (
    _build_model,
    _make_aligner,
    _prepare_data,
    load_config,
)

METHODS = ("randomKTA", "quackKTA", "ccka")
LABELS = {
    "randomKTA": "RandomKTA",
    "quackKTA": "QUACK",
    "ccka": "CCKA",
}


def _kernel_matrix(aligner: Any, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Evaluate a rectangular kernel matrix for either supported backend."""
    weights = aligner.weights
    if hasattr(aligner, "test_kernel_matrix"):
        if len(left) == len(right):
            return np.asarray(
                aligner.regular_kernel_matrix(weights, left), dtype=float
            )
        return np.asarray(
            aligner.test_kernel_matrix(weights, right, left), dtype=float
        )

    import torch

    left_tensor = torch.as_tensor(left, dtype=torch.float32)
    right_tensor = torch.as_tensor(right, dtype=torch.float32)
    rows = np.repeat(left_tensor.numpy(), len(right), axis=0)
    cols = np.tile(right_tensor.numpy(), (len(left), 1))
    values = aligner.kernel_model.forward(
        torch.as_tensor(rows, dtype=torch.float32),
        torch.as_tensor(cols, dtype=torch.float32),
        weights,
    )
    return values.detach().cpu().numpy().reshape(len(left), len(right))


def _fit_method(
    method: str,
    data: tuple[np.ndarray, ...],
    settings: dict[str, Any],
    seed: int,
    ansatz: str,
    centroids: int,
) -> tuple[Any, float]:
    model = _build_model(
        settings,
        data[0].shape[1],
        {"type": "none", "level": 0.0},
        0,
        seed,
        ansatz,
    )
    aligner = _make_aligner(
        method,
        model,
        data,
        settings,
        seed,
        centroids=centroids,
        initialization="regular",
        update_kao=True,
        update_co=True,
    )
    history = aligner.align()
    return aligner, float(history["final_svm_metrics"]["test_accuracy"])


def _plot_boundaries(
    data: tuple[np.ndarray, ...],
    fitted: dict[str, tuple[Any, float]],
    output: Path,
    *,
    grid_size: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X_train, _, X_test, y_train, _, y_test = data
    active = np.flatnonzero(X_train.std(axis=0) > 1e-9)
    if len(active) != 2:
        raise ValueError(
            "Decision-boundary plotting requires exactly two non-constant "
            f"features; found {len(active)}."
        )

    X_all = np.concatenate((X_train, X_test))
    lo = X_all[:, active].min(axis=0)
    hi = X_all[:, active].max(axis=0)
    padding = 0.08 * np.maximum(hi - lo, 1e-6)
    x_values = np.linspace(lo[0] - padding[0], hi[0] + padding[0], grid_size)
    y_values = np.linspace(lo[1] - padding[1], hi[1] + padding[1], grid_size)
    gx, gy = np.meshgrid(x_values, y_values)
    grid = np.broadcast_to(
        X_train.mean(axis=0), (gx.size, X_train.shape[1])
    ).copy()
    grid[:, active[0]], grid[:, active[1]] = gx.ravel(), gy.ravel()

    fig, axes = plt.subplots(1, len(METHODS), figsize=(12, 3.8), sharex=True, sharey=True)
    for axis, method in zip(axes, METHODS):
        aligner, test_accuracy = fitted[method]
        train_kernel = _kernel_matrix(aligner, X_train, X_train)
        grid_kernel = _kernel_matrix(aligner, grid, X_train)
        classifier = SVC(kernel="precomputed", C=float(aligner.svm_c))
        classifier.fit(train_kernel, y_train)
        prediction = classifier.predict(grid_kernel).reshape(gx.shape)

        axis.contourf(
            gx,
            gy,
            prediction,
            levels=[-1, 0, 1],
            alpha=0.28,
            cmap="coolwarm",
        )
        for label, color in zip(np.unique(y_train), ("#2166ac", "#b2182b")):
            train_mask = y_train == label
            test_mask = y_test == label
            axis.scatter(
                X_train[train_mask, active[0]],
                X_train[train_mask, active[1]],
                c=color,
                marker="o",
                s=28,
                edgecolors="black",
                linewidths=0.35,
                label=f"Train {label}",
            )
            axis.scatter(
                X_test[test_mask, active[0]],
                X_test[test_mask, active[1]],
                c=color,
                marker="^",
                s=34,
                edgecolors="black",
                linewidths=0.35,
                label=f"Test {label}",
            )
        axis.set_title(f"{LABELS[method]}\ntest accuracy = {test_accuracy:.2f}")
        axis.set_xlabel(f"Feature {active[0] + 1}")
        axis.grid(alpha=0.15)
    axes[0].set_ylabel(f"Feature {active[1] + 1}")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_from_config(
    config: dict[str, Any],
    *,
    output_dir: Path,
) -> list[Path]:
    datasets = config.get("datasets", ["corners"])
    if datasets != ["corners"]:
        raise ValueError("The decision-boundary experiment is defined for corners only.")
    methods = tuple(config.get("experiments", {}).get("methods", METHODS))
    if methods != METHODS:
        raise ValueError(f"Expected methods in this order: {METHODS}.")

    settings = dict(config["training"])
    centroids = int(settings.get("centroids", [4])[0])
    ansatze = config.get("ansatze", ["crz_ring"])
    grid_size = int(config.get("decision_boundaries", {}).get("grid_size", 180))
    outputs = []
    for seed in (int(value) for value in config.get("seeds", [0])):
        data = _prepare_data("corners", seed, settings)
        for ansatz in ansatze:
            fitted = {
                method: _fit_method(
                    method, data, settings, seed, ansatz, centroids
                )
                for method in METHODS
            }
            path = output_dir / f"decision_boundaries_corners_{ansatz}_seed{seed}.png"
            _plot_boundaries(data, fitted, path, grid_size=grid_size)
            outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RandomKTA, QUACK, and CCKA decision boundaries on corners."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = run_from_config(load_config(args.config), output_dir=args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
