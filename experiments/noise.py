from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from experiments.accuracy_vs_executions import ROOT, run_method
from experiments.experiments import CONFIG_DIR, _prepare_data

LOGGER = logging.getLogger("ccka.noise")
DEFAULT_NOISE_LEVELS = (0.01, 0.03, 0.05, 0.1)
NOISE_TYPES = ("coherent", "depolarizing")
METHODS = (("quackKTA", "not_applicable"), ("ccka", "kmeans"))


def _validate_noise_levels(levels: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    validated = tuple(float(level) for level in levels)
    if len(validated) != 4:
        raise ValueError("Exactly four noise levels are required.")
    if any(not 0.0 <= level <= 1.0 for level in validated):
        raise ValueError("Noise levels must be within [0, 1].")
    if len(set(validated)) != len(validated):
        raise ValueError("Noise levels must be unique.")
    return validated


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    styles = {
        ("coherent", "quackKTA"): ("#1f77b4", "o", "-"),
        ("coherent", "ccka"): ("#ff7f0e", "o", "-"),
        ("depolarizing", "quackKTA"): ("#1f77b4", "s", "--"),
        ("depolarizing", "ccka"): ("#ff7f0e", "s", "--"),
    }
    labels = {
        "quackKTA": "QUACK",
        "ccka": "CCKA (K-means)",
    }
    noise_types = tuple(
        noise_type
        for noise_type in NOISE_TYPES
        if any(row["noise_type"] == noise_type for row in rows)
    )
    for noise_type in noise_types:
        for method, _ in METHODS:
            values = sorted(
                (
                    row for row in rows
                    if row["noise_type"] == noise_type
                    and row["method"] == method
                ),
                key=lambda row: row["noise_level"],
            )
            color, marker, line_style = styles[(noise_type, method)]
            ax.plot(
                [row["noise_level"] for row in values],
                [row["best_validation_accuracy"] * 100 for row in values],
                color=color,
                marker=marker,
                linestyle=line_style,
                linewidth=1.8,
                label=f"{labels[method]} - {noise_type}",
            )
    ax.set_xlabel("Noise level")
    ax.set_ylabel("Best validation accuracy (%)")
    noise_label = (
        "coherent and depolarizing"
        if len(noise_types) == len(NOISE_TYPES)
        else noise_types[0]
    )
    ax.set_title(f"Checkerboard accuracy under {noise_label} noise")
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
            "Compare CCKA and QUACK accuracy across selected noise levels "
            "on checkerboard."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "default.yaml",
        help="YAML file supplying the training settings.",
    )
    parser.add_argument(
        "--noise-type",
        choices=("both", *NOISE_TYPES),
        default="both",
        help="Noise type to run (default: both).",
    )
    parser.add_argument(
        "--noise-levels",
        nargs=5,
        type=float,
        default=DEFAULT_NOISE_LEVELS,
        metavar="P",
        help="Five noise levels in [0, 1].",
    )
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--target-accuracy", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--centroids", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1.5)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix for the single chart and its CSV data.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    noise_levels = _validate_noise_levels(args.noise_levels)
    if args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive.")
    if not 0.0 < args.target_accuracy <= 1.0:
        raise ValueError("--target-accuracy must be in (0, 1].")
    if args.centroids <= 0:
        raise ValueError("--centroids must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = dict(config["training"])
    settings.update(
        {
            "epochs": args.max_iterations,
            "learning_rate": args.learning_rate,
            "centroid_lr": args.learning_rate,
            "sub_centroid_lr": args.learning_rate,
            "test_as_validation": True,
            "inference_method": "svm",
            "collect_diagnostics": False,
        }
    )
    for method, _ in METHODS:
        settings[f"{method}_learning_rate"] = args.learning_rate
        settings[f"{method}_inference_method"] = "svm"
    shots = args.shots
    if shots is None:
        shot_values = config.get("noise", {}).get("shots", [256])
        shots = int(shot_values[0])
    if shots <= 0:
        raise ValueError("--shots must be positive.")

    data = _prepare_data("checkerboard", args.seed, settings)
    rows = []
    noise_types = (
        NOISE_TYPES if args.noise_type == "both" else (args.noise_type,)
    )
    for noise_type in noise_types:
        for noise_level in noise_levels:
            for method, initialization in METHODS:
                LOGGER.info(
                    "Running %s with %s noise %.3f (shots=%d, K=%d)",
                    method,
                    noise_type,
                    noise_level,
                    shots,
                    args.centroids,
                )
                curve = run_method(
                    method,
                    data,
                    settings,
                    args.seed,
                    args.centroids,
                    args.target_accuracy,
                    args.max_iterations,
                    initialization,
                    noise_type=noise_type,
                    noise_level=noise_level,
                    shots=shots,
                )
                best = max(curve, key=lambda row: row["validation_accuracy"])
                final = curve[-1]
                rows.append(
                    {
                        "dataset": "checkerboard",
                        "method": method,
                        "initialization": initialization,
                        "noise_type": noise_type,
                        "noise_level": noise_level,
                        "shots": shots,
                        "seed": args.seed,
                        "centroids": args.centroids if method == "ccka" else None,
                        "learning_rate": args.learning_rate,
                        "inference_method": "svm",
                        "optimization_iterations": final[
                            "optimization_iterations"
                        ],
                        "training_pair_evaluations": final[
                            "training_circuit_executions"
                        ],
                        "target_reached": any(
                            row["target_reached"] for row in curve
                        ),
                        "best_checkpoint_iteration": best[
                            "optimization_iterations"
                        ],
                        "best_validation_accuracy": best[
                            "validation_accuracy"
                        ],
                        "restored_test_accuracy": final[
                            "restored_test_accuracy"
                        ],
                    }
                )

    prefix = args.output_prefix or (
        ROOT / "results" / "checkerboard_noise_ccka_quack"
    )
    _write_csv(prefix.with_suffix(".csv"), rows)
    _plot(prefix.with_suffix(".png"), rows)
    print(
        "noise_type,noise_level,method,best_validation_accuracy,"
        "best_checkpoint_iteration,training_pair_evaluations"
    )
    for row in rows:
        print(
            f"{row['noise_type']},{row['noise_level']:.3f},{row['method']},"
            f"{row['best_validation_accuracy']:.4f},"
            f"{row['best_checkpoint_iteration']},"
            f"{row['training_pair_evaluations']}"
        )
    LOGGER.info("Wrote noise sweep chart and data using prefix %s", prefix)


if __name__ == "__main__":
    main()
