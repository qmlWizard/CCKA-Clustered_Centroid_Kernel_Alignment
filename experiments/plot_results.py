from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "dataset",
    "method",
    "method_label",
    "ablation",
    "initialization",
    "centroids_per_class",
    "noise_type",
    "noise_level",
    "shots",
    "seed",
    "test_accuracy",
    "circuit_executions",
    "wall_time_seconds",
}
IDENTITY_COLUMNS = [
    "dataset",
    "method",
    "ablation",
    "initialization",
    "centroids_per_class",
    "noise_type",
    "noise_level",
    "shots",
    "seed",
]


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")


def _csv_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Results path does not exist: {source}")
    summaries = sorted(source.rglob("summary.csv"))
    if summaries:
        return summaries
    return sorted(
        path
        for path in source.rglob("*.csv")
        if path.name not in {"summary_table.csv"}
    )


def load_results(source: Path) -> pd.DataFrame:
    paths = _csv_paths(source)
    if not paths:
        raise FileNotFoundError(f"No experiment CSV files found under {source}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame["_source"] = str(path)
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    results["centroids_per_class"] = pd.to_numeric(
        results["centroids_per_class"], errors="coerce"
    )
    for column in (
        "noise_level",
        "seed",
        "shots",
        "test_accuracy",
        "circuit_executions",
        "wall_time_seconds",
    ):
        results[column] = pd.to_numeric(results[column], errors="coerce")
    results["noise_type"] = results["noise_type"].fillna("none").str.lower()
    results["ablation"] = results["ablation"].fillna("none")
    results["initialization"] = results["initialization"].fillna(
        "not_applicable"
    )
    results = results.drop_duplicates(
        subset=IDENTITY_COLUMNS + ["test_accuracy", "circuit_executions"]
    )
    return results


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _primary_rows(results: pd.DataFrame) -> pd.DataFrame:
    return results[
        (results["method"] == "ccka")
        & (results["initialization"] == "regular")
    ]


def _noiseless(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows["noise_type"] == "none"]


def _full_ccka(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows["ablation"] == "full"]


def _plot_complexity(results: pd.DataFrame, output: Path) -> int:
    rows = results[
        (results["noise_type"] == "none")
        & (results["ablation"].isin(["none", "full"]))
    ]
    datasets = sorted(rows["dataset"].dropna().unique())
    methods = sorted(rows["method"].dropna().unique())
    if not datasets or not methods:
        return 0

    outputs = 0
    for metric, filename, title, ylabel in (
        ("wall_time_seconds", "complexity_time.png", "Runtime", "Seconds"),
        (
            "circuit_executions",
            "complexity_circuits.png",
            "Circuit evaluations",
            "Forward-pair calls",
        ),
    ):
        columns = 4
        row_count = math.ceil(len(datasets) / columns)
        fig, axes = plt.subplots(
            row_count,
            columns,
            figsize=(4.5 * columns, 3.5 * row_count),
            squeeze=False,
        )
        for index, dataset in enumerate(datasets):
            axis = axes.flat[index]
            subset = rows[rows["dataset"] == dataset]
            means = subset.groupby("method")[metric].mean().reindex(methods)
            axis.bar(range(len(methods)), means.values, color="steelblue")
            axis.set_title(dataset.replace("_", " "))
            axis.set_xticks(range(len(methods)))
            axis.set_xticklabels(methods, rotation=35, ha="right")
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.25)
        for axis in axes.flat[len(datasets) :]:
            axis.set_visible(False)
        fig.suptitle(f"{title} by method and dataset (noiseless)", y=1.01)
        fig.tight_layout()
        _save(fig, output / filename)
        outputs += 1
    return outputs


def _parse_history(value: object, source: str) -> list[float]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{source} has no per-epoch validation history. "
            "Run experiments with the updated runner before plotting epoch curves."
        )
    try:
        values = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid validation history in {source}") from error
    if not isinstance(values, list) or not values:
        raise ValueError(f"Empty validation history in {source}")
    return [float(value) for value in values]


def _plot_epoch_accuracy(results: pd.DataFrame, output: Path) -> int:
    rows = _noiseless(results)
    rows = rows[
        ((rows["method"] == "ccka") & (rows["ablation"] == "full")
         & (rows["initialization"] == "regular"))
        | (rows["method"].isin(["fullKTA", "randomKTA", "quackKTA"])
           & (rows["ablation"] == "none"))
    ]
    if "validation_accuracy_history" not in rows.columns:
        raise ValueError(
            "CSV results do not include validation_accuracy_history; "
            "rerun experiments with the updated runner."
        )
    rows = rows.copy()
    rows["history"] = [
        _parse_history(value, source)
        for value, source in zip(
            rows["validation_accuracy_history"], rows["_source"]
        )
    ]
    plots = 0
    styles = {
        "fullKTA": ("#4c9bd6", "--"),
        "randomKTA": ("#83ad45", "--"),
        "quackKTA": ("#29947f", "--"),
        "ccka": ("#e95d36", "-"),
    }
    method_names = {
        "fullKTA": "FullKTA",
        "randomKTA": "RandomKTA",
        "quackKTA": "QuackKTA",
        "ccka": "CCKA (ours)",
    }
    for dataset in sorted(rows["dataset"].dropna().unique()):
        dataset_rows = rows[rows["dataset"] == dataset]
        fig, axis = plt.subplots(figsize=(11, 7))
        for (method, centroid_count), curve_rows in dataset_rows.groupby(
            ["method", "centroids_per_class"], dropna=False, sort=True
        ):
            epoch_values: dict[int, list[float]] = {}
            for history in curve_rows["history"]:
                for epoch, value in enumerate(history, start=1):
                    epoch_values.setdefault(epoch, []).append(value)
            if not epoch_values:
                continue
            epochs = np.asarray(sorted(epoch_values))
            means = np.asarray(
                [np.mean(epoch_values[epoch]) for epoch in epochs]
            )
            minima = np.asarray(
                [np.min(epoch_values[epoch]) for epoch in epochs]
            )
            maxima = np.asarray(
                [np.max(epoch_values[epoch]) for epoch in epochs]
            )
            method_label = method_names.get(
                method, str(curve_rows["method_label"].iloc[0])
            )
            label = (
                method_label
                if method != "ccka" or pd.isna(centroid_count)
                else f"{method_label}, K={int(centroid_count)}"
            )
            color, line_style = styles[method]
            line = axis.plot(
                epochs,
                means,
                color=color,
                linestyle=line_style,
                linewidth=1.8,
                label=label,
            )[0]
            axis.fill_between(
                epochs, minima, maxima, color=line.get_color(), alpha=0.12
            )
        axis.set(
            title=dataset.replace("_", " "),
            xlabel="Epoch",
            ylabel="Validation Accuracy",
            ylim=(0, 1),
        )
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        _save(fig, output / f"accuracy-epochs-{_slug(dataset)}.png")
        plots += 1
    return plots


def _mean_range(rows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        rows.groupby(group_columns)["test_accuracy"]
        .agg(mean="mean", minimum="min", maximum="max")
        .reset_index()
    )


def _plot_centroids(results: pd.DataFrame, output: Path) -> int:
    rows = _full_ccka(
        _noiseless(_primary_rows(results))
    ).dropna(subset=["centroids_per_class", "test_accuracy"])
    if rows.empty:
        return 0
    datasets = sorted(rows["dataset"].unique())
    for dataset in datasets:
        subset = rows[rows["dataset"] == dataset]
        summary = _mean_range(subset, ["centroids_per_class"])
        fig, axis = plt.subplots(figsize=(7, 5))
        axis.plot(summary["centroids_per_class"], summary["mean"], marker="o")
        axis.fill_between(
            summary["centroids_per_class"],
            summary["minimum"],
            summary["maximum"],
            alpha=0.2,
        )
        axis.set(
            title=f"CCKA centroids vs accuracy: {dataset}",
            xlabel="Centroids per class (K)",
            ylabel="Test accuracy",
            ylim=(0, 1),
        )
        axis.grid(alpha=0.25)
        fig.tight_layout()
        _save(fig, output / "centroids_by_dataset" / f"{_slug(dataset)}.png")

    per_dataset = (
        rows.groupby(["dataset", "centroids_per_class"])["test_accuracy"]
        .mean()
        .reset_index()
    )
    summary = _mean_range(per_dataset, ["centroids_per_class"])
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(summary["centroids_per_class"], summary["mean"], marker="o")
    axis.fill_between(
        summary["centroids_per_class"],
        summary["minimum"],
        summary["maximum"],
        alpha=0.2,
    )
    axis.set(
        title="CCKA centroids vs accuracy (dataset average)",
        xlabel="Centroids per class (K)",
        ylabel="Mean test accuracy",
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, output / "centroids_average.png")
    return len(datasets) + 1


def _plot_noise(
    results: pd.DataFrame, output: Path, centroids: int
) -> int:
    rows = _primary_rows(results)
    rows = rows[
        (rows["ablation"] == "full")
        & (rows["centroids_per_class"] == centroids)
        & (rows["noise_type"].isin(["none", "coherent", "depolarizing", "depolarising"]))
    ].copy()
    rows["noise_type"] = rows["noise_type"].replace(
        {"depolarising": "depolarizing"}
    )
    outputs = 0
    for noise_type in ("coherent", "depolarizing"):
        subset = rows[
            (rows["noise_type"] == noise_type)
            | (rows["noise_type"] == "none")
        ].copy()
        if subset.empty or noise_type not in set(subset["noise_type"]):
            continue
        subset["noise_level"] = np.where(
            subset["noise_type"] == "none", 0.0, subset["noise_level"]
        )
        # Average shot settings within each seed before computing seed ranges.
        per_seed = (
            subset.groupby(["dataset", "seed", "noise_level"])["test_accuracy"]
            .mean()
            .reset_index()
        )
        summary = _mean_range(per_seed, ["dataset", "noise_level"])
        datasets = sorted(summary["dataset"].unique())
        fig, axis = plt.subplots(figsize=(11, 7))
        for dataset in datasets:
            data = summary[summary["dataset"] == dataset].sort_values(
                "noise_level"
            )
            line = axis.plot(
                data["noise_level"], data["mean"], marker="o", label=dataset
            )[0]
            axis.fill_between(
                data["noise_level"],
                data["minimum"],
                data["maximum"],
                color=line.get_color(),
                alpha=0.12,
            )
        axis.set(
            title=f"CCKA accuracy under {noise_type} noise (K={centroids})",
            xlabel="Noise level (0 is noiseless)",
            ylabel="Test accuracy",
            ylim=(0, 1),
        )
        axis.grid(alpha=0.25)
        axis.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        _save(fig, output / f"noise-{noise_type}.png")
        outputs += 1
    return outputs


def _plot_ablations(results: pd.DataFrame, output: Path) -> int:
    rows = _primary_rows(results)
    rows = _noiseless(rows)
    rows = rows[
        (rows["centroids_per_class"] == 1) & rows["test_accuracy"].notna()
    ]
    if rows.empty:
        return 0
    order = ["full", "main_only", "sub_only", "no_kao", "no_co"]
    rows = rows[rows["ablation"].isin(order)]
    rows = rows.drop_duplicates(
        subset=["dataset", "ablation", "seed", "test_accuracy"]
    )
    datasets = sorted(rows["dataset"].unique())
    columns = 4
    row_count = math.ceil(len(datasets) / columns)
    fig, axes = plt.subplots(
        row_count,
        columns,
        figsize=(4.5 * columns, 3.5 * row_count),
        squeeze=False,
    )
    for index, dataset in enumerate(datasets):
        axis = axes.flat[index]
        subset = rows[rows["dataset"] == dataset]
        summary = _mean_range(subset, ["ablation"]).set_index("ablation")
        names = [name for name in order if name in summary.index]
        means = summary.loc[names, "mean"].to_numpy()
        lower = means - summary.loc[names, "minimum"].to_numpy()
        upper = summary.loc[names, "maximum"].to_numpy() - means
        axis.bar(range(len(names)), means, color="darkseagreen")
        axis.errorbar(
            range(len(names)), means, yerr=np.vstack((lower, upper)),
            fmt="none", ecolor="black", capsize=3,
        )
        axis.set_title(dataset.replace("_", " "))
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(names, rotation=35, ha="right")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Test accuracy")
        axis.grid(axis="y", alpha=0.25)
    for axis in axes.flat[len(datasets) :]:
        axis.set_visible(False)
    fig.suptitle("CCKA ablation accuracy by dataset (K=1)", y=1.01)
    fig.tight_layout()
    _save(fig, output / "ablations_by_dataset.png")

    per_dataset = (
        rows.groupby(["dataset", "ablation"])["test_accuracy"]
        .mean()
        .reset_index()
    )
    summary = _mean_range(per_dataset, ["ablation"]).set_index("ablation")
    names = [name for name in order if name in summary.index]
    means = summary.loc[names, "mean"].to_numpy()
    lower = means - summary.loc[names, "minimum"].to_numpy()
    upper = summary.loc[names, "maximum"].to_numpy() - means
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(range(len(names)), means, color="darkseagreen")
    axis.errorbar(
        range(len(names)), means, yerr=np.vstack((lower, upper)),
        fmt="none", ecolor="black", capsize=4,
    )
    axis.set(
        title="CCKA ablation accuracy (average across datasets, K=1)",
        ylabel="Mean test accuracy",
        ylim=(0, 1),
    )
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=25, ha="right")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, output / "ablations_average.png")
    return 2


def make_plots(
    results_path: Path,
    output: Path,
    *,
    noise_centroids: int = 2,
) -> list[Path]:
    results = load_results(results_path)
    output.mkdir(parents=True, exist_ok=True)
    count = _plot_complexity(results, output)
    count += _plot_epoch_accuracy(results, output)
    count += _plot_centroids(results, output)
    count += _plot_noise(results, output, noise_centroids)
    count += _plot_ablations(results, output)
    return sorted(output.rglob("*.png")) if count else []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot experiment metrics from summary CSV files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results"),
        help="A run directory, summary CSV, or directory containing run folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plots"),
        help="Directory for generated PNG figures.",
    )
    parser.add_argument(
        "--noise-centroids",
        type=int,
        default=2,
        help="CCKA K used for coherent/depolarizing noise plots.",
    )
    args = parser.parse_args()
    paths = make_plots(args.input, args.output, noise_centroids=args.noise_centroids)
    if not paths:
        raise SystemExit("No plots were generated; check the CSV contents.")
    print(f"Saved {len(paths)} plots under {args.output}")


if __name__ == "__main__":
    main()
