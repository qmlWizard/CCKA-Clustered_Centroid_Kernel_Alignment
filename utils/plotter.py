"""
plotter.py

A drop-in plotting + analysis toolkit to make your Results section reviewer-proof.

What it covers (each function is optional — use only what you need):
1) Statistical significance & stability: accuracy mean±std across runs + paired tests (t-test, Wilcoxon).
2) Ablations: accuracy vs number of sub-centroids; circuit executions vs sub-centroids.
3) Noise robustness: accuracy vs noise level, with/without mitigation, across methods.
4) Runtime efficiency: wall-clock training time per epoch/method; circuit executions vs dataset size.
5) Convergence: kernel alignment (or loss) vs iteration.
6) Real hardware: easy table builder for small-device results.
7) Embedding visualization: PCA/t-SNE before vs after optimization.
8) Convenience: flexible CSV schema support + consistent plot saving.

Assumed logging schema (flexible, but this is the default expectation):
- Per-iteration logs (one CSV per run): columns like
  ['dataset','method','run_id','iteration','accuracy','alignment','loss','circuits','time_sec']
  'method' in {'CCKA','QUACK','QKA','Random','RBF','RF','LR','MLP',...}
- Per-run summary (optional): columns like
  ['dataset','method','run_id','accuracy','val_accuracy','test_accuracy','train_time_sec','circuits_total']

If your column names differ, pass column_name overrides to the functions.

All plots use matplotlib (no seaborn), single-axes per figure, and do not set explicit colors,
as per your rendering constraints. Figures are saved as PNGs by default.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------
# Utilities
# ----------------------------

@dataclass
class ColumnMap:
    dataset: str = "dataset"
    method: str = "method"
    run_id: str = "run_id"
    iteration: str = "iteration"
    accuracy: str = "accuracy"
    alignment: str = "alignment"
    loss: str = "loss"
    circuits: str = "circuits"
    time_sec: str = "time_sec"
    # per-run summary
    test_accuracy: str = "test_accuracy"
    val_accuracy: str = "val_accuracy"
    train_time_sec: str = "train_time_sec"
    circuits_total: str = "circuits_total"
    # ablation knobs
    subcentroids: str = "subcentroids"  # number of sub-centroids per class
    centroids_total: str = "centroids_total"
    # noise
    noise_level: str = "noise_level"
    mitigation: str = "mitigation"  # str or bool: 'on'/'off' or True/False


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csvs(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
    if not frames:
        raise FileNotFoundError("No CSVs could be loaded from the given paths.")
    return pd.concat(frames, ignore_index=True)


def save_fig(fig: plt.Figure, outdir: Path, name: str, dpi: int = 800) -> Path:
    _ensure_dir(outdir)
    outfile = outdir / f"{name}.png"
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return outfile


# ----------------------------
# 1) Statistical significance & stability
# ----------------------------

def accuracy_stats_table(
    df: pd.DataFrame,
    cols: ColumnMap = ColumnMap(),
    group_by: Tuple[str, ...] = ("dataset", "method")
) -> pd.DataFrame:
    """
    Returns a table with mean, std, n for accuracy grouped by (dataset, method).
    Accepts either per-run 'test_accuracy' or single 'accuracy'. If both exist, prefers 'test_accuracy'.
    """
    metric = cols.test_accuracy if cols.test_accuracy in df.columns else cols.accuracy
    needed = [getattr(cols, c) for c in group_by] + [metric]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    group = df.groupby([getattr(cols, c) for c in group_by])[metric]
    out = group.agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"mean": "acc_mean", "std": "acc_std", "count": "n"})
    return out


def paired_significance_vs_baseline(
    df: pd.DataFrame,
    baseline: str,
    cols: ColumnMap = ColumnMap(),
    dataset_filter: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Performs paired t-test and Wilcoxon signed-rank test between each method and the baseline,
    pairing by (dataset, run_id). Returns a tidy table.
    """
    metric = cols.test_accuracy if cols.test_accuracy in df.columns else cols.accuracy
    needed = [cols.dataset, cols.method, cols.run_id, metric]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    if dataset_filter is not None:
        df = df[df[cols.dataset].isin(dataset_filter)].copy()

    # Pivot to align pairs
    pivot = df.pivot_table(
        index=[cols.dataset, cols.run_id],
        columns=cols.method,
        values=metric
    )

    if baseline not in pivot.columns:
        raise ValueError(f"Baseline method '{baseline}' not found in data columns: {list(pivot.columns)}")

    results = []
    base_vals = pivot[baseline].dropna()
    for meth in pivot.columns:
        if meth == baseline:
            continue
        paired = pivot[[baseline, meth]].dropna()
        if paired.empty:
            continue
        b = paired[baseline].values
        m = paired[meth].values
        # Paired t-test
        t_stat, t_p = stats.ttest_rel(m, b, alternative="greater")
        # Wilcoxon (requires non-zero diffs)
        diffs = m - b
        nonzero = diffs[np.abs(diffs) > 1e-12]
        if len(nonzero) >= 5:  # Wilcoxon needs some pairs
            try:
                w_stat, w_p = stats.wilcoxon(nonzero, alternative="greater")
            except Exception:
                w_stat, w_p = np.nan, np.nan
        else:
            w_stat, w_p = np.nan, np.nan

        results.append({
            "method": meth,
            "baseline": baseline,
            "pairs": len(paired),
            "t_stat": t_stat,
            "t_p_greater": t_p,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p_greater": w_p
        })
    return pd.DataFrame(results)


def plot_accuracy_bars(
    stats_df: pd.DataFrame,
    outdir: Path,
    title: str = "Accuracy (mean ± std) by dataset and method",
    rotate_xticks: int = 20
) -> Path:
    """
    Expects a table from accuracy_stats_table() with columns:
    [dataset, method, acc_mean, acc_std].
    Creates one bar plot per dataset.
    """
    paths = []
    datasets = sorted(stats_df["dataset"].unique())
    for d in datasets:
        sub = stats_df[stats_df["dataset"] == d]
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        x = np.arange(len(sub))
        means = sub["acc_mean"].values
        stds = sub["acc_std"].values
        ax.bar(x, means, yerr=stds, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["method"].tolist(), rotation=rotate_xticks)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title}\nDataset: {d}")
        paths.append(save_fig(fig, outdir, f"acc_bar_{d}"))
    return paths


# ----------------------------
# 2) Ablations
# ----------------------------

def plot_ablation_subcentroids(
    df: pd.DataFrame,
    outdir: Path,
    cols: ColumnMap = ColumnMap(),
    dataset: Optional[str] = None,
    method: str = "CCKA",
    metric_col: Optional[str] = None,
    title: str = "Ablation: sub-centroids per class vs metric"
) -> Path:
    """
    Plots metric (default: test_accuracy/accuracy) vs number of sub-centroids (per class).
    """
    metric = metric_col or (cols.test_accuracy if cols.test_accuracy in df.columns else cols.accuracy)
    needed = [cols.method, cols.subcentroids, metric]
    if dataset is not None:
        needed.append(cols.dataset)
        df = df[df[cols.dataset] == dataset].copy()
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    sub = df[df[cols.method] == method]
    agg = sub.groupby(cols.subcentroids)[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(agg[cols.subcentroids].values, agg[metric].values, marker="o")
    ax.set_xlabel("Sub-centroids per class")
    ax.set_ylabel(metric)
    ttl = title if dataset is None else f"{title} — {dataset}"
    ax.set_title(ttl)
    return save_fig(fig, outdir, f"ablation_subcentroids_{dataset or 'all'}")


def plot_ablation_circuits_vs_subcentroids(
    df: pd.DataFrame,
    outdir: Path,
    cols: ColumnMap = ColumnMap(),
    dataset: Optional[str] = None,
    method: str = "CCKA",
    title: str = "Circuit executions vs sub-centroids per class"
) -> Path:
    needed = [cols.method, cols.subcentroids, cols.circuits]
    if dataset is not None:
        needed.append(cols.dataset)
        df = df[df[cols.dataset] == dataset].copy()
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")
    sub = df[df[cols.method] == method]
    agg = sub.groupby(cols.subcentroids)[cols.circuits].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(agg[cols.subcentroids].values, agg[cols.circuits].values, marker="o")
    ax.set_xlabel("Sub-centroids per class")
    ax.set_ylabel("Avg circuits per iteration")
    ttl = title if dataset is None else f"{title} — {dataset}"
    ax.set_title(ttl)
    return save_fig(fig, outdir, f"ablation_circuits_{dataset or 'all'}")


# ----------------------------
# 3) Noise robustness
# ----------------------------

def plot_noise_robustness(
    df: pd.DataFrame,
    outdir: Path,
    cols: ColumnMap = ColumnMap(),
    dataset: Optional[str] = None,
    methods: Optional[List[str]] = None,
    title: str = "Accuracy vs noise level"
) -> List[Path]:
    """
    Expects columns: [dataset?, method, noise_level, mitigation?, test_accuracy/accuracy].
    Generates one plot with mitigation OFF, one with mitigation ON (if present).
    """
    metric = cols.test_accuracy if cols.test_accuracy in df.columns else cols.accuracy
    needed = [cols.method, cols.noise_level, metric]
    if cols.mitigation in df.columns:
        needed.append(cols.mitigation)
    if dataset is not None:
        needed.append(cols.dataset)
        df = df[df[cols.dataset] == dataset].copy()
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    if methods is None:
        methods = sorted(df[cols.method].unique())

    paths = []
    if cols.mitigation in df.columns:
        for mit in sorted(df[cols.mitigation].astype(str).unique()):
            sub = df[df[cols.mitigation].astype(str) == mit]
            fig, ax = plt.subplots(figsize=(7.5, 4.0))
            for m in methods:
                sm = sub[sub[cols.method] == m]
                agg = sm.groupby(cols.noise_level)[metric].mean().reset_index()
                ax.plot(agg[cols.noise_level].values, agg[metric].values, marker="o", label=m)
            ax.set_xlabel("Noise level")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{title} — mitigation={mit}" + (f" — {dataset}" if dataset else ""))
            ax.legend()
            paths.append(save_fig(fig, outdir, f"noise_{mit}_{dataset or 'all'}"))
    else:
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        for m in methods:
            sm = df[df[cols.method] == m]
            agg = sm.groupby(cols.noise_level)[metric].mean().reset_index()
            ax.plot(agg[cols.noise_level].values, agg[metric].values, marker="o", label=m)
        ax.set_xlabel("Noise level")
        ax.set_ylabel("Accuracy")
        ax.set_title(title + (f" — {dataset}" if dataset else ""))
        ax.legend()
        paths.append(save_fig(fig, outdir, f"noise_{dataset or 'all'}"))
    return paths


# ----------------------------
# 4) Runtime efficiency & scaling
# ----------------------------

def plot_runtime_per_epoch(
    df: pd.DataFrame,
    outdir: Path,
    cols: ColumnMap = ColumnMap(),
    dataset: Optional[str] = None,
    title: str = "Wall-clock training time per epoch"
) -> Path:
    needed = [cols.method, cols.time_sec]
    if dataset is not None:
        needed.append(cols.dataset)
        df = df[df[cols.dataset] == dataset].copy()
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")
    agg = df.groupby(cols.method)[cols.time_sec].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(agg[cols.method].values, agg[cols.time_sec].values)
    ax.set_ylabel("Seconds / epoch (avg)")
    ax.set_title(title + (f" — {dataset}" if dataset else ""))
    return save_fig(fig, outdir, f"runtime_epoch_{dataset or 'all'}")


def plot_circuits_vs_datasize(
    df: pd.DataFrame,
    outdir: Path,
    cols: ColumnMap = ColumnMap(),
    title: str = "Circuit executions vs dataset size",
    size_col: str = "n_samples"
) -> Path:
    """
    df should contain columns: [method, n_samples, circuits] aggregated per iteration or total.
    """
    needed = [cols.method, cols.circuits, size_col]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for m in sorted(df[cols.method].unique()):
        sm = df[df[cols.method] == m]
        agg = sm.groupby(size_col)[cols.circuits].mean().reset_index()
        ax.plot(agg[size_col].values, agg[cols.circuits].values, marker="o", label=m)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size (log)")
    ax.set_ylabel("Avg circuits (log)")
    ax.set_title(title)
    ax.legend()
    return save_fig(fig, outdir, "circuits_scaling")


# ----------------------------
# 5) Convergence
# ----------------------------

def plot_convergence(
    df: pd.DataFrame,
    outdir: Path,
    cols: ColumnMap = ColumnMap(),
    dataset: Optional[str] = None,
    methods: Optional[List[str]] = None,
    y: str = "alignment",
    title: str = "Convergence (metric vs iteration)"
) -> List[Path]:
    """
    Plots metric (alignment/loss/accuracy) vs iteration. One figure per method to keep it clean.
    """
    metric_col = getattr(cols, y) if hasattr(cols, y) else y
    needed = [cols.method, cols.iteration, metric_col]
    if dataset is not None:
        needed.append(cols.dataset)
        df = df[df[cols.dataset] == dataset].copy()
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    if methods is None:
        methods = sorted(df[cols.method].unique())

    paths = []
    for m in methods:
        sub = df[df[cols.method] == m]
        agg = sub.groupby(cols.iteration)[metric_col].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.plot(agg[cols.iteration].values, agg[metric_col].values)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric_col)
        ax.set_title(f"{title} — {m}" + (f" — {dataset}" if dataset else ""))
        paths.append(save_fig(fig, outdir, f"convergence_{y}_{m}_{dataset or 'all'}"))
    return paths


# ----------------------------
# 6) Real hardware table builder
# ----------------------------

def build_hardware_results_table(
    entries: List[Dict],
    outpath: Path
) -> Path:
    """
    entries: list of dicts with keys like
        {'device':'ibmq_falcon','dataset':'moons','method':'CCKA','shots':8192,
         'accuracy':0.86,'circuits':800,'notes':'readout-mitigation on'}
    """
    df = pd.DataFrame(entries)
    df.to_csv(outpath, index=False)
    return outpath


# ----------------------------
# 7) Embedding visualization (PCA/t-SNE)
# ----------------------------

def plot_embeddings_2d(
    X_before: np.ndarray,
    X_after: np.ndarray,
    y: np.ndarray,
    outdir: Path,
    title_before: str = "Embeddings before optimization",
    title_after: str = "Embeddings after CCKA optimization",
    method: str = "pca"
) -> List[Path]:
    """
    X_*: [n_samples, n_features] quantum feature vectors (e.g., from state overlaps or post-embedding classical features).
    y: labels [n_samples].
    method: 'pca' or 'tsne'.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    def reduce2d(X: np.ndarray) -> np.ndarray:
        if method.lower() == "pca":
            return PCA(n_components=2).fit_transform(X)
        elif method.lower() == "tsne":
            return TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(X)
        else:
            raise ValueError("method must be 'pca' or 'tsne'")

    P = reduce2d(X_before)
    Q = reduce2d(X_after)

    paths = []
    # Before
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for lab in np.unique(y):
        idx = (y == lab)
        ax.scatter(P[idx, 0], P[idx, 1], label=str(lab), s=16)
    ax.set_title(title_before)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend()
    paths.append(save_fig(fig, outdir, "embeddings_before"))

    # After
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for lab in np.unique(y):
        idx = (y == lab)
        ax.scatter(Q[idx, 0], Q[idx, 1], label=str(lab), s=16)
    ax.set_title(title_after)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend()
    paths.append(save_fig(fig, outdir, "embeddings_after"))
    return paths


# ----------------------------
# 8) Convenience: dump stats to markdown/JSON
# ----------------------------

def export_stats_markdown(
    acc_stats: pd.DataFrame,
    significance: pd.DataFrame,
    outpath: Path
) -> Path:
    lines = ["# Results Stats Summary", "", "## Accuracy mean±std", ""]
    lines.append(acc_stats.to_markdown(index=False))
    lines += ["", "## Paired significance vs baseline", ""]
    lines.append(significance.to_markdown(index=False))
    outpath.write_text("\n".join(lines))
    return outpath


def export_json(obj, outpath: Path) -> Path:
    import json
    with open(outpath, "w") as f:
        json.dump(obj, f, indent=2)
    return outpath


# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from matplotlib.colors import ListedColormap
# from sklearn.decomposition import PCA
# from concurrent.futures import ThreadPoolExecutor
# import os
# import torch
# import pandas as pd

# # Set consistent style and color palette
# plt.style.use('seaborn-v0_8')
# cmap_background = ListedColormap(["#ffcccb", "#add8e6"])
# color_pos = '#f6932a'
# color_neg = '#3cb0ea'
# background_color = "#eaeaf2"


# def to_numpy(x):
#         return x.detach().cpu().numpy() if hasattr(x, "detach") else x

# def kernel_heatmap(K, path, title="Kernel Heatmap"):
#     """Plot a kernel matrix as a heatmap."""
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_facecolor(background_color)
#     sns.heatmap(K, cmap='viridis', ax=ax)
#     ax.set_xlabel("Samples", fontsize=12)
#     ax.set_ylabel("Samples", fontsize=12)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.tight_layout()
#     plt.savefig(f"{path}/{title}.png", dpi=800, bbox_inches='tight')
#     plt.close(fig)


# def alignment_progress_over_iterations(alignment_arrs, path, title="Alignment Progress Over Iterations"):
#     """
#     Plot alignment progress. Accepts 1 or 2 arrays. If 2, label them as Centroid 1 and Centroid -1.
#     """
#     # Convert tensors to numpy arrays
#     processed = []
#     for arr in alignment_arrs:
#         if torch.is_tensor(arr):
#             arr = arr.detach().cpu().numpy()
#         if isinstance(arr, list):
#             arr = np.array(arr)
#         processed.append(arr)

#     # Plot setup
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_facecolor(background_color)

#     # Plot logic
#     if len(processed) == 2 and len(processed[1]) > 0:
#         ax.plot(processed[0], label='Centroid 1', color=color_pos)
#         ax.plot(processed[1], label='Centroid -1', color=color_neg)
#     else:
#         ax.plot(processed[0], label='Alignment', color=color_pos)

#     # Axis and title setup
#     ax.set_xlabel("Iteration", fontsize=12)
#     ax.set_ylabel("Alignment Score", fontsize=12)
#     ax.grid(True, linestyle='--', alpha=0.6)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.legend(loc='upper right', fontsize=10, frameon=True)

#     # Save
#     os.makedirs(path, exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(f"{path}/{title}.png", dpi=800, bbox_inches='tight')
#     plt.close(fig)

# def execution_reduction_vs_performance_trade_off(executions, performances, path, title="Execution Reduction vs Performance Trade-off"):
#     full_exec = max(executions)
#     reduction = [(full_exec - e) / full_exec * 100 for e in executions]
#     performance_retention = [p / performances[0] * 100 for p in performances]

#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_facecolor(background_color)
#     ax.plot(reduction, performance_retention, 'o-', color=color_neg, markersize=8)
#     for i in range(len(reduction)):
#         ax.text(reduction[i]+1, performance_retention[i], f"Run {i+1}", fontsize=9)
#     ax.set_title(title, fontsize=14, fontweight='bold')
#     ax.set_xlabel("Execution Reduction (%)", fontsize=12)
#     ax.set_ylabel("Performance Retention (%)", fontsize=12)
#     ax.grid(axis='both', linestyle='--', alpha=0.6)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.tight_layout()
#     plt.savefig(f"{path}/{title}.png", dpi=800, bbox_inches='tight')
#     plt.close(fig)


# def plot_initial_final_accuracies(initial_train_acc, initial_test_acc,
#                                   final_train_acc, final_test_acc,
#                                   path, title="Initial vs Final Accuracies"):
#     """
#     Plot initial and final training/testing accuracy in a grouped bar chart.
#     """
#     os.makedirs(path, exist_ok=True)

#     labels = ['Training Accuracy', 'Testing Accuracy']
#     initial_scores = [initial_train_acc * 100, initial_test_acc * 100]
#     final_scores = [final_train_acc * 100, final_test_acc * 100]

#     x = np.arange(len(labels))  # Label locations
#     width = 0.35  # Width of the bars

#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_facecolor(background_color)
#     bars1 = ax.bar(x - width / 2, initial_scores, width, label='Initial', color=color_neg)
#     bars2 = ax.bar(x + width / 2, final_scores, width, label='Final', color=color_pos)

#     # Add labels and titles
#     ax.set_ylabel('Accuracy (%)', fontsize=12)
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, fontsize=12)
#     ax.legend()
#     ax.set_ylim(0, 100)

#     # Add values on top of bars
#     for bars in [bars1, bars2]:
#         for bar in bars:
#             height = bar.get_height()
#             ax.annotate(f'{height:.1f}%',
#                         xy=(bar.get_x() + bar.get_width() / 2, height),
#                         xytext=(0, 3),  # Offset above bar
#                         textcoords="offset points",
#                         ha='center', va='bottom', fontsize=10)

#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.grid(axis='y', linestyle='--', alpha=0.6)

#     plt.tight_layout()
#     plt.savefig(f"{path}/{title}.png", dpi=800, bbox_inches='tight')
#     plt.close(fig)

# def decision_boundary(model, training_data, training_labels, test_data, test_labels,
#                       kernel, compute_test_kernel_row, path, title="decision_boundary"):

#     training_data = to_numpy(training_data)
#     training_labels = to_numpy(training_labels)
#     test_data = to_numpy(test_data)
#     test_labels = to_numpy(test_labels)

#     # Create a high-res mesh grid for decision surface
#     x_min, x_max = np.vstack([training_data, test_data])[:, 0].min() - 1, np.vstack([training_data, test_data])[:, 0].max() + 1
#     y_min, y_max = np.vstack([training_data, test_data])[:, 1].min() - 1, np.vstack([training_data, test_data])[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
#     grid_points = np.c_[xx.ravel(), yy.ravel()]

#     # Compute kernel values for mesh grid (as test points vs training)
#     with ThreadPoolExecutor() as executor:
#         futures = [executor.submit(compute_test_kernel_row, kernel, grid_points, training_data, i)
#                    for i in range(len(grid_points))]
#         kernel_rows = [f.result() for f in futures]

#     kernel_matrix = np.stack(kernel_rows)
#     Z = model.predict(kernel_matrix).reshape(xx.shape)

#     # Compute predictions on test set
#     with ThreadPoolExecutor() as executor:
#         futures = [executor.submit(compute_test_kernel_row, kernel, test_data, training_data, i)
#                    for i in range(test_data.shape[0])]
#         test_rows = [f.result() for f in futures]

#     test_kernel_matrix = np.stack(test_rows)
#     test_predictions = model.predict(test_kernel_matrix)

#     decision_table = pd.DataFrame({
#         "x": test_data[:, 0],
#         "y": test_data[:, 1],
#         "true_label": test_labels,
#         "predicted_label": test_predictions
#     })

#     # Plot decision boundary with soft contours
#     fig, ax = plt.subplots(figsize=(6, 6))

#     # Background with soft contourf
#     contour = ax.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=["#eba152", "#7ec7ec"], alpha=0.4)

#     # Training points
#     ax.scatter(training_data[training_labels == 1][:, 0], training_data[training_labels == 1][:, 1],
#                c="#3cb0ea", s=60, label="Train +1")
#     ax.scatter(training_data[training_labels == -1][:, 0], training_data[training_labels == -1][:, 1],
#                c="#f6932a", s=60, label="Train -1")

#     # Test points with outlined circle markers (no fill)
#     ax.scatter(test_data[test_labels == 1][:, 0], test_data[test_labels == 1][:, 1],
#             facecolors='none', edgecolors='#3cb0ea', s=60, linewidths=1.2, marker='o', label="Test +1")

#     ax.scatter(test_data[test_labels == -1][:, 0], test_data[test_labels == -1][:, 1],
#             facecolors='none', edgecolors="#f6932a", s=60, linewidths=1.2, marker='o', label="Test -1")


#     # Style adjustments to mimic paper
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     for spine in ax.spines.values():
#         spine.set_visible(False)

#     ax.legend(loc='upper right', fontsize=8, frameon=False)

#     os.makedirs(path, exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(f"{path}/{title}.png", dpi=800, bbox_inches='tight')
#     plt.close()

#     return decision_table


# def decision_boundary_pennylane(model, training_data, training_labels, test_data, test_labels,
#                       kernel_fn, path, title="decision_boundary"):

#     # Convert tensors to numpy arrays for plotting and model prediction
#     if torch.is_tensor(training_data):
#         training_data = training_data.detach()
#     if torch.is_tensor(training_labels):
#         training_labels = training_labels.detach()
#     if torch.is_tensor(test_data):
#         test_data = test_data.detach()
#     if torch.is_tensor(test_labels):
#         test_labels = test_labels.detach()

#     training_data_np = training_data
#     training_labels_np = training_labels
#     test_data_np = test_data
#     test_labels_np = test_labels

#     # Create mesh grid
#     x_min, x_max = np.vstack([training_data_np, test_data_np])[:, 0].min() - 1, np.vstack([training_data_np, test_data_np])[:, 0].max() + 1
#     y_min, y_max = np.vstack([training_data_np, test_data_np])[:, 1].min() - 1, np.vstack([training_data_np, test_data_np])[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
#     grid_points = np.c_[xx.ravel(), yy.ravel()]
#     grid_tensor = torch.tensor(grid_points, dtype=training_data.dtype)

#     # Compute kernel between mesh points and training data
#     x_0 = grid_tensor.repeat_interleave(training_data.shape[0], dim=0)
#     x_1 = training_data.repeat(grid_tensor.shape[0], 1)
#     kernel_matrix = kernel_fn(x_0, x_1).to(torch.float32).reshape(grid_tensor.shape[0], training_data.shape[0])
#     kernel_matrix_np = kernel_matrix.detach().numpy()

#     # Predict using SVC model trained on precomputed kernel
#     Z = model.predict(kernel_matrix_np).reshape(xx.shape)

#     # Compute test predictions for overlay
#     x_0_test = test_data.repeat_interleave(training_data.shape[0], dim=0)
#     x_1_test = training_data.repeat(test_data.shape[0], 1)
#     test_kernel_matrix = kernel_fn(x_0_test, x_1_test).to(torch.float32).reshape(test_data.shape[0], training_data.shape[0])
#     test_kernel_matrix_np = test_kernel_matrix.detach().numpy()
#     test_predictions = model.predict(test_kernel_matrix_np)

#     decision_table = pd.DataFrame({
#         "x": test_data_np[:, 0],
#         "y": test_data_np[:, 1],
#         "true_label": test_labels_np,
#         "predicted_label": test_predictions
#     })

#     # Plot decision surface and points
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=["#eba152", "#7ec7ec"], alpha=0.4)

#     ax.scatter(training_data_np[training_labels_np == 1][:, 0], training_data_np[training_labels_np == 1][:, 1],
#                c="#3cb0ea", s=60, label="Train +1")
#     ax.scatter(training_data_np[training_labels_np == -1][:, 0], training_data_np[training_labels_np == -1][:, 1],
#                c="#f6932a", s=60, label="Train -1")

#     ax.scatter(test_data_np[test_labels_np == 1][:, 0], test_data_np[test_labels_np == 1][:, 1],
#                facecolors='none', edgecolors='#3cb0ea', s=60, linewidths=1.2, marker='o', label="Test +1")
#     ax.scatter(test_data_np[test_labels_np == -1][:, 0], test_data_np[test_labels_np == -1][:, 1],
#                facecolors='none', edgecolors="#f6932a", s=60, linewidths=1.2, marker='o', label="Test -1")

#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     for spine in ax.spines.values():
#         spine.set_visible(False)

#     ax.legend(loc='upper right', fontsize=8, frameon=False)

#     os.makedirs(path, exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(f"{path}/{title}.png", dpi=800, bbox_inches='tight')
#     plt.close()

#     return decision_table

