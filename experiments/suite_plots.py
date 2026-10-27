"""Plot suite for the experiment configs in configs/exp*.yaml.

    python -m experiments.suite_plots --results-dir results --output-dir results/plots

Every experiment CSV that exists is plotted; missing ones are skipped. Curves
show mean +- std over seeds. Hyper-parameters are never selected on test data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.plot_results import load_results

METHODS = ["fullKTA", "randomKTA", "quackKTA", "ccka"]
COLORS = dict(zip(METHODS, ["tab:blue", "tab:gray", "tab:orange", "tab:green"]))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _noiseless(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["noise_type"].isin(["none", "shots_only"])) & (df["noise_level"] == 0)]


def _main_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = _noiseless(df)
    df = df[df["ablation"].isin(["none", "full"])]
    return df[(df["method"] != "ccka") | (df["initialization"] == "regular")]


def grouped_bars(df: pd.DataFrame, out: Path, metric: str = "test_accuracy", prefix: str = "") -> None:
    df = _main_rows(df)
    if "ansatz" not in df:
        df = df.assign(ansatz="crz_ring")
    for ansatz, sub in df.groupby("ansatz"):
        g = sub.groupby(["dataset", "method"])[metric].agg(["mean", "std"]).reset_index()
        datasets = sorted(g["dataset"].unique())
        fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(datasets)), 3.6))
        width = 0.8 / len(METHODS)
        for i, m in enumerate(METHODS):
            gm = g[g["method"] == m].set_index("dataset").reindex(datasets)
            ax.bar(np.arange(len(datasets)) + i * width, gm["mean"], width,
                   yerr=gm["std"], label=m, color=COLORS[m], capsize=2)
        ax.set_xticks(np.arange(len(datasets)) + 0.4 - width / 2, datasets, rotation=45, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"ansatz: {ansatz}")
        ax.legend(fontsize=7, ncol=4)
        _save(fig, out / f"{prefix}main_bars_{ansatz}.png")


def heatmaps(df: pd.DataFrame, out: Path) -> None:
    df = _main_rows(df)
    if "ansatz" not in df:
        return
    g = df.groupby(["ansatz", "method"])["test_accuracy"].mean().unstack()
    g = g[[m for m in METHODS if m in g]]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(g.values, cmap="viridis")
    ax.set_xticks(range(g.shape[1]), g.columns, rotation=30)
    ax.set_yticks(range(g.shape[0]), g.index)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            ax.text(j, i, f"{g.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, label="mean test accuracy")
    _save(fig, out / "heatmap_ansatz_x_method.png")
    for method, sub in df.groupby("method"):
        g = sub.groupby(["ansatz", "dataset"])["test_accuracy"].mean().unstack()
        fig, ax = plt.subplots(figsize=(0.7 * g.shape[1] + 2, 3.2))
        im = ax.imshow(g.values, cmap="viridis", vmin=0.4, vmax=1.0)
        ax.set_xticks(range(g.shape[1]), g.columns, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(g.shape[0]), g.index)
        ax.set_title(method)
        fig.colorbar(im)
        _save(fig, out / f"heatmap_ansatz_x_dataset_{method}.png")


def pareto(df: pd.DataFrame, out: Path) -> None:
    df = _main_rows(df)
    keys = ["dataset", "method"] + (["ansatz"] if "ansatz" in df else [])
    g = df.groupby(keys)[["test_accuracy", "circuit_executions"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for m in METHODS:
        s = g[g["method"] == m]
        ax.scatter(s["circuit_executions"], s["test_accuracy"], label=m, color=COLORS[m], alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("training circuit evaluations")
    ax.set_ylabel("test accuracy")
    ax.legend()
    _save(fig, out / "pareto_accuracy_vs_cost.png")


def sweep(df: pd.DataFrame, x: str, out: Path, name: str, methods=METHODS,
          logx: bool = False, flat_methods=(), metric: str = "test_accuracy") -> None:
    """One panel per dataset: metric vs x, one line per method.
    Methods in flat_methods have no x (e.g. QUACK in a K sweep) and are drawn flat."""
    df = df.copy()
    df[x] = pd.to_numeric(df[x], errors="coerce")
    datasets = sorted(df["dataset"].unique())
    cols = min(4, len(datasets))
    rows = int(np.ceil(len(datasets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows), squeeze=False)
    for ax, dataset in zip(axes.ravel(), datasets):
        sub = df[df["dataset"] == dataset]
        for m in methods:
            s = sub[sub["method"] == m]
            if s.empty:
                continue
            if m in flat_methods:
                ax.axhline(s[metric].mean(), color=COLORS[m], ls="--", label=m)
                continue
            g = s.groupby(x)[metric].agg(["mean", "std"]).dropna(subset=["mean"]).fillna(0)
            ax.plot(g.index, g["mean"], marker="o", color=COLORS[m], label=m)
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=COLORS[m], alpha=0.15)
        if logx:
            ax.set_xscale("log")
        ax.set_title(dataset, fontsize=9)
        ax.set_xlabel(x)
        ax.set_ylabel(metric)
    for ax in axes.ravel()[len(datasets):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    _save(fig, out / f"{name}.png")


def k_sweep(df: pd.DataFrame, out: Path) -> None:
    df = _main_rows(df)
    ccka = df[df["method"] == "ccka"].copy()
    ccka["K"] = pd.to_numeric(ccka["centroids_per_class"])
    quack = df[df["method"] == "quackKTA"].assign(K=np.nan)
    both = pd.concat([ccka, quack])
    for ansatz, sub in both.groupby("ansatz") if "ansatz" in both else [("all", both)]:
        sweep(sub, "K", out, f"k_sweep_{ansatz}", methods=["quackKTA", "ccka"], flat_methods=("quackKTA",))
    # accuracy vs K with the evaluation cost on the top axis
    datasets = sorted(ccka["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.2), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        g = ccka[ccka["dataset"] == dataset].groupby("K")[["test_accuracy", "circuit_executions"]].mean()
        ax.plot(g.index, g["test_accuracy"], marker="o", color=COLORS["ccka"])
        ax.set_xscale("log", base=2)
        ax.set_title(dataset, fontsize=9)
        ax.set_xlabel("K")
        ax.set_ylabel("test accuracy")
        top = ax.secondary_xaxis("top", functions=(
            lambda k, g=g: np.interp(k, g.index, g["circuit_executions"]),
            lambda c, g=g: np.interp(c, g["circuit_executions"], g.index)))
        top.set_xlabel("circuit evals", fontsize=7)
    _save(fig, out / "k_sweep_with_cost_axis.png")
    # optimal K (by validation) vs number of modes for the gmm datasets
    modes = {"gmm_modes_1": 1, "gmm_modes_2": 2, "gmm_modes_4": 4}
    rows = []
    for dataset, n_modes in modes.items():
        g = ccka[ccka["dataset"] == dataset].groupby("K")["best_validation_accuracy"].mean()
        if not g.empty:
            rows.append((n_modes, g.idxmax()))
    if rows:
        fig, ax = plt.subplots(figsize=(4, 3.2))
        ax.plot(*zip(*rows), marker="o")
        ax.set_xlabel("modes per class")
        ax.set_ylabel("best K (validation)")
        _save(fig, out / "optimal_K_vs_modes.png")
        fig, ax = plt.subplots(figsize=(4.4, 3.4))
        for n_modes, d in [(v, k) for k, v in modes.items()]:
            g = ccka[ccka["dataset"] == d].groupby("K")["test_accuracy"].mean()
            ax.plot(g.index, g.values, marker="o", label=f"{n_modes} modes")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K")
        ax.set_ylabel("test accuracy")
        ax.legend()
        _save(fig, out / "accuracy_vs_K_by_modes.png")


def _heat_grid(df: pd.DataFrame, row: str, col: str, out: Path, name: str) -> None:
    for dataset, sub in df[df["method"] == "ccka"].groupby("dataset"):
        g = sub.groupby([row, col])["test_accuracy"].mean().unstack()
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        im = ax.imshow(g.values, cmap="viridis")
        ax.set_xticks(range(g.shape[1]), [f"{c:g}" for c in g.columns])
        ax.set_yticks(range(g.shape[0]), [f"{r:g}" for r in g.index])
        ax.set_xlabel(col)
        ax.set_ylabel(row)
        ax.set_title(dataset)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                ax.text(j, i, f"{g.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, label="test accuracy")
        _save(fig, out / f"{name}_{dataset}.png")


def ablation(df: pd.DataFrame, out: Path) -> None:
    df = _noiseless(df).copy()
    bad = set(df["dataset"]) - {"corners", "checkerboard"}
    if bad:
        raise ValueError(f"Ablation plots are restricted to corners/checkerboard, found {sorted(bad)}")
    df["variant"] = np.where(
        (df["method"] == "quackKTA"), "QUACK",
        np.where(df["ablation"] == "full",
                 np.where(df["initialization"] == "regular", "full CCKA", "clustering:" + df["initialization"].astype(str)),
                 df["ablation"].astype(str)))
    df = df[df["method"].isin(["ccka", "quackKTA"])]
    fig, axes = plt.subplots(1, df["dataset"].nunique(), figsize=(5.5 * df["dataset"].nunique(), 3.8), squeeze=False)
    for ax, (dataset, sub) in zip(axes[0], df.groupby("dataset")):
        base = sub[sub["variant"] == "full CCKA"].groupby("seed")["test_accuracy"].mean()
        names, deltas, errs = [], [], []
        for variant, s in sub.groupby("variant"):
            if variant == "full CCKA":
                continue
            d = (s.groupby("seed")["test_accuracy"].mean() - base).dropna()
            names.append(variant); deltas.append(d.mean()); errs.append(d.std(ddof=1) if len(d) > 1 else 0)
        order = np.argsort(deltas)
        ax.barh(np.array(names)[order], np.array(deltas)[order], xerr=np.array(errs)[order], color="tab:red", capsize=2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(dataset)
        ax.set_xlabel("Δ test accuracy vs full CCKA")
    _save(fig, out / "ablation_delta_bars.png")

    for dataset, sub in df.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        for variant, s in sub.groupby("variant"):
            curves = [json.loads(v) for v in s["validation_accuracy_history"].dropna()]
            n = min(len(c) for c in curves)
            ax.plot(np.mean([c[:n] for c in curves], axis=0), label=variant)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation accuracy")
        ax.legend(fontsize=6)
        ax.set_title(dataset)
        _save(fig, out / f"ablation_learning_curves_{dataset}.png")


def noise(df: pd.DataFrame, out: Path) -> None:
    df = df[df["ablation"].isin(["none", "full"])]
    zero = df[(df["noise_type"] == "none")]
    types = [t for t in df["noise_type"].unique() if t != "none"]
    fig, axes = plt.subplots(1, len(types), figsize=(3.6 * len(types), 3.2), squeeze=False)
    for ax, t in zip(axes[0], types):
        for m in METHODS:
            s = pd.concat([zero[zero["method"] == m].assign(noise_level=0.0),
                           df[(df["noise_type"] == t) & (df["method"] == m)]])
            g = s.groupby("noise_level")["test_accuracy"].agg(["mean", "std"]).fillna(0)
            if len(g) > 1:
                ax.errorbar(range(len(g)), g["mean"], g["std"], marker="o", color=COLORS[m], label=m, capsize=2)
                ax.set_xticks(range(len(g)), [f"{v:g}" for v in g.index], rotation=45)
        ax.set_title(t)
        ax.set_xlabel("noise level")
        ax.set_ylabel("test accuracy")
    axes[0][0].legend(fontsize=7)
    _save(fig, out / "noise_accuracy_vs_level.png")
    g = df[df["noise_level"] > 0].groupby(["noise_type", "method"])["test_accuracy"].mean().unstack()
    if not g.empty:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        im = ax.imshow(g.values, cmap="viridis")
        ax.set_xticks(range(g.shape[1]), g.columns, rotation=30)
        ax.set_yticks(range(g.shape[0]), g.index)
        fig.colorbar(im, label="mean accuracy over noise levels")
        _save(fig, out / "noise_type_x_method_heatmap.png")
    # variance across seeds of the final accuracy as a proxy for noise-induced variability
    v = df.groupby(["method", "noise_level"])["test_accuracy"].var().unstack(0)
    if len(v) > 1:
        fig, ax = plt.subplots(figsize=(4.4, 3.4))
        for m in v.columns:
            ax.plot(v.index, v[m], marker="o", label=m, color=COLORS.get(m))
        ax.set_xlabel("noise level")
        ax.set_ylabel("variance of test accuracy across seeds/types")
        ax.legend(fontsize=7)
        _save(fig, out / "noise_result_variance.png")


def multiclass(df: pd.DataFrame, out: Path) -> None:
    g = _main_rows(df).groupby(["dataset", "multiclass_strategy", "method"])["test_accuracy"].agg(["mean", "std"]).reset_index()
    datasets = sorted(g["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 3.2), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        s = g[g["dataset"] == d]
        for i, strat in enumerate(sorted(s["multiclass_strategy"].unique())):
            t = s[s["multiclass_strategy"] == strat].set_index("method").reindex(METHODS[::2][:1] + ["quackKTA", "ccka"]).dropna(how="all")
            ax.bar(np.arange(len(t)) + i * 0.4, t["mean"], 0.4, yerr=t["std"], label=strat)
            ax.set_xticks(np.arange(len(t)) + 0.2, t.index, rotation=30)
        ax.set_title(d, fontsize=9)
    axes[0][0].legend()
    _save(fig, out / "multiclass_strategy.png")


def depth(df: pd.DataFrame, out: Path) -> None:
    df = _main_rows(df)
    for dataset, sub in df.groupby("dataset"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        for ax, m in zip(axes, ["fullKTA", "quackKTA", "ccka"]):
            for ansatz, s in sub[sub["method"] == m].groupby("ansatz"):
                g = s.groupby("reps")["test_accuracy"].mean()
                ax.plot(g.index, g.values, marker="o", label=ansatz)
            ax.set_title(f"{dataset}: {m}")
            ax.set_xlabel("layers L")
            ax.set_ylabel("test accuracy")
        axes[0].legend(fontsize=6)
        _save(fig, out / f"accuracy_vs_depth_{dataset}.png")
    g = df[df["method"] == "ccka"].groupby(["ansatz", "n_params"])["test_accuracy"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for ansatz, s in g.groupby("ansatz"):
        ax.plot(s["n_params"], s["test_accuracy"], marker="o", label=ansatz)
    ax.set_xlabel("parameter count")
    ax.set_ylabel("CCKA test accuracy")
    ax.legend(fontsize=6)
    _save(fig, out / "accuracy_vs_parameter_count.png")


def optimizer(df: pd.DataFrame, out: Path) -> None:
    df = _main_rows(df)
    g = df.groupby(["optimizer", "init_scale", "method"])["test_accuracy"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for (opt, scale), s in g.groupby(["optimizer", "init_scale"]):
        s = s.set_index("method").reindex(METHODS[::2][:1] + ["quackKTA", "ccka"]).dropna()
        ax.plot(s.index, s["test_accuracy"], marker="o", label=f"{opt}, init×{scale:g}")
    ax.set_ylabel("test accuracy")
    ax.legend(fontsize=6)
    _save(fig, out / "optimizer_init_sensitivity.png")

    # epochs-to-plateau: first epoch reaching 98% of the best validation accuracy
    plateau = []
    for _, r in df.iterrows():
        h = json.loads(r["validation_accuracy_history"])
        if h:
            target = 0.98 * max(h)
            plateau.append((r["method"], int(np.argmax(np.asarray(h) >= target))))
    p = pd.DataFrame(plateau, columns=["method", "epochs_to_plateau"])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    methods = [m for m in METHODS if m in set(p["method"])]
    ax.boxplot([p[p["method"] == m]["epochs_to_plateau"] for m in methods], tick_labels=methods)
    ax.set_ylabel("epochs to plateau")
    _save(fig, out / "epochs_to_plateau_box.png")


EXPERIMENTS = {
    "exp01_main_benchmark": lambda df, out: (grouped_bars(df, out), grouped_bars(df, out, "test_f1_macro", "f1_"), heatmaps(df, out), pareto(df, out)),
    "exp02_multiclass": multiclass,
    "exp03_ablation_structure": ablation,
    "exp04_ablation_lambda_grid": lambda df, out: _heat_grid(df, "lambda_kao", "lambda_co", out, "heatmap_lambda_kao_x_lambda_co"),
    "exp05_ablation_lr_grid": lambda df, out: _heat_grid(df, "centroid_lr", "sub_centroid_lr", out, "heatmap_mclr_x_cclr"),
    "exp06_k_sweep": k_sweep,
    "exp07_ntrain": lambda df, out: sweep(_main_rows(df), "n_train", out, "data_regime_ntrain", logx=True),
    "exp08_label_noise": lambda df, out: sweep(_main_rows(df), "label_noise", out, "data_regime_label_noise"),
    "exp09_imbalance": lambda df, out: (sweep(_main_rows(df), "class_imbalance", out, "data_regime_imbalance"),
                                           sweep(_main_rows(df), "class_imbalance", out, "data_regime_imbalance_f1", metric="test_f1_macro")),
    "exp10_feature_dim": lambda df, out: sweep(_main_rows(df), "n_qubits", out, "data_regime_feature_dim"),
    "exp11_noise": noise,
    "exp12_shots": lambda df, out: sweep(df[df["ablation"].isin(["none", "full"])].assign(shots=lambda d: d["shots"]), "shots", out, "accuracy_vs_shots", logx=True),
    "exp13_ansatz_depth": depth,
    "exp14_optimizer_init": optimizer,
    "exp15_synthetic_claims": lambda df, out: (grouped_bars(df, out), pareto(df, out)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()
    for name, fn in EXPERIMENTS.items():
        path = args.results_dir / name  # directory written by main.py
        if not path.exists():
            print(f"skip {name}: not found")
            continue
        df = load_results(path)
        out = args.output_dir / name
        print(f"plotting {name} -> {out}")
        fn(df, out)


if __name__ == "__main__":
    main()
