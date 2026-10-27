"""Section 7: kernel analysis on corners / checkerboard (non-Ray).

For every ansatz, method (Init, FullKTA, QUACK, CCKA) and seed this records:
global KTA, centred KTA, effective rank (exp of the Shannon entropy of the
normalised eigenvalues), off-diagonal kernel variance (concentration), the
class-block gap (mean in-class minus mean cross-class kernel value) and test
accuracy. For seed 0 it also saves kernel heatmaps (sorted by class),
eigenspectra, off-diagonal violins, pairwise CKA between methods and decision
boundaries. Across all runs it plots global KTA vs test accuracy.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.svm import SVC

from experiments.experiments import (
    ROOT,
    _build_model,
    _make_aligner,
    _prepare_data,
    load_config,
)

LOGGER = logging.getLogger(__name__)
NOISE = {"type": "none", "level": 0.0}
MODES = ("init", "fullKTA", "quackKTA", "ccka")
LABELS = {"init": "Init", "fullKTA": "FullKTA", "quackKTA": "QUACK", "ccka": "CCKA"}
ALLOWED = {"corners", "checkerboard"}


def _center(kernel: np.ndarray) -> np.ndarray:
    n = len(kernel)
    h = np.eye(n) - np.ones((n, n)) / n
    return h @ kernel @ h


def kta(kernel: np.ndarray, y: np.ndarray, centred: bool = False) -> float:
    target = np.outer(y, y).astype(float)
    if centred:
        kernel, target = _center(kernel), _center(target)
    return float(
        (kernel * target).sum()
        / (np.linalg.norm(kernel) * np.linalg.norm(target) + 1e-12)
    )


def effective_rank(kernel: np.ndarray) -> float:
    eig = np.clip(np.linalg.eigvalsh(kernel), 0, None)
    p = eig / max(eig.sum(), 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def linear_cka(k1: np.ndarray, k2: np.ndarray) -> float:
    c1, c2 = _center(k1), _center(k2)
    return float(
        (c1 * c2).sum() / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-12)
    )


def block_gap(kernel: np.ndarray, y: np.ndarray) -> float:
    same = y[:, None] == y[None, :]
    off = ~np.eye(len(y), dtype=bool)
    return float(kernel[same & off].mean() - kernel[~same].mean())


def offdiag(kernel: np.ndarray) -> np.ndarray:
    return kernel[~np.eye(len(kernel), dtype=bool)]


def analyse_run(dataset, ansatz, seed, settings, centroids):
    data = _prepare_data(dataset, seed, settings)
    ytr = np.asarray(data[3])
    results: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        method = "fullKTA" if mode == "init" else mode
        model = _build_model(settings, data[0].shape[1], NOISE, 0, seed, ansatz)
        aligner = _make_aligner(
            method, model, data, settings, seed, centroids=centroids,
            initialization="regular", update_kao=True, update_co=True,
        )
        if mode == "init":
            metrics = aligner.svm_training(aligner.xtrain, aligner.ytrain)
            test_acc = float(
                metrics.get("test_accuracy", metrics["validation_accuracy"])
            )
        else:
            history = aligner.align()
            test_acc = float(history["final_svm_metrics"]["test_accuracy"])
        weights = aligner.weights
        ktr, _, _ = aligner._evaluation_kernel_matrices(
            weights, aligner.xtrain, aligner.xtrain[:1], None
        )
        if ktr.shape[0] != ktr.shape[1]:
            ktr = aligner.regular_kernel_matrix(weights, aligner.xtrain)
        ktr = np.asarray(ktr, dtype=float)
        order = np.argsort(ytr, kind="stable")
        ks = ktr[np.ix_(order, order)]
        ys = ytr[order]
        results[mode] = {
            "kernel": ks, "y": ys, "aligner": aligner, "weights": weights,
            "row": {
                "dataset": dataset, "ansatz": ansatz, "seed": seed,
                "method": mode, "global_kta": kta(ks, ys),
                "centred_kta": kta(ks, ys, centred=True),
                "effective_rank": effective_rank(ks),
                "offdiag_variance": float(offdiag(ks).var()),
                "offdiag_mean": float(offdiag(ks).mean()),
                "class_block_gap": block_gap(ks, ys),
                "test_accuracy": test_acc,
            },
        }
    return data, results


def analyse_task(dataset, ansatz, seed, settings, centroids, out, make_plots):
    """Worker entry point: returns only picklable metric rows."""
    data, results = analyse_run(dataset, ansatz, seed, settings, centroids)
    if make_plots:
        Path(out).mkdir(parents=True, exist_ok=True)
        _plots(dataset, ansatz, data, results, Path(out))
    return [r["row"] for r in results.values()]


def _plots(dataset, ansatz, data, results, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stem = f"{dataset}_{ansatz}"
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, mode in zip(axes, MODES):
        im = ax.imshow(results[mode]["kernel"], vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"{LABELS[mode]} (KTA={results[mode]['row']['global_kta']:.2f})")
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(out / f"kernel_heatmaps_{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for mode in MODES:
        eig = np.clip(np.linalg.eigvalsh(results[mode]["kernel"])[::-1], 1e-12, None)
        ax.plot(eig / eig.sum(), label=LABELS[mode])
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("normalised eigenvalue")
    ax.legend()
    fig.savefig(out / f"eigenspectrum_{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.violinplot([offdiag(results[m]["kernel"]) for m in MODES], showmedians=True)
    ax.set_xticks(range(1, len(MODES) + 1), [LABELS[m] for m in MODES])
    ax.set_ylabel("off-diagonal kernel value")
    fig.savefig(out / f"offdiag_violin_{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    cka = np.array([[linear_cka(results[a]["kernel"], results[b]["kernel"])
                     for b in MODES] for a in MODES])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    im = ax.imshow(cka, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(MODES)), [LABELS[m] for m in MODES], rotation=45)
    ax.set_yticks(range(len(MODES)), [LABELS[m] for m in MODES])
    for i in range(len(MODES)):
        for j in range(len(MODES)):
            ax.text(j, i, f"{cka[i, j]:.2f}", ha="center", va="center", color="w")
    fig.colorbar(im)
    fig.savefig(out / f"cka_similarity_{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    xtr = np.asarray(results["fullKTA"]["aligner"].xtrain)
    ytr = np.asarray(data[3])
    active = np.where(xtr.std(axis=0) > 1e-9)[0]
    if len(active) == 2:
        lo, hi = xtr[:, active].min(0), xtr[:, active].max(0)
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 25), np.linspace(lo[1], hi[1], 25))
        grid = np.zeros((gx.size, xtr.shape[1]), dtype=np.float32)
        grid[:, active[0]], grid[:, active[1]] = gx.ravel(), gy.ravel()
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
        for ax, mode in zip(axes, MODES):
            al, w = results[mode]["aligner"], results[mode]["weights"]
            ktr = np.asarray(al.regular_kernel_matrix(w, al.xtrain), dtype=float)
            kgrid = np.asarray(al.test_kernel_matrix(w, al.xtrain, grid), dtype=float)
            clf = SVC(kernel="precomputed", C=1.0).fit(ktr, ytr)
            zz = clf.predict(kgrid).reshape(gx.shape)
            ax.contourf(gx, gy, zz, alpha=0.3, cmap="coolwarm")
            ax.scatter(xtr[:, active[0]], xtr[:, active[1]], c=ytr, cmap="coolwarm", s=12, edgecolor="k", linewidths=0.3)
            ax.set_title(LABELS[mode])
        fig.savefig(out / f"decision_boundary_{stem}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _summary_plots(rows: list[dict[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for mode in MODES:
        sub = df[df["method"] == mode]
        ax.scatter(sub["global_kta"], sub["test_accuracy"], label=LABELS[mode], alpha=0.7)
    corr = df[["global_kta", "test_accuracy"]].corr().iloc[0, 1]
    ax.set_title(f"Pearson r = {corr:.2f}")
    ax.set_xlabel("global KTA")
    ax.set_ylabel("test accuracy")
    ax.legend()
    fig.savefig(out / "kta_vs_accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    g = df.groupby(["ansatz", "method"])["effective_rank"].mean().unstack()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(g[list(MODES)].values, cmap="viridis")
    ax.set_xticks(range(len(MODES)), [LABELS[m] for m in MODES])
    ax.set_yticks(range(len(g)), g.index)
    fig.colorbar(im, label="effective rank")
    fig.savefig(out / "effective_rank_ansatz_method.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    g = df.groupby(["ansatz", "method"])["offdiag_variance"].mean().unstack()[list(MODES)]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    g.plot.bar(ax=ax, logy=True)
    ax.set_ylabel("off-diagonal variance (concentration)")
    fig.savefig(out / "kernel_concentration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "base.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "kernel_analysis")
    parser.add_argument("--datasets", nargs="+", default=["corners", "checkerboard"])
    parser.add_argument("--ansatze", nargs="+", default=["crz_ring", "hardware_efficient", "iqp", "zz_feature_map", "strongly_entangling", "separable"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--centroids", type=int, default=4)
    args = parser.parse_args()
    bad = set(args.datasets) - ALLOWED
    if bad:
        raise SystemExit(f"Kernel analysis is restricted to corners/checkerboard, got {sorted(bad)}")
    logging.basicConfig(level=logging.INFO)
    settings = dict(load_config(args.config)["training"])
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    settings["max_samples"] = min(int(settings.get("max_samples", 200)), 200)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for ansatz in args.ansatze:
            for seed in args.seeds:
                LOGGER.info("kernel analysis %s %s seed %d", dataset, ansatz, seed)
                rows.extend(analyse_task(
                    dataset, ansatz, seed, settings, args.centroids,
                    args.output_dir, seed == args.seeds[0],
                ))
    write_outputs(rows, args.output_dir)


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "kernel_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _summary_plots(rows, output_dir)


if __name__ == "__main__":
    main()
