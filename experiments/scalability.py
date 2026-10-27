"""Section 3: measured scalability (non-Ray, single node).

Sweeps the training-set size N, qubit count, layer count and Nystrom landmark
count, recording circuit evaluations per epoch, wall-clock per epoch, and the
split of the full pipeline into alignment / kernel construction / inference.
Only corners and checkerboard are used (scalability is an "extra" study).
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.svm import SVC

from experiments.experiments import (
    ROOT,
    _build_model,
    _make_aligner,
    _prepare_data,
    _run_one,
    load_config,
)

LOGGER = logging.getLogger(__name__)
METHODS = ("fullKTA", "quackKTA", "ccka")
NOISE = {"type": "none", "level": 0.0}


def _timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def measure(
    dataset: str,
    method: str,
    settings: dict[str, Any],
    seed: int,
    centroids: int,
    ansatz: str,
) -> dict[str, Any]:
    data = _prepare_data(dataset, seed, settings)
    model = _build_model(settings, data[0].shape[1], NOISE, 0, seed, ansatz)
    aligner = _make_aligner(
        method, model, data, settings, seed, centroids=centroids,
        initialization="regular", update_kao=True, update_co=True,
    )
    epochs = int(settings["epochs"])
    history, align_seconds = _timed(aligner.align)
    train_executions = int(history["circuit_executions"])

    weights = aligner.weights
    xtr, xte = aligner.xtrain, aligner.xtest
    (ktr, _, _), t_train_kernel = _timed(
        lambda: aligner._evaluation_kernel_matrices(weights, xtr, xte[:1], None)
    )
    (_, kte, _), t_all = _timed(
        lambda: aligner._evaluation_kernel_matrices(weights, xtr, xte, None)
    )
    t_inference_kernel = max(t_all - t_train_kernel, 0.0)
    ktr, kte = np.asarray(ktr), np.asarray(kte)
    clf = SVC(kernel="precomputed", C=float(settings.get("svm_c", 1.0)))
    _, t_svm_fit = _timed(lambda: clf.fit(ktr, np.asarray(aligner.ytrain)))
    _, t_svm_predict = _timed(lambda: clf.predict(kte))
    return {
        "dataset": dataset,
        "method": method,
        "ansatz": ansatz,
        "seed": seed,
        "n_train": int(len(xtr)),
        "n_test": int(len(xte)),
        "n_qubits": int(data[0].shape[1]),
        "reps": int(settings["reps"]),
        "centroids": centroids if method == "ccka" else 0,
        "inference_landmarks": settings.get("inference_landmarks"),
        "epochs": epochs,
        "train_circuit_executions": train_executions,
        "executions_per_epoch": train_executions / max(epochs, 1),
        "seconds_per_epoch": align_seconds / max(epochs, 1),
        "total_train_seconds": align_seconds,
        "kernel_construction_seconds": t_train_kernel + t_svm_fit,
        "inference_seconds": t_inference_kernel + t_svm_predict,
        "inference_kernel_evaluations": int(len(xte) * (
            len(xtr)
            if settings.get("inference_landmarks") == "all"
            else min(int(settings.get("inference_landmarks", 32)), len(xtr))
        )),
        "test_accuracy": float(history["final_svm_metrics"]["test_accuracy"]),
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    """One independent measurement per task so they can be farmed out."""
    config = load_config(args.config)
    base = dict(config["training"])
    base.update(epochs=args.epochs, n_qubits=args.qubits, reps=args.layers)
    sweeps: list[tuple[str, dict[str, Any]]] = []
    sweeps += [("n_train", {"n_train": n, "max_samples": 2 * n + 50})
               for n in args.n_values]
    sweeps += [("n_qubits", {"n_qubits": q}) for q in args.qubit_values]
    sweeps += [("reps", {"reps": r}) for r in args.layer_values]
    sweeps += [("landmarks", {"inference_landmarks": m})
               for m in args.landmark_values]
    tasks: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for sweep, overrides in sweeps:
            settings = {**base, "max_samples": max(args.n_train * 2 + 50, 200),
                        "n_train": args.n_train, **overrides}
            for method in METHODS:
                if sweep == "landmarks" and method == "quackKTA":
                    continue
                # FullKTA needs O(N^2) circuits per epoch: skip beyond the cap.
                if (
                    method == "fullKTA"
                    and overrides.get("n_train", 0) > args.max_full_n
                ):
                    continue
                for seed in args.seeds:
                    tasks.append({
                        "dataset": dataset, "method": method,
                        "settings": settings, "seed": seed,
                        "centroids": args.centroids, "ansatz": args.ansatz,
                        "sweep": sweep,
                    })
    return tasks


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    task = dict(task)
    sweep = task.pop("sweep")
    row = measure(**task)
    row["sweep"] = sweep
    return row


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in build_tasks(args):
        LOGGER.info("%s %s %s", task["dataset"], task["sweep"], task["method"])
        rows.append(run_task(task))
        _write(args.output_dir / "scalability.csv", rows)
    return rows


def plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(rows)
    colors = {"fullKTA": "tab:blue", "quackKTA": "tab:orange", "ccka": "tab:green"}

    def lines(sweep, x, y, ylabel, name, logx=True, logy=True, methods=METHODS):
        sub = df[df["sweep"] == sweep]
        if sub.empty:
            return
        fig, ax = plt.subplots(figsize=(5, 4))
        for method in methods:
            m = sub[sub["method"] == method]
            if m.empty:
                continue
            g = m.groupby(x)[y].agg(["mean", "std"]).fillna(0)
            ax.errorbar(g.index, g["mean"], g["std"], marker="o",
                        label=method, color=colors[method], capsize=2)
        ax.set_xlabel(x)
        ax.set_ylabel(ylabel)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.legend()
        fig.savefig(output_dir / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    lines("n_train", "n_train", "executions_per_epoch",
          "circuit evaluations / epoch", "scal_executions_vs_N")
    lines("n_train", "n_train", "seconds_per_epoch", "seconds / epoch",
          "scal_time_per_epoch_vs_N")
    lines("n_train", "n_train", "total_train_seconds", "total training seconds",
          "scal_total_time_vs_N")
    lines("n_qubits", "n_qubits", "seconds_per_epoch", "seconds / epoch",
          "scal_time_vs_qubits", logx=False)
    lines("reps", "reps", "seconds_per_epoch", "seconds / epoch",
          "scal_time_vs_layers", logx=False)
    lines("landmarks", "inference_landmarks", "test_accuracy",
          "test accuracy", "scal_accuracy_vs_nystrom_landmarks",
          logy=False, methods=("fullKTA", "ccka"))

    sub = df[df["sweep"] == "n_train"]
    if not sub.empty:
        stages = ["total_train_seconds", "kernel_construction_seconds",
                  "inference_seconds"]
        g = sub.groupby(["method", "n_train"])[stages].mean()
        fig, ax = plt.subplots(figsize=(9, 4))
        labels = [f"{m}\nN={n}" for m, n in g.index]
        bottom = np.zeros(len(g))
        for stage in stages:
            ax.bar(labels, g[stage].values, bottom=bottom,
                   label=stage.replace("_seconds", "").replace("_", " "))
            bottom += g[stage].values
        ax.set_ylabel("seconds")
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=60, fontsize=7)
        fig.savefig(output_dir / "scal_pipeline_stacked.png", dpi=180,
                    bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "base.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "scalability")
    parser.add_argument("--datasets", nargs="+", default=["corners", "checkerboard"])
    parser.add_argument("--ansatz", default="crz_ring")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--centroids", type=int, default=4)
    parser.add_argument("--qubits", type=int, default=5)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--n-train", type=int, default=64,
                        help="N used while sweeping qubits/layers/landmarks.")
    parser.add_argument("--max-full-n", type=int, default=1000,
                        help="Largest N at which FullKTA is measured.")
    parser.add_argument("--n-values", type=int, nargs="+", default=[100, 500, 1000, 5000, 10000])
    parser.add_argument("--qubit-values", type=int, nargs="+", default=[3, 5, 8, 10])
    parser.add_argument("--layer-values", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10])
    parser.add_argument("--landmark-values", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run(args)
    plot(rows, args.output_dir)


if __name__ == "__main__":
    main()
