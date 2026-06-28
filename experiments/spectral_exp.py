"""
Experiment: Spectral / Effective-Rank analysis of learned kernels
=================================================================

This is the decisive test of the low-rank discriminative alignment theory.

THEORY (see CCKA_behaviour_analysis.md)
---------------------------------------
CCKA achieves high accuracy at low global KTA because it performs IMPLICIT
LOW-RANK kernel alignment: it concentrates label information in the top 1-2
eigenvectors of the kernel while leaving the rest of the spectrum
label-orthogonal. Global KTA penalizes this (its denominator counts all
eigenvalues) but the SVM does not care (it only needs ONE discriminative
direction).

This experiment measures, for every method at every epoch:

  1. Per-eigenvector label alignment:  a_i = |v_i^T y_unit|
     where v_i are the eigenvectors of the centered kernel and
     y_unit = y / ||y||.  a_i in [0,1], a_i=1 means eigenvector i points
     exactly along the label direction.

  2. Eigenvalue spectrum:  lambda_i (sorted descending).

  3. Label-energy concentration:
       top1_energy = lambda_1 * a_1^2 / sum_j(lambda_j * a_j^2)
       top2_energy = (lambda_1 a_1^2 + lambda_2 a_2^2) / sum_j(lambda_j a_j^2)
     Fraction of total label-aligned energy captured by the top 1 / 2 eigenvectors.
     HIGH concentration = low-rank alignment (CCKA hypothesis).

  4. Effective rank of the label-aligned subspace:
       weights w_i = lambda_i * a_i^2  (label-aligned energy per eigenvector)
       p_i = w_i / sum_j w_j
       eff_rank = exp( -sum_i p_i log p_i )   (entropy-based effective rank)
     eff_rank ~ 1  => all label info in one direction (rank-1, CCKA)
     eff_rank large => label info spread across many directions (FullKTA)

  5. Participation ratio (alternative effective rank):
       PR = (sum_i w_i)^2 / sum_i (w_i^2)

  6. KTA decomposition:
       global_kta = sum_i lambda_i a_i^2 / (||K||_F * ||T||_F)
     We report the cumulative contribution of the top-k eigenvectors to KTA,
     showing how much of the (already low) global KTA comes from just the
     leading directions.

PREDICTIONS
-----------
  - CCKA: eff_rank ~ 1-2, top1_energy > 0.7, even at peak accuracy
  - FullKTA: eff_rank >> 2, label energy spread across many eigenvectors
  - At CCKA's collapse (epoch ~64): top eigenvector label alignment a_1 should
    DROP sharply -- the single discriminative direction lost alignment.
  - Plot eff_rank vs accuracy: CCKA should achieve high accuracy at LOW eff_rank,
    proving rank-1 sufficiency. Other methods need higher eff_rank.

Outputs
-------
  {dataset}_spectral_results.csv         per-epoch spectral metrics
  {dataset}_spec1_eff_rank_vs_budget.pdf effective rank over training
  {dataset}_spec2_eff_rank_vs_acc.pdf    effective rank vs accuracy (KEY PLOT)
  {dataset}_spec3_top_alignment.pdf      top-eigenvector label alignment over time
  {dataset}_spec4_spectrum_snapshots.pdf eigenvalue + alignment spectra (init/peak/final)
  {dataset}_spec5_kta_decomposition.pdf  cumulative KTA contribution by rank
  {dataset}_spectral_full.pdf            combined figure

Usage
-----
  python exp_spectral.py --dataset checkerboard
  python exp_spectral.py --dataset checkerboard --dry-run
  python exp_spectral.py --dataset checkerboard --results-csv checkerboard_spectral_results.csv
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Backend import
# ---------------------------------------------------------------------------

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

if not DRY_RUN:
    try:
        import jax
        import jax.numpy as jnp
        import optax as ox
        from ccka.models.kernel import KernelModel
        from ccka.circuits.angleEmbeddingKernel import quackEmbeddingCircuit
        from ccka.aligner.kta import (
            fullKTA, centroidBasedKTA, quackKTA, randomKTA, greedyKTA
        )
        from sklearn.svm import SVC
        BACKEND_AVAILABLE = True
    except ImportError:
        print("[WARNING] ccka backend not importable -- switching to DRY_RUN mode.")
        BACKEND_AVAILABLE = False
        DRY_RUN = True
else:
    BACKEND_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_PATHS = {
    "corners":      "../data/corners.npy",
    "checkerboard": "../data/checkerboard_dataset.npy",
    "donuts":       "../data/donuts.npy",
}

DEFAULT_CENTROIDS = 4
EPOCHS = 100
ALL_METHODS = ["fullKTA", "randomKTA", "quackKTA", "centroidBasedKTA", "greedyKTA"]

METHOD_COLORS = {
    "fullKTA":          "#185FA5",
    "randomKTA":        "#639922",
    "quackKTA":         "#0F6E56",
    "centroidBasedKTA": "#D85A30",
    "greedyKTA":        "#BA7517",
}
METHOD_LABELS = {
    "fullKTA":          "FullKTA",
    "randomKTA":        "RandomKTA",
    "quackKTA":         "QuackKTA",
    "centroidBasedKTA": "CCKA (ours)",
    "greedyKTA":        "GreedyKTA",
}
CCKA = "centroidBasedKTA"


# ---------------------------------------------------------------------------
# Core spectral metrics (pure numpy)
# ---------------------------------------------------------------------------

def spectral_metrics(K: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """
    Compute the full suite of spectral / effective-rank metrics for a centered
    kernel matrix K against labels y.

    Returns a dict of scalar metrics (see module docstring for definitions).
    """
    N = len(y)
    y_unit = y.astype(float) / (np.linalg.norm(y) + 1e-12)

    # Symmetrize for numerical safety, then eigendecompose
    Ksym = 0.5 * (K + K.T)
    eigvals, eigvecs = np.linalg.eigh(Ksym)

    # Sort descending
    order   = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Clip tiny/negative eigenvalues to 0 for energy accounting
    lam = np.clip(eigvals, 0.0, None)

    # Per-eigenvector label alignment a_i = |v_i^T y_unit|
    a = np.abs(eigvecs.T @ y_unit)          # shape (N,)

    # Label-aligned energy per eigenvector: w_i = lambda_i * a_i^2
    w = lam * (a ** 2)
    w_sum = w.sum() + 1e-12
    p = w / w_sum

    # Effective rank (entropy-based) of the label-aligned subspace
    nz = p > 1e-12
    eff_rank = float(np.exp(-np.sum(p[nz] * np.log(p[nz]))))

    # Participation ratio
    pr = float((w.sum() ** 2) / (np.sum(w ** 2) + 1e-12))

    # Top-k label energy concentration
    top1_energy = float(p[0]) if N >= 1 else 0.0
    top2_energy = float(p[0] + p[1]) if N >= 2 else top1_energy
    top3_energy = float(p[:3].sum()) if N >= 3 else top2_energy

    # Global KTA from spectrum (should match direct computation)
    # KTA = sum_i lambda_i a_i^2 / (||K||_F * ||T||_F)
    # ||K||_F = sqrt(sum lambda_i^2) (using signed eigenvalues),  ||T||_F = N
    K_fro = np.sqrt(np.sum(eigvals ** 2)) + 1e-12
    T_fro = float(N)
    kta_spectral = float(np.sum(eigvals * (a ** 2)) / (K_fro * T_fro))

    # Cumulative KTA fraction from top-k eigenvectors
    kta_contrib = eigvals * (a ** 2) / (K_fro * T_fro)
    kta_top1_frac = float(kta_contrib[0] / (kta_spectral + 1e-12)) if N >= 1 else 0.0
    kta_top2_frac = float(kta_contrib[:2].sum() / (kta_spectral + 1e-12)) if N >= 2 else kta_top1_frac

    return {
        "eff_rank":        eff_rank,
        "participation_ratio": pr,
        "top1_label_energy": top1_energy,
        "top2_label_energy": top2_energy,
        "top3_label_energy": top3_energy,
        "top1_alignment":  float(a[0]),         # |v_1 . y|
        "top2_alignment":  float(a[1]) if N >= 2 else 0.0,
        "max_alignment":   float(a.max()),      # best-aligned eigenvector
        "argmax_alignment_rank": int(np.argmax(a)),  # which eigenvector is most aligned
        "kta_spectral":    kta_spectral,
        "kta_top1_frac":   kta_top1_frac,
        "kta_top2_frac":   kta_top2_frac,
        "lambda1":         float(eigvals[0]),
        "lambda2":         float(eigvals[1]) if N >= 2 else 0.0,
        "spectral_decay":  float(eigvals[0] / (abs(eigvals[1]) + 1e-12)) if N >= 2 else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-epoch training + spectral analysis
# ---------------------------------------------------------------------------

def run_spectral_experiment(
    method: str, dataset: str, dataset_path: str,
    centroids: int, num_iterations: int,
) -> tuple[list[dict[str, Any]], dict]:
    """
    Train one method, computing spectral metrics at every epoch.

    Returns (rows, snapshot_store) where:
      rows = per-epoch dicts of scalar metrics
      snapshot_store = full eigenvalue + alignment arrays at init/peak/final
                       for the spectrum-snapshot plot
    """
    import jax.numpy as jnp

    data = np.load(dataset_path, allow_pickle=True).item()
    X = jnp.asarray(np.concatenate([data["x_train"], data["x_test"]], axis=0))
    y = jnp.asarray(np.concatenate([data["y_train"], data["y_test"]], axis=0))

    kernel = quackEmbeddingCircuit(num_qubits=5, reps=6, reupload=True)
    model  = KernelModel(circuit=kernel)
    common = dict(kernel_model=model, data=X, labels=y,
                  matrix_type="regular", split_size=0.5)

    if method == "fullKTA":
        aligner = fullKTA(**common, learning_rate=0.1, optimizer="adam", epochs=num_iterations)
    elif method == "randomKTA":
        aligner = randomKTA(**common, random_samples=centroids, landmark_points=10,
                            learning_rate=0.1, optimizer="adam", epochs=num_iterations)
    elif method == "greedyKTA":
        aligner = greedyKTA(**common, greedy_samples=centroids, landmark_points=10,
                            learning_rate=0.1, optimizer="adam", epochs=num_iterations)
    elif method == "quackKTA":
        aligner = quackKTA(**common, centroids=centroids, clustering="regular",
                           lambda_co=0.001, lambda_kao=0.001, epochs=num_iterations)
    elif method == "centroidBasedKTA":
        aligner = centroidBasedKTA(**common, clustering="regular", centroids=centroids,
                                   learning_rate=0.2, centroid_lr=0.01, sub_centroid_lr=0.01,
                                   lambda_co=0.001, lambda_kao=0.001, epochs=num_iterations)
    else:
        raise ValueError(method)

    print(f"  Training {METHOD_LABELS[method]} for {num_iterations} epochs...")

    rows = []
    snapshots = {}  # epoch -> (eigvals, alignments)
    y_np = np.asarray(aligner.ytrain)

    # Reset to ensure clean per-epoch trajectory mirroring the aligner's rule
    aligner.weights   = aligner.kernel_model.circuit.init_weights()
    aligner.opt_state = aligner._optimizer.init(aligner.weights)
    if hasattr(aligner, "_kao_weight_opt_state"):
        aligner._kao_weight_opt_state = aligner._kao_weight_optimizer.init(aligner.weights)
    if hasattr(aligner, "_kao_sub_opt_state"):
        aligner._kao_sub_opt_state = aligner._kao_sub_optimizer.init(aligner.sub_centroids)
    if hasattr(aligner, "_co_main_opt_state"):
        aligner._co_main_opt_state = aligner._co_main_optimizer.init(aligner.main_centroids)
    if hasattr(aligner, "_kao_opt_state") and not hasattr(aligner, "_kao_weight_opt_state"):
        aligner._kao_opt_state = aligner._kao_optimizer.init(aligner.weights)
    if hasattr(aligner, "_co_opt_state"):
        aligner._co_opt_state = aligner._co_optimizer.init(aligner.main_centroids)

    unique_labels = np.unique(y_np)
    n_cls = len(unique_labels)
    y_raw = aligner.sub_centroid_labels if hasattr(aligner, "sub_centroid_labels") else None

    best_acc = -1.0
    best_epoch = 0

    for epoch in range(num_iterations):
        # ── Replicate the aligner's per-epoch update rule ──────────────────
        if method == "centroidBasedKTA":
            cl_kao = unique_labels[epoch % n_cls]
            main_idx = int(jnp.argmax(aligner.main_centroid_labels == cl_kao))
            main_centroid = aligner.main_centroids[main_idx]
            for _ in range(10):
                aligner.weights, aligner.sub_centroids = aligner._kao_joint_update(
                    main_centroid, y_raw, l=float(cl_kao))
            main_centroid = aligner.main_centroids[main_idx]
            for _ in range(10):
                aligner.main_centroids = aligner._co_main_update(
                    cl=cl_kao, y_raw=y_raw, l=-float(cl_kao))
        elif method == "quackKTA":
            cl_kao = unique_labels[epoch % n_cls]
            main_idx = int(jnp.argmax(aligner.main_centroid_labels == cl_kao))
            main_centroid = aligner.main_centroids[main_idx]
            y_kao = jnp.where(aligner.ytrain == cl_kao, 1.0, -1.0)
            if epoch % 2 == 0:
                for _ in range(10):
                    aligner.weights = aligner._kao_weight_update_quack(
                        main_centroid, aligner.xtrain, y_kao, l=float(cl_kao))
            else:
                for _ in range(10):
                    aligner.main_centroids = aligner._main_centroid_gradient_update_quack(
                        main_idx, aligner.xtrain, y_kao, l=-float(cl_kao))
        else:
            X_b, y_b = aligner._get_batch(epoch)
            grads = aligner._grad_fn(aligner.weights, X_b, y_b)
            updates, aligner.opt_state = aligner._optimizer.update(grads, aligner.opt_state)
            aligner.weights = ox.apply_updates(aligner.weights, updates)

        # ── Build centered kernel + SVM ───────────────────────────────────
        K_train = np.asarray(
            aligner._apply_centering(aligner._kernel_matrix(aligner.weights, aligner.xtrain)))
        svm = SVC(kernel="precomputed", C=1.0, max_iter=10_000)
        svm.fit(K_train, y_np)
        K_test = np.asarray(
            aligner.test_kernel_matrix(aligner.weights, aligner.xtrain, aligner.xtest))
        test_acc = float(np.mean(svm.predict(K_test) == np.asarray(aligner.ytest)))

        # ── Spectral metrics ──────────────────────────────────────────────
        sm = spectral_metrics(K_train, y_np)
        sm.update({
            "method": method, "dataset": dataset, "epoch": epoch + 1,
            "centroids": centroids, "test_accuracy": test_acc,
            "n_sv": int(len(svm.support_)),
        })
        rows.append(sm)

        if test_acc > best_acc:
            best_acc, best_epoch = test_acc, epoch + 1

        # Save spectrum snapshots at init / mid / will-fill-peak / final
        if epoch == 0 or epoch == num_iterations // 2 or epoch == num_iterations - 1:
            Ksym = 0.5 * (K_train + K_train.T)
            ev, evec = np.linalg.eigh(Ksym)
            o = np.argsort(ev)[::-1]
            y_unit = y_np.astype(float) / (np.linalg.norm(y_np) + 1e-12)
            snapshots[epoch + 1] = (ev[o].copy(), np.abs((evec[:, o].T @ y_unit)).copy())

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}/{num_iterations}  acc={test_acc:.3f}  "
                  f"eff_rank={sm['eff_rank']:.2f}  top1_energy={sm['top1_label_energy']:.3f}  "
                  f"top1_align={sm['top1_alignment']:.3f}")

    return rows, snapshots


# ---------------------------------------------------------------------------
# Run all methods
# ---------------------------------------------------------------------------

_snapshot_store: dict[str, dict] = {}

def run_all(dataset: str) -> pd.DataFrame:
    dataset_path = DATASET_PATHS[dataset]
    all_rows = []
    for method in ALL_METHODS:
        print(f"\n[{METHOD_LABELS[method]}]")
        rows, snaps = run_spectral_experiment(
            method, dataset, dataset_path, DEFAULT_CENTROIDS, EPOCHS)
        all_rows.extend(rows)
        _snapshot_store[method] = snaps
    df = pd.DataFrame(all_rows)
    df.to_csv(f"{dataset}_spectral_results.csv", index=False)
    print(f"\n[Saved] {dataset}_spectral_results.csv  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _smooth(y, w=5):
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(np.pad(y, (w//2, w//2), mode="edge"), k, mode="valid")[:len(y)]


def plot_eff_rank_vs_budget(df, dataset, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))
    for m in ALL_METHODS:
        sub = df[df["method"] == m].sort_values("epoch")
        if sub.empty:
            continue
        lw = 2.5 if m == CCKA else 1.5
        ls = "-" if m == CCKA else "--"
        ax.plot(sub["epoch"], _smooth(sub["eff_rank"]),
                color=METHOD_COLORS[m], label=METHOD_LABELS[m], linewidth=lw, linestyle=ls)
    ax.axhline(1.0, color="gray", ls=":", lw=1.0, label="rank-1 (pure low-rank)")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Effective rank of label-aligned subspace", fontsize=10)
    ax.set_title(f"Effective rank over training — {dataset}\n"
                 "CCKA expected near 1 (rank-1); FullKTA much higher", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    if standalone:
        plt.tight_layout(); plt.savefig(f"{dataset}_spec1_eff_rank_vs_budget.pdf", dpi=150, bbox_inches="tight"); plt.close()


def plot_eff_rank_vs_accuracy(df, dataset, ax=None):
    """KEY PLOT: low effective rank + high accuracy = rank-1 sufficiency proof."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))
    for m in ALL_METHODS:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        ms = 70 if m == CCKA else 25
        mk = "D" if m == CCKA else "o"
        al = 0.8 if m == CCKA else 0.4
        ax.scatter(sub["eff_rank"], sub["test_accuracy"],
                   c=METHOD_COLORS[m], label=METHOD_LABELS[m], s=ms, marker=mk, alpha=al, linewidths=0)
    ax.set_xlabel("Effective rank of label-aligned subspace", fontsize=11)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title(f"Effective rank vs accuracy — {dataset}\n"
                 "CCKA (◆) achieving high accuracy at LOW eff-rank = rank-1 sufficiency",
                 fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    if standalone:
        plt.tight_layout(); plt.savefig(f"{dataset}_spec2_eff_rank_vs_acc.pdf", dpi=150, bbox_inches="tight"); plt.close()


def plot_top_alignment(df, dataset, ax=None):
    """Top-eigenvector label alignment over time. CCKA collapse = sharp drop."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))
    for m in ALL_METHODS:
        sub = df[df["method"] == m].sort_values("epoch")
        if sub.empty:
            continue
        lw = 2.5 if m == CCKA else 1.5
        ls = "-" if m == CCKA else "--"
        ax.plot(sub["epoch"], _smooth(sub["max_alignment"]),
                color=METHOD_COLORS[m], label=METHOD_LABELS[m], linewidth=lw, linestyle=ls)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Max eigenvector–label alignment  max_i |v_i·ŷ|", fontsize=10)
    ax.set_title(f"Best-aligned eigenvector over training — {dataset}\n"
                 "CCKA collapse (~ep64) should show a sharp drop here", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25); ax.set_ylim(0, 1.02)
    if standalone:
        plt.tight_layout(); plt.savefig(f"{dataset}_spec3_top_alignment.pdf", dpi=150, bbox_inches="tight"); plt.close()


def plot_spectrum_snapshots(dataset):
    """Eigenvalue spectrum + per-eigenvector label alignment at init/mid/final."""
    if not _snapshot_store:
        print("[spectrum snapshots] no live snapshot data -- skipping (run live).")
        return
    methods = [m for m in ALL_METHODS if m in _snapshot_store and _snapshot_store[m]]
    n = len(methods)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.6 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for ri, m in enumerate(methods):
        snaps = _snapshot_store[m]
        keys = sorted(snaps.keys())
        labels = ["Init", "Mid", "Final"][:len(keys)]
        for ci, (ep, lab) in enumerate(zip(keys, labels)):
            ax = axes[ri, ci]
            eigvals, align = snaps[ep]
            idx = np.arange(1, len(eigvals) + 1)
            # eigenvalue magnitude (bars) + alignment (line, twin axis)
            ax.bar(idx, np.clip(eigvals, 0, None), color=METHOD_COLORS[m], alpha=0.4, width=0.9)
            ax.set_yscale("log")
            ax.set_ylabel("eigenvalue (log)", fontsize=8, color=METHOD_COLORS[m])
            ax2 = ax.twinx()
            ax2.plot(idx, align, color="black", linewidth=1.4, marker="o", markersize=2)
            ax2.set_ylabel("|v·ŷ|", fontsize=8)
            ax2.set_ylim(0, 1.02)
            ax.set_xlabel("eigenvector rank", fontsize=8)
            ax.set_title(f"{lab} (epoch {ep})", fontsize=9)
            if ci == 0:
                ax.text(-0.28, 0.5, METHOD_LABELS[m], transform=ax.transAxes,
                        fontsize=10, fontweight="bold", color=METHOD_COLORS[m],
                        rotation=90, va="center")
    fig.suptitle(f"Eigenvalue spectrum (bars) + label alignment (line) — {dataset}\n"
                 "CCKA: label alignment concentrated in few eigenvectors (low-rank)",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    plt.savefig(f"{dataset}_spec4_spectrum_snapshots.pdf", dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Saved] {dataset}_spec4_spectrum_snapshots.pdf")


def plot_kta_decomposition(df, dataset, ax=None):
    """How much of global KTA comes from just the top-1 / top-2 eigenvectors."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))
    for m in ALL_METHODS:
        sub = df[df["method"] == m].sort_values("epoch")
        if sub.empty:
            continue
        lw = 2.5 if m == CCKA else 1.5
        ax.plot(sub["epoch"], _smooth(sub["kta_top1_frac"]),
                color=METHOD_COLORS[m], linewidth=lw, linestyle="-",
                label=f"{METHOD_LABELS[m]} top-1")
        ax.plot(sub["epoch"], _smooth(sub["kta_top2_frac"]),
                color=METHOD_COLORS[m], linewidth=lw*0.7, linestyle="--", alpha=0.6,
                label=f"{METHOD_LABELS[m]} top-2")
    ax.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Fraction of global KTA from top-k eigenvectors", fontsize=10)
    ax.set_title(f"KTA decomposition by rank — {dataset}\n"
                 "Near 1.0 = essentially all KTA in the top 1-2 directions (low-rank)",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.25)
    if standalone:
        plt.tight_layout(); plt.savefig(f"{dataset}_spec5_kta_decomposition.pdf", dpi=150, bbox_inches="tight"); plt.close()


def plot_full(df, dataset):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Spectral / Effective-Rank Analysis — {dataset.upper()}\n"
                 "Testing the low-rank discriminative alignment theory of CCKA",
                 fontsize=13, y=0.99)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    plot_eff_rank_vs_budget(df, dataset, ax=fig.add_subplot(gs[0, 0]))
    plot_eff_rank_vs_accuracy(df, dataset, ax=fig.add_subplot(gs[0, 1]))
    plot_top_alignment(df, dataset, ax=fig.add_subplot(gs[1, 0]))
    plot_kta_decomposition(df, dataset, ax=fig.add_subplot(gs[1, 1]))
    plt.savefig(f"{dataset}_spectral_full.pdf", dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Saved] {dataset}_spectral_full.pdf")


# ---------------------------------------------------------------------------
# Synthetic data for dry-run
# ---------------------------------------------------------------------------

def _synthetic(dataset):
    rng = np.random.default_rng(0)
    rows = []
    def sig(t, x0, k): return 1.0/(1.0+np.exp(-k*(t-x0)))
    ep = np.arange(1, EPOCHS+1); t = (ep-1)/(EPOCHS-1)
    specs = {
        # CCKA: low eff_rank (~1.5), high top1 energy, collapse at ep64
        "centroidBasedKTA": dict(
            acc=np.where(t<0.64, 0.5+0.5*sig(t,0.35,9), 0.5),
            eff_rank=1.2+0.6*sig(t,0.4,6), top1=0.78-0.05*np.sin(t*20),
            align=np.where(t<0.64, 0.55+0.3*sig(t,0.3,8), 0.25)),
        "fullKTA": dict(
            acc=0.55+0.44*sig(t,0.45,8), eff_rank=2.5+5.0*sig(t,0.5,5),
            top1=0.45+0.1*sig(t,0.5,5), align=0.5+0.35*sig(t,0.4,6)),
        "quackKTA": dict(
            acc=0.52+0.31*sig(t,0.4,8), eff_rank=2.0+3.0*sig(t,0.5,5),
            top1=0.5+0.15*sig(t,0.5,5), align=0.45+0.3*sig(t,0.45,6)),
        "randomKTA": dict(
            acc=0.5+0.45*sig(t,0.55,7), eff_rank=2.2+3.5*sig(t,0.55,5),
            top1=0.5+0.12*sig(t,0.55,5), align=0.45+0.3*sig(t,0.55,6)),
        "greedyKTA": dict(
            acc=0.5+0.23*sig(t,0.45,7), eff_rank=1.8+2.0*sig(t,0.5,5),
            top1=0.55+0.1*sig(t,0.5,5), align=0.45+0.25*sig(t,0.5,6)),
    }
    for m, sp in specs.items():
        for i in range(EPOCHS):
            acc = float(np.clip(sp["acc"][i] + rng.normal(0,0.03), 0, 1))
            er  = float(max(1.0, sp["eff_rank"][i] + rng.normal(0,0.15)))
            t1  = float(np.clip(sp["top1"][i] + rng.normal(0,0.03), 0, 1))
            al  = float(np.clip(sp["align"][i] + rng.normal(0,0.03), 0, 1))
            rows.append(dict(
                method=m, dataset=dataset, epoch=i+1, centroids=DEFAULT_CENTROIDS,
                test_accuracy=acc, n_sv=20, eff_rank=er, participation_ratio=er*1.2,
                top1_label_energy=t1, top2_label_energy=min(1.0,t1+0.12),
                top3_label_energy=min(1.0,t1+0.18), top1_alignment=al,
                top2_alignment=al*0.6, max_alignment=al, argmax_alignment_rank=0,
                kta_spectral=0.1+0.4*al, kta_top1_frac=t1, kta_top2_frac=min(1.0,t1+0.12),
                lambda1=1.0, lambda2=0.5, spectral_decay=2.0))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Spectral / effective-rank analysis of learned kernels.")
    ap.add_argument("--dataset", default="checkerboard", choices=list(DATASET_PATHS))
    ap.add_argument("--results-csv", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    global DRY_RUN
    if args.dry_run:
        DRY_RUN = True
    dataset = args.dataset

    if args.results_csv:
        print(f"[Loading {args.results_csv}]")
        df = pd.read_csv(args.results_csv)
    elif not DRY_RUN:
        df = run_all(dataset)
    else:
        print("[DRY_RUN] generating synthetic spectral data...")
        df = _synthetic(dataset)

    # Quick numeric summary of the key claim
    print(f"\n{'='*70}")
    print("  KEY METRIC: effective rank at best accuracy per method")
    print(f"{'='*70}")
    for m in ALL_METHODS:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        best = sub.loc[sub["test_accuracy"].idxmax()]
        print(f"  {METHOD_LABELS[m]:14s}: best_acc={best['test_accuracy']:.3f}  "
              f"eff_rank={best['eff_rank']:.2f}  top1_energy={best['top1_label_energy']:.3f}")
    print(f"{'='*70}\n")

    plot_eff_rank_vs_budget(df, dataset)
    plot_eff_rank_vs_accuracy(df, dataset)
    plot_top_alignment(df, dataset)
    plot_kta_decomposition(df, dataset)
    plot_spectrum_snapshots(dataset)
    plot_full(df, dataset)
    print("\nDone.")


if __name__ == "__main__":
    main()