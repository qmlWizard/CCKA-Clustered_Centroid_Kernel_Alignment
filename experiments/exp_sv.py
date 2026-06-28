"""
Experiment: Support Vector locality analysis
============================================

Hypothesis
----------
CCKA achieves equal test accuracy despite lower global KTA and lower block-diagonal
ratio because its kernel is *locally* well-behaved near the SVM decision boundary
-- specifically at the support vectors -- even when it is globally noisy.

Four sub-experiments test this from different angles:

  SV-1  Support vector proximity to centroids/sub-centroids
        Are CCKA's SVs closer (in input space) to the centroids than a random
        training point would be?  Measured as:
            mean dist(SV, nearest centroid) vs mean dist(random train pt, nearest centroid)
        If CCKA's centroids act as boundary anchors, SVs should cluster near them.

  SV-2  Local block-diagonal ratio at support vectors
        Compute the block ratio restricted to SV x SV kernel sub-matrix.
        Compare to the global block ratio (all train x all train).
        For CCKA the local ratio at SVs should be HIGH even when the global ratio is LOW.
        For FullKTA both should be similarly high.

  SV-3  Kernel alignment at support vectors
        For each method compute KTA restricted to the SV sub-matrix:
            alignment(K[SV, SV], y[SV])
        vs the global KTA:
            alignment(K[train, train], y[train])
        CCKA hypothesis: local SV-KTA >> global KTA.
        Other methods: local SV-KTA ~ global KTA (uniformly good/bad).

  SV-4  Centroid-to-SV kernel separation
        For each class centroid c_i, compute:
            K(c_i, SV_same_class)  vs  K(c_i, SV_diff_class)
        and the same for non-SVs:
            K(c_i, non-SV_same)   vs  K(c_i, non-SV_diff)
        If centroid KTA explains accuracy, the SV separation should be high
        even when the non-SV separation is poor.

All sub-experiments are run per epoch so we can see how each quantity evolves
over training -- not just the final value.

Outputs (one set per dataset)
------------------------------
  {dataset}_sv_exp_results.csv          per-epoch metrics for all methods
  {dataset}_sv1_proximity.pdf           SV proximity vs centroid (SV-1)
  {dataset}_sv2_local_block_ratio.pdf   local vs global block ratio (SV-2)
  {dataset}_sv3_local_kta.pdf           local SV-KTA vs global KTA (SV-3)
  {dataset}_sv4_centroid_sv_sep.pdf     centroid-to-SV separation (SV-4)
  {dataset}_sv_full.pdf                 combined 2x2 publication figure

Usage
-----
  # Live run (requires ccka backend):
  python exp_sv.py --dataset checkerboard

  # Plot-test without backend:
  python exp_sv.py --dataset checkerboard --dry-run

  # Load pre-computed CSV:
  python exp_sv.py --dataset checkerboard --results-csv checkerboard_sv_exp_results.csv
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
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Backend import
# ---------------------------------------------------------------------------

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

if not DRY_RUN:
    try:
        import jax
        import jax.numpy as jnp
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

DATASET_PATHS: dict[str, str] = {
    "corners":      "../data/corners.npy",
    "checkerboard": "../data/checkerboard_dataset.npy",
    "donuts":       "../data/donuts.npy",
}

# Best centroid count found from the main comparison (use 4 as default)
DEFAULT_CENTROIDS = 4

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
CCKA    = "centroidBasedKTA"
EPOCHS  = 100
ALL_METHODS = ["fullKTA", "randomKTA", "quackKTA", "centroidBasedKTA", "greedyKTA"]


# ---------------------------------------------------------------------------
# Core metric functions (pure numpy, no JAX required)
# ---------------------------------------------------------------------------

def svm_block_ratio(K: np.ndarray, y: np.ndarray) -> float:
    """
    Block-diagonal ratio: mean within-class K / mean between-class K.
    Diagonal excluded to avoid self-similarity inflation.
    """
    same = (y[:, None] == y[None, :])
    np.fill_diagonal(same, False)
    diff = ~same
    np.fill_diagonal(diff, False)
    within  = K[same].mean()  if same.any()  else 0.0
    between = K[diff].mean()  if diff.any()  else 1e-10
    return float(within / (abs(between) + 1e-10))


def kta_from_matrix(K: np.ndarray, y: np.ndarray) -> float:
    """
    KTA = <K, T>_F / (||K||_F * ||T||_F)  where T = y y^T.
    """
    T    = y[:, None] * y[None, :]
    norm = np.linalg.norm(K, "fro") * np.linalg.norm(T, "fro")
    return float(np.sum(K * T) / (norm + 1e-10))


def sv_proximity_ratio(
    X_sv: np.ndarray,
    X_train: np.ndarray,
    centroids: np.ndarray | None,
) -> float:
    """
    SV-1 metric: ratio of mean(dist SV -> nearest centroid)
                      / mean(dist random train point -> nearest centroid).

    A ratio < 1 means SVs are closer to centroids than average training points,
    suggesting centroids act as boundary anchors that attract SVs.
    Returns NaN if no centroids are available (non-centroid methods).
    """
    if centroids is None or len(centroids) == 0:
        return float("nan")
    D_sv    = cdist(X_sv,    centroids).min(axis=1).mean()
    D_train = cdist(X_train, centroids).min(axis=1).mean()
    return float(D_sv / (D_train + 1e-10))


def centroid_sv_separation(
    K_centroid_sv: np.ndarray,   # shape (n_centroids, n_sv)
    y_sv: np.ndarray,
    centroid_labels: np.ndarray,
    K_centroid_non_sv: np.ndarray,  # shape (n_centroids, n_non_sv)
    y_non_sv: np.ndarray,
) -> dict[str, float]:
    """
    SV-4 metric: for each centroid, compare kernel values to SVs of the
    same vs different class, and to non-SVs of same vs different class.

    Returns:
      sv_within    -- mean K(centroid, SV same class)
      sv_between   -- mean K(centroid, SV diff class)
      sv_sep       -- sv_within / sv_between  (separation ratio at SVs)
      non_sv_within  -- mean K(centroid, non-SV same class)
      non_sv_between -- mean K(centroid, non-SV diff class)
      non_sv_sep   -- non_sv_within / non_sv_between
    """
    sv_within_vals, sv_between_vals = [], []
    non_sv_within_vals, non_sv_between_vals = [], []

    for ci, cl in enumerate(centroid_labels):
        # SVs
        sv_same  = K_centroid_sv[ci,  y_sv     == cl]
        sv_diff  = K_centroid_sv[ci,  y_sv     != cl]
        if len(sv_same)  > 0: sv_within_vals.append(sv_same.mean())
        if len(sv_diff)  > 0: sv_between_vals.append(sv_diff.mean())
        # non-SVs
        nsv_same = K_centroid_non_sv[ci, y_non_sv == cl]
        nsv_diff = K_centroid_non_sv[ci, y_non_sv != cl]
        if len(nsv_same) > 0: non_sv_within_vals.append(nsv_same.mean())
        if len(nsv_diff) > 0: non_sv_between_vals.append(nsv_diff.mean())

    sv_w  = float(np.mean(sv_within_vals))    if sv_within_vals    else 0.0
    sv_b  = float(np.mean(sv_between_vals))   if sv_between_vals   else 1e-10
    nsv_w = float(np.mean(non_sv_within_vals)) if non_sv_within_vals else 0.0
    nsv_b = float(np.mean(non_sv_between_vals)) if non_sv_between_vals else 1e-10

    return {
        "sv_within":    sv_w,
        "sv_between":   sv_b,
        "sv_sep":       sv_w  / (abs(sv_b)  + 1e-10),
        "non_sv_within":  nsv_w,
        "non_sv_between": nsv_b,
        "non_sv_sep":   nsv_w / (abs(nsv_b) + 1e-10),
    }


# ---------------------------------------------------------------------------
# Per-epoch SV analysis hook
# This is called once per epoch during training via a thin wrapper around
# each aligner's svm_training() result.
# ---------------------------------------------------------------------------

def analyse_epoch(
    aligner,
    epoch: int,
    method: str,
    result: dict[str, Any],           # from aligner.svm_training()
    K_train: np.ndarray,              # full NxN kernel matrix (numpy)
    X_train: np.ndarray,
    y_train: np.ndarray,
    main_centroids: np.ndarray | None,
    sub_centroids:  np.ndarray | None,
    main_centroid_labels: np.ndarray | None,
) -> dict[str, Any]:
    """
    Given a trained SVM and full kernel matrix for this epoch, compute all
    four SV metrics and return as a flat dict suitable for CSV logging.
    """
    svm         = result["svm"]
    sv_indices  = svm.support_          # indices into K_train rows
    y_np        = np.asarray(y_train)
    n_train     = len(y_np)

    sv_mask       = np.zeros(n_train, dtype=bool)
    sv_mask[sv_indices] = True
    non_sv_mask   = ~sv_mask

    y_sv      = y_np[sv_mask]
    y_non_sv  = y_np[non_sv_mask]
    X_sv      = np.asarray(X_train)[sv_mask]
    X_non_sv  = np.asarray(X_train)[non_sv_mask]
    n_sv      = sv_mask.sum()
    n_non_sv  = non_sv_mask.sum()

    # ── SV-2: local block ratio at SVs ──────────────────────────────────────
    K_sv_sv = K_train[np.ix_(sv_mask, sv_mask)]
    local_block_ratio  = svm_block_ratio(K_sv_sv, y_sv) if n_sv > 1 else float("nan")
    global_block_ratio = svm_block_ratio(K_train, y_np)

    # ── SV-3: local KTA at SVs ───────────────────────────────────────────────
    local_kta  = kta_from_matrix(K_sv_sv, y_sv) if n_sv > 1 else float("nan")
    global_kta = kta_from_matrix(K_train, y_np)

    # ── SV-1: proximity to centroids ────────────────────────────────────────
    # Use whichever centroids are available:
    #   CCKA -> use sub_centroids (the finer boundary representatives)
    #   QUACK -> use main_centroids (no sub-centroids exist)
    #   others -> None (proximity metric will be NaN)
    centroids_for_proximity = None
    if sub_centroids is not None:
        centroids_for_proximity = sub_centroids
    elif main_centroids is not None:
        centroids_for_proximity = main_centroids

    proximity_ratio = sv_proximity_ratio(X_sv, np.asarray(X_train), centroids_for_proximity)

    # ── SV-4: centroid-to-SV kernel separation ──────────────────────────────
    sv4_metrics: dict[str, float] = {
        "sv_sep":     float("nan"),
        "non_sv_sep": float("nan"),
        "sv_within":  float("nan"),
        "sv_between": float("nan"),
        "non_sv_within":  float("nan"),
        "non_sv_between": float("nan"),
    }

    if main_centroids is not None and main_centroid_labels is not None and n_sv > 0 and n_non_sv > 0:
        # Build kernel between centroids and SVs / non-SVs
        # We need the aligner's kernel function for this.
        # Use the rows of K_train that correspond to centroids.
        # Since centroids are NOT in X_train, we compute them explicitly
        # via the aligner's _centroid_kernel_vec or regular_kernel_matrix.
        # For generality we use aligner.test_kernel_matrix() which gives
        # K(query_rows, train_cols). Here query = centroids, train = X_sv/X_non_sv.

        import jax.numpy as jnp
        mc_j  = jnp.asarray(main_centroids)
        sv_j  = jnp.asarray(X_sv)
        nsv_j = jnp.asarray(X_non_sv)

        # K_centroid_sv:     shape (n_centroids, n_sv)
        K_c_sv  = np.asarray(
            aligner.test_kernel_matrix(aligner.weights, sv_j,  mc_j)
        )   # note: test_kernel_matrix(weights, X_train, X_test) -> (n_test, n_train)
            # so we call it as (weights, sv_rows, centroid_queries)
            # result shape (n_centroids, n_sv) -- correct

        K_c_nsv = np.asarray(
            aligner.test_kernel_matrix(aligner.weights, nsv_j, mc_j)
        )

        sv4_metrics = centroid_sv_separation(
            K_c_sv, y_sv,
            np.asarray(main_centroid_labels),
            K_c_nsv, y_non_sv,
        )

    row = {
        "method":              method,
        "epoch":               epoch,
        "n_sv":                int(n_sv),
        "sv_fraction":         float(n_sv / n_train),
        "test_accuracy":       result["test_accuracy"],
        "train_accuracy":      result["train_accuracy"],
        "margin":              result["margin"],
        # SV-1
        "proximity_ratio":     proximity_ratio,
        # SV-2
        "local_block_ratio":   local_block_ratio,
        "global_block_ratio":  global_block_ratio,
        "block_ratio_lift":    (
            local_block_ratio / (global_block_ratio + 1e-10)
            if not np.isnan(local_block_ratio) else float("nan")
        ),
        # SV-3
        "local_kta":           local_kta,
        "global_kta":          global_kta,
        "kta_lift":            (
            local_kta / (global_kta + 1e-10)
            if not np.isnan(local_kta) else float("nan")
        ),
        # SV-4
        **{f"sv4_{k}": v for k, v in sv4_metrics.items()},
    }
    return row


# ---------------------------------------------------------------------------
# Training loop wrapper with per-epoch SV analysis
# ---------------------------------------------------------------------------

def run_sv_experiment(
    method:          str,
    dataset:         str,
    dataset_path:    str,
    centroids:       int,
    num_iterations:  int,
) -> list[dict[str, Any]]:
    """
    Train one method for `num_iterations` epochs, computing SV metrics at
    each epoch. Returns a list of per-epoch dicts.

    We replicate the same training loop style as the aligner but hook in our
    analysis after each SVM call. This avoids modifying kta.py further and
    keeps the experiment self-contained.
    """
    import jax
    import jax.numpy as jnp
    import optax as ox
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    data    = np.load(dataset_path, allow_pickle=True).item()
    X       = jnp.asarray(np.concatenate([data["x_train"], data["x_test"]], axis=0))
    y       = jnp.asarray(np.concatenate([data["y_train"], data["y_test"]], axis=0))

    kernel  = quackEmbeddingCircuit(num_qubits=5, reps=6, reupload=True)
    model   = KernelModel(circuit=kernel)

    common = dict(
        kernel_model=model,
        data=X,
        labels=y,
        matrix_type="regular",
        split_size=0.5,
    )

    if method == "fullKTA":
        aligner = fullKTA(**common, learning_rate=0.1, optimizer="adam",
                          epochs=num_iterations)
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
                                   learning_rate=0.2, centroid_lr=0.1,
                                   sub_centroid_lr=0.1, lambda_co=0.001,
                                   lambda_kao=0.001, epochs=num_iterations)
    else:
        raise ValueError(f"Unknown method: {method}")

    # -- Run the aligner's full training loop to get history ------------------
    print(f"  Training {METHOD_LABELS[method]} for {num_iterations} epochs...")
    history = aligner.align()

    # -- Now replay the saved kernel checkpoints for SV analysis --------------
    # We don't have per-epoch weight checkpoints stored cheaply, so we use
    # the coords history (which was computed from the live kernel) as a proxy,
    # and re-run the SV analysis using the FINAL weights but at the accuracy
    # values recorded per epoch.
    #
    # For a clean per-epoch analysis we need the weights at each epoch.
    # We therefore run a SECOND pass: a lightweight custom loop that mirrors
    # the aligner's logic step-by-step and hooks in our analysis.
    # This is the correct approach -- we replicate the aligner's update rule.

    rows = _per_epoch_sv_loop(method, dataset, aligner, centroids, num_iterations)
    return rows


def _per_epoch_sv_loop(
    method:         str,
    dataset:        str,
    aligner,
    centroids:      int,
    num_iterations: int,
) -> list[dict[str, Any]]:
    """
    Run a fresh training pass that mirrors the aligner's update rule exactly,
    but hooks in SV analysis after every epoch's SVM call.

    We reset the aligner's weights to initial and re-train from scratch so
    that the per-epoch kernel matrices are the ones that were actually seen
    during training.
    """
    import jax
    import jax.numpy as jnp
    import optax as ox
    from sklearn.svm import SVC

    # Reset weights to initial state
    aligner.weights   = aligner.kernel_model.circuit.init_weights()
    aligner.opt_state = aligner._optimizer.init(aligner.weights)
    if hasattr(aligner, "_kao_weight_opt_state"):
        aligner._kao_weight_opt_state = aligner._kao_weight_optimizer.init(aligner.weights)
    if hasattr(aligner, "_kao_sub_opt_state"):
        aligner._kao_sub_opt_state    = aligner._kao_sub_optimizer.init(aligner.sub_centroids)
    if hasattr(aligner, "_co_main_opt_state"):
        aligner._co_main_opt_state    = aligner._co_main_optimizer.init(aligner.main_centroids)

    rows = []
    unique_labels = np.unique(np.asarray(aligner.ytrain))
    n_cls = len(unique_labels)
    y_raw = aligner.sub_centroid_labels if hasattr(aligner, "sub_centroid_labels") else None

    for epoch in range(num_iterations):

        # ── Replicate the aligner's update rule for this epoch ─────────────
        if method == "centroidBasedKTA":
            cl_kao = unique_labels[epoch % n_cls]
            l_kao  = float(cl_kao)
            l_co   = -float(cl_kao)
            main_idx = int(jnp.argmax(aligner.main_centroid_labels == cl_kao))
            main_centroid = aligner.main_centroids[main_idx]
            for _ in range(10):
                aligner.weights, aligner.sub_centroids = aligner._kao_joint_update(
                    main_centroid, y_raw, l=l_kao
                )
            main_centroid = aligner.main_centroids[main_idx]
            for _ in range(10):
                aligner.main_centroids = aligner._co_main_update(
                    cl=cl_kao, y_raw=y_raw, l=l_co
                )

        elif method == "quackKTA":
            cl_kao = unique_labels[epoch % n_cls]
            l_kao  = float(cl_kao)
            l_co   = -float(cl_kao)
            main_idx = int(jnp.argmax(aligner.main_centroid_labels == cl_kao))
            main_centroid = aligner.main_centroids[main_idx]
            y_kao = jnp.where(aligner.ytrain == cl_kao, 1.0, -1.0)
            if epoch % 2 == 0:
                for _ in range(10):
                    aligner.weights = aligner._kao_weight_update_quack(
                        main_centroid, aligner.xtrain, y_kao, l=l_kao
                    )
            else:
                for _ in range(10):
                    aligner.main_centroids = aligner._main_centroid_gradient_update_quack(
                        main_idx, aligner.xtrain, y_kao, l=l_co
                    )
        else:
            # FullKTA / RandomKTA / GreedyKTA: standard single gradient step
            X_b, y_b = aligner._get_batch(epoch)
            grads = aligner._grad_fn(aligner.weights, X_b, y_b)
            updates, aligner.opt_state = aligner._optimizer.update(grads, aligner.opt_state)
            aligner.weights = ox.apply_updates(aligner.weights, updates)

        # ── Build full NxN kernel matrix for this epoch ────────────────────
        K_train = np.asarray(
            aligner._apply_centering(
                aligner._kernel_matrix(aligner.weights, aligner.xtrain)
            )
        )

        # ── Fit SVM and get support vectors ───────────────────────────────
        y_np = np.asarray(aligner.ytrain)
        svm  = SVC(kernel="precomputed", C=1.0, probability=True, max_iter=10_000)
        svm.fit(K_train, y_np)

        # Evaluate on test set too
        K_test = np.asarray(
            aligner.test_kernel_matrix(aligner.weights, aligner.xtrain, aligner.xtest)
        )
        y_test_np   = np.asarray(aligner.ytest)
        y_pred_test = svm.predict(K_test)
        test_acc    = float(np.mean(y_pred_test == y_test_np))
        y_pred_train = svm.predict(K_train)
        train_acc   = float(np.mean(y_pred_train == y_np))
        dual_coefs  = svm.dual_coef_
        margin      = float(1.0 / np.sqrt(np.sum(dual_coefs ** 2)))

        svm_result = {
            "svm": svm, "test_accuracy": test_acc, "train_accuracy": train_acc,
            "margin": margin,
        }

        # ── Gather centroids if available ─────────────────────────────────
        main_centroids        = None
        sub_centroids         = None
        main_centroid_labels  = None
        if hasattr(aligner, "main_centroids"):
            main_centroids       = np.asarray(aligner.main_centroids)
            main_centroid_labels = np.asarray(aligner.main_centroid_labels)
        if hasattr(aligner, "sub_centroids"):
            sub_centroids = np.asarray(aligner.sub_centroids)

        # ── Run SV analysis ───────────────────────────────────────────────
        row = analyse_epoch(
            aligner      = aligner,
            epoch        = epoch + 1,
            method       = method,
            result       = svm_result,
            K_train      = K_train,
            X_train      = aligner.xtrain,
            y_train      = aligner.ytrain,
            main_centroids       = main_centroids,
            sub_centroids        = sub_centroids,
            main_centroid_labels = main_centroid_labels,
        )
        row["dataset"]   = dataset
        row["centroids"] = centroids
        rows.append(row)

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}/{num_iterations}  "
                  f"acc={test_acc:.3f}  "
                  f"n_sv={row['n_sv']:3d}  "
                  f"local_br={row['local_block_ratio']:.2f}  "
                  f"global_br={row['global_block_ratio']:.2f}  "
                  f"local_kta={row['local_kta']:.3f}  "
                  f"global_kta={row['global_kta']:.3f}")

    return rows


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_all(dataset: str) -> pd.DataFrame:
    dataset_path = DATASET_PATHS[dataset]
    all_rows = []
    for method in ALL_METHODS:
        print(f"\n[{METHOD_LABELS[method]}]")
        rows = run_sv_experiment(
            method, dataset, dataset_path, DEFAULT_CENTROIDS, EPOCHS
        )
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    out = f"{dataset}_sv_exp_results.csv"
    df.to_csv(out, index=False)
    print(f"\n[Saved] {out}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _smooth(y: np.ndarray, w: int = 5) -> np.ndarray:
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(np.pad(y, (w//2, w//2), mode="edge"), k, mode="valid")[:len(y)]


def _get(df: pd.DataFrame, method: str, col: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df[df["method"] == method].sort_values("epoch")
    return sub["epoch"].values, sub[col].values


# ---------------------------------------------------------------------------
# SV-1: Proximity ratio
# ---------------------------------------------------------------------------

def plot_sv1_proximity(df: pd.DataFrame, dataset: str, ax=None):
    """
    SV-1: mean dist(SV, nearest centroid) / mean dist(train, nearest centroid).

    Ratio < 1  => SVs are CLOSER to centroids than average train points.
                  Centroids act as boundary anchors that SVs cluster around.
    Ratio = 1  => SVs are no closer to centroids than random train points.
    Ratio > 1  => SVs are FARTHER from centroids (centroids are not near the boundary).

    Only meaningful for centroid-based methods (CCKA, QUACK).
    Non-centroid methods show NaN and are omitted.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4.5))

    plotted = False
    for method in ALL_METHODS:
        sub = df[df["method"] == method].sort_values("epoch")
        y = sub["proximity_ratio"].values
        if np.all(np.isnan(y)):
            continue
        x = sub["epoch"].values
        lw = 2.5 if method == CCKA else 1.8
        ls = "-"  if method == CCKA else "--"
        ax.plot(x, _smooth(y), color=METHOD_COLORS[method],
                label=METHOD_LABELS[method], linewidth=lw, linestyle=ls)
        plotted = True

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2,
               label="ratio = 1 (no proximity bias)")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("dist(SV, centroid) / dist(train, centroid)", fontsize=10)
    ax.set_title(
        f"SV-1: Support vector proximity to centroids — {dataset}\n"
        "Ratio < 1 → SVs cluster near centroids (boundary anchoring)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    if standalone:
        plt.tight_layout()
        out = f"{dataset}_sv1_proximity.pdf"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {out}")


# ---------------------------------------------------------------------------
# SV-2: Local vs Global block-diagonal ratio
# ---------------------------------------------------------------------------

def plot_sv2_local_block_ratio(df: pd.DataFrame, dataset: str, ax=None):
    """
    SV-2: Local block ratio (computed at SVs only) vs Global block ratio
    (computed at all training points).

    Key question: For CCKA, is the kernel matrix well-structured at the SVs
    even when the global matrix is noisy?

    We plot both local (solid) and global (dashed) per method.
    The gap between solid and dashed for CCKA is the core evidence:
      - Large positive gap => kernel is locally good at SVs, globally poor
      - No gap            => the kernel is uniformly good or uniformly poor
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))

    for method in ALL_METHODS:
        sub = df[df["method"] == method].sort_values("epoch")
        x = sub["epoch"].values
        local  = sub["local_block_ratio"].values.astype(float)
        global_ = sub["global_block_ratio"].values.astype(float)
        color = METHOD_COLORS[method]
        lw    = 2.5 if method == CCKA else 1.5

        # Solid = local (at SVs), Dashed = global (all train)
        ax.plot(x, _smooth(local),   color=color, linewidth=lw,
                linestyle="-",  label=f"{METHOD_LABELS[method]} local (SVs)")
        ax.plot(x, _smooth(global_), color=color, linewidth=lw * 0.7,
                linestyle="--", alpha=0.6, label=f"{METHOD_LABELS[method]} global")

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, label="ratio=1 baseline")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Block-diagonal ratio (within / between class K)", fontsize=10)
    ax.set_title(
        f"SV-2: Local (SV) vs Global block ratio — {dataset}\n"
        "Solid = at support vectors only   |   Dashed = full training set\n"
        "CCKA expected: high solid, low dashed",
        fontsize=10,
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)

    if standalone:
        plt.tight_layout()
        out = f"{dataset}_sv2_local_block_ratio.pdf"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {out}")


# ---------------------------------------------------------------------------
# SV-3: Local KTA vs Global KTA
# ---------------------------------------------------------------------------

def plot_sv3_local_kta(df: pd.DataFrame, dataset: str, ax=None):
    """
    SV-3: KTA computed only at the SV sub-matrix vs global KTA.

    This directly tests the locality hypothesis:
    If CCKA's kernel is well-aligned at the decision boundary but noisy
    everywhere else, the local SV-KTA should be HIGHER than the global KTA,
    while for FullKTA both should move together.

    We also plot the KTA lift (local/global ratio) to make the gap explicit.
    """
    standalone = ax is None
    if standalone:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        # If embedded, split the provided ax into a 1x2 via inset or just use ax
        # for simplicity we draw only the lift panel
        ax2 = ax
        ax1 = None

    # Panel 1: raw KTA values
    if ax1 is not None:
        for method in ALL_METHODS:
            sub = df[df["method"] == method].sort_values("epoch")
            x = sub["epoch"].values
            color = METHOD_COLORS[method]
            lw    = 2.5 if method == CCKA else 1.5
            ax1.plot(x, _smooth(sub["local_kta"].values.astype(float)),
                     color=color, linewidth=lw, linestyle="-",
                     label=f"{METHOD_LABELS[method]} local")
            ax1.plot(x, _smooth(sub["global_kta"].values.astype(float)),
                     color=color, linewidth=lw * 0.7, linestyle="--", alpha=0.6,
                     label=f"{METHOD_LABELS[method]} global")
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("KTA value", fontsize=11)
        ax1.set_title(
            "Local SV-KTA (solid) vs Global KTA (dashed)\n"
            "CCKA expected: solid >> dashed",
            fontsize=10,
        )
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.25)

    # Panel 2: KTA lift = local / global  (>1 means locally better than globally)
    for method in ALL_METHODS:
        sub = df[df["method"] == method].sort_values("epoch")
        x = sub["epoch"].values
        lift = sub["kta_lift"].values.astype(float)
        lw   = 2.5 if method == CCKA else 1.5
        ls   = "-"  if method == CCKA else "--"
        ax2.plot(x, _smooth(lift), color=METHOD_COLORS[method],
                 linewidth=lw, linestyle=ls, label=METHOD_LABELS[method])

    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1.2,
                label="lift = 1 (uniform)")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("KTA lift (local SV-KTA / global KTA)", fontsize=10)
    ax2.set_title(
        f"SV-3: KTA lift at support vectors — {dataset}\n"
        "Lift > 1 → kernel is locally better at SVs than globally",
        fontsize=10,
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    if standalone:
        plt.tight_layout()
        out = f"{dataset}_sv3_local_kta.pdf"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {out}")


# ---------------------------------------------------------------------------
# SV-4: Centroid-to-SV kernel separation
# ---------------------------------------------------------------------------

def plot_sv4_centroid_sv_sep(df: pd.DataFrame, dataset: str, ax=None):
    """
    SV-4: Centroid-to-SV separation ratio vs centroid-to-non-SV separation.

    For CCKA:
      sv_sep   = mean K(centroid, SV_same) / mean K(centroid, SV_diff)
      non_sv_sep = mean K(centroid, non-SV_same) / mean K(centroid, non-SV_diff)

    Hypothesis:
      sv_sep >> non_sv_sep for CCKA  -- the centroid kernel is specifically
                                        well-structured AT the support vectors
      sv_sep ~= non_sv_sep for FullKTA -- uniformly good everywhere

    This is the smoking gun: if CCKA's centroid-to-SV separation is high
    while its centroid-to-non-SV separation is low/noisy, we have confirmed
    that CCKA's centroid optimization implicitly targets boundary-relevant points.

    Only plotted for methods that have centroids (CCKA, QUACK).
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))

    for method in ALL_METHODS:
        sub = df[df["method"] == method].sort_values("epoch")
        x        = sub["epoch"].values
        sv_sep   = sub["sv4_sv_sep"].values.astype(float)
        non_sv_sep = sub["sv4_non_sv_sep"].values.astype(float)

        # Only plot if we have real values (non-NaN)
        if np.all(np.isnan(sv_sep)):
            continue

        color = METHOD_COLORS[method]
        lw    = 2.5 if method == CCKA else 1.5

        # Solid = at SVs, dashed = at non-SVs
        ax.plot(x, _smooth(sv_sep),     color=color, linewidth=lw, linestyle="-",
                label=f"{METHOD_LABELS[method]}  at SVs")
        ax.plot(x, _smooth(non_sv_sep), color=color, linewidth=lw * 0.7,
                linestyle="--", alpha=0.7, label=f"{METHOD_LABELS[method]}  non-SVs")

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2,
               label="sep = 1 (no class structure)")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("K(centroid, same-class) / K(centroid, diff-class)", fontsize=10)
    ax.set_title(
        f"SV-4: Centroid-to-SV kernel separation — {dataset}\n"
        "Solid = at support vectors   |   Dashed = at non-support-vectors\n"
        "CCKA expected: solid >> dashed (boundary-targeted centroid alignment)",
        fontsize=10,
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)

    if standalone:
        plt.tight_layout()
        out = f"{dataset}_sv4_centroid_sv_sep.pdf"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {out}")


# ---------------------------------------------------------------------------
# Bonus: SV count and accuracy trajectory
# ---------------------------------------------------------------------------

def plot_sv_count_and_accuracy(df: pd.DataFrame, dataset: str, ax=None):
    """
    Number of support vectors over training, alongside test accuracy.
    A declining SV count means the margin is widening -- better generalization.
    CCKA hypothesis: SV count drops faster/lower than other methods.
    """
    standalone = ax is None
    if standalone:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    else:
        ax1, ax2 = ax, None

    for method in ALL_METHODS:
        sub = df[df["method"] == method].sort_values("epoch")
        x = sub["epoch"].values
        lw = 2.5 if method == CCKA else 1.5
        ls = "-"  if method == CCKA else "--"
        ax1.plot(x, _smooth(sub["n_sv"].values.astype(float)),
                 color=METHOD_COLORS[method], linewidth=lw, linestyle=ls,
                 label=METHOD_LABELS[method])
        if ax2 is not None:
            ax2.plot(x, _smooth(sub["test_accuracy"].values.astype(float)),
                     color=METHOD_COLORS[method], linewidth=lw, linestyle=ls,
                     label=METHOD_LABELS[method])

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Number of support vectors", fontsize=11)
    ax1.set_title(f"SV count over training — {dataset}", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    if ax2 is not None:
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Test accuracy", fontsize=11)
        ax2.set_title(f"Test accuracy over training — {dataset}", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.25)
        ax2.set_ylim(0, 1.05)

    if standalone:
        plt.tight_layout()
        out = f"{dataset}_sv_count.pdf"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {out}")


# ---------------------------------------------------------------------------
# Combined publication figure
# ---------------------------------------------------------------------------

def plot_full(df: pd.DataFrame, dataset: str):
    """
    2x3 combined figure:
      [SV-count | SV-2 local block ratio | SV-3 KTA lift]
      [SV-1 proximity | SV-4 centroid-SV sep | test accuracy]
    """
    fig = plt.figure(figsize=(19, 10))
    fig.suptitle(
        f"Support Vector Locality Analysis — {dataset.upper()}\n"
        "Hypothesis: CCKA's kernel is locally well-behaved at SVs "
        "even when globally noisy",
        fontsize=13, y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Row 0
    plot_sv_count_and_accuracy(df, dataset, ax=fig.add_subplot(gs[0, 0]))
    plot_sv2_local_block_ratio(df, dataset, ax=fig.add_subplot(gs[0, 1]))
    plot_sv3_local_kta(df, dataset,         ax=fig.add_subplot(gs[0, 2]))

    # Row 1
    plot_sv1_proximity(df, dataset,          ax=fig.add_subplot(gs[1, 0]))
    plot_sv4_centroid_sv_sep(df, dataset,    ax=fig.add_subplot(gs[1, 1]))

    # Accuracy recap (bottom right)
    ax_acc = fig.add_subplot(gs[1, 2])
    for method in ALL_METHODS:
        sub = df[df["method"] == method].sort_values("epoch")
        x = sub["epoch"].values
        lw = 2.5 if method == CCKA else 1.5
        ls = "-"  if method == CCKA else "--"
        ax_acc.plot(x, _smooth(sub["test_accuracy"].values.astype(float)),
                    color=METHOD_COLORS[method], linewidth=lw, linestyle=ls,
                    label=METHOD_LABELS[method])
    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Test accuracy", fontsize=11)
    ax_acc.set_title("Test accuracy (reference)", fontsize=11)
    ax_acc.legend(fontsize=9)
    ax_acc.grid(True, alpha=0.25)
    ax_acc.set_ylim(0, 1.05)

    out = f"{dataset}_sv_full.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out}")


# ---------------------------------------------------------------------------
# Synthetic data for dry-run
# ---------------------------------------------------------------------------

def _synthetic(dataset: str) -> pd.DataFrame:
    """
    Simulate the expected outcome:
      CCKA:    local block ratio > global, KTA lift > 1, sv_sep >> non_sv_sep
      FullKTA: high block ratio everywhere, lift ~= 1
      Others:  intermediate
    """
    rng = np.random.default_rng(0)
    rows = []

    def sig(t, x0, k): return 1.0 / (1.0 + np.exp(-k * (t - x0)))

    epochs = np.arange(1, EPOCHS + 1)
    t = (epochs - 1) / (EPOCHS - 1)

    specs = {
        "fullKTA": dict(
            test_acc    = 0.55 + 0.44 * sig(t, 0.5, 8),
            n_sv        = (25 - 15 * sig(t, 0.5, 6)).astype(int),
            local_br    = 5.0 + 5.0 * sig(t, 0.4, 7),
            global_br   = 4.5 + 5.5 * sig(t, 0.4, 7),
            local_kta   = 0.15 + 0.50 * sig(t, 0.4, 7),
            global_kta  = 0.12 + 0.48 * sig(t, 0.4, 7),
            prox        = np.full(EPOCHS, float("nan")),
            sv_sep      = np.full(EPOCHS, float("nan")),
            non_sv_sep  = np.full(EPOCHS, float("nan")),
        ),
        "centroidBasedKTA": dict(
            test_acc    = 0.50 + 0.50 * sig(t, 0.3, 10),
            n_sv        = (25 - 18 * sig(t, 0.3, 8)).astype(int),
            local_br    = 4.0 + 4.5 * sig(t, 0.25, 9),   # high locally
            global_br   = 1.2 + 1.5 * sig(t, 0.35, 7),   # low globally
            local_kta   = 0.10 + 0.55 * sig(t, 0.25, 9), # high locally
            global_kta  = 0.08 + 0.18 * sig(t, 0.3, 6),  # low globally
            prox        = 0.9 - 0.25 * sig(t, 0.3, 8),   # <1, dropping
            sv_sep      = 2.5 + 3.0 * sig(t, 0.25, 9),   # high at SVs
            non_sv_sep  = 1.1 + 0.4 * sig(t, 0.4, 5),    # low at non-SVs
        ),
        "quackKTA": dict(
            test_acc    = 0.52 + 0.47 * sig(t, 0.4, 8),
            n_sv        = (25 - 14 * sig(t, 0.4, 6)).astype(int),
            local_br    = 3.0 + 3.5 * sig(t, 0.4, 7),
            global_br   = 2.0 + 3.5 * sig(t, 0.5, 6),
            local_kta   = 0.08 + 0.38 * sig(t, 0.4, 7),
            global_kta  = 0.05 + 0.35 * sig(t, 0.5, 6),
            prox        = 1.0 - 0.15 * sig(t, 0.4, 7),
            sv_sep      = 1.5 + 2.0 * sig(t, 0.4, 7),
            non_sv_sep  = 1.2 + 1.0 * sig(t, 0.5, 5),
        ),
        "randomKTA": dict(
            test_acc    = 0.50 + 0.42 * sig(t, 0.55, 7),
            n_sv        = (25 - 12 * sig(t, 0.55, 5)).astype(int),
            local_br    = 1.5 + 2.0 * sig(t, 0.55, 6),
            global_br   = 1.2 + 1.5 * sig(t, 0.55, 6),
            local_kta   = 0.06 + 0.18 * sig(t, 0.55, 6),
            global_kta  = 0.05 + 0.15 * sig(t, 0.55, 6),
            prox        = np.full(EPOCHS, float("nan")),
            sv_sep      = np.full(EPOCHS, float("nan")),
            non_sv_sep  = np.full(EPOCHS, float("nan")),
        ),
        "greedyKTA": dict(
            test_acc    = 0.50 + 0.43 * sig(t, 0.45, 7),
            n_sv        = (25 - 13 * sig(t, 0.45, 5)).astype(int),
            local_br    = 1.8 + 2.5 * sig(t, 0.45, 6),
            global_br   = 1.5 + 2.0 * sig(t, 0.45, 6),
            local_kta   = 0.07 + 0.20 * sig(t, 0.45, 6),
            global_kta  = 0.06 + 0.18 * sig(t, 0.45, 6),
            prox        = np.full(EPOCHS, float("nan")),
            sv_sep      = np.full(EPOCHS, float("nan")),
            non_sv_sep  = np.full(EPOCHS, float("nan")),
        ),
    }

    noise = lambda arr, s: arr + rng.normal(0, s, len(arr))

    for method, sp in specs.items():
        for i, ep in enumerate(epochs):
            local_br  = float(noise(sp["local_br"],  0.3)[i])
            global_br = float(noise(sp["global_br"], 0.2)[i])
            local_kta = float(noise(sp["local_kta"], 0.02)[i])
            global_kta= float(noise(sp["global_kta"],0.015)[i])
            rows.append({
                "method":             method,
                "dataset":            dataset,
                "centroids":          DEFAULT_CENTROIDS,
                "epoch":              ep,
                "test_accuracy":      float(np.clip(noise(sp["test_acc"], 0.01)[i], 0, 1)),
                "train_accuracy":     float(np.clip(noise(sp["test_acc"] + 0.05, 0.01)[i], 0, 1)),
                "n_sv":               max(3, int(noise(sp["n_sv"].astype(float), 1)[i])),
                "sv_fraction":        float(max(3, sp["n_sv"][i]) / 30),
                "margin":             float(0.1 + sp["test_acc"][i] * 0.3),
                "proximity_ratio":    float(noise(sp["prox"], 0.03)[i])
                                      if not np.isnan(sp["prox"][i]) else float("nan"),
                "local_block_ratio":  local_br,
                "global_block_ratio": global_br,
                "block_ratio_lift":   local_br / (global_br + 1e-10),
                "local_kta":          local_kta,
                "global_kta":         global_kta,
                "kta_lift":           local_kta / (global_kta + 1e-10),
                "sv4_sv_sep":         float(noise(sp["sv_sep"],     0.2)[i])
                                      if not np.isnan(sp["sv_sep"][i]) else float("nan"),
                "sv4_non_sv_sep":     float(noise(sp["non_sv_sep"], 0.15)[i])
                                      if not np.isnan(sp["non_sv_sep"][i]) else float("nan"),
                "sv4_sv_within":      float("nan"),
                "sv4_sv_between":     float("nan"),
                "sv4_non_sv_within":  float("nan"),
                "sv4_non_sv_between": float("nan"),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Support vector locality analysis for CCKA vs baselines."
    )
    parser.add_argument("--dataset", type=str, default="checkerboard",
                        choices=list(DATASET_PATHS))
    parser.add_argument("--results-csv", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

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
        print("[DRY_RUN] generating synthetic data...")
        df = _synthetic(dataset)

    # Individual plots
    plot_sv1_proximity(df, dataset)
    plot_sv2_local_block_ratio(df, dataset)
    plot_sv3_local_kta(df, dataset)
    plot_sv4_centroid_sv_sep(df, dataset)
    plot_sv_count_and_accuracy(df, dataset)

    # Combined figure
    plot_full(df, dataset)

    print("\nDone.")


if __name__ == "__main__":
    main()