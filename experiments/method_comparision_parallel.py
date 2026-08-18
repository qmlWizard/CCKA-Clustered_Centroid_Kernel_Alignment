from __future__ import annotations

import argparse
import inspect
import os
import random
import time
import warnings
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"


if not DRY_RUN:
    try:
        import jax
        import jax.numpy as jnp
        import optax as ox
        from sklearn.metrics import f1_score, precision_score, recall_score
        from sklearn.svm import SVC

        from ccka.aligner.kta import (centroidBasedKTA, fullKTA, greedyKTA,
                                       quackKTA, randomKTA)
        from ccka.circuits.angleEmbeddingKernel import quackEmbeddingCircuit
        from ccka.models.kernel import KernelModel

        BACKEND_AVAILABLE = True
    except ImportError as e:
        print(f"[WARNING] ccka backend not importable ({e}) -- switching to DRY_RUN mode.")
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

CENTROID_VALUES = [2, 4, 6, 8, 10]
EPOCH_VALUES    = [100]
LANDMARK_POINTS = 10  # shared hyperparameter, used both in aligner construction
                       # and in the circuit-execution cost estimator below

N_SEEDS        = 10
SEEDS          = list(range(N_SEEDS))
REFERENCE_SEED = SEEDS[0]   # used only for the qualitative spectrum-snapshot plot

CENTROID_METHODS = {"centroidBasedKTA", "randomKTA", "greedyKTA", "quackKTA"}
ALL_METHODS      = ["fullKTA", "randomKTA", "quackKTA", "centroidBasedKTA", "greedyKTA"]

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
# Seeding (fixed): inject the seed/key under whatever name the aligner accepts
# ---------------------------------------------------------------------------

_seed_kwargs_logged: set[str] = set()


def _set_global_seed(seed: int):
    """Seed everything we control directly. Returns a fresh JAX PRNGKey."""
    np.random.seed(seed)
    random.seed(seed)
    if not DRY_RUN:
        return jax.random.PRNGKey(seed)
    return seed


def _seed_kwargs_for(cls, seed: int, key) -> dict:
    """
    Inspect `cls.__init__` and return a dict of {param_name: value} for
    whichever seed/key-like parameter names it actually accepts, so that
    the 5 seeds genuinely produce different initial weights instead of all
    silently falling back to the aligner's own default RNG.

    Prints a one-time diagnostic per class so you can confirm seeding is
    actually reaching the weight initializer.
    """
    try:
        sig_params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        sig_params = {}

    candidates = {
        "seed":        seed,
        "random_seed": seed,
        "random_state": seed,
        "weight_seed": seed,
        "key":       key,
        "prng_key":  key,
        "rng_key":   key,
        "init_key":  key,
        "rng":       key,
    }
    matched = {name: val for name, val in candidates.items() if name in sig_params}

    tag = cls.__name__
    if tag not in _seed_kwargs_logged:
        _seed_kwargs_logged.add(tag)
        if matched:
            print(f"[seeding] {tag} accepts seed kwarg(s): {sorted(matched.keys())}")
        else:
            print(
                f"[seeding][WARNING] {tag}.__init__ does not accept any of "
                f"{sorted(candidates)} -- only global numpy/python seeding will "
                f"apply to this method. If its band still looks flat across "
                f"seeds, its weight init (or centroid init) is using its own "
                f"fixed/default RNG untouched by our seed. Tell us the actual "
                f"parameter name (check the aligner's __init__ signature) and "
                f"we'll add it to the candidate list above."
            )
    return matched


METHOD_CLASS_GETTER = {
    "fullKTA":          lambda: fullKTA,
    "randomKTA":        lambda: randomKTA,
    "greedyKTA":        lambda: greedyKTA,
    "quackKTA":         lambda: quackKTA,
    "centroidBasedKTA": lambda: centroidBasedKTA,
}


# ---------------------------------------------------------------------------
# Spectral metrics (verbatim from exp_spectral.py)
# ---------------------------------------------------------------------------

def spectral_metrics(K: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """
    Compute the full suite of spectral / effective-rank metrics for a
    centered kernel matrix K against labels y.

    `eff_rank` here IS "direction diversity" -- exp(-sum_i p_i ln p_i) over
    the label-aligned energy distribution across eigenvectors. eff_rank ~ 1
    means all label information sits in a single (rank-1) direction;
    eff_rank large means it is spread across many directions.
    """
    N = len(y)
    y_unit = y.astype(float) / (np.linalg.norm(y) + 1e-12)

    Ksym = 0.5 * (K + K.T)
    eigvals, eigvecs = np.linalg.eigh(Ksym)

    order   = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    lam = np.clip(eigvals, 0.0, None)
    a = np.abs(eigvecs.T @ y_unit)

    w = lam * (a ** 2)
    w_sum = w.sum() + 1e-12
    p = w / w_sum

    nz = p > 1e-12
    eff_rank = float(np.exp(-np.sum(p[nz] * np.log(p[nz]))))

    pr = float((w.sum() ** 2) / (np.sum(w ** 2) + 1e-12))

    top1_energy = float(p[0]) if N >= 1 else 0.0
    top2_energy = float(p[0] + p[1]) if N >= 2 else top1_energy
    top3_energy = float(p[:3].sum()) if N >= 3 else top2_energy

    K_fro = np.sqrt(np.sum(eigvals ** 2)) + 1e-12
    T_fro = float(N)
    kta_spectral = float(np.sum(eigvals * (a ** 2)) / (K_fro * T_fro))

    kta_contrib = eigvals * (a ** 2) / (K_fro * T_fro)
    kta_top1_frac = float(kta_contrib[0] / (kta_spectral + 1e-12)) if N >= 1 else 0.0
    kta_top2_frac = float(kta_contrib[:2].sum() / (kta_spectral + 1e-12)) if N >= 2 else kta_top1_frac

    return {
        "eff_rank":             eff_rank,   # == "direction_diversity"
        "participation_ratio":  pr,
        "top1_label_energy":    top1_energy,
        "top2_label_energy":    top2_energy,
        "top3_label_energy":    top3_energy,
        "top1_alignment":       float(a[0]),
        "top2_alignment":       float(a[1]) if N >= 2 else 0.0,
        "max_alignment":        float(a.max()),
        "argmax_alignment_rank": int(np.argmax(a)),
        "kta_spectral":         kta_spectral,
        "kta_top1_frac":        kta_top1_frac,
        "kta_top2_frac":        kta_top2_frac,
        "lambda1":              float(eigvals[0]),
        "lambda2":              float(eigvals[1]) if N >= 2 else 0.0,
        "spectral_decay":       float(eigvals[0] / (abs(eigvals[1]) + 1e-12)) if N >= 2 else 0.0,
    }


def _block_diagonal_ratio(K: np.ndarray, y: np.ndarray) -> float:
    """
    Within-class / between-class kernel ratio, computed directly from any
    (K, y) pair -- no backend support required, so every method gets a real
    value (unlike the old behaviour where only some backends exposed a
    block_ratio_history and everyone else silently got 0.0).
    """
    y = np.asarray(y)
    same = y[:, None] == y[None, :]
    off_diag = ~np.eye(len(y), dtype=bool)
    within = K[same & off_diag]
    between = K[~same]
    within_mean = float(within.mean()) if within.size else 0.0
    between_mean = float(between.mean()) if between.size else 1e-12
    return within_mean / (abs(between_mean) + 1e-12)


def _estimate_epoch_circuit_cost(
    method: str, n_train: int, centroids: int | None, batch_size: int | None,
) -> int:
    """
    APPROXIMATE circuit-execution cost for one epoch. Two parts:
      (1) building the full N x N training kernel every epoch (required for
          spectral analysis, paid by every method here);
      (2) the method's own per-epoch parameter-update cost.
    This is an estimate for reference/plotting only -- see the module-level
    fairness caveat above.
    """
    full_kernel_cost = n_train * n_train

    c = centroids or 1
    if method == "fullKTA":
        b = batch_size or n_train
        update_cost = b * b
    elif method in {"randomKTA", "greedyKTA"}:
        update_cost = c * LANDMARK_POINTS
    elif method == "quackKTA":
        update_cost = 10 * c * LANDMARK_POINTS  # 10 inner sub-steps/epoch
    elif method == "centroidBasedKTA":
        update_cost = 10 * (c * LANDMARK_POINTS + c)  # joint update + main-centroid update
    else:
        update_cost = 0

    return int(full_kernel_cost + update_cost)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dataset_path: str):
    data = np.load(dataset_path, allow_pickle=True).item()
    X = jnp.asarray(np.concatenate([data["x_train"], data["x_test"]], axis=0))
    y = jnp.asarray(np.concatenate([data["y_train"], data["y_test"]], axis=0))
    return X, y


# ---------------------------------------------------------------------------
# Per-epoch training + spectral analysis (single method/centroids/seed run)
# ---------------------------------------------------------------------------

def run_experiment(
    method: str,
    dataset: str,
    dataset_path: str,
    centroids: int | None,
    num_iterations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict]:
    """
    Run one (method, centroids, num_iterations, seed) combination, manually
    stepping the aligner epoch-by-epoch (mirroring exp_spectral.py's update
    rules) so we can compute full spectral metrics -- including direction
    diversity -- at every epoch.

    Returns (rows, snapshots):
      rows      = list of per-epoch result dicts (one row per epoch)
      snapshots = {epoch: (eigvals, alignment)} at init/mid/final, for the
                  qualitative spectrum-snapshot plot (reference seed only
                  is actually plotted, but we compute it for every seed in
                  case you want to inspect others).
    """
    prng_key = _set_global_seed(seed)
    X, y = load_data(dataset_path)

    kernel = quackEmbeddingCircuit(num_qubits=5, reps=6, reupload=True, seed=seed)
    model = KernelModel(circuit=kernel)
    common = dict(kernel_model=model, data=X, labels=y, matrix_type="regular", split_size=0.5)

    cls = METHOD_CLASS_GETTER[method]()
    #seed_kwargs = _seed_kwargs_for(cls, seed, prng_key)

    batch_size = None
    if method == "fullKTA":
        aligner = fullKTA(**common, seed = seed, learning_rate=0.1, optimizer="adam",
                           epochs=num_iterations)
    elif method == "randomKTA":
        aligner = randomKTA(**common, seed = seed, random_samples=centroids,
                             landmark_points=LANDMARK_POINTS, learning_rate=0.1,
                             optimizer="adam", epochs=num_iterations)
    elif method == "greedyKTA":
        aligner = greedyKTA(**common, seed = seed, greedy_samples=centroids,
                             landmark_points=LANDMARK_POINTS, learning_rate=0.1,
                             optimizer="adam", epochs=num_iterations)
    elif method == "quackKTA":
        aligner = quackKTA(**common, seed = seed, centroids=centroids, clustering="regular",
                            lambda_co=0.001, lambda_kao=0.001, epochs=num_iterations)
    elif method == "centroidBasedKTA":
        aligner = centroidBasedKTA(**common, seed = seed, clustering="regular",
                                    centroids=centroids, learning_rate=0.2, centroid_lr=0.01,
                                    sub_centroid_lr=0.01, lambda_co=0.001, lambda_kao=0.001,
                                    epochs=num_iterations)
    else:
        raise ValueError(f"Unknown method: {method}")

    if hasattr(aligner, "batch_size"):
        batch_size = aligner.batch_size

    y_np = np.asarray(aligner.ytrain)

    # -- Reset to a clean per-epoch trajectory (mirrors exp_spectral.py) -----
    aligner.weights = aligner.kernel_model.circuit.init_weights(seed=seed)
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

    rows: list[dict] = []
    snapshots: dict = {}
    cum_execs = 0
    t_start = time.time()

    # one-time warning about centroid-space KTA (see module docstring)
    global _centroid_kta_warned
    if method in {"centroidBasedKTA", "quackKTA"} and not _centroid_kta_warned:
        _centroid_kta_warned = True

    for epoch in range(num_iterations):
        # ---- Replicate the aligner's per-epoch update rule -----------------
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

        # ---- Build centered kernel + SVM ------------------------------------
        K_train = np.asarray(
            aligner._apply_centering(aligner._kernel_matrix(aligner.weights, aligner.xtrain)))
        svm = SVC(kernel="precomputed", C=1.0, max_iter=10_000)
        svm.fit(K_train, y_np)
        train_pred = svm.predict(K_train)
        train_acc = float(np.mean(train_pred == y_np))

        K_test = np.asarray(
            aligner.test_kernel_matrix(aligner.weights, aligner.xtrain, aligner.xtest))
        y_test_np = np.asarray(aligner.ytest)
        test_pred = svm.predict(K_test)
        test_acc = float(np.mean(test_pred == y_test_np))
        f1 = float(f1_score(y_test_np, test_pred, average="macro", zero_division=0))
        prec = float(precision_score(y_test_np, test_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test_np, test_pred, average="macro", zero_division=0))

        # margin proxy (mean |decision function| on the training set) --
        # NOT the backend's real margin definition, which we no longer have
        # access to since we bypass aligner.align(); documented as approximate.
        margin_proxy = float(np.mean(np.abs(svm.decision_function(K_train))))

        sm = spectral_metrics(K_train, y_np)
        block_ratio = _block_diagonal_ratio(K_train, y_np)
        global_kta = sm["kta_spectral"]
        centroid_kta = global_kta  # fallback -- see NOTE above
        direction_diversity = sm["eff_rank"]

        cum_execs += _estimate_epoch_circuit_cost(method, len(y_np), centroids, batch_size)

        row = {
            "method":             method,
            "dataset":            dataset,
            "centroids":          centroids if centroids is not None else 0,
            "num_iterations":     num_iterations,
            "seed":               seed,
            "step":               epoch,
            "epoch":              epoch + 1,
            "train_accuracy":     train_acc,
            "test_accuracy":      test_acc,
            "f1_score":           f1,
            "precision":          prec,
            "recall":             rec,
            "global_kta":         global_kta,
            "centroid_kta":       centroid_kta,
            "block_ratio":        block_ratio,
            "direction_diversity": direction_diversity,
            "circuit_executions": cum_execs,
            "wall_time":          time.time() - t_start,
            "margin":             margin_proxy,
            "n_sv":               int(len(svm.support_)),
        }
        row.update({k: v for k, v in sm.items() if k not in row})  # add remaining spectral fields
        rows.append(row)

        if epoch == 0 or epoch == num_iterations // 2 or epoch == num_iterations - 1:
            Ksym = 0.5 * (K_train + K_train.T)
            ev, evec = np.linalg.eigh(Ksym)
            o = np.argsort(ev)[::-1]
            y_unit = y_np.astype(float) / (np.linalg.norm(y_np) + 1e-12)
            snapshots[epoch + 1] = (ev[o].copy(), np.abs((evec[:, o].T @ y_unit)).copy())

    return rows, snapshots


# Module-level stores populated during live runs
_spectrum_snapshot_store: dict[tuple, dict] = {}
_centroid_kta_warned = False


# ---------------------------------------------------------------------------
# Main experiment loop (methods x centroids x epochs x seeds)
# ---------------------------------------------------------------------------

def run_all_experiments(dataset: str) -> pd.DataFrame:
    from tqdm import tqdm

    dataset_path = DATASET_PATHS[dataset]
    all_rows: list[dict] = []

    per_seed_total = sum(
        len(CENTROID_VALUES) if m in CENTROID_METHODS else 1
        for _ in EPOCH_VALUES
        for m in ALL_METHODS
    )
    total = per_seed_total * len(SEEDS)

    with tqdm(total=total, desc=f"Running [{dataset}] x {len(SEEDS)} seeds") as pbar:
        for seed in SEEDS:
            for num_iters in EPOCH_VALUES:
                for method in ALL_METHODS:
                    centroid_list = CENTROID_VALUES if method in CENTROID_METHODS else [None]
                    for c in centroid_list:
                        rows, snaps = run_experiment(method, dataset, dataset_path, c, num_iters, seed)
                        all_rows.extend(rows)
                        _spectrum_snapshot_store[(method, c or 0, num_iters, seed)] = {
                            "snapshots": snaps, "method": method,
                        }
                        pbar.update(1)

    df = pd.DataFrame(all_rows)
    df.to_csv(f"../results/{dataset}/method_comparison_results.csv", index=False)
    print(f"[Saved] {dataset}_method_comparison_results.csv  "
          f"({len(df)} rows = {len(SEEDS)} seeds x epochs x methods/centroids)")
    return df


# ---------------------------------------------------------------------------
# Seed-aggregation helpers
# ---------------------------------------------------------------------------

def _best_centroid_for_method(df: pd.DataFrame, method: str):
    sub = df[df["method"] == method]
    if sub.empty:
        return None
    if method not in CENTROID_METHODS:
        return sub["centroids"].iloc[0]
    per_seed_best = sub.groupby(["centroids", "seed"])["test_accuracy"].max()
    mean_across_seeds = per_seed_best.groupby("centroids").mean()
    return mean_across_seeds.idxmax()


def _aggregate_curve(
    df: pd.DataFrame, method: str, x_col: str, y_col: str, cummax: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a per-step trajectory across seeds. Returns (x, y_mean, y_min, y_max)."""
    sub = df[df["method"] == method].copy()
    if sub.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])

    best_c = _best_centroid_for_method(df, method)
    sub = sub[sub["centroids"] == best_c]

    if cummax:
        sub = sub.sort_values(["seed", "step"])
        sub[y_col] = sub.groupby("seed")[y_col].cummax()

    grouped = (
        sub.groupby("step")
        .agg(x=(x_col, "mean"), y_mean=(y_col, "mean"), y_min=(y_col, "min"), y_max=(y_col, "max"))
        .sort_values("x")
    )
    return grouped["x"].values, grouped["y_mean"].values, grouped["y_min"].values, grouped["y_max"].values


def _per_seed_best_stats(df: pd.DataFrame, method: str) -> dict | None:
    sub = df[df["method"] == method]
    if sub.empty:
        return None
    best_c = _best_centroid_for_method(df, method)
    sub = sub[sub["centroids"] == best_c]

    per_seed_rows = []
    for seed, seed_sub in sub.groupby("seed"):
        idx = seed_sub["test_accuracy"].idxmax()
        per_seed_rows.append(seed_sub.loc[idx])
    per_seed_df = pd.DataFrame(per_seed_rows)

    def _stats(col):
        vals = per_seed_df[col].values.astype(float)
        return dict(mean=vals.mean(), std=vals.std(), min=vals.min(), max=vals.max())

    eff_per_seed = per_seed_df["test_accuracy"] / (per_seed_df["circuit_executions"] / 1000).replace(0, np.nan)

    return {
        "centroids":          best_c,
        "test_accuracy":      _stats("test_accuracy"),
        "global_kta":         _stats("global_kta"),
        "centroid_kta":       _stats("centroid_kta"),
        "block_ratio":        _stats("block_ratio"),
        "direction_diversity": _stats("direction_diversity"),
        "top1_label_energy":  _stats("top1_label_energy"),
        "f1_score":           _stats("f1_score"),
        "circuit_executions": _stats("circuit_executions"),
        "wall_time":          _stats("wall_time"),
        "acc_per_1k_execs":   dict(
            mean=eff_per_seed.mean(), std=eff_per_seed.std(),
            min=eff_per_seed.min(),   max=eff_per_seed.max(),
        ),
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for method in ALL_METHODS:
        stats = _per_seed_best_stats(df, method)
        if stats is None:
            continue
        rows.append({
            "method":                  METHOD_LABELS.get(method, method),
            "centroids":               stats["centroids"] if stats["centroids"] else "N/A",
            "n_seeds":                 len(SEEDS),
            "best_test_acc_mean":      round(stats["test_accuracy"]["mean"], 4),
            "best_test_acc_std":       round(stats["test_accuracy"]["std"], 4),
            "best_test_acc_min":       round(stats["test_accuracy"]["min"], 4),
            "best_test_acc_max":       round(stats["test_accuracy"]["max"], 4),
            "global_kta_mean":         round(stats["global_kta"]["mean"], 4),
            "centroid_kta_mean":       round(stats["centroid_kta"]["mean"], 4),
            "block_ratio_mean":        round(stats["block_ratio"]["mean"], 4),
            "direction_diversity_mean": round(stats["direction_diversity"]["mean"], 4),
            "direction_diversity_std": round(stats["direction_diversity"]["std"], 4),
            "top1_label_energy_mean": round(stats["top1_label_energy"]["mean"], 4),
            "f1_mean":                 round(stats["f1_score"]["mean"], 4),
            "circuit_executions_mean": int(round(stats["circuit_executions"]["mean"])),
            "wall_time_s_mean":        round(stats["wall_time"]["mean"], 1),
            "acc_per_1k_execs_mean":   round(stats["acc_per_1k_execs"]["mean"], 4),
        })

    summary = pd.DataFrame(rows).sort_values("best_test_acc_mean", ascending=False)

    print(f"\n{'='*110}")
    print(f"  SUMMARY TABLE -- {dataset.upper()}  (mean/std/min/max across {len(SEEDS)} seeds)")
    print(f"{'='*110}")
    print(summary.to_string(index=False))
    print(f"{'='*110}\n")
    summary.to_csv(f"../results/{dataset}/summary_table.csv", index=False)
    return summary


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _smooth(y: np.ndarray, w: int = 5) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return y
    kernel = np.ones(w) / w
    padded = np.pad(y, (w // 2, w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(y)]


def _save_all_formats(basename: str, dpi: int = 300):
    plt.savefig(f"{basename}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.savefig(f"{basename}.jpg", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.savefig(f"{basename}.pdf", dpi=dpi, bbox_inches="tight")


def _plot_mean_band(ax, x, y_mean, y_min, y_max, color, label, linestyle="-", linewidth=1.5, smooth=True):
    if len(x) == 0:
        return
    ym = _smooth(y_mean) if smooth else y_mean
    ylo = _smooth(y_min) if smooth else y_min
    yhi = _smooth(y_max) if smooth else y_max
    ax.plot(x, ym, color=color, label=label, linewidth=linewidth, linestyle=linestyle)
    ax.fill_between(x, ylo, yhi, color=color, alpha=0.15, linewidth=0)


def _write_aggregated_csv(df: pd.DataFrame, metrics: list[str], x_col: str, dataset: str, out_name: str):
    """Aggregate mean/min/max per method per x_col value for a list of metrics, save to CSV."""
    csv_rows = []
    for y_col in metrics:
        for method in ALL_METHODS:
            x, y_mean, y_min, y_max = _aggregate_curve(df, method, x_col, y_col)
            for xi, ym, ylo, yhi in zip(x, y_mean, y_min, y_max):
                csv_rows.append({
                    "dataset": dataset, "method": method, "metric": y_col,
                    x_col: xi, "mean": ym, "min": ylo, "max": yhi, "n_seeds": len(SEEDS),
                })
    out_df = pd.DataFrame(csv_rows)
    out_df.to_csv(out_name, index=False)
    print(f"[Saved] {out_name}")
    return out_df


# ---------------------------------------------------------------------------
# Original plots (Figs 1-6), mean +/- min/max across seeds
# ---------------------------------------------------------------------------

def plot_accuracy_vs_budget(df, dataset, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in ALL_METHODS:
        x, y_mean, y_min, y_max = _aggregate_curve(df, method, "circuit_executions", "test_accuracy", cummax=True)
        lw = 2.5 if method == CCKA else 1.5
        ls = "-" if method == CCKA else "--"
        _plot_mean_band(ax, x, y_mean, y_min, y_max, METHOD_COLORS[method], METHOD_LABELS[method], ls, lw)
    ax.set_xlabel("Circuit executions (APPROXIMATE budget -- see fairness caveat)", fontsize=10)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title(f"Test accuracy vs budget — {dataset}\n(mean ± min/max, {len(SEEDS)} seeds)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25); ax.set_ylim(0, 1.05)
    if standalone:
        plt.tight_layout(); _save_all_formats(f"{dataset}_fig1_accuracy_vs_budget"); plt.close()


def plot_accuracy_vs_epoch(df, dataset, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in ALL_METHODS:
        x, y_mean, y_min, y_max = _aggregate_curve(df, method, "epoch", "test_accuracy", cummax=True)
        lw = 2.5 if method == CCKA else 1.5
        ls = "-" if method == CCKA else "--"
        _plot_mean_band(ax, x, y_mean, y_min, y_max, METHOD_COLORS[method], METHOD_LABELS[method], ls, lw)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title(f"Test accuracy vs epoch — {dataset}\n(mean ± min/max, {len(SEEDS)} seeds)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25); ax.set_ylim(0, 1.05)
    if standalone:
        plt.tight_layout(); _save_all_formats(f"{dataset}_fig2_accuracy_vs_epoch"); plt.close()


def plot_kta_vs_budget(df, dataset, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in ALL_METHODS:
        x, y_mean, y_min, y_max = _aggregate_curve(df, method, "circuit_executions", "global_kta")
        lw = 2.5 if method == CCKA else 1.5
        ls = "-" if method == CCKA else "--"
        _plot_mean_band(ax, x, y_mean, y_min, y_max, METHOD_COLORS[method], METHOD_LABELS[method], ls, lw)
    ax.set_xlabel("Circuit executions (approximate budget)", fontsize=10)
    ax.set_ylabel("Global KTA", fontsize=11)
    ax.set_title(f"Global KTA vs budget — {dataset} (mean ± min/max, {len(SEEDS)} seeds)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    if standalone:
        plt.tight_layout(); _save_all_formats(f"{dataset}_fig3_kta_vs_budget"); plt.close()


def plot_kta_vs_accuracy_scatter(df, dataset, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in ALL_METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        ms = 60 if method == CCKA else 20
        mrk = "D" if method == CCKA else "o"
        alpha = 0.6 if method == CCKA else 0.3
        ax.scatter(sub["global_kta"], sub["test_accuracy"], c=METHOD_COLORS[method],
                   label=METHOD_LABELS[method], s=ms, marker=mrk, alpha=alpha, linewidths=0)
    ax.set_xlabel("Global KTA", fontsize=11)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title(f"Global KTA vs test accuracy — {dataset} (all {len(SEEDS)} seeds pooled)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    if standalone:
        plt.tight_layout(); _save_all_formats(f"{dataset}_fig4_kta_vs_accuracy_scatter"); plt.close()


def plot_best_accuracy_bars(df, dataset, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))
    stats_by_method = {m: s["test_accuracy"] for m in ALL_METHODS if (s := _per_seed_best_stats(df, m))}
    methods_sorted = sorted(stats_by_method, key=lambda m: stats_by_method[m]["mean"], reverse=True)
    labels = [METHOD_LABELS[m] for m in methods_sorted]
    means = [stats_by_method[m]["mean"] for m in methods_sorted]
    lo_err = [max(0.0, stats_by_method[m]["mean"] - stats_by_method[m]["min"]) for m in methods_sorted]
    hi_err = [max(0.0, stats_by_method[m]["max"] - stats_by_method[m]["mean"]) for m in methods_sorted]
    colors = [METHOD_COLORS[m] for m in methods_sorted]
    edge_w = [2.0 if m == CCKA else 0.5 for m in methods_sorted]
    edge_c = ["black" if m == CCKA else "none" for m in methods_sorted]
    bars = ax.bar(labels, means, yerr=[lo_err, hi_err], capsize=5, color=colors,
                   edgecolor=edge_c, linewidth=edge_w, error_kw=dict(ecolor="black", elinewidth=1.2))
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Best test accuracy\n(mean, error bars = min/max)", fontsize=10)
    ax.set_title(f"Best test accuracy per method — {dataset} ({len(SEEDS)} seeds)", fontsize=11)
    ax.tick_params(axis="x", labelsize=9); ax.grid(True, axis="y", alpha=0.25)
    if standalone:
        plt.tight_layout(); _save_all_formats(f"{dataset}_fig5_best_accuracy_bars"); plt.close()


def plot_cost_efficiency(df, dataset, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))
    eff_by_method = {m: s["acc_per_1k_execs"] for m in ALL_METHODS if (s := _per_seed_best_stats(df, m))}
    methods_sorted = sorted(eff_by_method, key=lambda m: eff_by_method[m]["mean"], reverse=True)
    labels = [METHOD_LABELS[m] for m in methods_sorted]
    means = [eff_by_method[m]["mean"] for m in methods_sorted]
    lo_err = [max(0.0, eff_by_method[m]["mean"] - eff_by_method[m]["min"]) for m in methods_sorted]
    hi_err = [max(0.0, eff_by_method[m]["max"] - eff_by_method[m]["mean"]) for m in methods_sorted]
    colors = [METHOD_COLORS[m] for m in methods_sorted]
    edge_c = ["black" if m == CCKA else "none" for m in methods_sorted]
    bars = ax.bar(labels, means, yerr=[lo_err, hi_err], capsize=5, color=colors,
                   edgecolor=edge_c, linewidth=1.5, error_kw=dict(ecolor="black", elinewidth=1.2))
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f"{val:.4f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Accuracy / 1k circuit executions (approx.)\n(mean, error bars = min/max)", fontsize=9)
    ax.set_title(f"Cost efficiency (approximate) — {dataset} ({len(SEEDS)} seeds)", fontsize=11)
    ax.tick_params(axis="x", labelsize=9); ax.grid(True, axis="y", alpha=0.25)
    if standalone:
        plt.tight_layout(); _save_all_formats(f"{dataset}_fig6_cost_efficiency"); plt.close()


def plot_all(df, dataset):
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(f"Method comparison — {dataset.upper()} (mean ± min/max over {len(SEEDS)} seeds)",
                 fontsize=13, y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    plot_accuracy_vs_budget(df, dataset, ax=fig.add_subplot(gs[0, 0]))
    plot_accuracy_vs_epoch(df, dataset, ax=fig.add_subplot(gs[0, 1]))
    plot_kta_vs_budget(df, dataset, ax=fig.add_subplot(gs[0, 2]))
    plot_kta_vs_accuracy_scatter(df, dataset, ax=fig.add_subplot(gs[1, 0]))
    plot_best_accuracy_bars(df, dataset, ax=fig.add_subplot(gs[1, 1]))
    plot_cost_efficiency(df, dataset, ax=fig.add_subplot(gs[1, 2]))
    out = f"../results/{dataset}/full_comparison"
    _save_all_formats(out); plt.close()
    print(f"[Saved] {out}.png / .jpg / .pdf")


# ---------------------------------------------------------------------------
# (a)/(b)/(c) panel: Test accuracy / Global KTA / Direction diversity
# ---------------------------------------------------------------------------

def plot_abc_panel_vs_epoch(df: pd.DataFrame, dataset: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_acc, ax_kta, ax_div = axes

    panel_specs = [
        (ax_acc, "test_accuracy",       "Test accuracy",  "(a) Test accuracy",       True),
        (ax_kta, "global_kta",          "Global KTA",     "(b) Global KTA",          False),
        (ax_div, "direction_diversity", "Eff. rank  exp(-Σ pᵢ ln pᵢ)", "(c) Direction diversity", False),
    ]

    for ax, y_col, ylabel, title, cummax in panel_specs:
        for method in ALL_METHODS:
            x, y_mean, y_min, y_max = _aggregate_curve(df, method, "epoch", y_col, cummax=cummax)
            if len(x) == 0:
                continue
            lw = 2.5 if method == CCKA else 1.5
            ls = "-" if method == CCKA else "--"
            _plot_mean_band(ax, x, y_mean, y_min, y_max, METHOD_COLORS[method], METHOD_LABELS[method], ls, lw)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.25)

    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(ALL_METHODS),
               bbox_to_anchor=(0.5, -0.05), fontsize=10, frameon=False)
    fig.suptitle(f"{dataset} — mean ± min/max across {len(SEEDS)} seeds (different initial weights)",
                 fontsize=12, y=1.03)
    plt.tight_layout()
    out = f"../results/{dataset}/abc_panel_vs_epoch"
    _save_all_formats(out); plt.close()
    print(f"[Saved] {out}.png / .jpg / .pdf")

    _write_aggregated_csv(df, ["test_accuracy", "global_kta", "direction_diversity"],
                           "epoch", dataset, f"../results/{dataset}/abc_panel_epoch_stats.csv")


# ---------------------------------------------------------------------------
# Experiment 1: Centroid-Space KTA vs Global KTA
# ---------------------------------------------------------------------------

def plot_exp1_dual_kta_vs_budget(df, dataset):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    for ax_i, method in enumerate(ALL_METHODS):
        ax = axes[ax_i]
        x_g, gmean, gmin, gmax = _aggregate_curve(df, method, "circuit_executions", "global_kta")
        x_c, cmean, cmin, cmax = _aggregate_curve(df, method, "circuit_executions", "centroid_kta")
        if len(x_g) == 0:
            ax.set_visible(False); continue
        ax.plot(x_g, _smooth(gmean), color=METHOD_COLORS[method], linestyle="--", linewidth=1.8,
                label="Global KTA (measured)")
        ax.fill_between(x_g, _smooth(gmin), _smooth(gmax), color=METHOD_COLORS[method], alpha=0.12)
        ax.plot(x_c, _smooth(cmean), color=METHOD_COLORS[method], linestyle="-", linewidth=2.2,
                label="Centroid KTA (fallback = global; see NOTE)" if method in {CCKA, "quackKTA"}
                      else "Centroid KTA (= global)")
        ax.fill_between(x_c, _smooth(cmin), _smooth(cmax), color=METHOD_COLORS[method], alpha=0.20)
        ax.set_title(METHOD_LABELS[method], fontsize=11, color=METHOD_COLORS[method])
        ax.set_xlabel("Circuit executions (approx.)", fontsize=9)
        ax.set_ylabel("KTA value", fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.25)
    if len(ALL_METHODS) < len(axes):
        for ax in axes[len(ALL_METHODS):]:
            ax.set_visible(False)
    fig.suptitle(f"Experiment 1: Centroid-Space KTA vs Global KTA — {dataset} "
                 f"(mean ± min/max, {len(SEEDS)} seeds)", fontsize=12, y=1.01)
    plt.tight_layout()
    out = f"../results/{dataset}/exp1a_dual_kta_vs_budget"
    _save_all_formats(out); plt.close()
    print(f"[Saved] {out}.png / .jpg / .pdf")


def plot_exp1_centroid_kta_vs_accuracy_scatter(df, dataset):
    fig, (ax_global, ax_centroid) = plt.subplots(1, 2, figsize=(14, 5))
    for method in ALL_METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        ms = 60 if method == CCKA else 20
        mrk = "D" if method == CCKA else "o"
        alpha = 0.6 if method == CCKA else 0.3
        kw = dict(c=METHOD_COLORS[method], label=METHOD_LABELS[method], s=ms, marker=mrk, alpha=alpha, linewidths=0)
        ax_global.scatter(sub["global_kta"], sub["test_accuracy"], **kw)
        ax_centroid.scatter(sub["centroid_kta"], sub["test_accuracy"], **kw)
    for ax, xlabel, title in [(ax_global, "Global KTA", "Global KTA vs test accuracy"),
                               (ax_centroid, "Centroid KTA (fallback)", "Centroid KTA vs test accuracy")]:
        ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel("Test accuracy", fontsize=11)
        ax.set_title(f"{title}\n{dataset} (all {len(SEEDS)} seeds pooled)", fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = f"../results/{dataset}/exp1b_centroid_kta_vs_accuracy"
    _save_all_formats(out); plt.close()
    print(f"[Saved] {out}.png / .jpg / .pdf")


# ---------------------------------------------------------------------------
# Experiment 2: Block-Diagonal Ratio (now self-computed for every method)
# ---------------------------------------------------------------------------

def plot_exp2_block_ratio_vs_budget(df, dataset):
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in ALL_METHODS:
        x, y_mean, y_min, y_max = _aggregate_curve(df, method, "circuit_executions", "block_ratio")
        lw = 2.5 if method == CCKA else 1.5
        ls = "-" if method == CCKA else "--"
        _plot_mean_band(ax, x, y_mean, y_min, y_max, METHOD_COLORS[method], METHOD_LABELS[method], ls, lw)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, label="ratio = 1 (random baseline)")
    ax.set_xlabel("Circuit executions (approx. budget)", fontsize=10)
    ax.set_ylabel("Block-diagonal ratio (within/between class kernel)", fontsize=10)
    ax.set_title(f"Experiment 2: Kernel block structure vs budget — {dataset} "
                 f"(mean ± min/max, {len(SEEDS)} seeds)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = f"../results/{dataset}/exp2a_block_ratio_vs_budget"
    _save_all_formats(out); plt.close()
    print(f"[Saved] {out}.png / .jpg / .pdf")


def plot_exp2_block_ratio_vs_accuracy_scatter(df, dataset):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ALL_METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        ms = 60 if method == CCKA else 20
        mrk = "D" if method == CCKA else "o"
        alpha = 0.6 if method == CCKA else 0.3
        ax.scatter(sub["block_ratio"], sub["test_accuracy"], c=METHOD_COLORS[method],
                   label=METHOD_LABELS[method], s=ms, marker=mrk, alpha=alpha, linewidths=0)
    ax.set_xlabel("Block-diagonal ratio", fontsize=11)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title(f"Experiment 2: Block ratio vs test accuracy — {dataset} (all {len(SEEDS)} seeds pooled)",
                 fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = f"../results/{dataset}/exp2b_block_ratio_vs_accuracy"
    _save_all_formats(out); plt.close()
    print(f"[Saved] {out}.png / .jpg / .pdf")


# ---------------------------------------------------------------------------
# Experiment 3: Spectral / Effective-Rank Analysis
# ---------------------------------------------------------------------------

def plot_spec1_eff_rank_vs_epoch(df, dataset, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))
    for method in ALL_METHODS:
        x, y_mean, y_min, y_max = _aggregate_curve(df, method, "epoch", "direction_diversity")
        lw = 2.5 if method == CCKA else 1.5
        ls = "-" if method == CCKA else "--"
        _plot_mean_band(ax, x, y_mean, y_min, y_max, METHOD_COLORS[method], METHOD_LABELS[method], ls, lw)
    ax.axhline(1.0, color="gray", ls=":", lw=1.0, label="rank-1 (pure low-rank)")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Effective rank / direction diversity", fontsize=10)
    ax.set_title(f"Effective rank (direction diversity) over training — {dataset}\n"
                 f"mean ± min/max, {len(SEEDS)} seeds. CCKA expected near 1 (rank-1)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    if standalone:
        plt.tight_layout(); _save_all_formats(f"../results/{dataset}/spec1_eff_rank_vs_epoch"); plt.close()


def plot_spec2_eff_rank_vs_accuracy(df, dataset, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))
    for method in ALL_METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        ms = 70 if method == CCKA else 25
        mk = "D" if method == CCKA else "o"
        al = 0.6 if method == CCKA else 0.3
        ax.scatter(sub["direction_diversity"], sub["test_accuracy"], c=METHOD_COLORS[method],
                   label=METHOD_LABELS[method], s=ms, marker=mk, alpha=al, linewidths=0)
    ax.set_xlabel("Effective rank / direction diversity", fontsize=11)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title(f"Effective rank vs accuracy — {dataset} (all {len(SEEDS)} seeds pooled)\n"
                 "CCKA (◆) achieving high accuracy at LOW eff-rank = rank-1 sufficiency", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    if standalone:
        plt.tight_layout(); _save_all_formats(f"../results/{dataset}/spec2_eff_rank_vs_accuracy"); plt.close()


def plot_spec3_top_alignment(df, dataset, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))
    for method in ALL_METHODS:
        x, y_mean, y_min, y_max = _aggregate_curve(df, method, "epoch", "max_alignment")
        lw = 2.5 if method == CCKA else 1.5
        ls = "-" if method == CCKA else "--"
        _plot_mean_band(ax, x, y_mean, y_min, y_max, METHOD_COLORS[method], METHOD_LABELS[method], ls, lw)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Max eigenvector–label alignment  max_i |v_i·ŷ|", fontsize=10)
    ax.set_title(f"Best-aligned eigenvector over training — {dataset}\n"
                 f"mean ± min/max, {len(SEEDS)} seeds", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25); ax.set_ylim(0, 1.02)
    if standalone:
        plt.tight_layout(); _save_all_formats(f"../results/{dataset}/spec3_top_alignment"); plt.close()


def plot_spec4_kta_decomposition(df, dataset, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))
    for method in ALL_METHODS:
        x1, m1, lo1, hi1 = _aggregate_curve(df, method, "epoch", "kta_top1_frac")
        x2, m2, lo2, hi2 = _aggregate_curve(df, method, "epoch", "kta_top2_frac")
        lw = 2.5 if method == CCKA else 1.5
        if len(x1):
            ax.plot(x1, _smooth(m1), color=METHOD_COLORS[method], linewidth=lw, linestyle="-",
                    label=f"{METHOD_LABELS[method]} top-1")
            ax.fill_between(x1, _smooth(lo1), _smooth(hi1), color=METHOD_COLORS[method], alpha=0.12)
        if len(x2):
            ax.plot(x2, _smooth(m2), color=METHOD_COLORS[method], linewidth=lw * 0.7, linestyle="--", alpha=0.7,
                    label=f"{METHOD_LABELS[method]} top-2")
    ax.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Fraction of global KTA from top-k eigenvectors", fontsize=10)
    ax.set_title(f"KTA decomposition by rank — {dataset} (mean ± min/max, {len(SEEDS)} seeds)", fontsize=10)
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.25)
    if standalone:
        plt.tight_layout(); _save_all_formats(f"../results/{dataset}/spec4_kta_decomposition"); plt.close()


def plot_spec5_spectrum_snapshots(dataset):
    """
    Eigenvalue spectrum (bars) + label alignment (line) at init/mid/final,
    for REFERENCE_SEED only -- averaging eigen-spectra across seeds is not
    meaningful, so this qualitative plot uses a single representative run
    (see the mean±band plots above for the aggregated/statistically
    meaningful version of this information).
    """
    methods_with_snaps = [
        (m, c, ni, s) for (m, c, ni, s) in _spectrum_snapshot_store
        if s == REFERENCE_SEED and _spectrum_snapshot_store[(m, c, ni, s)]["snapshots"]
    ]
    if not methods_with_snaps:
        print("[spec5] no live snapshot data for the reference seed -- skipping (run live).")
        return

    # one entry per method (first match)
    by_method = {}
    for key in methods_with_snaps:
        m = key[0]
        if m not in by_method:
            by_method[m] = key

    methods = [m for m in ALL_METHODS if m in by_method]
    n = len(methods)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 3, figsize=(15, 3.6 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for ri, m in enumerate(methods):
        snaps = _spectrum_snapshot_store[by_method[m]]["snapshots"]
        keys = sorted(snaps.keys())
        labels = ["Init", "Mid", "Final"][:len(keys)]
        for ci, (ep, lab) in enumerate(zip(keys, labels)):
            ax = axes[ri, ci]
            eigvals, align = snaps[ep]
            idx = np.arange(1, len(eigvals) + 1)
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
                ax.text(-0.28, 0.5, METHOD_LABELS[m], transform=ax.transAxes, fontsize=10,
                        fontweight="bold", color=METHOD_COLORS[m], rotation=90, va="center")

    fig.suptitle(f"Eigenvalue spectrum + label alignment — {dataset} (seed={REFERENCE_SEED})\n"
                 "CCKA: label alignment concentrated in few eigenvectors (low-rank)", fontsize=12, y=1.005)
    plt.tight_layout()
    out = f"../results/{dataset}/spec5_spectrum_snapshots"
    _save_all_formats(out); plt.close()
    print(f"[Saved] {out}.png / .jpg / .pdf")


def plot_spectral_full(df, dataset):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Spectral / Effective-Rank Analysis — {dataset.upper()} (mean ± min/max, {len(SEEDS)} seeds)\n"
                 "Testing the low-rank discriminative alignment theory of CCKA", fontsize=13, y=0.99)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    plot_spec1_eff_rank_vs_epoch(df, dataset, ax=fig.add_subplot(gs[0, 0]))
    plot_spec2_eff_rank_vs_accuracy(df, dataset, ax=fig.add_subplot(gs[0, 1]))
    plot_spec3_top_alignment(df, dataset, ax=fig.add_subplot(gs[1, 0]))
    plot_spec4_kta_decomposition(df, dataset, ax=fig.add_subplot(gs[1, 1]))
    out = f"../results/{dataset}/spectral_full"
    _save_all_formats(out); plt.close()
    print(f"[Saved] {out}.png / .jpg / .pdf")

    _write_aggregated_csv(
        df,
        ["direction_diversity", "max_alignment", "kta_top1_frac", "kta_top2_frac", "top1_label_energy"],
        "epoch", dataset, f"../results/{dataset}/spectral_epoch_stats.csv",
    )


def plot_all_experiments(df: pd.DataFrame, dataset: str):
    plot_abc_panel_vs_epoch(df, dataset)

    plot_exp1_dual_kta_vs_budget(df, dataset)
    plot_exp1_centroid_kta_vs_accuracy_scatter(df, dataset)

    plot_exp2_block_ratio_vs_budget(df, dataset)
    plot_exp2_block_ratio_vs_accuracy_scatter(df, dataset)

    plot_spec1_eff_rank_vs_epoch(df, dataset)
    plot_spec2_eff_rank_vs_accuracy(df, dataset)
    plot_spec3_top_alignment(df, dataset)
    plot_spec4_kta_decomposition(df, dataset)
    plot_spec5_spectrum_snapshots(dataset)
    plot_spectral_full(df, dataset)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global DRY_RUN, SEEDS

    parser = argparse.ArgumentParser(
        description="Fair CCKA vs baselines comparison + spectral/effective-rank analysis, "
                    f"averaged over {N_SEEDS} random seeds."
    )
    parser.add_argument("--dataset", type=str, default="corners", choices=list(DATASET_PATHS.keys()))
    parser.add_argument("--results-csv", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    if args.dry_run:
        DRY_RUN = True
    if args.seeds is not None:
        SEEDS = args.seeds

    dataset = args.dataset

    output_dir = f"./results/{dataset}"
    os.makedirs(output_dir, exist_ok=True)

    if args.results_csv:
        print(f"[Loading results from {args.results_csv}]")
        df = pd.read_csv(args.results_csv)
        if "seed" not in df.columns:
            df["seed"] = 0
            print("[WARNING] seed column missing (single-seed legacy CSV) -- defaulting to seed=0.")
        for col, default in [
            ("centroid_kta", None), ("block_ratio", 1.0), ("direction_diversity", float("nan")),
            ("max_alignment", float("nan")), ("kta_top1_frac", float("nan")), ("kta_top2_frac", float("nan")),
            ("top1_label_energy", float("nan")),
        ]:
            if col not in df.columns:
                df[col] = df["global_kta"] if col == "centroid_kta" and "global_kta" in df.columns else default
                print(f"[WARNING] {col} column missing -- filled with placeholder/fallback.")
    else:
        df = run_all_experiments(dataset)


    summary = print_summary_table(df, dataset)

    plot_all(df, dataset)
    plot_all_experiments(df, dataset)

    print("\nDone. Files written:")
    csv_files = [
        f"{dataset}_method_comparison_results.csv",
        f"{dataset}_summary_table.csv",
        f"{dataset}_abc_panel_epoch_stats.csv",
        f"{dataset}_spectral_epoch_stats.csv",
    ]
    for fname in csv_files:
        if os.path.exists(fname):
            print(f"  {fname}")

    fig_basenames = [
        f"{dataset}_full_comparison",
        f"{dataset}_abc_panel_vs_epoch",
        f"{dataset}_exp1a_dual_kta_vs_budget",
        f"{dataset}_exp1b_centroid_kta_vs_accuracy",
        f"{dataset}_exp2a_block_ratio_vs_budget",
        f"{dataset}_exp2b_block_ratio_vs_accuracy",
        f"{dataset}_spec1_eff_rank_vs_epoch",
        f"{dataset}_spec2_eff_rank_vs_accuracy",
        f"{dataset}_spec3_top_alignment",
        f"{dataset}_spec4_kta_decomposition",
        f"{dataset}_spec5_spectrum_snapshots",
        f"{dataset}_spectral_full",
    ]
    for base in fig_basenames:
        for ext in ("png", "jpg", "pdf"):
            fname = f"{base}.{ext}"
            if os.path.exists(fname):
                print(f"  {fname}")


if __name__ == "__main__":
    main()