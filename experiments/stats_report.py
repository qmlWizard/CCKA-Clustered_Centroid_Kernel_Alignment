"""Section 11 / 1 / 9: statistical reporting.

Reads result CSVs written by main.py and produces
  * mean +- std and 95% CI tables per (dataset, ansatz, method),
  * paired Wilcoxon tests of every method vs FullKTA and vs QUACK (paired by seed),
  * "best" markers only when the best method is significantly better than
    every other method (otherwise ties are listed, nothing is bolded),
  * average-rank / Nemenyi critical-difference plot,
  * Kendall's tau between method rankings of different ansatze + rank-stability plot,
  * a claims report for the synthetic datasets (CCKA - QUACK gap and p-value).

Hyper-parameters that vary within a method (e.g. K) are chosen by mean
*validation* accuracy, never by test accuracy.
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiments.datasets import SYNTHETIC_CLAIMS
from experiments.plot_results import OPTIONAL_IDENTITY_COLUMNS, load_results

METHOD_ORDER = ["fullKTA", "randomKTA", "quackKTA", "ccka"]
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}
ALPHA = 0.05


def prepare(results: pd.DataFrame) -> pd.DataFrame:
    """Noiseless, non-ablation rows; one hyper-parameter setting per cell,
    selected by validation accuracy."""
    df = results[
        results["noise_type"].isin(["none", "shots_only"])
        & (results["noise_level"] == 0)
        & results["ablation"].isin(["none", "full"])
        & (
            (results["method"] != "ccka")
            | (results["initialization"] == "regular")
        )
    ].copy()
    if "ansatz" not in df:
        df["ansatz"] = "crz_ring"
    df["ansatz"] = df["ansatz"].fillna("crz_ring")
    cfg_cols = ["centroids_per_class"] + [
        c for c in OPTIONAL_IDENTITY_COLUMNS
        if c in df and c not in {"ansatz", "n_classes"} and df[c].nunique(dropna=False) > 1
    ]
    df["_cfg"] = df[cfg_cols].astype(str).agg("|".join, axis=1)
    key = ["dataset", "ansatz", "method"]
    selector = (
        "best_validation_accuracy"
        if "best_validation_accuracy" in df
        else "validation_accuracy"
    )
    val = df.groupby(key + ["_cfg"])[selector].mean()
    best = val.groupby(level=key).idxmax()
    keep = pd.MultiIndex.from_tuples(list(best.values), names=key + ["_cfg"])
    idx = df.set_index(key + ["_cfg"]).index.isin(keep)
    return df[idx]


def ci95(values: np.ndarray) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / math.sqrt(n))


def paired_p(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(joined) < 3 or np.allclose(joined["a"], joined["b"]):
        return float("nan")
    try:
        return float(stats.wilcoxon(joined["a"], joined["b"]).pvalue)
    except ValueError:
        return float("nan")


def summary_table(df: pd.DataFrame, metric: str = "test_accuracy") -> pd.DataFrame:
    rows = []
    for (dataset, ansatz), cell in df.groupby(["dataset", "ansatz"]):
        per = {m: g.set_index("seed")[metric] for m, g in cell.groupby("method")}
        means = {m: s.mean() for m, s in per.items()}
        best = max(means, key=means.get)
        others = [m for m in per if m != best]
        pvals = {m: paired_p(per[best], per[m]) for m in others}
        supported = bool(others) and all(p < ALPHA for p in pvals.values())
        ties = [m for m, p in pvals.items() if not p < ALPHA]
        for m, s in per.items():
            rows.append({
                "dataset": dataset, "ansatz": ansatz, "method": m,
                "n_seeds": len(s), "mean": s.mean(), "std": s.std(ddof=1) if len(s) > 1 else np.nan,
                "ci95": ci95(s.values),
                "p_vs_fullKTA": paired_p(s, per["fullKTA"]) if "fullKTA" in per and m != "fullKTA" else np.nan,
                "p_vs_quackKTA": paired_p(s, per["quackKTA"]) if "quackKTA" in per and m != "quackKTA" else np.nan,
                "is_best": m == best,
                "best_supported": m == best and supported,
                "tied_with_best": m != best and m in ties,
            })
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame, metric: str) -> str:
    lines = [f"# {metric}: mean ± std [95% CI]", "",
             "**bold** = best and significantly better than all others (Wilcoxon, p<0.05); "
             "`≈` = statistically tied with the best method; `*`/`†` = p<0.05 vs FullKTA / QUACK.", ""]
    for (dataset, ansatz), cell in table.groupby(["dataset", "ansatz"]):
        lines += [f"## {dataset} / {ansatz}", "", "| method | n | mean ± std | 95% CI | p vs FullKTA | p vs QUACK |", "|---|---|---|---|---|---|"]
        for m in METHOD_ORDER:
            r = cell[cell["method"] == m]
            if r.empty:
                continue
            r = r.iloc[0]
            text = f"{r['mean']:.3f} ± {r['std']:.3f}"
            if r["best_supported"]:
                text = f"**{text}**"
            elif r["tied_with_best"] or r["is_best"]:
                text += " ≈" if r["tied_with_best"] else " (best, not significant)"
            sig = ("*" if r["p_vs_fullKTA"] < ALPHA else "") + ("†" if r["p_vs_quackKTA"] < ALPHA else "")
            lines.append(f"| {m} | {int(r['n_seeds'])} | {text}{sig} | ±{r['ci95']:.3f} | {r['p_vs_fullKTA']:.3g} | {r['p_vs_quackKTA']:.3g} |")
        lines.append("")
    return "\n".join(lines)


def rank_matrix(table: pd.DataFrame) -> pd.DataFrame:
    wide = table.pivot_table(index=["dataset", "ansatz"], columns="method", values="mean")
    wide = wide[[m for m in METHOD_ORDER if m in wide]].dropna()
    return wide.rank(axis=1, ascending=False)


def plot_cd(ranks: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    avg = ranks.mean().sort_values()
    k, n = len(avg), len(ranks)
    cd = NEMENYI_Q05.get(k, 2.85) * math.sqrt(k * (k + 1) / (6 * n))
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.set_xlim(k + 0.5, 0.5)
    ax.set_ylim(0, 1)
    ax.hlines(0.6, 1, k, color="k")
    for i in range(1, k + 1):
        ax.vlines(i, 0.57, 0.63, color="k")
        ax.text(i, 0.67, str(i), ha="center")
    for j, (m, r) in enumerate(avg.items()):
        ax.plot([r, r], [0.6, 0.45 - 0.08 * j], color="tab:blue")
        ax.text(r, 0.42 - 0.08 * j, f"{m} ({r:.2f})", ha="center", va="top", fontsize=8)
    ax.hlines(0.92, 1, 1 + cd, color="r", lw=2)
    ax.text(1 + cd / 2, 0.95, f"CD={cd:.2f} (N={n})", ha="center", color="r", fontsize=8)
    ax.axis("off")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def kendall_by_ansatz(table: pd.DataFrame) -> pd.DataFrame:
    wide = table.pivot_table(index="dataset", columns=["ansatz", "method"], values="mean")
    ansatze = sorted({a for a, _ in wide.columns})
    out = pd.DataFrame(index=ansatze, columns=ansatze, dtype=float)
    for a, b in itertools.product(ansatze, ansatze):
        taus = []
        for dataset in wide.index:
            ra, rb = wide.loc[dataset, a], wide.loc[dataset, b]
            both = pd.concat([ra, rb], axis=1).dropna()
            if len(both) >= 3:
                tau = stats.kendalltau(both.iloc[:, 0], both.iloc[:, 1]).statistic
                if not np.isnan(tau):
                    taus.append(tau)
        out.loc[a, b] = float(np.mean(taus)) if taus else np.nan
    return out


def plot_rank_stability(tau: pd.DataFrame, ranks: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(tau.values.astype(float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(tau)), tau.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(tau)), tau.index)
    for i in range(len(tau)):
        for j in range(len(tau)):
            ax.text(j, i, f"{tau.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="mean Kendall τ (method ranking)")
    fig.savefig(out / "rank_stability_kendall.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    per_ansatz = ranks.groupby(level="ansatz").mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in per_ansatz.columns:
        ax.plot(per_ansatz.index, per_ansatz[m], marker="o", label=m)
    ax.invert_yaxis()
    ax.set_ylabel("average rank (1 = best)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend()
    fig.savefig(out / "rank_stability_across_ansatze.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def claims_report(df: pd.DataFrame) -> str:
    lines = ["# Synthetic dataset claims", "", "| dataset | behaviour | expected | CCKA | QUACK | FullKTA | CCKA-QUACK | p |", "|---|---|---|---|---|---|---|---|"]
    for dataset, claim in SYNTHETIC_CLAIMS.items():
        cell = df[df["dataset"] == dataset]
        if cell.empty:
            continue
        per = {m: g.groupby("seed")["test_accuracy"].mean() for m, g in cell.groupby("method")}
        if "ccka" not in per or "quackKTA" not in per:
            continue
        gap = (per["ccka"] - per["quackKTA"]).dropna()
        f = lambda m: f"{per[m].mean():.3f}" if m in per else "-"
        lines.append(f"| {dataset} | {claim.behaviour} | {claim.expected} | {f('ccka')} | {f('quackKTA')} | {f('fullKTA')} | {gap.mean():+.3f} | {paired_p(per['ccka'], per['quackKTA']):.3g} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="CSV file or directory of result CSVs.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/stats"))
    parser.add_argument("--metrics", nargs="+", default=["test_accuracy", "test_f1_macro"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = prepare(load_results(args.results))
    for metric in args.metrics:
        if metric not in df:
            continue
        table = summary_table(df, metric)
        table.to_csv(args.output_dir / f"summary_{metric}.csv", index=False)
        (args.output_dir / f"summary_{metric}.md").write_text(to_markdown(table, metric), encoding="utf-8")
        if metric == "test_accuracy":
            ranks = rank_matrix(table)
            if len(ranks) >= 2:
                ranks.to_csv(args.output_dir / "ranks.csv")
                plot_cd(ranks, args.output_dir / "critical_difference.png")
                tau = kendall_by_ansatz(table)
                tau.to_csv(args.output_dir / "kendall_tau_ansatze.csv")
                if len(tau) >= 2:
                    plot_rank_stability(tau, ranks, args.output_dir)
    (args.output_dir / "claims_report.md").write_text(claims_report(df), encoding="utf-8")


if __name__ == "__main__":
    main()
