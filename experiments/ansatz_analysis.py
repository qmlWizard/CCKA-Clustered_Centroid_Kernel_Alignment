"""Section 9: ansatz analysis (non-Ray; independent of any dataset).

For every ansatz: expressibility (KL divergence of the state-fidelity
distribution from Haar), entangling capability (mean Meyer-Wallach measure),
gradient variance of kernel entries vs qubits and vs depth (barren-plateau
check), and circuit depth / gate count / parameter count.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml

from ccka.circuits.angleEmbeddingKernel import quackEmbeddingCircuit
from ccka.circuits.ansatz import ansatz_names
from experiments.experiments import ROOT

LOGGER = logging.getLogger(__name__)


def _circuit(n_qubits: int, reps: int, ansatz: str):
    return quackEmbeddingCircuit(
        num_qubits=n_qubits, reps=reps, reupload=True, ansatz=ansatz
    )


def _state_fn(circ, n_qubits: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="numpy")
    def state(x, w):
        circ._build_circuit(x, w, apply_noise=False)
        return qml.state()

    return state


def meyer_wallach(state: np.ndarray, n: int) -> float:
    psi = state.reshape([2] * n)
    total = 0.0
    for k in range(n):
        rho = np.tensordot(
            np.moveaxis(psi, k, 0), np.conj(np.moveaxis(psi, k, 0)),
            axes=(list(range(1, n)), list(range(1, n))),
        ) if n > 1 else np.outer(psi, psi.conj())
        total += 1.0 - float(np.real(np.trace(rho @ rho)))
    return 2.0 * total / n


def expressibility_kl(fidelities: np.ndarray, dim: int, bins: int = 75) -> float:
    hist, edges = np.histogram(fidelities, bins=bins, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    cdf = lambda f: 1.0 - (1.0 - f) ** (dim - 1)  # Haar CDF
    q = cdf(edges[1:]) - cdf(edges[:-1])
    mask = (p > 0) & (q > 0)
    return float((p[mask] * np.log(p[mask] / q[mask])).sum())


def expressibility_and_entanglement(ansatz, n_qubits, reps, samples, rng):
    circ = _circuit(n_qubits, reps, ansatz)
    state = _state_fn(circ, n_qubits)
    n_w = 3 * reps * n_qubits
    fid, mw = [], []
    for _ in range(samples):
        x = rng.uniform(0, np.pi, n_qubits)
        a = np.asarray(state(x, rng.uniform(-np.pi, np.pi, n_w)))
        b = np.asarray(state(x, rng.uniform(-np.pi, np.pi, n_w)))
        fid.append(abs(np.vdot(a, b)) ** 2)
        mw.append(meyer_wallach(a, n_qubits))
    return expressibility_kl(np.asarray(fid), 2 ** n_qubits), float(np.mean(mw))


def gradient_variance(ansatz, n_qubits, reps, samples, rng):
    import jax
    import jax.numpy as jnp

    circ = _circuit(n_qubits, reps, ansatz)
    dev = qml.device("default.qubit", wires=n_qubits)
    qnode = qml.QNode(circ.kernel_circuit, dev, interface="jax", diff_method="backprop")
    kernel = jax.jit(qnode)
    grad = jax.jit(jax.grad(lambda w, x1, x2: kernel(x1, x2, w)))
    n_w = 3 * reps * n_qubits
    grads = []
    for _ in range(samples):
        w = jnp.asarray(rng.uniform(-np.pi, np.pi, n_w))
        x1 = jnp.asarray(rng.uniform(0, np.pi, n_qubits))
        x2 = jnp.asarray(rng.uniform(0, np.pi, n_qubits))
        grads.append(np.asarray(grad(w, x1, x2)))
    return float(np.var(np.stack(grads), axis=0).mean())


def circuit_costs(ansatz, n_qubits, reps):
    circ = _circuit(n_qubits, reps, ansatz)
    dev = qml.device("default.qubit", wires=n_qubits)
    qnode = qml.QNode(circ.kernel_circuit, dev)
    w = np.zeros(3 * reps * n_qubits)
    specs = qml.specs(qnode)(np.zeros(n_qubits), np.zeros(n_qubits), w)
    res = specs["resources"]
    return int(res.depth), int(res.num_gates), int(w.size)


def analyse_unit(ansatz: str, n: int, reps: int, samples: int,
                 max_state_qubits: int, seed: int) -> dict[str, Any]:
    """One (ansatz, qubits, depth) cell; seeded per cell so it is order-independent."""
    rng = np.random.default_rng([seed, n, reps, sum(map(ord, ansatz))])
    kl, mw = (
        expressibility_and_entanglement(ansatz, n, reps, samples, rng)
        if n <= max_state_qubits else (np.nan, np.nan)
    )
    depth, gates, params = circuit_costs(ansatz, n, reps)
    return {
        "ansatz": ansatz, "n_qubits": n, "reps": reps,
        "expressibility_kl": kl, "meyer_wallach": mw,
        "grad_variance": gradient_variance(ansatz, n, reps, samples, rng),
        "circuit_depth": depth, "gate_count": gates, "n_params": params,
    }


def run(args) -> list[dict[str, Any]]:
    rows = []
    for ansatz in args.ansatze:
        for n in args.qubits:
            for reps in args.depths:
                LOGGER.info("ansatz analysis %s n=%d L=%d", ansatz, n, reps)
                rows.append(analyse_unit(
                    ansatz, n, reps, args.samples, args.max_state_qubits, args.seed
                ))
    return rows


def plot(rows, out: Path, accuracy_csv: Path | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(rows)
    ref_reps = sorted(df["reps"].unique())[len(df["reps"].unique()) // 2]
    ref_n = sorted(df["n_qubits"].unique())[len(df["n_qubits"].unique()) // 2]

    def curves(sub, x, y, name, ylabel, logy):
        fig, ax = plt.subplots(figsize=(5, 4))
        for ansatz, g in sub.groupby("ansatz"):
            g = g.sort_values(x)
            ax.plot(g[x], g[y], marker="o", label=ansatz)
        ax.set_xlabel(x)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=7)
        fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    curves(df[df["reps"] == ref_reps], "n_qubits", "grad_variance",
           "grad_variance_vs_qubits", "Var[dK/dw]", True)
    curves(df[df["n_qubits"] == ref_n], "reps", "grad_variance",
           "grad_variance_vs_depth", "Var[dK/dw]", True)
    curves(df[df["reps"] == ref_reps].dropna(subset=["expressibility_kl"]),
           "n_qubits", "expressibility_kl", "expressibility_vs_qubits",
           "KL from Haar (lower = more expressive)", False)
    curves(df[df["reps"] == ref_reps].dropna(subset=["meyer_wallach"]),
           "n_qubits", "meyer_wallach", "entangling_capability",
           "mean Meyer-Wallach Q", False)
    curves(df[df["n_qubits"] == ref_n], "reps", "gate_count",
           "gate_count_vs_depth", "gate count", False)

    if accuracy_csv is not None and accuracy_csv.exists():
        acc = pd.read_csv(accuracy_csv)
        if {"ansatz", "method", "test_accuracy", "reps"} <= set(acc.columns):
            acc = acc[acc["method"] == "ccka"].groupby(["ansatz", "reps"])[
                "test_accuracy"].mean().reset_index()
            m = acc.merge(df[df["n_qubits"] == ref_n], on=["ansatz", "reps"])
            m = m.dropna(subset=["expressibility_kl"])
            if not m.empty:
                fig, ax = plt.subplots(figsize=(5, 4))
                for ansatz, g in m.groupby("ansatz"):
                    ax.scatter(g["expressibility_kl"], g["test_accuracy"], label=ansatz)
                ax.set_xlabel("expressibility KL from Haar")
                ax.set_ylabel("CCKA test accuracy")
                ax.legend(fontsize=7)
                fig.savefig(out / "expressibility_vs_accuracy.png", dpi=180, bbox_inches="tight")
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(5, 4))
                for ansatz, g in m.groupby("ansatz"):
                    ax.scatter(g["n_params"], g["test_accuracy"], label=ansatz)
                ax.set_xlabel("parameter count")
                ax.set_ylabel("CCKA test accuracy")
                ax.legend(fontsize=7)
                fig.savefig(out / "params_vs_accuracy.png", dpi=180, bbox_inches="tight")
                plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "ansatz_analysis")
    parser.add_argument("--ansatze", nargs="+", default=list(ansatz_names()))
    parser.add_argument("--qubits", type=int, nargs="+", default=[3, 5, 8, 10])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-state-qubits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accuracy-csv", type=Path, default=ROOT / "results" / "exp13_ansatz_depth.csv",
                        help="Optional exp13 output used for expressibility/params vs accuracy.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run(args)
    with (args.output_dir / "ansatz_metrics.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.output_dir, args.accuracy_csv)


if __name__ == "__main__":
    main()
