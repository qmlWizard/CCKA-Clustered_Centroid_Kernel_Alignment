from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import pytest

from ccka.circuits.angleEmbeddingKernel import quackEmbeddingCircuit
from ccka.circuits.ansatz import ansatz_names
from experiments import stats_report
from experiments.datasets import GENERATORS, SYNTHETIC_CLAIMS, generate_dataset
from experiments.experiments import (
    ABLATION_DATASETS,
    DATASETS,
    iter_experiment_jobs,
    load_config,
)
from experiments.multiclass import normalise_strategy
from experiments.plot_results import load_results

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("ansatz", ansatz_names())
def test_ansatz_kernel_is_valid(ansatz):
    circuit = quackEmbeddingCircuit(num_qubits=3, reps=2, reupload=True, ansatz=ansatz)
    dev = qml.device("default.qubit", wires=3)
    kernel = qml.QNode(circuit.kernel_circuit, dev)
    rng = np.random.default_rng(0)
    w = rng.uniform(-np.pi, np.pi, 3 * 2 * 3)
    x1, x2 = rng.uniform(0, np.pi, (2, 3))
    assert float(kernel(x1, x1, w)) == pytest.approx(1.0, abs=1e-6)
    assert float(kernel(x1, x2, w)) == pytest.approx(float(kernel(x2, x1, w)), abs=1e-6)
    assert 0.0 <= float(kernel(x1, x2, w)) <= 1.0 + 1e-9


def test_unknown_ansatz_rejected():
    with pytest.raises(Exception):
        quackEmbeddingCircuit(num_qubits=2, reps=1, ansatz="nope")


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_generators_shapes_and_labels(name):
    X, y = generate_dataset(name, 0, 120)
    assert len(X) == len(y) > 0
    assert len(np.unique(y)) >= 2


def test_every_synthetic_dataset_has_a_claim_and_is_registered():
    assert set(SYNTHETIC_CLAIMS) <= set(DATASETS)
    assert set(GENERATORS) == set(SYNTHETIC_CLAIMS)


def test_multiclass_strategy_aliases():
    assert normalise_strategy("ova") == normalise_strategy("ovr")
    assert normalise_strategy("one_vs_one") == normalise_strategy("ovo")
    with pytest.raises(ValueError):
        normalise_strategy("bogus")


def test_load_config_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "datasets: [a]\ntraining: {epochs: 5, reps: 2}\nseeds: [0]\n"
    )
    (tmp_path / "child.yaml").write_text(
        "base: base.yaml\ntraining: {epochs: 9}\n"
    )
    config = load_config(tmp_path / "child.yaml")
    assert config["training"] == {"epochs": 9, "reps": 2}
    assert config["datasets"] == ["a"]
    assert "base" not in config


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("exp*.yaml")))
def test_experiment_configs_are_valid(path):
    config = load_config(path)
    names = config["datasets"]
    assert set(names) <= set(DATASETS)
    assert len(config["seeds"]) >= 3
    unknown = set(config.get("ansatze", [])) - set(ansatz_names())
    assert not unknown
    variants = config["experiments"].get("ccka_ablations") or {}
    if variants:
        assert set(names) <= set(ABLATION_DATASETS)


def test_ablations_only_run_on_ablation_datasets():
    config = load_config(ROOT / "configs" / "exp03_ablation_structure.yaml")
    config["datasets"] = ["corners", "moons"]
    config["seeds"] = [0]
    config["training"]["max_samples"] = 40
    seen = {}
    for job in iter_experiment_jobs(config):
        seen.setdefault(job["dataset"], set()).add(job["ablation"])
    assert "main_only" in seen["corners"]
    assert seen["moons"] <= {"full", "none"}


def test_stats_helpers():
    kernel = np.eye(4)
    y = np.array([1, 1, -1, -1])
    assert stats_report is not None
    from experiments.kernel_analysis import effective_rank, kta, linear_cka

    assert effective_rank(kernel) == pytest.approx(4.0)
    assert -1 <= kta(np.outer(y, y).astype(float), y) <= 1
    assert kta(np.outer(y, y).astype(float), y) == pytest.approx(1.0)
    assert linear_cka(kernel, kernel) == pytest.approx(1.0)


def test_best_is_only_bold_when_significant():
    seeds = list(range(10))
    rows = []
    for method, offset in (("fullKTA", 0.0), ("quackKTA", 0.0), ("ccka", 0.1)):
        for s in seeds:
            rows.append({
                "dataset": "d", "ansatz": "crz_ring", "method": method,
                "seed": s, "test_accuracy": 0.6 + offset + 0.001 * s,
            })
    table = stats_report.summary_table(pd.DataFrame(rows))
    best = table[table["is_best"]].iloc[0]
    assert best["method"] == "ccka" and best["best_supported"]
    tie = pd.DataFrame(rows).assign(
        test_accuracy=lambda d: d["test_accuracy"].where(d["method"] != "ccka", 0.6 + 0.001 * d["seed"])
    )
    table = stats_report.summary_table(tie)
    assert not table["best_supported"].any()


def test_load_results_keeps_rows_from_different_ansatze(tmp_path):
    base = {
        "dataset": "d", "method": "ccka", "method_label": "CCKA", "ablation": "full",
        "initialization": "regular", "centroids_per_class": 2, "noise_type": "none",
        "noise_level": 0.0, "shots": None, "seed": 0, "test_accuracy": 0.5,
        "circuit_executions": 10, "wall_time_seconds": 1.0,
    }
    pd.DataFrame([{**base, "ansatz": "iqp"}, {**base, "ansatz": "separable"}]).to_csv(
        tmp_path / "a.csv", index=False
    )
    assert len(load_results(tmp_path / "a.csv")) == 2
