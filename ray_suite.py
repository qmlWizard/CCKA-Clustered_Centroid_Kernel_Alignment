"""Run the whole CCKA experiment suite in parallel on one Ray cluster.

Every experiment config (configs/exp*.yaml) is expanded into independent jobs
and *all* of them, together with the sharded analyses (budget, kernel,
ansatz, scalability), are submitted to the same pool at once, so the cluster
stays busy until the last task finishes. Stats and plots run at the end.

    python ray_suite.py --address auto                  # everything
    python ray_suite.py --only exp06_k_sweep kernel     # subset
    python ray_suite.py --address local --smoke         # tiny local check

Scalability timings must not compete for CPUs. If nodes were started with the
custom resource ``timing`` and no schedulable CPUs (see jobs/run_all.slurm)
those tasks run there, one at a time per node with the whole node to
themselves; otherwise they run alone after
everything else.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import traceback
from argparse import Namespace
from itertools import zip_longest
from pathlib import Path
from typing import Any

import ray

from ray_cluster import init_ray_cluster, cleanup_ray
from experiments import ansatz_analysis, kernel_analysis, scalability
from experiments.ansatz_analysis import analyse_unit
from experiments.experiments import (
    ROOT,
    iter_experiment_jobs,
    load_config,
    run_experiment_job,
)
from main import _experiment_filename, _write_csv

LOGGER = logging.getLogger("ccka.suite")
ANALYSES = ("budget", "kernel", "ansatz", "scalability")
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    "PYTHONPATH": str(ROOT),
}


def _experiment_configs(root: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted((root / "configs").glob("exp*.yaml"))}


def _safe(fn):
    """Wrap a task so one failure is reported instead of killing the suite."""
    def wrapper(*args, **kwargs):
        try:
            return {"ok": True, "value": fn(*args, **kwargs)}
        except Exception:
            return {"ok": False, "error": traceback.format_exc()}
    wrapper.__name__ = getattr(fn, "__name__", "task")
    return wrapper


def _budget_task(dataset: str, seed: int, out: str, centroids: int,
                 epochs: int | None) -> str:
    cmd = [
        sys.executable, "-m", "experiments.accuracy_vs_executions",
        "--config", str(ROOT / "configs" / "base.yaml"),
        "--dataset", dataset, "--seed", str(seed),
        "--methods", "fullKTA", "quackKTA", "ccka",
        "--centroids", str(centroids), "--target-accuracy", "0.9",
        "--run-full-iterations", "--output-prefix", out,
    ]
    if epochs is not None:
        cmd += ["--max-iterations", str(epochs)]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True, text=True)
    return out


def _shrink(config: dict[str, Any]) -> None:
    """Tiny version of a config for --smoke: same code paths, seconds per job."""
    training = config["training"]
    training.update(epochs=1, reps=1, max_samples=48, inference_landmarks=4,
                    quack_inner_steps=1, ccka_inner_steps=1)
    training["n_qubits"] = 3
    training["centroids"] = training["centroids"][:2]
    config["seeds"] = config["seeds"][:1]
    config["datasets"] = config["datasets"][:2]
    config["ansatze"] = config.get("ansatze", ["crz_ring"])[:2]
    config["noise"]["levels"] = [0.01]
    config["noise"]["shots"] = config["noise"]["shots"][:1]
    config["noise"]["types"] = config["noise"]["types"][:3]
    grid = config.get("training_grid") or {}
    for key, values in grid.items():
        grid[key] = [8, 16] if key == "n_train" else values[:2]
        if key == "n_qubits":
            grid[key] = [2, 3]


def _submit_experiments(names, configs, args, runner):
    """Round-robin over configs so every experiment makes progress at once."""
    per_config: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        config = load_config(configs[name])
        if args.seeds:
            config["seeds"] = args.seeds
        if args.smoke:
            _shrink(config)
        per_config[name] = list(iter_experiment_jobs(config))
        LOGGER.info("%s: %d jobs", name, len(per_config[name]))
    data_refs: dict[int, Any] = {}
    pending: dict[Any, str] = {}
    for batch in zip_longest(*per_config.values()):
        for name, job in zip(per_config, batch):
            if job is None:
                continue
            key = id(job["data"])
            if key not in data_refs:
                data_refs[key] = ray.put(job["data"])
            light = {k: v for k, v in job.items() if k != "data"}
            ref = runner.remote(light, data_refs[key])
            pending[ref] = name
    return pending, {n: len(j) for n, j in per_config.items()}


def run(args: Namespace) -> int:
    root = ROOT
    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    configs = _experiment_configs(root)
    wanted = args.only or (list(configs) + list(ANALYSES))
    exp_names = [n for n in wanted if n in configs and n not in args.skip]
    analyses = [n for n in wanted if n in ANALYSES and n not in args.skip]
    unknown = set(wanted) - set(configs) - set(ANALYSES)
    if unknown:
        raise SystemExit(f"Unknown names: {sorted(unknown)}")

    try:
        init_ray_cluster(address=args.address, wait_for_all_nodes=(args.address == "auto"))
        resources = ray.cluster_resources()
        LOGGER.info("Cluster: %s CPUs on %d nodes", resources.get("CPU", 0),
                    len([n for n in ray.nodes() if n.get("Alive")]))
        env = {"env_vars": SINGLE_THREAD_ENV}

        exp_runner = ray.remote(_safe(run_experiment_job)).options(
            num_cpus=1, runtime_env=env)
        pending, totals = _submit_experiments(exp_names, configs, args, exp_runner)
        owner = dict(pending)  # ref -> experiment name
        kind: dict[Any, tuple] = {r: ("exp", n) for r, n in owner.items()}

        smoke = args.smoke
        seeds = args.seeds or ([0] if smoke else [0, 1, 2, 3, 4])
        light_task = lambda fn: ray.remote(_safe(fn)).options(num_cpus=1, runtime_env=env)
        kernel_rows_expected = 0
        if "kernel" in analyses:
            settings = dict(load_config(root / "configs" / "base.yaml")["training"])
            settings["max_samples"] = 40 if smoke else 200
            if smoke:
                settings.update(epochs=1, reps=1, inference_landmarks=4)
            ansatze = ["crz_ring"] if smoke else list(
                kernel_analysis_ansatze())
            remote = light_task(kernel_analysis.analyse_task)
            for dataset in ("corners", "checkerboard"):
                for ansatz in ansatze:
                    for seed in seeds:
                        ref = remote.remote(
                            dataset, ansatz, seed, settings, 4,
                            str(out_root / "kernel_analysis"), seed == seeds[0])
                        kind[ref] = ("kernel", f"{dataset}/{ansatz}/{seed}")
                        kernel_rows_expected += 1
        if "ansatz" in analyses:
            remote = light_task(analyse_unit)
            from ccka.circuits.ansatz import ansatz_names
            qubits, depths = ([3], [1, 2]) if smoke else ([3, 5, 8, 10], [1, 2, 4, 6, 8, 10])
            for ansatz in (["crz_ring"] if smoke else ansatz_names()):
                for n in qubits:
                    for reps in depths:
                        ref = remote.remote(ansatz, n, reps, 10 if smoke else 100, 8, 0)
                        kind[ref] = ("ansatz", f"{ansatz}/{n}/{reps}")
        if "budget" in analyses:
            remote = light_task(_budget_task)
            for dataset in ("corners", "checkerboard", "spirals"):
                for seed in (seeds if smoke else range(10)):
                    out = out_root / "budget" / f"{dataset}_seed{seed}"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    ref = remote.remote(dataset, seed, str(out), 4, 2 if smoke else None)
                    kind[ref] = ("budget", f"{dataset}/{seed}")

        scal_args = None
        scal_tasks: list[dict[str, Any]] = []
        if "scalability" in analyses:
            scal_args = scalability_args(smoke, seeds)
            scal_tasks = scalability.build_tasks(scal_args)
        has_timing = resources.get("timing", 0) > 0
        scal_remote = ray.remote(_safe(scalability.run_task))
        if has_timing:
            scal_remote = scal_remote.options(num_cpus=0, resources={"timing": 1})
            for task in scal_tasks:
                kind[scal_remote.remote(task)] = ("scal", task["sweep"])
            scal_tasks = []

        results: dict[str, list] = {}
        done_by_exp: dict[str, int] = {n: 0 for n in exp_names}
        failures: list[str] = []
        exp_rows: dict[str, list[dict[str, Any]]] = {n: [] for n in exp_names}

        def flush_scal(rows):
            if rows:
                scalability._write(out_root / "scalability" / "scalability.csv", rows)

        def drain(refs):
            refs = list(refs)
            while refs:
                ready, refs = ray.wait(refs, num_returns=1)
                ref = ready[0]
                tag, name = kind[ref]
                result = ray.get(ref)
                if not result["ok"]:
                    failures.append(f"{tag}:{name}\n{result['error']}")
                    LOGGER.error("FAILED %s:%s\n%s", tag, name, result["error"])
                    continue
                value = result["value"]
                if tag == "exp":
                    row = value
                    exp_rows[name].append(row)
                    folder = out_root / name
                    _write_csv(folder / _experiment_filename(row), [row])
                    done_by_exp[name] += 1
                    if done_by_exp[name] % 25 == 0 or done_by_exp[name] == totals[name]:
                        _write_csv(folder / "summary.csv", exp_rows[name])
                        LOGGER.info("%s: %d/%d", name, done_by_exp[name], totals[name])
                elif tag == "scal":
                    results.setdefault("scal", []).append(value)
                    if len(results["scal"]) % 10 == 0:
                        flush_scal(results["scal"])
                else:
                    results.setdefault(tag, []).append(value)

        drain(kind)

        if scal_tasks:  # no dedicated timing nodes: run alone with every CPU
            node_cpus = int(max(n["Resources"].get("CPU", 1) for n in ray.nodes() if n.get("Alive")))
            LOGGER.info("Running scalability alone, %d CPUs per task", node_cpus)
            alone = ray.remote(_safe(scalability.run_task)).options(num_cpus=node_cpus)
            refs = []
            for task in scal_tasks:
                ref = alone.remote(task)
                kind[ref] = ("scal", task["sweep"])
                refs.append(ref)
            drain(refs)

        if results.get("scal"):
            flush_scal(results["scal"])
            scalability.plot(results["scal"], out_root / "scalability")
        if results.get("kernel"):
            rows = [row for rows in results["kernel"] for row in rows]
            kernel_analysis.write_outputs(rows, out_root / "kernel_analysis")
        if results.get("ansatz"):
            import csv
            folder = out_root / "ansatz_analysis"
            folder.mkdir(parents=True, exist_ok=True)
            rows = results["ansatz"]
            with (folder / "ansatz_metrics.csv").open("w", newline="", encoding="utf-8") as h:
                writer = csv.DictWriter(h, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            ansatz_analysis.plot(rows, folder, out_root / "exp13_ansatz_depth" / "summary.csv")

        if not args.no_report:
            _report(out_root, exp_names, args)

        if failures:
            (out_root / "failures.log").write_text("\n\n".join(failures), encoding="utf-8")
            LOGGER.error("%d task(s) failed; see %s", len(failures), out_root / "failures.log")
        return 1 if failures else 0
    finally:
        try:
            ray.shutdown()
        finally:
            cleanup_ray()


def kernel_analysis_ansatze():
    from ccka.circuits.ansatz import ansatz_names
    return ansatz_names()


def scalability_args(smoke: bool, seeds) -> Namespace:
    if smoke:
        return Namespace(
            config=ROOT / "configs" / "base.yaml", datasets=["corners"],
            ansatz="crz_ring", seeds=list(seeds)[:1], epochs=1, centroids=2,
            qubits=3, layers=1, n_train=16, max_full_n=1000,
            n_values=[16, 32], qubit_values=[3], layer_values=[1],
            landmark_values=[4],
        )
    return Namespace(
        config=ROOT / "configs" / "base.yaml", datasets=["corners", "checkerboard"],
        ansatz="crz_ring", seeds=[0, 1, 2], epochs=3, centroids=4, qubits=5,
        layers=2, n_train=64, max_full_n=1000,
        n_values=[100, 500, 1000, 5000, 10000], qubit_values=[3, 5, 8, 10],
        layer_values=[1, 2, 4, 6, 8, 10], landmark_values=[4, 8, 16, 32, 64],
    )


def _report(out_root: Path, exp_names, args) -> None:
    """Stats tables and plots from whatever finished (runs on the driver)."""
    def call(*cmd):
        proc = subprocess.run([sys.executable, "-m", *cmd], cwd=ROOT,
                              capture_output=True, text=True)
        if proc.returncode:
            LOGGER.error("%s failed:\n%s", cmd[0], proc.stderr[-2000:])

    for name, target in (("exp01_main_benchmark", "main_benchmark"),
                         ("exp15_synthetic_claims", "synthetic_claims")):
        if (out_root / name / "summary.csv").exists():
            call("experiments.stats_report", str(out_root / name),
                 "--output-dir", str(out_root / "stats" / target))
    call("experiments.suite_plots", "--results-dir", str(out_root),
         "--output-dir", str(out_root / "plots"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--address", default="auto")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--only", nargs="+", help="Experiment configs / analyses to run.")
    parser.add_argument("--skip", nargs="+", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override seeds (analyses and, in --smoke, experiments).")
    parser.add_argument("--smoke", action="store_true", help="Tiny settings for a quick check.")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
