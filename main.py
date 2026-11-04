from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from ccka_experiments.experiments import (
    CONFIG_DIR,
    ROOT,
    iter_experiment_jobs,
    load_config,
    run_experiment_job,
)


LOGGER = logging.getLogger("ccka.runner")


# ============================================================
# LOGGING (tqdm-safe)
# ============================================================

class _TqdmHandler(logging.StreamHandler):
    """Emit log lines through tqdm.write so they don't corrupt the progress bar."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def setup_logging(level: int = logging.INFO) -> None:
    handler = _TqdmHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


# ============================================================
# CSV: config fields first, then metrics
# ============================================================

_OUTCOME_PREFIXES = (
    "test_",
    "validation",
    "best_validation",
    "circuit_executions",
    "wall_time",
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _scalar(value: Any) -> Any:
    """Make a value CSV-friendly (lists/dicts/objects -> str)."""
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:  # numpy scalar
        return value.item()
    return str(value)


def format_row(trial_id: int, job: dict[str, Any], result: dict[str, Any],
               job_wall: float, exp_name: str) -> dict[str, Any]:
    """exp_name, trial_id | cfg_* | run_* | metric_*

    cfg_*    : every job field (minus the data object) plus every non-outcome
               field the job returned (dataset, method_label, seed, ...).
    metric_* : every outcome field (test_*, validation*, circuit_executions*, wall_time*).
    """
    row: dict[str, Any] = {"exp_name": exp_name, "trial_id": f"{trial_id:05d}"}

    for key, value in job.items():
        row[f"cfg_{key}"] = _scalar(value)
    for key, value in result.items():
        if not key.startswith(_OUTCOME_PREFIXES):
            row.setdefault(f"cfg_{key}", _scalar(value))

    row["run_host"] = result.get("run_host") or platform.node()
    row["run_timestamp"] = datetime.now().isoformat(timespec="seconds")
    row["run_job_wall_time"] = job_wall

    for key, value in result.items():
        if key.startswith(_OUTCOME_PREFIXES):
            row[f"metric_{key}"] = _scalar(value)
    return row


def save_combined_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    first = [c for c in ("exp_name", "trial_id") if any(c in r for r in rows)]
    seen: set[str] = set()
    for r in rows:
        seen.update(r)
    cfg_cols = sorted(c for c in seen if c.startswith("cfg_"))
    run_cols = sorted(c for c in seen if c.startswith("run_"))
    metric_cols = sorted(c for c in seen if c.startswith("metric_"))
    other = sorted(seen - set(first) - set(cfg_cols) - set(run_cols) - set(metric_cols))
    fieldnames = first + cfg_cols + run_cols + other + metric_cols

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _experiment_filename(result: dict[str, Any]) -> str:
    fields = [
        result["dataset"],
        result["method_label"],
        result["noise_type"],
        f"noise-{result['noise_level']:g}",
        f"shots-{result['shots'] if result['shots'] is not None else 'exact'}",
        f"seed-{result['seed']}",
        f"K-{result['centroids_per_class'] or 'na'}",
        result["initialization"],
        result["ablation"],
    ]
    identity = {k: v for k, v in result.items() if not k.startswith(_OUTCOME_PREFIXES)}
    digest = hashlib.sha1(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    return _slug("_".join(map(str, fields))) + f"_{digest}.csv"


# ============================================================
# RAY DISTRIBUTION
# ============================================================

def _job_worker(job: dict[str, Any], data: Any) -> dict[str, Any]:
    """Runs on a Ray worker; never raises so one bad job can't kill the sweep."""
    setup_logging()
    t0 = time.perf_counter()
    try:
        # Match thread counts to the CPUs Ray assigned; defaults oversubscribe parallel workers.
        import ray

        cpus = ray.get_runtime_context().get_assigned_resources().get("CPU")
        if cpus:
            os.environ["OMP_NUM_THREADS"] = str(max(1, int(cpus)))
            try:
                import torch

                torch.set_num_threads(max(1, int(cpus)))
            except ImportError:
                pass
        result = run_experiment_job(job, data)
        return {"ok": True, "result": result, "wall": time.perf_counter() - t0}
    except Exception as e:
        return {
            "ok": False,
            "error": f"{e!r}\n{traceback.format_exc()}",
            "wall": time.perf_counter() - t0,
        }


def run_distributed(
    config: dict[str, Any],
    *,
    output_dir: Path,
    rcfg: dict[str, Any],
    exp_name: str,
    address: Optional[str] = None,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    jobs = list(iter_experiment_jobs(config, smoke=smoke))
    if not jobs:
        LOGGER.warning("Configuration produced no experiment jobs.")
        return []

    import ray

    output_dir.mkdir(parents=True, exist_ok=True)
    per_experiment_dir = output_dir / "experiments"
    csv_path = output_dir / f"{output_dir.name}.csv"

    init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
    if address == "local":
        address = None
    if address:
        init_kwargs["address"] = address
        wd = rcfg.get("working_dir")
        # Shipping the working dir is only needed for explicit remote clusters;
        # an auto-discovered cluster shares this filesystem.
        if wd and address != "auto":
            init_kwargs["runtime_env"] = {
                "working_dir": str(Path(wd).resolve()),
                "excludes": ["results", ".git", ".venv", "__pycache__", "docs"],
            }
    try:
        ray.init(**init_kwargs)
    except Exception as error:
        if address != "auto":
            raise
        LOGGER.warning(
            "Could not use the existing Ray cluster (%s); starting a local Ray runtime. "
            "Run `ray stop --force` to clear stale clusters.", error,
        )
        ray.shutdown()
        ray.init(ignore_reinit_error=True)
    print("Ray resources:", ray.cluster_resources(),
          "| nodes:", len([n for n in ray.nodes() if n["Alive"]]))

    # num_cpus_per_trial: null -> all CPUs on a single node, else 1.
    num_cpus = rcfg.get("num_cpus_per_trial")
    if num_cpus is None:
        alive = [n for n in ray.nodes() if n["Alive"]]
        num_cpus = ray.cluster_resources().get("CPU", 1) if len(alive) == 1 else 1

    remote_job = ray.remote(
        num_cpus=num_cpus,
        num_gpus=rcfg.get("num_gpus_per_trial", 0),
        max_retries=rcfg.get("max_retries", 0),
    )(_job_worker)

    # Each distinct prepared dataset goes into the object store once.
    data_refs: dict[int, Any] = {}
    queue = list(enumerate(jobs))
    max_in_flight = rcfg.get("max_in_flight") or len(queue)
    pending: list[Any] = []
    meta: dict[Any, tuple[int, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    pbar = tqdm(total=len(queue), desc=exp_name, unit="job")

    def submit_next() -> None:
        i, job = queue.pop(0)
        key = id(job["data"])
        if key not in data_refs:
            data_refs[key] = ray.put(job["data"])
        job_cfg = {k: v for k, v in job.items() if k != "data"}
        ref = remote_job.remote(job_cfg, data_refs[key])
        pending.append(ref)
        meta[ref] = (i, job_cfg)

    try:
        while queue and len(pending) < max_in_flight:
            submit_next()

        while pending:
            done, pending[:] = ray.wait(pending, num_returns=1)
            ref = done[0]
            i, job_cfg = meta.pop(ref)
            try:
                out = ray.get(ref)
            except Exception as e:  # worker/node died after exhausting retries
                out = {"ok": False, "error": repr(e), "wall": 0.0}
            pbar.update(1)

            if out["ok"]:
                result = out["result"]
                row = format_row(i, job_cfg, result, out["wall"], exp_name)
                rows.append(row)
                rows.sort(key=lambda r: r["trial_id"])
                save_combined_csv(rows, csv_path)  # incremental
                save_combined_csv([row], per_experiment_dir / _experiment_filename(result))
            else:
                failed.append({
                    "trial_id": f"{i:05d}",
                    **{f"cfg_{k}": _scalar(v) for k, v in job_cfg.items()},
                    "error": out["error"],
                })
                tqdm.write(f"Job {i} failed:\n{out['error']}")

            if queue:
                submit_next()
    except BaseException:
        for reference in pending:
            ray.cancel(reference, force=False)
        raise
    finally:
        pbar.close()
        ray.shutdown()

    if failed:
        save_combined_csv(failed, output_dir / "failed.csv")
        print(f"WARNING: {len(failed)} job(s) failed; see {output_dir / 'failed.csv'}")
    print(f"Results CSV: {csv_path.resolve()}  ({len(rows)}/{len(jobs)} jobs)")
    return rows


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run configured CCKA experiments distributed with Ray."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "exp01_main_benchmark.yaml")
    parser.add_argument("--address", default=None,
                        help="Ray address ('auto', 'local' or host:port); overrides ray.address in the YAML")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the job count and exit")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (defaults to results/<exp_name>_<timestamp>).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper()))
    config = load_config(args.config)

    exp_name = re.sub(r"[^\w\-.]", "_", str(config.get("exp_name", args.config.stem)))
    run_name = f"{exp_name}_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = args.output_dir or ROOT / "results" / run_name

    if config.get("experiment_type") == "decision_boundaries":
        from ccka_experiments.decision_boundaries import run_from_config

        paths = run_from_config(config, output_dir=output_dir)
        LOGGER.info("Saved %d decision-boundary figures under %s", len(paths), output_dir)
        return

    if args.dry_run:
        print(f"Total jobs: {sum(1 for _ in iter_experiment_jobs(config, smoke=args.smoke)):,}")
        return

    rcfg = config.get("ray") or {}
    rows = run_distributed(
        config,
        output_dir=output_dir,
        rcfg=rcfg,
        exp_name=exp_name,
        address=args.address or rcfg.get("address"),
        smoke=args.smoke,
    )
    LOGGER.info("Saved %d experiment results under %s", len(rows), output_dir)


if __name__ == "__main__":
    main()
