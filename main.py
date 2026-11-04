from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import ray
from tqdm import tqdm

from ccka_experiments.experiments import (
    CONFIG_DIR,
    ROOT,
    iter_experiment_jobs,
    load_config,
    run_experiment_job,
)


LOGGER = logging.getLogger("ccka.ray_runner")


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


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# ============================================================
# RAY RUNTIME (local / existing cluster / Slurm)
# ============================================================

class RayAgent:
    """Own the Ray runtime for local, existing-cluster, or Slurm execution."""

    def __init__(
        self,
        *,
        num_cpus_per_node: Optional[int] = None,
        port: int = 6379,
        startup_timeout: float = 180,
        local_num_cpus: Optional[int] = None,
        runtime_env: Optional[dict[str, Any]] = None,
    ) -> None:
        self.num_cpus_per_node = num_cpus_per_node or int(
            os.environ.get(
                "RAY_CPUS",
                os.environ.get(
                    "SLURM_CPUS_ON_NODE",
                    os.environ.get("SLURM_CPUS_PER_TASK", _available_cpus()),
                ),
            )
        )
        self.port = port
        self.startup_timeout = startup_timeout
        self.local_num_cpus = local_num_cpus
        self.runtime_env = runtime_env
        self.nodes: list[str] = []
        self.ray_address: Optional[str] = None
        self.local_mode = False
        self._processes: list[subprocess.Popen[bytes]] = []
        self._owns_cluster = False

    @staticmethod
    def slurm_available() -> bool:
        return bool(os.environ.get("SLURM_JOB_NODELIST")) and all(
            shutil.which(command) for command in ("srun", "scontrol")
        )

    def _slurm_node_count(self) -> int:
        nodelist = os.environ["SLURM_JOB_NODELIST"]
        return len(
            subprocess.check_output(
                ["scontrol", "show", "hostnames", nodelist], text=True
            ).split()
        )

    def initialise(self, *, address: str = "auto", expected_nodes: Optional[int] = None):
        if address == "local":
            self.local_mode = True
            expected_nodes = expected_nodes or 1
            self._connect_local()
            return self._runtime_summary(expected_nodes)

        configured_address = os.environ.get("RAY_ADDRESS") if address == "auto" else address

        if configured_address and configured_address != "auto":
            self.ray_address = configured_address
        elif self.slurm_available() and self._slurm_node_count() > 1:
            self._create_slurm_cluster()
            expected_nodes = expected_nodes or len(self.nodes)
        else:
            # Single node (or no Slurm): plain local Ray, no srun/ray start needed.
            self.local_mode = True
            expected_nodes = expected_nodes or 1
            self._connect_local()
            return self._runtime_summary(expected_nodes)

        self._connect_existing(expected_nodes=expected_nodes)
        return self._runtime_summary(expected_nodes)

    def _create_slurm_cluster(self) -> None:
        nodelist = os.environ["SLURM_JOB_NODELIST"]

        self.nodes = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist], text=True
        ).split()
        if not self.nodes:
            raise RuntimeError("SLURM_JOB_NODELIST did not contain any nodes.")

        head_node = self.nodes[0]
        node_count = len(self.nodes)

        head_ip = subprocess.check_output(
            ["srun", "--overlap", "-N1", "-n1", "-w", head_node, "hostname", "-I"],
            text=True,
        ).split()[0]

        self.ray_address = f"{head_ip}:{self.port}"
        os.environ["RAY_ADDRESS"] = self.ray_address

        LOGGER.info(
            "Starting Ray cluster: head=%s address=%s nodes=%d CPUs/node=%d",
            head_node, self.ray_address, node_count, self.num_cpus_per_node,
        )

        # SLURM_PROCID 0 -> Ray head; 1..N-1 -> workers (retry while head starts).
        worker_script = r"""
    set -eu

    if [ "${SLURM_PROCID}" = "0" ]; then
        exec ray start \
            --head \
            --node-ip-address="${RAY_HEAD_IP}" \
            --port="${RAY_PORT}" \
            --num-cpus="${RAY_CPUS}" \
            --disable-usage-stats \
            --block
    else
        attempt=1
        while [ "$attempt" -le 30 ]; do
            if ray start \
                --address="${RAY_ADDRESS}" \
                --num-cpus="${RAY_CPUS}" \
                --disable-usage-stats \
                --block; then
                exit 0
            fi

            echo "Ray worker startup attempt ${attempt} failed on $(hostname)" >&2
            attempt=$((attempt + 1))
            sleep 2
        done

        echo "Ray worker failed after 30 attempts on $(hostname)" >&2
        exit 1
    fi
    """

        env = os.environ.copy()
        env["RAY_HEAD_IP"] = head_ip
        env["RAY_PORT"] = str(self.port)
        env["RAY_CPUS"] = str(self.num_cpus_per_node)
        env["RAY_ADDRESS"] = self.ray_address
        # Use the driver's own ray binary on every node (avoids login-shell PATH issues).
        env["RAY_BIN"] = shutil.which("ray") or "ray"
        worker_script = worker_script.replace("ray start", '"${RAY_BIN}" start')

        command = [
            "srun",
            "--overlap",
            f"--nodes={node_count}",
            f"--ntasks={node_count}",
            "--ntasks-per-node=1",
            "--kill-on-bad-exit=1",
            "bash",
            "-c",  # was -lc: a login shell can reset PATH / conda env
            worker_script,
        ]
        self._start_process(command, "Ray cluster launch", env=env)
        self._owns_cluster = True

    def _start_process(self, command: list[str], label: str, env=None) -> None:
        log_path = Path(f"ray_cluster_{os.environ.get('SLURM_JOB_ID', 'local')}.log")
        log = log_path.open("wb")
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env
        )
        self._processes.append(process)
        time.sleep(3)
        if process.poll() is not None:
            raise RuntimeError(f"{label} failed to start; see {log_path}")
        LOGGER.info("%s logging to %s", label, log_path)

    def _connect_local(self) -> None:
        num_cpus = self.local_num_cpus or self.num_cpus_per_node
        LOGGER.info("Starting local Ray runtime with %d CPUs", num_cpus)
        ray.init(
            num_cpus=num_cpus,
            ignore_reinit_error=True,
            include_dashboard=False,
            logging_level="WARNING",
        )
        self.ray_address = "local"

    def _connect_existing(self, *, expected_nodes: Optional[int]) -> None:
        if not self.ray_address:
            raise RuntimeError("No Ray address is available.")
        init_kwargs: dict[str, Any] = {
            "address": self.ray_address,
            "ignore_reinit_error": True,
            "include_dashboard": False,
            "logging_level": "WARNING",
        }
        if self.runtime_env:
            init_kwargs["runtime_env"] = self.runtime_env

        deadline = time.monotonic() + self.startup_timeout
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                ray.init(**init_kwargs)
                break
            except Exception as error:
                last_error = error
                time.sleep(2)
        else:
            raise RuntimeError(
                f"Could not connect to Ray at {self.ray_address}: {last_error}"
            ) from last_error

        if expected_nodes:
            while time.monotonic() < deadline:
                alive = sum(1 for node in ray.nodes() if node.get("Alive"))
                if alive >= expected_nodes:
                    return
                time.sleep(2)
            LOGGER.warning(
                "Only %d/%d Ray nodes joined before the startup timeout.",
                sum(1 for node in ray.nodes() if node.get("Alive")),
                expected_nodes,
            )

    def _runtime_summary(self, expected_nodes: Optional[int]):
        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        total_cpus = int(ray.cluster_resources().get("CPU", 0))
        if total_cpus <= 0:
            raise RuntimeError("Connected to Ray, but Ray reports zero CPUs.")
        LOGGER.info(
            "Ray ready: address=%s nodes=%d CPUs=%d expected_nodes=%s",
            self.ray_address, len(alive_nodes), total_cpus, expected_nodes,
        )
        return total_cpus, alive_nodes

    def stop(self) -> None:
        if ray.is_initialized():
            ray.shutdown()

        if not self._owns_cluster and not self._processes:
            return

        for process in self._processes:
            if process.poll() is None:
                process.terminate()
        for process in self._processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        self._processes.clear()

        if self._owns_cluster and self.nodes:
            try:
                subprocess.run(
                    [
                        "srun", "--overlap",
                        f"--nodes={len(self.nodes)}",
                        f"--ntasks={len(self.nodes)}",
                        "--ntasks-per-node=1",
                        "ray", "stop", "--force",
                    ],
                    capture_output=True, text=True, timeout=60, check=False,
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                LOGGER.warning("Ray cleanup encountered an error: %s", exc)

        self._owns_cluster = False


def build_runtime_env(rcfg: dict[str, Any]) -> Optional[dict[str, Any]]:
    wd = rcfg.get("working_dir")
    if not wd:
        return None
    return {
        "working_dir": str(Path(wd).resolve()),
        "excludes": rcfg.get("excludes", ["results", ".git", "__pycache__", "docs"]),
    }


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
    smoke: bool = False,
) -> list[dict[str, Any]]:
    jobs = list(iter_experiment_jobs(config, smoke=smoke))
    if not jobs:
        LOGGER.warning("Configuration produced no experiment jobs.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    per_experiment_dir = output_dir / "experiments"
    csv_path = output_dir / f"{output_dir.name}.csv"

    alive_nodes = [n for n in ray.nodes() if n.get("Alive", False)]
    if not alive_nodes:
        raise RuntimeError("Ray reports no alive nodes.")
    total_cpus = ray.cluster_resources().get("CPU", 0.0)
    print("Ray resources:", ray.cluster_resources(), "| nodes:", len(alive_nodes))

    # num_cpus_per_trial: null -> all CPUs on a single node, else 1.
    num_cpus = rcfg.get("num_cpus_per_trial")
    if num_cpus is None:
        num_cpus = total_cpus if len(alive_nodes) == 1 else 1
    if num_cpus <= 0:
        raise RuntimeError("num_cpus_per_trial resolved to zero; cannot schedule jobs.")

    remote_job = ray.remote(
        num_cpus=num_cpus,
        num_gpus=rcfg.get("num_gpus_per_trial", 0),
        max_retries=rcfg.get("max_retries", 0),
    )(_job_worker)

    # Each distinct prepared dataset goes into the object store once.
    data_refs: dict[int, Any] = {}

    def submit(job: dict[str, Any]) -> Any:
        key = id(job["data"])
        if key not in data_refs:
            data_refs[key] = ray.put(job["data"])
        job_without_data = {k: v for k, v in job.items() if k != "data"}
        return remote_job.remote(job_without_data, data_refs[key])

    queue = list(enumerate(jobs))
    max_in_flight = rcfg.get("max_in_flight") or len(queue)
    pending: list[Any] = []
    meta: dict[Any, tuple[int, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    pbar = tqdm(total=len(queue), desc=exp_name, unit="job")

    def submit_next() -> None:
        i, job = queue.pop(0)
        ref = submit(job)
        pending.append(ref)
        meta[ref] = (i, {k: v for k, v in job.items() if k != "data"})

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
                LOGGER.info(
                    "Saved experiment %d/%d: %s / %s",
                    len(rows), len(jobs),
                    result.get("dataset"), result.get("method_label"),
                )
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
        description="Distribute configured CCKA experiments across a Ray cluster."
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
    address = args.address or rcfg.get("address") or "auto"

    agent = RayAgent(
        num_cpus_per_node=rcfg.get("num_cpus_per_node"),
        port=rcfg.get("port", 6379),
        startup_timeout=rcfg.get("startup_timeout", 180),
        local_num_cpus=rcfg.get("local_num_cpus"),
        # runtime_env only matters when attaching to a remote cluster by address
        runtime_env=build_runtime_env(rcfg) if address not in ("auto", "local") else None,
    )
    try:
        agent.initialise(address=address, expected_nodes=rcfg.get("expected_nodes"))
        rows = run_distributed(
            config, output_dir=output_dir, rcfg=rcfg, exp_name=exp_name, smoke=args.smoke
        )
        LOGGER.info("Saved %d experiment results under %s", len(rows), output_dir)
    finally:
        agent.stop()


if __name__ == "__main__":
    main()