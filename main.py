from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
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


def _fmt_num(value: Any) -> str:
    try:
        return f"{value:g}"
    except (TypeError, ValueError):
        return str(value)


def _experiment_filename(result: dict[str, Any]) -> str:
    shots = result.get("shots")
    fields = [
        result.get("dataset", "na"),
        result.get("method_label", "na"),
        result.get("noise_type", "na"),
        f"noise-{_fmt_num(result.get('noise_level'))}",
        f"shots-{shots if shots is not None else 'exact'}",
        f"seed-{result.get('seed', 'na')}",
        f"K-{result.get('centroids_per_class') or 'na'}",
        result.get("initialization", "na"),
        result.get("ablation", "na"),
    ]
    identity = {k: v for k, v in result.items() if not k.startswith(_OUTCOME_PREFIXES)}
    digest = hashlib.sha1(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    return _slug("_".join(map(str, fields))) + f"_{digest}.csv"


# ============================================================
# RESOURCE DETECTION (SLURM-aware, no config file needed)
# ============================================================

@dataclass
class Resources:
    nodes: list[str]
    cpus_per_node: int
    gpus_per_node: int
    in_slurm: bool
    job_id: str = ""

    @property
    def multi_node(self) -> bool:
        return len(self.nodes) > 1

    @property
    def total_cpus(self) -> int:
        return len(self.nodes) * self.cpus_per_node

    def describe(self) -> str:
        where = f"SLURM job {self.job_id}" if self.in_slurm else "standalone host"
        return (f"{where}: {len(self.nodes)} node(s) x {self.cpus_per_node} CPU "
                f"x {self.gpus_per_node} GPU = {self.total_cpus} CPUs")


def _split_top_level(s: str) -> list[str]:
    """Split on commas that are not inside [...]."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p for p in parts if p]


def _expand_nodelist(nodelist: str) -> list[str]:
    """Expand 'cbcn[0065-0066,0118]' -> hostnames. Prefers `scontrol`, falls back to a parser."""
    try:
        out = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.split()
        if out:
            return out
    except (OSError, subprocess.SubprocessError):
        pass

    nodes: list[str] = []
    for part in _split_top_level(nodelist):
        m = re.fullmatch(r"(.*?)\[([^\]]+)\](.*)", part)
        if not m:
            nodes.append(part)
            continue
        prefix, body, suffix = m.groups()
        for item in body.split(","):
            if "-" in item:
                lo, hi = item.split("-", 1)
                width = len(lo)
                nodes += [f"{prefix}{i:0{width}d}{suffix}" for i in range(int(lo), int(hi) + 1)]
            else:
                nodes.append(f"{prefix}{item}{suffix}")
    return nodes


def _detect_cpus() -> int:
    for var in ("SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK"):
        value = os.environ.get(var)
        if value and value.isdigit():
            return int(value)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _detect_gpus() -> int:
    value = os.environ.get("SLURM_GPUS_ON_NODE")
    if value and value.isdigit():
        return int(value)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([v for v in visible.split(",") if v.strip()])
    return 0


def detect_resources() -> Resources:
    env = os.environ
    nodelist = env.get("SLURM_JOB_NODELIST") or env.get("SLURM_NODELIST")
    in_slurm = bool(env.get("SLURM_JOB_ID") and nodelist)
    nodes = _expand_nodelist(nodelist) if in_slurm and nodelist else []
    if not nodes:
        in_slurm = False
        nodes = [socket.gethostname().split(".")[0]]
    return Resources(
        nodes=nodes,
        cpus_per_node=_detect_cpus(),
        gpus_per_node=_detect_gpus(),
        in_slurm=in_slurm,
        job_id=env.get("SLURM_JOB_ID", ""),
    )


# ============================================================
# SELF-MANAGED RAY CLUSTER (head on this node, workers via srun)
# ============================================================

def _ray_bin() -> str:
    candidate = Path(sys.executable).parent / "ray"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("ray")
    if found:
        return found
    raise RuntimeError("Could not locate the `ray` executable next to the Python interpreter or on PATH.")


def _local_ip() -> str:
    host = socket.gethostname()
    try:
        for info in socket.getaddrinfo(host, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                return ip
    except socket.gaierror:
        pass
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=10).stdout.split()
        for ip in out:
            if "." in ip and not ip.startswith("127."):
                return ip
    except (OSError, subprocess.SubprocessError):
        pass
    return socket.gethostbyname(host)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class ManagedCluster:
    """Starts `ray start --head` locally and `ray start --address=...` on every other
    allocated node through srun. Tear down with stop()."""

    def __init__(self, res: Resources, log_dir: Path, srun_args: str = ""):
        self.res = res
        self.log_dir = log_dir
        self.srun_args = shlex.split(srun_args)
        self.ray = _ray_bin()
        self.address = ""
        self.procs: list[tuple[str, subprocess.Popen, Path]] = []
        self.temp_dir = f"/tmp/ray_{os.environ.get('USER', 'user')}_{res.job_id or os.getpid()}"
        self._head_started = False

        me = socket.gethostname().split(".")[0]
        names = [n.split(".")[0] for n in res.nodes]
        self.head_node = me if me in names else names[0]
        self.worker_nodes = [n for n in res.nodes if n.split(".")[0] != self.head_node]

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("RAY_USAGE_STATS_ENABLED", "0")
        # make `ccka_experiments` importable on every node (shared filesystem)
        import ccka_experiments
        pkg_parent = str(Path(ccka_experiments.__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [pkg_parent, env.get("PYTHONPATH", "")]))
        return env

    def start(self) -> str:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        env = self._env()
        ip, port = _local_ip(), _free_port()
        self.address = f"{ip}:{port}"
        cpus, gpus = self.res.cpus_per_node, self.res.gpus_per_node

        LOGGER.info("Starting Ray head on %s (%s)", self.head_node, self.address)
        with (self.log_dir / f"head_{self.head_node}.log").open("w") as log:
            subprocess.run(
                [self.ray, "start", "--head", f"--node-ip-address={ip}", f"--port={port}",
                 f"--num-cpus={cpus}", f"--num-gpus={gpus}", f"--temp-dir={self.temp_dir}",
                 "--include-dashboard=false"],
                check=True, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        self._head_started = True

        LOGGER.info("Launching Ray workers on %d node(s) via srun", len(self.worker_nodes))
        for node in self.worker_nodes:
            log_path = self.log_dir / f"worker_{node}.log"
            cmd = [
                "srun", "--nodes=1", "--ntasks=1", f"--cpus-per-task={cpus}",
                "-w", node, "--export=ALL", *self.srun_args,
                self.ray, "start", f"--address={self.address}",
                f"--num-cpus={cpus}", f"--num-gpus={gpus}", f"--temp-dir={self.temp_dir}",
                "--block",
            ]
            proc = subprocess.Popen(cmd, env=env, stdout=log_path.open("w"), stderr=subprocess.STDOUT)
            self.procs.append((node, proc, log_path))
        return self.address

    def wait_for_nodes(self, ray: Any, timeout: float) -> None:
        expected = len(self.res.nodes)
        t0, last_report, reported_dead = time.time(), 0.0, set()
        while True:
            alive = sum(1 for n in ray.nodes() if n["Alive"])
            if alive >= expected:
                LOGGER.info("All %d Ray nodes joined (%.0fs).", alive, time.time() - t0)
                return
            for node, proc, log_path in self.procs:
                if proc.poll() is not None and node not in reported_dead:
                    reported_dead.add(node)
                    tail = ""
                    try:
                        tail = "".join(log_path.read_text(errors="replace").splitlines(True)[-5:])
                    except OSError:
                        pass
                    LOGGER.warning("Worker launcher for %s exited (code %s). Log tail:\n%s",
                                   node, proc.returncode, tail)
            if time.time() - t0 > timeout:
                LOGGER.warning("Only %d/%d nodes joined within %.0fs; continuing with what is available "
                               "(late nodes will still be used once they join).", alive, expected, timeout)
                return
            if time.time() - last_report > 15:
                LOGGER.info("Waiting for Ray nodes: %d/%d joined", alive, expected)
                last_report = time.time()
            time.sleep(2)

    def stop(self) -> None:
        for _, proc, _ in self.procs:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.time() + 15
        for _, proc, _ in self.procs:
            try:
                proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._head_started:
            subprocess.run([self.ray, "stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ============================================================
# RAY DISTRIBUTION
# ============================================================

def _job_worker(
    job: dict[str, Any], data: Any, max_attempts: int = 2
) -> dict[str, Any]:
    """Runs on a Ray worker; never raises so one bad job can't kill the sweep."""
    setup_logging()
    t0 = time.perf_counter()
    error_text = ""
    for attempt in range(1, max_attempts + 1):
        try:
            # Match thread counts to the CPUs Ray assigned; defaults oversubscribe parallel workers.
            import ray

            cpus = ray.get_runtime_context().get_assigned_resources().get("CPU")
            if cpus:
                n = str(max(1, int(cpus)))
                for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                    os.environ[var] = n
                try:
                    import torch

                    torch.set_num_threads(int(n))
                except ImportError:
                    pass
            result = run_experiment_job(job, data)
            return {
                "ok": True,
                "result": result,
                "wall": time.perf_counter() - t0,
                "attempts": attempt,
            }
        except Exception as error:
            error_text = f"{error!r}\n{traceback.format_exc()}"
            if attempt < max_attempts:
                LOGGER.warning(
                    "Job for dataset %s, seed %s failed on attempt %d/%d; retrying: %r",
                    job.get("dataset", "unknown"),
                    job.get("seed", "unknown"),
                    attempt,
                    max_attempts,
                    error,
                )

    return {
        "ok": False,
        "error": error_text,
        "wall": time.perf_counter() - t0,
        "attempts": max_attempts,
    }


def _connect_existing(ray: Any, address: str, rcfg: dict[str, Any]) -> bool:
    """Connect to an already running cluster. Returns False only for 'auto' with nothing to connect to."""
    kwargs: dict[str, Any] = {"ignore_reinit_error": True, "address": address}
    wd = rcfg.get("working_dir")
    # Shipping the working dir is only needed for explicit remote clusters.
    if wd and address != "auto":
        kwargs["runtime_env"] = {
            "working_dir": str(Path(wd).resolve()),
            "excludes": ["results", ".git", ".venv", "__pycache__", "docs"],
        }
    try:
        ray.init(**kwargs)
        return True
    except Exception as error:
        if address != "auto":
            raise
        LOGGER.info("No existing Ray cluster found (%s); will start one.", error)
        ray.shutdown()
        return False


def run_distributed(
    config: dict[str, Any],
    *,
    output_dir: Path,
    rcfg: dict[str, Any],
    exp_name: str,
    address: Optional[str] = None,
    smoke: bool = False,
    cpus_per_trial: Optional[int] = None,
    srun_args: str = "",
    join_timeout: float = 300.0,
) -> list[dict[str, Any]]:
    jobs = list(iter_experiment_jobs(config, smoke=smoke))
    if not jobs:
        LOGGER.warning("Configuration produced no experiment jobs.")
        return []

    import ray

    output_dir.mkdir(parents=True, exist_ok=True)
    per_experiment_dir = output_dir / "experiments"
    csv_path = output_dir / f"{output_dir.name}.csv"

    res = detect_resources()
    LOGGER.info("Detected %s", res.describe())

    cluster: Optional[ManagedCluster] = None
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    pbar = None

    def flush_csv() -> None:
        rows.sort(key=lambda r: r["trial_id"])
        save_combined_csv(rows, csv_path)

    try:
        # ---- 1. get a cluster -------------------------------------------------
        connected = False
        if address and address != "local":
            connected = _connect_existing(ray, address, rcfg)

        if not connected:
            multi = res.multi_node and address != "local"
            if multi and shutil.which("srun") is None:
                LOGGER.warning("Multi-node allocation detected but `srun` is not available; using this node only.")
                multi = False
            if multi:
                cluster = ManagedCluster(res, output_dir / "ray_logs", srun_args)
                try:
                    ray.init(address=cluster.start(), ignore_reinit_error=True)
                    cluster.wait_for_nodes(ray, join_timeout)
                except Exception as error:
                    LOGGER.warning("Multi-node Ray startup failed (%r); falling back to single node.", error)
                    ray.shutdown()
                    cluster.stop()
                    cluster = None
                    multi = False
            if not multi:
                ray.init(num_cpus=res.cpus_per_node,
                         num_gpus=res.gpus_per_node or None,
                         ignore_reinit_error=True)

        alive = [n for n in ray.nodes() if n["Alive"]]
        total = ray.cluster_resources()
        total_cpus = int(total.get("CPU", 1))
        node_cpus = int(max((n.get("Resources", {}).get("CPU", 1) for n in alive), default=1))
        print("Ray resources:", total, "| nodes:", len(alive))

        # ---- 2. size the tasks from what the cluster actually has --------------
        if cpus_per_trial is None:
            cpus_per_trial = rcfg.get("num_cpus_per_trial")  # optional override in YAML
        if cpus_per_trial is None:
            cpus_per_trial = node_cpus
        cpus_per_trial = max(1, min(int(cpus_per_trial), node_cpus))

        gpus_per_trial = rcfg.get("num_gpus_per_trial", 0) or 0
        if gpus_per_trial and not total.get("GPU"):
            LOGGER.warning("num_gpus_per_trial=%s requested but no GPU found; using 0.", gpus_per_trial)
            gpus_per_trial = 0

        slots = max(1, total_cpus // cpus_per_trial)
        max_in_flight = rcfg.get("max_in_flight") or slots
        max_job_attempts = int(rcfg.get("max_job_attempts", 2))
        max_ray_retries = int(rcfg.get("max_retries", 2))
        max_in_flight = int(max_in_flight)
        if max_in_flight < 1:
            raise ValueError("ray.max_in_flight must be at least 1.")
        if max_job_attempts < 1:
            raise ValueError("ray.max_job_attempts must be at least 1.")
        if max_ray_retries < 0:
            raise ValueError("ray.max_retries must be non-negative.")
        LOGGER.info("%d jobs | %d CPU/job | %d parallel slots | max in flight %d",
                    len(jobs), cpus_per_trial, slots, max_in_flight)

        remote_job = ray.remote(
            num_cpus=cpus_per_trial,
            num_gpus=gpus_per_trial,
            max_retries=max_ray_retries,
        )(_job_worker)

        # ---- 3. run the sweep --------------------------------------------------
        data_refs: dict[int, Any] = {}  # each distinct dataset goes into the object store once
        queue = deque(enumerate(jobs))
        pending: list[Any] = []
        meta: dict[Any, tuple[int, dict[str, Any]]] = {}
        pbar = tqdm(total=len(queue), desc=exp_name, unit="job")
        last_save = time.time()

        def submit_next() -> None:
            i, job = queue.popleft()
            key = id(job["data"])
            if key not in data_refs:
                data_refs[key] = ray.put(job["data"])
            job_cfg = {k: v for k, v in job.items() if k != "data"}
            ref = remote_job.remote(job_cfg, data_refs[key], max_job_attempts)
            pending.append(ref)
            meta[ref] = (i, job_cfg)

        try:
            while queue and len(pending) < max_in_flight:
                submit_next()

            while pending:
                done, remaining = ray.wait(pending, num_returns=1, timeout=60)
                if not done:
                    n_alive = sum(1 for n in ray.nodes() if n["Alive"])
                    tqdm.write(f"[heartbeat] in flight={len(pending)} queued={len(queue)} "
                               f"done={len(rows)} failed={len(failed)} nodes={n_alive}")
                    continue
                pending[:] = remaining
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
                    try:
                        save_combined_csv([row], per_experiment_dir / _experiment_filename(result))
                    except Exception as e:
                        LOGGER.warning("Could not write per-experiment CSV for job %d: %r", i, e)
                    if time.time() - last_save > 5:  # throttle full rewrites
                        flush_csv()
                        last_save = time.time()
                else:
                    failed.append({
                        "trial_id": f"{i:05d}",
                        **{f"cfg_{k}": _scalar(v) for k, v in job_cfg.items()},
                        "error": out["error"],
                        "attempts": out.get("attempts", "unknown"),
                    })
                    tqdm.write(f"Job {i} failed:\n{out['error']}")

                if queue:
                    submit_next()
        except BaseException:
            for reference in pending:
                ray.cancel(reference, force=False)
            raise
    finally:
        if pbar is not None:
            pbar.close()
        flush_csv()
        if failed:
            save_combined_csv(failed, output_dir / "failed.csv")
            print(f"WARNING: {len(failed)} job(s) failed; see {output_dir / 'failed.csv'}")
        try:
            ray.shutdown()
        finally:
            if cluster is not None:
                cluster.stop()

    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(jobs)} job(s) failed after retries; "
            f"see {output_dir / 'failed.csv'}."
        )

    print(f"Results CSV: {csv_path.resolve()}  ({len(rows)}/{len(jobs)} jobs)")
    return rows


# ============================================================
# MAIN
# ============================================================

def _install_signal_handlers() -> None:
    """Turn SLURM's SIGTERM into KeyboardInterrupt so the Ray cluster is torn down cleanly."""
    def handler(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, handler)


def main() -> None:
    # Nodes of one SLURM job share a trusted network; Ray's token auth would block
    # workers from joining unless the same token is distributed. Override by exporting
    # RAY_AUTH_MODE yourself before launching.
    os.environ.setdefault("RAY_AUTH_MODE", "disabled")

    parser = argparse.ArgumentParser(
        description="Run configured CCKA experiments distributed with Ray."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "exp01_main_benchmark.yaml")
    parser.add_argument(
        "--address", default=None,
        help="Default (unset): auto-detect. Multi-node SLURM job -> start head + workers; otherwise single node. "
             "'local' forces single node. 'auto' or host:port connects to an existing cluster "
             "(falls back to the default behaviour for 'auto' if none is running). Overrides ray.address in the YAML.",
    )
    parser.add_argument("--cpus-per-trial", type=int, default=None,
                        help="CPUs per job (default: derived from cluster size and job count).")
    parser.add_argument("--srun-args", default="",
                        help="Extra arguments for the srun that launches each worker node, e.g. '--overlap'.")
    parser.add_argument("--join-timeout", type=float, default=300.0,
                        help="Seconds to wait for worker nodes to join the Ray cluster.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the job count and detected resources, then exit")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (defaults to results/<exp_name>_<timestamp>).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper()))
    _install_signal_handlers()
    config = load_config(args.config)

    exp_name = re.sub(r"[^\w\-.]", "_", str(config.get("exp_name", args.config.stem)))
    LOGGER.info("Experiment: %s", exp_name)
    run_name = f"{exp_name}_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = args.output_dir or ROOT / "results" / run_name

    if config.get("experiment_type") == "decision_boundaries":
        from ccka_experiments.decision_boundaries import run_from_config

        paths = run_from_config(config, output_dir=output_dir)
        LOGGER.info("Saved %d decision-boundary figures under %s", len(paths), output_dir)
        return

    if args.dry_run:
        print(detect_resources().describe())
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
        cpus_per_trial=args.cpus_per_trial,
        srun_args=args.srun_args,
        join_timeout=args.join_timeout,
    )
    LOGGER.info("Saved %d experiment results under %s", len(rows), output_dir)


if __name__ == "__main__":
    main()