from __future__ import annotations

import argparse
import csv
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import ray
import yaml

from experiments.experiments import (
    CONFIG_DIR,
    ROOT,
    iter_experiment_jobs,
    run_experiment_job,
)


LOGGER = logging.getLogger("ccka.ray_runner")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _experiment_filename(row: dict[str, Any]) -> str:
    fields = [
        row["dataset"],
        row["method_label"],
        row["noise_type"],
        f"noise-{row['noise_level']:g}",
        f"shots-{row['shots'] if row['shots'] is not None else 'exact'}",
        f"seed-{row['seed']}",
        f"K-{row['centroids_per_class'] or 'na'}",
        row["initialization"],
        row["ablation"],
    ]
    return _slug("_".join(map(str, fields))) + ".csv"

def run_distributed(config: dict[str, Any], *, output_dir: Path, smoke: bool = False,) -> list[dict[str, Any]]:
    jobs = list(iter_experiment_jobs(config, smoke=smoke))
    if not jobs:
        LOGGER.warning("Configuration produced no experiment jobs.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    data_refs: dict[tuple[str, int], Any] = {}
    remote_runner = ray.remote(run_experiment_job)

    alive_nodes = [node for node in ray.nodes() if node.get("Alive", False)]
    if not alive_nodes:
        raise RuntimeError("Ray reports no alive nodes.")
    single_node = len(alive_nodes) == 1
    single_node_cpus = None
    if single_node:
        single_node_cpus = ray.cluster_resources().get("CPU", 0.0)
        if single_node_cpus <= 0:
            raise RuntimeError(
                "Ray reports no CPUs on the single node; cannot schedule an experiment."
            )
        LOGGER.info(
            "Single-node Ray runtime detected; running experiments sequentially "
            "with %.1f CPUs reserved per experiment.",
            single_node_cpus,
        )
    else:
        LOGGER.info("Ray runtime has %d nodes; scheduling experiments in parallel.", len(alive_nodes))

    def submit(job: dict[str, Any]) -> Any:
        key = (job["dataset"], job["seed"])
        if key not in data_refs:
            data_refs[key] = ray.put(job["data"])
        job_without_data = {name: value for name, value in job.items() if name != "data"}
        runner = (
            remote_runner.options(num_cpus=single_node_cpus)
            if single_node
            else remote_runner
        )
        return runner.remote(job_without_data, data_refs[key])

    rows: list[dict[str, Any]] = []
    pending = []
    try:
        if single_node:
            for job in jobs:
                row = ray.get(submit(job))
                rows.append(row)
                _write_csv(output_dir / _experiment_filename(row), [row])
                _write_csv(output_dir / "summary.csv", rows)
                LOGGER.info(
                    "Saved experiment %d/%d: %s / %s",
                    len(rows),
                    len(jobs),
                    row["dataset"],
                    row["method_label"],
                )
        else:
            pending = [submit(job) for job in jobs]
            while pending:
                ready, pending = ray.wait(pending, num_returns=1)
                row = ray.get(ready[0])
                rows.append(row)
                _write_csv(output_dir / _experiment_filename(row), [row])
                _write_csv(output_dir / "summary.csv", rows)
                LOGGER.info(
                    "Saved experiment %d/%d: %s / %s",
                    len(rows),
                    len(jobs),
                    row["dataset"],
                    row["method_label"],
                )
    except Exception:
        for reference in pending:
            ray.cancel(reference, force=False)
        raise
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distribute configured CCKA experiments across a Ray cluster."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "default.yaml",
        help="YAML experiment configuration.",
    )
    parser.add_argument(
        "--address",
        default="auto",
        help="Ray cluster address; use 'local' to start a local Ray runtime.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to results/<config>-<timestamp>).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_name = (
        f"{args.config.stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    output_dir = args.output_dir or ROOT / "results" / run_name

    ray.init(address=args.address, ignore_reinit_error=True)
    try:
        rows = run_distributed(
            config,
            output_dir=output_dir,
            smoke=args.smoke,
        )
        LOGGER.info("Saved %d experiment results under %s", len(rows), output_dir)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
