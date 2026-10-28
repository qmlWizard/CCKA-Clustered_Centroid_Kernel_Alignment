"""Start or connect to Ray, with a local single-node fallback."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from typing import Optional

import ray

LOGGER = logging.getLogger("ccka.distribute")


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


class RayAgent:
    """Own the Ray runtime for local, existing-cluster, or Slurm execution."""

    def __init__(
        self,
        *,
        num_cpus_per_node: Optional[int] = None,
        port: int = 6379,
        startup_timeout: float = 180,
        local_num_cpus: Optional[int] = None,
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

    def initialise(self, *, address: str = "auto", expected_nodes: Optional[int] = None):
        configured_address = (
            os.environ.get("RAY_ADDRESS") if address == "auto" else address
        )
        if configured_address and configured_address != "auto":
            self.ray_address = configured_address
        elif self.slurm_available():
            self._create_slurm_cluster()
            expected_nodes = expected_nodes or len(self.nodes)
        else:
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
        head_ip = subprocess.check_output(
            ["srun", "-N1", "-n1", "-w", head_node, "hostname", "-I"],
            text=True,
        ).split()[0]
        self.ray_address = f"{head_ip}:{self.port}"
        os.environ["RAY_ADDRESS"] = self.ray_address
        LOGGER.info("Starting Ray head on %s at %s", head_node, self.ray_address)
        self._start_process(
            [
                "srun",
                "--overlap",
                "-N1",
                "-n1",
                "-w",
                head_node,
                "ray",
                "start",
                "--head",
                f"--node-ip-address={head_ip}",
                f"--port={self.port}",
                f"--num-cpus={self.num_cpus_per_node}",
                "--disable-usage-stats",
                "--block",
            ],
            "Ray head",
        )

        for node in self.nodes[1:]:
            self._start_process(
                [
                    "srun",
                    "--overlap",
                    "-N1",
                    "-n1",
                    "-w",
                    node,
                    "ray",
                    "start",
                    f"--address={self.ray_address}",
                    f"--num-cpus={self.num_cpus_per_node}",
                    "--disable-usage-stats",
                    "--block",
                ],
                f"Ray worker on {node}",
            )
        self._owns_cluster = True

    def _start_process(self, command: list[str], label: str) -> None:
        process = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        self._processes.append(process)
        time.sleep(1)
        if process.poll() is not None:
            stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
            raise RuntimeError(f"{label} failed to start:\n{stderr}")

    def _connect_local(self) -> None:
        num_cpus = self.local_num_cpus or _available_cpus()
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
        deadline = time.monotonic() + self.startup_timeout
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                ray.init(
                    address=self.ray_address,
                    ignore_reinit_error=True,
                    include_dashboard=False,
                    logging_level="WARNING",
                )
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
            self.ray_address,
            len(alive_nodes),
            total_cpus,
            expected_nodes,
        )
        return total_cpus, alive_nodes

    def stop(self) -> None:
        if ray.is_initialized():
            ray.shutdown()
        if not self._owns_cluster and not self._processes:
            return
        node_count = len(self.nodes)
        try:
            subprocess.run(
                [
                    "srun",
                    "--overlap",
                    f"--nodes={node_count}",
                    f"--ntasks={node_count}",
                    "--ntasks-per-node=1",
                    "ray",
                    "stop",
                    "--force",
                ],
                capture_output=True,
                timeout=60,
                check=False,
            )
        finally:
            for process in self._processes:
                if process.poll() is None:
                    process.terminate()
            self._processes.clear()
            self._owns_cluster = False
