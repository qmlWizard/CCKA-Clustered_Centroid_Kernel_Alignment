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

    def initialise(
        self,
        *,
        address: str = "auto",
        expected_nodes: Optional[int] = None,
    ):
        if address == "local":
            self.local_mode = True
            expected_nodes = expected_nodes or 1
            self._connect_local()
            return self._runtime_summary(expected_nodes)

        configured_address = (
            os.environ.get("RAY_ADDRESS")
            if address == "auto"
            else address
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
            ["scontrol", "show", "hostnames", nodelist],
            text=True,
        ).split()

        if not self.nodes:
            raise RuntimeError("SLURM_JOB_NODELIST did not contain any nodes.")

        head_node = self.nodes[0]
        node_count = len(self.nodes)

        # This is a short-lived Slurm step. It finishes before the main
        # multi-node Ray launch begins.
        head_ip = subprocess.check_output(
            [
                "srun",
                "--overlap",
                "-N1",
                "-n1",
                "-w",
                head_node,
                "hostname",
                "-I",
            ],
            text=True,
        ).split()[0]

        self.ray_address = f"{head_ip}:{self.port}"
        os.environ["RAY_ADDRESS"] = self.ray_address

        LOGGER.info(
            "Starting Ray cluster: head=%s address=%s nodes=%d CPUs/node=%d",
            head_node,
            self.ray_address,
            node_count,
            self.num_cpus_per_node,
        )

        # Each task receives a unique SLURM_PROCID:
        #   0       -> Ray head
        #   1..N-1  -> Ray workers
        #
        # Workers retry while the head starts. Once Ray starts successfully,
        # --block keeps the process alive.
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

        command = [
            "srun",
            "--overlap",
            f"--nodes={node_count}",
            f"--ntasks={node_count}",
            "--ntasks-per-node=1",
            "--kill-on-bad-exit=1",
            "bash",
            "-lc",
            worker_script,
        ]

        self._start_process(command, "Ray cluster launch")
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

        # Terminate the long-running srun process first.
        # This releases the persistent job step before cleanup.
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

        # Ask Ray to stop any processes that survived the srun shutdown.
        if self._owns_cluster and self.nodes:
            try:
                subprocess.run(
                    [
                        "srun",
                        "--overlap",
                        f"--nodes={len(self.nodes)}",
                        f"--ntasks={len(self.nodes)}",
                        "--ntasks-per-node=1",
                        "ray",
                        "stop",
                        "--force",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                LOGGER.warning("Ray cleanup encountered an error: %s", exc)

        self._owns_cluster = False
