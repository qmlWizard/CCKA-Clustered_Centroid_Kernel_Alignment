"""Ray cluster initialization utilities.

Handles starting the Ray head, joining workers, and waiting for all nodes.
"""
from __future__ import annotations

import logging
import os
import socket
import subprocess
import time
from typing import Optional

import ray

LOGGER = logging.getLogger(__name__)


def get_node_list() -> list[str]:
    """Get list of allocated nodes from SLURM."""
    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        raise RuntimeError("SLURM_JOB_NODELIST not set. Must be run under SLURM.")
    result = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip().split("\n")


def get_head_ip(head_node: str) -> str:
    """Get IP address of head node."""
    result = subprocess.run(
        ["srun", "--nodes=1", "--ntasks=1", f"--nodelist={head_node}",
         "hostname", "-I"],
        capture_output=True, text=True, check=True
    )
    return result.stdout.split()[0]


def is_gcs_ready(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if Ray GCS is reachable."""
    try:
        s = socket.socket()
        s.settimeout(timeout)
        s.connect((host, port))
        s.close()
        return True
    except Exception as e:
        LOGGER.debug(f"GCS not ready: {e}")
        return False


def wait_for_gcs(host: str, port: int, max_attempts: int = 30,
                 interval: float = 2.0) -> bool:
    """Wait for Ray GCS to become reachable."""
    for attempt in range(max_attempts):
        if is_gcs_ready(host, port):
            LOGGER.info(f"GCS ready at {host}:{port}")
            return True
        LOGGER.info(f"Attempt {attempt + 1}/{max_attempts}: GCS not ready, retrying...")
        time.sleep(interval)
    return False


def start_ray_head(head_node: str, head_ip: str, port: int, num_cpus: int) -> None:
    """Start Ray head node on the head node."""
    cmd = [
        "srun", "--nodes=1", "--ntasks=1", f"--nodelist={head_node}",
        "ray", "start", "--head",
        f"--node-ip-address={head_ip}",
        f"--port={port}",
        f"--num-cpus={num_cpus}",
        "--disable-usage-stats"
    ]
    LOGGER.info(f"Starting Ray head on {head_node} ({head_ip}:{port})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to start Ray head on {head_node}:\n{result.stderr}"
        )
    LOGGER.info("Ray head started successfully")


def start_ray_workers(address: str, node_list: list[str], num_cpus: int,
                      resources: Optional[dict] = None) -> None:
    """Start Ray workers on given nodes."""
    if not node_list:
        return
    nodelist_str = ",".join(node_list)
    cmd = [
        "srun", f"--nodes={len(node_list)}", f"--ntasks={len(node_list)}",
        "--ntasks-per-node=1", f"--nodelist={nodelist_str}",
        "ray", "start", f"--address={address}",
        f"--num-cpus={num_cpus}",
        "--disable-usage-stats"
    ]
    if resources:
        resources_str = ",".join(f'"{k}":{v}' for k, v in resources.items())
        cmd.append(f"--resources={{{resources_str}}}")

    LOGGER.info(f"Starting Ray workers on {len(node_list)} nodes...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start Ray workers:\n{result.stderr}")
    LOGGER.info(f"Ray workers started successfully ({len(node_list)} nodes)")


def wait_for_nodes(expected_count: int, max_attempts: int = 60,
                   interval: float = 5.0) -> int:
    """Wait for all expected nodes to join the Ray cluster."""
    for attempt in range(max_attempts):
        try:
            alive_nodes = sum(1 for n in ray.nodes() if n.get("Alive"))
            if alive_nodes >= expected_count:
                LOGGER.info(f"All {alive_nodes}/{expected_count} nodes joined")
                return alive_nodes
            LOGGER.info(
                f"Attempt {attempt + 1}/{max_attempts}: "
                f"{alive_nodes}/{expected_count} nodes joined, waiting..."
            )
        except Exception as e:
            LOGGER.debug(f"Error checking nodes: {e}")
        time.sleep(interval)

    alive_nodes = sum(1 for n in ray.nodes() if n.get("Alive"))
    LOGGER.warning(
        f"Timeout waiting for nodes. Got {alive_nodes}/{expected_count}"
    )
    return alive_nodes


def init_ray_cluster(address: str = "auto", wait_for_all_nodes: bool = True) -> str:
    """Initialize Ray cluster from SLURM environment.

    If address == "auto", starts the Ray head and workers from SLURM info.
    Otherwise, connects to an existing cluster.

    Returns the Ray address.
    """
    if address != "auto":
        LOGGER.info(f"Connecting to Ray cluster at {address}...")
        ray.init(address=address, ignore_reinit_error=True, logging_level="ERROR")
        return address

    # Get SLURM info
    nodes = get_node_list()
    head_node = nodes[0]
    head_ip = get_head_ip(head_node)
    port = int(os.environ.get("RAY_PORT", 6379))
    num_cpus = int(os.environ.get("RAY_CPUS",
                                  os.environ.get("SLURM_CPUS_ON_NODE",
                                                os.environ.get("SLURM_CPUS_PER_TASK", 1))))
    timing_nodes = int(os.environ.get("TIMING_NODES", 1))

    # Validate
    if timing_nodes >= len(nodes):
        timing_nodes = 0

    # Start head
    start_ray_head(head_node, head_ip, port, num_cpus)

    # Wait for GCS to be ready
    if not wait_for_gcs(head_ip, port):
        raise RuntimeError(f"Ray GCS at {head_ip}:{port} did not become reachable")

    # Start workers
    address = f"{head_ip}:{port}"

    split = len(nodes) - timing_nodes
    compute_workers = nodes[1:split]
    timing_workers = nodes[split:]

    if compute_workers:
        start_ray_workers(address, compute_workers, num_cpus)

    if timing_workers:
        start_ray_workers(address, timing_workers, 0,
                         resources={"timing": 1})

    # Initialize Ray client
    LOGGER.info(f"Initializing Ray at {address}...")
    ray.init(address=address, ignore_reinit_error=True, logging_level="ERROR")

    # Wait for all nodes to join
    if wait_for_all_nodes:
        wait_for_nodes(len(nodes))

    return address


def cleanup_ray() -> None:
    """Stop all Ray processes."""
    try:
        subprocess.run(
            ["srun", f"--nodes={os.environ.get('SLURM_JOB_NUM_NODES', 1)}",
             f"--ntasks={os.environ.get('SLURM_JOB_NUM_NODES', 1)}",
             "--ntasks-per-node=1",
             "ray", "stop", "--force"],
            capture_output=True, timeout=30
        )
    except Exception as e:
        LOGGER.warning(f"Error stopping Ray: {e}")
