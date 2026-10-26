from __future__ import annotations

from typing import Any

import main as ray_runner


def _jobs() -> list[dict[str, Any]]:
    return [
        {
            "dataset": "checkerboard",
            "seed": seed,
            "method_label": f"method-{seed}",
            "noise_type": "none",
            "noise_level": 0.0,
            "shots": 0,
            "centroids_per_class": None,
            "initialization": "not_applicable",
            "ablation": "none",
            "data": [seed],
        }
        for seed in range(3)
    ]


def _install_fake_ray(monkeypatch, *, nodes: list[dict[str, Any]], cpus: float):
    events: list[tuple[Any, ...]] = []

    class RemoteRunner:
        def __init__(self, num_cpus: float | None = None) -> None:
            self.num_cpus = num_cpus

        def options(self, *, num_cpus: float) -> RemoteRunner:
            events.append(("options", num_cpus))
            return RemoteRunner(num_cpus)

        def remote(self, job: dict[str, Any], data_ref: Any) -> tuple[Any, ...]:
            events.append(("submit", job["seed"], self.num_cpus))
            return job, data_ref

    monkeypatch.setattr(ray_runner.ray, "nodes", lambda: nodes)
    monkeypatch.setattr(
        ray_runner.ray, "cluster_resources", lambda: {"CPU": cpus}
    )
    monkeypatch.setattr(ray_runner.ray, "remote", lambda function: RemoteRunner())
    monkeypatch.setattr(ray_runner.ray, "put", lambda data: ("data-ref", data))
    monkeypatch.setattr(
        ray_runner,
        "iter_experiment_jobs",
        lambda config, smoke=False: iter(_jobs()),
    )

    def run_job(job: dict[str, Any], data: Any) -> dict[str, Any]:
        events.append(("run", job["seed"]))
        return {
            **job,
            "backend": "jax",
            "n_qubits": 5,
            "method": "test",
            "method_label": job["method_label"],
        }

    monkeypatch.setattr(ray_runner, "run_experiment_job", run_job)

    def ray_get(reference: tuple[dict[str, Any], Any]) -> dict[str, Any]:
        job, data = reference
        events.append(("get", job["seed"]))
        return run_job(job, data)

    monkeypatch.setattr(ray_runner.ray, "get", ray_get)

    def ray_wait(references: list[Any], *, num_returns: int):
        events.append(("wait", len(references)))
        return references[:num_returns], references[num_returns:]

    monkeypatch.setattr(ray_runner.ray, "wait", ray_wait)
    return events


def test_single_node_runs_one_experiment_at_a_time_with_all_cpus(
    monkeypatch, tmp_path
):
    events = _install_fake_ray(
        monkeypatch, nodes=[{"Alive": True}], cpus=8.0
    )

    rows = ray_runner.run_distributed({}, output_dir=tmp_path)

    assert len(rows) == 3
    assert [event[0] for event in events].count("submit") == 3
    assert [event[0] for event in events].count("options") == 3
    for index in range(3):
        assert events[index * 4 : index * 4 + 4] == [
            ("options", 8.0),
            ("submit", index, 8.0),
            ("get", index),
            ("run", index),
        ]
    assert (tmp_path / "summary.csv").is_file()


def test_multiple_nodes_keep_parallel_scheduling(monkeypatch, tmp_path):
    events = _install_fake_ray(
        monkeypatch,
        nodes=[{"Alive": True}, {"Alive": True}],
        cpus=8.0,
    )

    rows = ray_runner.run_distributed({}, output_dir=tmp_path)

    assert len(rows) == 3
    assert [event[0] for event in events[:3]] == [
        "submit",
        "submit",
        "submit",
    ]
    assert not any(event[0] == "options" for event in events)
