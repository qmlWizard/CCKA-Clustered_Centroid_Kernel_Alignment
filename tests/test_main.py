from __future__ import annotations

from typing import Any

import pytest

import main as runner


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


class _Ref:
    def __init__(self, function, job, data, max_attempts):
        self.function, self.job, self.data = function, job, data
        self.max_attempts = max_attempts


def _run(tmp_path, monkeypatch, rcfg, fail_seed=None, fail_once_seed=None):
    import ray

    events: list[tuple[str, int]] = []
    executions: dict[int, int] = {}

    class Remote:
        def __init__(self, function):
            self.function = function

        def remote(self, job, data_ref, max_attempts):
            events.append(("submit", job["seed"]))
            return _Ref(self.function, job, data_ref, max_attempts)

    class Context:
        def get_assigned_resources(self):
            return {"CPU": 1.0}

    monkeypatch.setattr(ray, "get_runtime_context", lambda: Context())
    monkeypatch.setattr(ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(ray, "shutdown", lambda: None)
    monkeypatch.setattr(ray, "cluster_resources", lambda: {"CPU": 4.0})
    monkeypatch.setattr(ray, "nodes", lambda: [{"Alive": True}])
    monkeypatch.setattr(ray, "put", lambda data: data)
    monkeypatch.setattr(ray, "remote", lambda **kwargs: Remote)
    monkeypatch.setattr(
        ray,
        "get",
        lambda ref: runner._job_worker(ref.job, ref.data, ref.max_attempts),
    )
    monkeypatch.setattr(
        ray,
        "wait",
        lambda refs, num_returns, timeout=None: (
            refs[:num_returns],
            refs[num_returns:],
        ),
    )
    monkeypatch.setattr(
        runner, "iter_experiment_jobs", lambda config, smoke=False: iter(_jobs())
    )

    def run_job(job, data):
        assert "data" not in job
        executions[job["seed"]] = executions.get(job["seed"], 0) + 1
        if job["seed"] == fail_seed:
            raise ValueError("boom")
        if job["seed"] == fail_once_seed and executions[job["seed"]] == 1:
            raise ValueError("transient failure")
        return {**job, "backend": "jax", "n_qubits": 5, "method": "test"}

    monkeypatch.setattr(runner, "run_experiment_job", run_job)
    failure = None
    try:
        rows = runner.run_distributed(
            {}, output_dir=tmp_path, rcfg=rcfg, exp_name="t", address="local"
        )
    except RuntimeError as error:
        rows = []
        failure = error
    return rows, events, executions, failure


def test_runs_all_jobs(monkeypatch, tmp_path):
    rows, events, _, failure = _run(tmp_path, monkeypatch, {"max_in_flight": 2})

    assert failure is None
    assert [r["trial_id"] for r in rows] == ["00000", "00001", "00002"]
    assert len(events) == 3
    assert (tmp_path / f"{tmp_path.name}.csv").is_file()
    assert len(list((tmp_path / "experiments").glob("*.csv"))) == 3


def test_failed_job_is_recorded_and_sweep_continues(monkeypatch, tmp_path):
    rows, _, executions, failure = _run(tmp_path, monkeypatch, {}, fail_seed=1)

    assert rows == []
    assert isinstance(failure, RuntimeError)
    assert executions[1] == 2
    assert (tmp_path / "failed.csv").is_file()


def test_transient_job_failure_is_retried(monkeypatch, tmp_path):
    rows, _, executions, failure = _run(
        tmp_path, monkeypatch, {}, fail_once_seed=1
    )

    assert failure is None
    assert len(rows) == 3
    assert executions[1] == 2
    assert not (tmp_path / "failed.csv").exists()
