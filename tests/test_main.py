from __future__ import annotations

from typing import Any

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
    def __init__(self, function, job, data):
        self.function, self.job, self.data = function, job, data


def _run(tmp_path, monkeypatch, rcfg, fail_seed=None):
    import ray

    events: list[tuple[str, int]] = []

    class Remote:
        def __init__(self, function):
            self.function = function

        def remote(self, job, data_ref):
            events.append(("submit", job["seed"]))
            return _Ref(self.function, job, data_ref)

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
    monkeypatch.setattr(ray, "get", lambda ref: ref.function(ref.job, ref.data))
    monkeypatch.setattr(
        ray, "wait", lambda refs, num_returns: (refs[:num_returns], refs[num_returns:])
    )
    monkeypatch.setattr(
        runner, "iter_experiment_jobs", lambda config, smoke=False: iter(_jobs())
    )

    def run_job(job, data):
        assert "data" not in job
        if job["seed"] == fail_seed:
            raise ValueError("boom")
        return {**job, "backend": "jax", "n_qubits": 5, "method": "test"}

    monkeypatch.setattr(runner, "run_experiment_job", run_job)
    rows = runner.run_distributed(
        {}, output_dir=tmp_path, rcfg=rcfg, exp_name="t", address="local"
    )
    return rows, events


def test_runs_all_jobs(monkeypatch, tmp_path):
    rows, events = _run(tmp_path, monkeypatch, {"max_in_flight": 2})

    assert [r["trial_id"] for r in rows] == ["00000", "00001", "00002"]
    assert len(events) == 3
    assert (tmp_path / f"{tmp_path.name}.csv").is_file()
    assert len(list((tmp_path / "experiments").glob("*.csv"))) == 3


def test_failed_job_is_recorded_and_sweep_continues(monkeypatch, tmp_path):
    rows, _ = _run(tmp_path, monkeypatch, {}, fail_seed=1)

    assert len(rows) == 2
    assert (tmp_path / "failed.csv").is_file()
