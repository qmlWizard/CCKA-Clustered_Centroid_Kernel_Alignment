import csv
import json

import pandas as pd

from experiments import plot_results
from experiments.plot_results import make_plots


def test_epoch_accuracy_plot_uses_reference_method_styles_per_dataset(
    tmp_path, monkeypatch
):
    style_by_method = {
        "fullKTA": ("#4c9bd6", "--"),
        "randomKTA": ("#83ad45", "--"),
        "quackKTA": ("#29947f", "--"),
        "ccka": ("#e95d36", "-"),
    }
    labels = {
        "fullKTA": "FullKTA",
        "randomKTA": "RandomKTA",
        "quackKTA": "QuackKTA",
        "ccka": "CCKA (ours)",
    }
    rows = [
        {
            "dataset": dataset,
            "method": method,
            "method_label": labels[method],
            "ablation": "full" if method == "ccka" else "none",
            "initialization": "regular" if method == "ccka" else "not_applicable",
            "centroids_per_class": 5 if method == "ccka" else None,
            "noise_type": "none",
            "validation_accuracy_history": json.dumps([0.5, 0.7, 0.9]),
            "_source": "test.csv",
        }
        for dataset in ("adult", "checkerboard")
        for method in style_by_method
    ]
    saved_figures = {}

    def capture_save(fig, path):
        saved_figures[path.name] = fig

    monkeypatch.setattr(plot_results, "_save", capture_save)
    assert plot_results._plot_epoch_accuracy(pd.DataFrame(rows), tmp_path) == 2

    for dataset in ("adult", "checkerboard"):
        figure = saved_figures[f"accuracy-epochs-{dataset}.png"]
        lines = {line.get_label(): line for line in figure.axes[0].lines}
        expected_labels = {
            **labels,
            "ccka": "CCKA (ours), K=5",
        }
        assert set(lines) == set(expected_labels.values())
        for method, label in labels.items():
            plot_label = expected_labels[method]
            assert lines[plot_label].get_color() == style_by_method[method][0]
            assert lines[plot_label].get_linestyle() == style_by_method[method][1]
        assert figure.axes[0].get_ylabel() == "Validation Accuracy"


def test_plot_results_writes_all_requested_figure_groups(tmp_path):
    csv_path = tmp_path / "summary.csv"
    fields = [
        "dataset",
        "method",
        "method_label",
        "ablation",
        "initialization",
        "centroids_per_class",
        "noise_type",
        "noise_level",
        "shots",
        "seed",
        "test_accuracy",
        "circuit_executions",
        "wall_time_seconds",
        "validation_accuracy_history",
    ]
    rows = []
    for dataset_index, dataset in enumerate(("adult", "checkerboard")):
        for seed in (0, 1):
            for ablation, centroid_count in (
                ("full", 1),
                ("full", 2),
                ("main_only", 1),
                ("sub_only", 1),
                ("no_kao", 1),
                ("no_co", 1),
            ):
                for noise_type, levels, shots in (
                    ("none", (0.0,), (None,)),
                    ("coherent", (0.01, 0.05), (256, 1024)),
                    ("depolarizing", (0.01, 0.05), (256, 1024)),
                ):
                    for noise_level in levels:
                        for shot in shots:
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "method": "ccka",
                                    "method_label": "CCKA",
                                    "ablation": ablation,
                                    "initialization": "regular",
                                    "centroids_per_class": centroid_count,
                                    "noise_type": noise_type,
                                    "noise_level": noise_level,
                                    "shots": shot,
                                    "seed": seed,
                                    "test_accuracy": 0.6 + 0.02 * seed
                                    + 0.01 * dataset_index,
                                    "circuit_executions": 1000 + 50 * centroid_count,
                                    "wall_time_seconds": 2.0 + centroid_count,
                                    "validation_accuracy_history": json.dumps(
                                        [0.5, 0.6 + 0.02 * seed]
                                    ),
                                }
                            )
            for method, method_label, centroid_count in (
                ("fullKTA", "All-Pairwise KTA (Full KTA)", None),
                ("randomKTA", "Random-Pairwise KTA", None),
                ("quackKTA", "QUACK", None),
            ):
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "method_label": method_label,
                        "ablation": "none",
                        "initialization": "not_applicable",
                        "centroids_per_class": centroid_count,
                        "noise_type": "none",
                        "noise_level": 0.0,
                        "shots": None,
                        "seed": seed,
                        "test_accuracy": 0.7 + 0.01 * seed,
                        "circuit_executions": 1000,
                        "wall_time_seconds": 2.0,
                        "validation_accuracy_history": json.dumps(
                            [0.5, 0.7 + 0.01 * seed]
                        ),
                    }
                )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "plots"
    plots = make_plots(csv_path, output, noise_centroids=2)

    assert len(plots) == 11
    assert (output / "complexity_time.png").is_file()
    assert (output / "complexity_circuits.png").is_file()
    assert (output / "accuracy-epochs-adult.png").is_file()
    assert (output / "accuracy-epochs-checkerboard.png").is_file()
    assert (output / "centroids_average.png").is_file()
    assert (output / "noise-coherent.png").is_file()
    assert (output / "noise-depolarizing.png").is_file()
    assert (output / "ablations_average.png").is_file()
