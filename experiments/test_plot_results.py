import csv
import json

from experiments.plot_results import make_plots


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
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "plots"
    plots = make_plots(csv_path, output, noise_centroids=2)

    assert len(plots) == 11
    assert (output / "complexity_time.png").is_file()
    assert (output / "complexity_circuits.png").is_file()
    assert (output / "accuracy-epochs-ccka-k-1.png").is_file()
    assert (output / "accuracy-epochs-ccka-k-2.png").is_file()
    assert (output / "centroids_average.png").is_file()
    assert (output / "noise-coherent.png").is_file()
    assert (output / "noise-depolarizing.png").is_file()
    assert (output / "ablations_average.png").is_file()
