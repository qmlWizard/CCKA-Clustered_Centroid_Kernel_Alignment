from ccka.models.kernel import KernelModel
from ccka.circuits.angleEmbeddingKernel import quackEmbeddingCircuit
from ccka.aligner.kta import fullKTA, centroidBasedKTA, quackKTA, randomKTA, greedyKTA
import pennylane as qml
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt 
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import time
import os
import warnings
warnings.filterwarnings("ignore")
from tqdm.auto import tqdm

LANDMARK_POINTS = 20

def load_data(dataset_path):
    data = jnp.load(dataset_path, allow_pickle=True).item()
    X = jnp.asarray(data['x_train'])
    y = jnp.asarray(data['y_train'])
    x_test = jnp.asarray(data['x_test'])
    y_test = jnp.asarray(data['y_test'])

    X_combined = jnp.concatenate([X, x_test], axis=0)
    y_combined = jnp.concatenate([y, y_test], axis=0)

    return X_combined, y_combined

def run_experiment(method, dataset, dataset_path, seed, centroids):

    print(f"Running {method} on {dataset}")

    centroids = centroids if centroids is not None else 10
    num_iterations = 100
    reps = 6
    lr = 2.0
    
    
    X, y = load_data(dataset_path)

    print('Labels: ', jnp.unique(y))
    print('Label Counts: ', jnp.bincount(y))

    nqubits = X.shape[1]

    kernel = quackEmbeddingCircuit(
                                num_qubits = nqubits,
                                reps = reps,
                                reupload = True,
                                seed = seed,
                        )

    print('Dataset Size: ', len(X))
    print('Qubits: ', X.shape[1])

    model = KernelModel(circuit=kernel)
    common = dict(kernel_model=model, data=X, labels=y, matrix_type="regular", split_size=0.7)
    
    if method == "fullKTA":
        aligner = fullKTA(**common, seed = seed, learning_rate=0.1, optimizer="adam",
                           epochs=num_iterations)
    elif method == "randomKTA":
        aligner = randomKTA(**common, seed = seed, random_samples=centroids,
                             landmark_points=LANDMARK_POINTS, learning_rate=0.1,
                             optimizer="adam", epochs=num_iterations)
    elif method == "greedyKTA":
        aligner = greedyKTA(**common, seed = seed, greedy_samples=centroids,
                             landmark_points=LANDMARK_POINTS, learning_rate=0.1,
                             optimizer="adam", epochs=num_iterations)
    elif method == "quackKTA":
        aligner = quackKTA(**common, seed = seed, centroids=centroids, clustering="regular",
                            lambda_co=0.001, lambda_kao=0.001, epochs=num_iterations)
    elif method == "centroidBasedKTA":
        aligner = centroidBasedKTA(**common, seed = seed, clustering="regular",
                                    centroids=centroids, learning_rate=0.2, centroid_lr=0.01,
                                    sub_centroid_lr=0.01, lambda_co=0.001, lambda_kao=0.001,
                                    epochs=num_iterations)
    else:
        raise ValueError(f"Unknown method: {method}")

    history = aligner.align()

    result = {
        'method': method,
        'dataset': dataset,
        'centroids': centroids,
        'num_iterations': num_iterations,
        'initial_training_accuracy': history['init_train_accuracy'],
        'final_training_accuracy': history['train_accuracy_history'][-1],
        'initial_testing_accuracy': history['init_test_accuracy'],
        'final_testing_accuracy': history['test_accuracy_history'][-1],
        'f1_score': history['f1_score_history'][-1],
        'precision': history['precision_score_history'][-1],
        'recall': history['recall_score_history'][-1],
        'training_time': history['time'],
        'circuit_executions': history['circuit_executions'],
        'initial_kernel_alignment': history['alignment_history'][0],
        'final_kernel_alignment': history['alignment_history'][-1]
    }

    return result


if __name__ == "__main__":

    datasets = {
        "banknote_authentication": "../data/banknote_authentication.npy",
        "breast_cancer": "../data/breast_cancer_wisconsin.npy",
        "german_credit": "../data/german_credit.npy",
        "pima_diabetes": "../data/pima_diabetes.npy",
        "qsar_biodegradation": "../data/qsar_biodegradation.npy"
    }

    methods = ["fullKTA", "randomKTA", "greedyKTA", "quackKTA", "centroidBasedKTA"]
    seeds = [1, 2, 3]
    centroids = [20]

    results = []

    total_runs = sum(
        len([None] if method == "fullKTA" else centroids)
        for method in methods
    ) * len(seeds) * len(datasets)

    # ----------------------------
    # Run experiments
    # ----------------------------

    with tqdm(total=total_runs, desc="Running Experiments") as pbar:

        for dataset_name, dataset_path in datasets.items():

            for method in methods:

                centroid_list = [None] if method == "fullKTA" else centroids

                for centroid in centroid_list:

                    for seed in seeds:

                        pbar.set_postfix(
                            dataset=dataset_name,
                            method=method,
                            seed=seed,
                            centroid=centroid if centroid is not None else "-"
                        )

                        result = run_experiment(
                            method,
                            dataset_name,
                            dataset_path,
                            seed,
                            centroid,
                        )

                        result["seed"] = seed
                        results.append(result)

                        pbar.update(1)


    # ==========================================================
    # Save per-seed results
    # ==========================================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "experiment_results_per_seed.csv",
        index=False,
    )

    print("Saved experiment_results_per_seed.csv")


    # ==========================================================
    # Aggregate across seeds
    # ==========================================================

    summary_df = (
        results_df
        .groupby(
            ["dataset", "method", "centroids"],
            dropna=False,
        )
        .agg(
            mean_accuracy=("final_testing_accuracy", "mean"),
            std_accuracy=("final_testing_accuracy", "std"),
            min_accuracy=("final_testing_accuracy", "min"),
            max_accuracy=("final_testing_accuracy", "max"),
            mean_f1=("f1_score", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_time=("training_time", "mean"),
        )
        .reset_index()
    )

    summary_df.to_csv(
        "experiment_results_summary.csv",
        index=False,
    )

    print("Saved experiment_results_summary.csv")


    # ==========================================================
    # Plot
    # ==========================================================

    METHOD_ORDER = [
        "fullKTA",
        "randomKTA",
        "greedyKTA",
        "quackKTA",
        "centroidBasedKTA",
    ]

    METHOD_LABELS = {
        "fullKTA": "Full",
        "randomKTA": "Random",
        "greedyKTA": "Greedy",
        "quackKTA": "Quack",
        "centroidBasedKTA": "CCKA",
    }

    colors = plt.cm.Set2(np.linspace(0, 1, len(METHOD_ORDER)))

    fig, ax = plt.subplots(figsize=(12, 6))

    datasets_list = list(datasets.keys())

    x = np.arange(len(datasets_list))

    width = 0.15

    for i, method in enumerate(METHOD_ORDER):

        vals = []
        err_low = []
        err_high = []

        for dataset in datasets_list:

            tmp = summary_df[
                (summary_df.dataset == dataset)
                &
                (summary_df.method == method)
            ]

            # For centroid methods use the largest centroid
            if method != "fullKTA":
                tmp = tmp.sort_values("centroids").tail(1)

            vals.append(tmp["mean_accuracy"].values[0])

            err_low.append(
                vals[-1] - tmp["min_accuracy"].values[0]
            )

            err_high.append(
                tmp["max_accuracy"].values[0] - vals[-1]
            )

        ax.bar(
            x + (i - 2) * width,
            vals,
            width,
            label=METHOD_LABELS[method],
            color=colors[i],
            edgecolor="black",
            linewidth=0.7,
            yerr=[err_low, err_high],
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets_list)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Final Test Accuracy")
    ax.set_xlabel("Dataset")
    ax.set_title("Comparison of KTA Methods")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    ax.legend(
        ncols=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
    )

    plt.tight_layout()

    plt.savefig(
        "method_comparison_barplot.png",
        dpi=600,
    )

    plt.show()