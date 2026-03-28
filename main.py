import numpy as np
import os
import time
import json
import yaml
import itertools

from ccka.models.kernel import KernelModel
from ccka.circuits.angleEmbeddingKernel import quackEmbeddingCircuit
from ccka.aligner.kta import centroidBasedKTA, fullKTA, randomKTA, greedyKTA

import jax.numpy as jnp


# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

def to_serializable(obj):
    import jax.numpy as jnp
    import numpy as np
    from sklearn.svm import SVC

    if isinstance(obj, SVC):
        return "SVC_model"  # or None

    if isinstance(obj, (jnp.ndarray,)):
        return np.array(obj).tolist()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.float32, np.float64, jnp.float32, jnp.float64)):
        return float(obj)

    if isinstance(obj, (np.int32, np.int64, jnp.int32, jnp.int64)):
        return int(obj)

    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]

    return obj

def gen_exp_name(config, dataset, method):
    return f"{method}_{dataset}_{int(time.time())}"


def save_results(exp_name, results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, f"{exp_name}.npy"), results)

    with open(os.path.join(output_dir, f"{exp_name}.json"), "w") as f:
        json.dump(to_serializable(results), f, indent=4)

    print(f"Saved: {exp_name}")


def load_dataset(base_path, dataset_name):
    path = os.path.join(base_path, f"{dataset_name}_dataset.npy")
    data = np.load(path, allow_pickle=True).item()

    X = jnp.asarray(data["x_train"])
    y = jnp.asarray(data["y_train"])
    x_test = jnp.asarray(data["x_test"])
    y_test = jnp.asarray(data["y_test"])

    return X, y, x_test, y_test


def build_kernel(config):
    kernel_cfg = config["kernel"]

    circuit = quackEmbeddingCircuit(
        num_qubits=kernel_cfg["num_qubits"],
        reps=kernel_cfg["num_layers"],
        reupload=kernel_cfg["data_reuploading"],
    )

    model = KernelModel(circuit=circuit)

    return model


# --------------------------------------------------
# GRID SEARCH HELPER
# --------------------------------------------------

def param_grid(param_dict):
    if param_dict is None:
        return [{}]

    keys = list(param_dict.keys())
    values = [v if isinstance(v, list) else [v] for v in param_dict.values()]

    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# --------------------------------------------------
# METHOD FACTORY
# --------------------------------------------------

def build_aligner(method_name, model, X, y, x_test, y_test, params):

    X_combined = jnp.concatenate([X, x_test], axis=0)
    y_combined = jnp.concatenate([y, y_test], axis=0)

    common_args = dict(
        kernel_model=model,
        data=X,
        labels=y,
        x_test=x_test,
        y_test=y_test,
    )

    if method_name == "centroid_based_kta":
        return centroidBasedKTA(**common_args, **params)

    elif method_name == "full_kta":
        return fullKTA(**common_args, **params)

    elif method_name == "random_kta":
        return randomKTA(**common_args, **params)

    elif method_name == "greedy_kta":
        return greedyKTA(**common_args, **params)

    else:
        raise ValueError(f"Unknown method: {method_name}")


# --------------------------------------------------
# CORE EXPERIMENT LOOP
# --------------------------------------------------

def run_experiment(config):

    output_dir = config["experiment"]["output_dir"]
    dataset_names = config["dataset"]["dataset_names"]

    results_all = {}

    for dataset_name in dataset_names:

        print(f"\n===== Dataset: {dataset_name} =====")

        X, y, x_test, y_test = load_dataset(
            config["dataset"]["dataset_path"], dataset_name
        )

        for method in config["alignment_methods"]:

            if not method.get("enabled", True):
                continue

            method_name = method["method_name"]
            print(f"\n--- Method: {method_name} ---")

            for params in param_grid(method.get("params")):

                print(f"Params: {params}")

                # Build kernel fresh each run
                model = build_kernel(config)

                aligner = build_aligner(
                    method_name, model, X, y, x_test, y_test, params
                )

                start = time.time()
                history = aligner.align()
                elapsed = time.time() - start

                exp_name = gen_exp_name(config, dataset_name, method_name)

                history = to_serializable(history)

                result = {
                    "dataset": dataset_name,
                    "method": method_name,
                    "params": params,
                    "history": history,
                }

                key = f"{exp_name}_{len(results_all)}"
                results_all[key] = result

                save_results(exp_name, result, output_dir)

    return results_all


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

def summarize_results(results):

    print("\n===== SUMMARY =====")

    for key, res in results.items():
        print(
            f"{res['method']} | {res['dataset']} | "
            f"Acc: {res['test_accuracy']:.3f} | "
            f"F1: {res['f1_score']:.3f} | "
            f"Time: {res['time']:.2f}s"
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    config_path = "experiments/configs/method_comparision.yaml"  # your YAML file

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results = run_experiment(config)
    summarize_results(results)