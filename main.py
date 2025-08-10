import torch
import ray
from ray import tune
from ray.air import session
from sklearn.model_selection import train_test_split
import yaml
import argparse
import shutil
from collections import namedtuple
import os
import datetime
import sys
import time
from pathlib import Path


# Custom Libraries
from utils.data_generator import DataGenerator
from utils.agent import TrainModel
from utils.helper import to_python_native, gen_experiment_name, set_seed, save_model_state, _now_iso, _safe_bool_str, _ensure_dir, _append_csv_row
from utils.plotter import alignment_progress_over_iterations, plot_initial_final_accuracies
from utils.logger import Logger

# === Backend Configuration ===
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print("Select device: ", device)

set_seed(42)

# === NEW/UPDATED: CSV helpers and fixed columns ===
PER_ITER_COLUMNS = [
    "dataset","method","run_id","iteration","accuracy","alignment","loss",
    "circuits","time_sec","subcentroids","noise_level","mitigation","n_samples"
]
PER_RUN_COLUMNS = [
    "dataset","method","run_id","test_accuracy","train_time_sec","circuits_total",
    "subcentroids","noise_level","mitigation","n_samples"
]

# === END helpers ===

def train(config):

    logger = Logger(dataset_name=config['name'], log_dir=config['ray_logging_path'], mirror_json=False)

    repeat_idx = int(config.get('repeat', 1))
    
    # Build paths
    ray_logging_path = Path(config['ray_logging_path'])
    per_iter_csv = ray_logging_path / "per_iter_logs.csv"
    per_run_csv = ray_logging_path / "per_run_summary.csv"

    name_str = f"_{config['name']}_{config['n_qubits']}_{config['ansatz']}_{config['ansatz_layers']}_{config['optimizer']}_{config['lr']}_{config['mclr']}_{config['cclr']}_{config['train_method']}_{config['lambda_kao']}_{config['lambda_co']}_{config['clusters']}_Kmeans_{config['use_kmeans']}"

    data_generator = DataGenerator(
        dataset_name=config['name'],
        file_path=config['file'],
        n_samples=config['n_samples'],
        noise=config['noise'],
        num_sectors=config['num_sectors'],
        points_per_sector=config['points_per_sector'],
        grid_size=config['grid_size'],
        sampling_radius=config['sampling_radius']
    )

    training_data, training_labels, testing_data, testing_labels = data_generator.generate_dataset()
    training_data = torch.tensor(training_data.to_numpy(), dtype=torch.float32, requires_grad=True)
    testing_data = torch.tensor(testing_data.to_numpy(), dtype=torch.float32, requires_grad=True)
    training_labels = torch.tensor(training_labels.to_numpy(), dtype=torch.int)
    testing_labels = torch.tensor(testing_labels.to_numpy(), dtype=torch.int)
    multi_class = True if int(training_labels.unique().numel()) == 2 else False

    # === NEW/UPDATED: run identifiers and static fields ===
    exp_name = gen_experiment_name(config)
    run_id = f"{exp_name}_rep{repeat_idx}"
    dataset_name = config['name']
    method = str(config.get('train_method', 'CCKA')) or "CCKA"
    subcentroids = config.get('clusters', "")
    noise_level = config.get('noise', "")
    mitigation = _safe_bool_str(config.get('mitigation', ""))  # not in your search_space yet; remains blank
    n_samples = config.get('n_samples', "")

    # === Kernel/model ===
    kernel = Qkernel(
        device=config['device'],
        n_qubits=config['n_qubits'],
        trainable=config['trainable'],
        input_scaling=config['input_scaling'],
        data_reuploading=config['data_reuploading'],
        ansatz=config['ansatz'],
        ansatz_layers=config['ansatz_layers']
    )

    agent = TrainModel(
        kernel=kernel,
        training_data=training_data,
        training_labels=training_labels,
        testing_data=testing_data,
        testing_labels=testing_labels,
        optimizer=config['optimizer'],
        lr=config['lr'],
        mclr=config['mclr'],
        cclr=config['cclr'],
        epochs=config['epochs'],
        train_method=config['train_method'],
        target_accuracy=config['target_accuracy'],
        get_alignment_every=config['get_alignment_every'],
        validate_every_epoch=config['validate_every_epoch'],
        base_path=config['base_path'],
        lambda_kao=config['lambda_kao'],
        lambda_co=config['lambda_co'],
        clusters=config['clusters'],
        get_decesion_boundary = config['decesion_boundary'],
        use_kmeans= config['use_kmeans'],
        logger = logger,
        config=config,
        run_id=run_id,
    )   

    # Pre-train evaluation
    if args.backend == 'qiskit':
        before_metrics = agent.evaluate_parallel(testing_data, testing_labels, 'before')
    else:
        before_metrics = agent.evaluate_test(testing_data, testing_labels, 'before')
    print(before_metrics)

    # === Training (timed)
    t0 = time.time()
    if multi_class:
        kernel, params, main_centroid, sub_centroid = agent.fit_multiclass(training_data, training_labels)
    else:
        kernel, params, main_centroid, sub_centroid = agent.fit_kernel(training_data, training_labels)
    train_time_sec = time.time() - t0
    print('Training Complete')

    # Post-train evaluation
    if args.backend == 'qiskit':
        after_metrics = agent.evaluate_parallel(testing_data, testing_labels, 'after')
    else:
        after_metrics = agent.evaluate_test(testing_data, testing_labels, 'after')
    print(after_metrics)

    # === Build per-run JSON-like metrics (Ray) and append to CSV ===
    circuits_total = after_metrics.get('executions', None)
    test_acc_final = after_metrics.get('testing_accuracy', None)

    # session.report for Ray dashboards (rich metrics bundle)
    ray_metrics = {
        "dataset": dataset_name,
        "method": method,
        "run_id": run_id,
        "num_layers": config['ansatz_layers'],
        "accuracy_train_init": before_metrics.get('training_accuracy', None),
        "accuracy_test_init": before_metrics.get('testing_accuracy', None),
        "alignment_train_init": before_metrics.get('alignment', None),
        "accuracy_train_final": after_metrics.get('training_accuracy', None),
        "accuracy_test_final": test_acc_final,
        "alignment_train_epochs": after_metrics.get('alignment_arr', None),
        "circuit_executions": circuits_total,
        "train_time_sec": train_time_sec,
        "subcentroids": subcentroids,
        "noise_level": noise_level,
        "mitigation": mitigation,
        "n_samples": n_samples,
    }
    session.report(to_python_native(ray_metrics))

    # Append per-run summary CSV row (exact schema)
    per_run_row = {
        "dataset": dataset_name,
        "method": method,
        "run_id": run_id,
        "test_accuracy": test_acc_final if test_acc_final is not None else "",
        "train_time_sec": f"{train_time_sec:.6f}",
        "circuits_total": circuits_total if circuits_total is not None else "",
        "subcentroids": subcentroids,
        "noise_level": noise_level,
        "mitigation": mitigation,
        "n_samples": n_samples,
    }

    logger.log_per_run(per_run_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This parser receives the yaml config file")
    parser.add_argument("--backend", default="qiskit")
    parser.add_argument("--config", default="configs/checkerboard.yaml")
    args = parser.parse_args()

    if args.backend == 'qiskit':
        from utils.qiskit.model import Qkernel
    elif args.backend == 'pennylane':
        from utils.pennylane.model import Qkernel

    with open(args.config) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    config = namedtuple("ObjectName", data.keys())(*data.values())

    path_from_home = str(Path.cwd())
    file_path = path_from_home + config.dataset['file']
    base_path = path_from_home + config.agent['base_path']

    ray.init(log_to_driver=False)
    search_space = {
        'repeat': tune.grid_search(list(range(1, config.ray_config['ray_num_trial_samples'] + 1))),
        'name': config.dataset['name'],
        'file': None if config.dataset['file'] == 'None' else file_path,
        'n_samples': config.dataset['n_samples'],
        'noise': config.dataset['noise'],
        'num_sectors': config.dataset['num_sectors'],
        'points_per_sector': config.dataset['points_per_sector'],
        'grid_size': config.dataset['grid_size'],
        'sampling_radius': config.dataset['sampling_radius'],
        'training_size': config.dataset['training_size'],
        'testing_size': config.dataset['testing_size'],
        'validation_size': config.dataset['validation_size'],
        'device': config.qkernel['device'],
        'n_qubits': tune.grid_search(config.qkernel['n_qubits']),
        'trainable': config.qkernel['trainable'],
        'input_scaling': config.qkernel['input_scaling'],
        'data_reuploading': config.qkernel['data_reuploading'],
        'ansatz': tune.grid_search(config.qkernel['ansatz']),
        'ansatz_layers': tune.grid_search(config.qkernel['ansatz_layers']),
        'optimizer': tune.grid_search(config.agent['optimizer']),
        'lr': tune.grid_search(config.agent['lr']),
        'mclr': tune.grid_search(config.agent['mclr']),
        'cclr': tune.grid_search(config.agent['cclr']),
        'epochs': tune.grid_search(config.agent['epochs']),
        'train_method': tune.grid_search(config.agent['train_method']),
        'target_accuracy': config.agent['target_accuracy'],
        'get_alignment_every': config.agent['get_alignment_every'],
        'validate_every_epoch': config.agent['validate_every_epoch'],
        'base_path': base_path,
        'lambda_kao': tune.grid_search(config.agent['lambda_kao']),
        'lambda_co': tune.grid_search(config.agent['lambda_co']),
        'clusters': tune.grid_search(config.agent['clusters']),
        'decesion_boundary': config.agent['decesion_boundary'],
        'use_kmeans': tune.grid_search(config.agent['use_kmeans']),
        'ray_logging_path': config.ray_config['ray_logging_path']  # directory for CSVs
    }

    def trial_name_creator(trial):
        return trial.__str__() + '_' + trial.experiment_tag + ','

    tuner = tune.Tuner(
        tune.with_resources(
            train,
            resources={"cpu": config.ray_config['num_cpus'], "gpu": config.ray_config['num_gpus']}
        ),
        tune_config=tune.TuneConfig(
            num_samples=config.ray_config['ray_num_trial_samples'],
            trial_dirname_creator=trial_name_creator,
        ),
        param_space=search_space,
    )

    tuner.fit()
    ray.shutdown()
