import pennylane as qml
import torch
from torch.utils.data import DataLoader, TensorDataset
import ray
from ray import tune
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import yaml
import argparse
import shutil
from collections import namedtuple
import json
import time
import os
import datetime

# Custom Libraries
from utils.model import Qkernel
from utils.data_generator import DataGenerator
from utils.agent import TrainModel
from utils.helper import tensor_to_list, to_python_native

# Backend Configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def train(config):
    # Dataset generation
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
    print("Dataset", config['clusters'])
    features, target = data_generator.generate_dataset()
    training_data, testing_data, training_labels, testing_labels = train_test_split(
        features, target, test_size=0.50, random_state=42)
    training_data = torch.tensor(
        training_data.to_numpy(), dtype=torch.float32, requires_grad=True)
    testing_data = torch.tensor(
        testing_data.to_numpy(), dtype=torch.float32, requires_grad=True)
    training_labels = torch.tensor(training_labels.to_numpy(), dtype=torch.int)
    testing_labels = torch.tensor(testing_labels.to_numpy(), dtype=torch.int)

    # Kernel initialization
    kernel = Qkernel(
        device=config['device'],
        n_qubits=config['n_qubits'],
        trainable=config['trainable'],
        input_scaling=config['input_scaling'],
        data_reuploading=config['data_reuploading'],
        ansatz=config['ansatz'],
        ansatz_layers=config['ansatz_layers']
    )

    # Agent initialization
    agent = TrainModel(
        kernel=kernel,
        training_data=training_data,
        training_labels=training_labels,
        testing_data=testing_data,
        testing_labels=testing_labels,
        optimizer=config['optimizer'],
        lr=config['lr'],
        epochs=config['epochs'],
        train_method=config['train_method'],
        target_accuracy=config['target_accuracy'],
        get_alignment_every=config['get_alignment_every'],
        validate_every_epoch=config['validate_every_epoch'],
        base_path=config['base_path'],
        lambda_kao=config['lambda_kao'],
        lambda_co=config['lambda_co'],
        clusters=config['clusters']
    )

    # Initial evaluation
    initial_metrics = agent.evaluate(testing_data, testing_labels)
    agent.fit_kernel(training_data, training_labels)
    after_metrics = agent.evaluate(testing_data, testing_labels)

    # Metrics for logging
    metrics = {
        "num_layers": config['ansatz_layers'],
        "accuracy_train_init": initial_metrics['training_accuracy'],
        "accuracy_test_init": initial_metrics['testing_accuracy'],
        "alignment_train_init": initial_metrics['alignment'],
        "accuracy_train_final": after_metrics['training_accuracy'],
        "accuracy_test_final": after_metrics['testing_accuracy'],
        "alignment_train_epochs": after_metrics['alignment_arr'],
        "circuit_executions": after_metrics['executions'],
    }

    metrics = to_python_native(metrics)
    # Log results for the trial
    print("Reporting metrics:", metrics)
    ray.train.report(metrics)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "This parser receives the yaml config file")
    parser.add_argument("--config", default = "configs/checkerboard.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        data = yaml.load(f, Loader = yaml.FullLoader)
    config = namedtuple("ObjectName", data.keys())(*data.values())

    ray.init(
             local_mode = config.ray_config['ray_local_mode'],
             num_cpus = config.ray_config['num_cpus'],
             num_gpus=config.ray_config['num_gpus'],
             include_dashboard = False
            )
    
    search_space = {
        'name':  config.dataset['name'],
        'file':  None if config.dataset['file'] == 'None' else config.dataset['file'],
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
        'n_qubits': config.qkernel['n_qubits'],
        'trainable': config.qkernel['trainable'],
        'input_scaling': config.qkernel['input_scaling'],
        'data_reuploading': config.qkernel['data_reuploading'],
        'ansatz': config.qkernel['ansatz'],
        'ansatz_layers': config.qkernel['ansatz_layers'],
        'optimizer': config.agent['optimizer'],
        'lr': tune.grid_search(config.agent['lr']),
        'epochs': config.agent['epochs'],
        'train_method': config.agent['train_method'],
        'target_accuracy': config.agent['target_accuracy'],
        'get_alignment_every': config.agent['get_alignment_every'],
        'validate_every_epoch': config.agent['validate_every_epoch'],
        'base_path': config.agent['base_path'],
        'lambda_kao': config.agent['lambda_kao'],
        'lambda_co': config.agent['lambda_co'],
        'clusters': tune.grid_search(config.agent['clusters']),
        'ray_logging_path': config.ray_config['ray_logging_path']
    }

    name = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + config.agent['train_method'] + '_'
    ray_path = os.getcwd() + '/Documents/developer/greedy_kernel_alignment/' + config.ray_config['ray_logging_path']
    path = ray_path + "/" + name

    os.makedirs(os.path.dirname(path + '/'), exist_ok=True)
    shutil.copy(args.config, path + '/alg_config.yml')

    def trial_name_creator(trial):
            return trial.__str__() + '_' + trial.experiment_tag + ','
    

    tuner = tune.Tuner(
            tune.with_resources(train, resources={"cpu": 20, "gpu": 2}),
            tune_config=tune.TuneConfig(num_samples=config.ray_config['ray_num_trial_samples'],
                                        trial_dirname_creator=trial_name_creator),
            param_space= search_space,
            )
        
    tuner.fit()
    ray.shutdown()
