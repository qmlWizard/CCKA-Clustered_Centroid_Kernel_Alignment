import torch
import ray
from ray import tune
from sklearn.model_selection import train_test_split
import yaml
import argparse
import shutil
from collections import namedtuple
import os
import datetime

# Custom Libraries
from utils.model import Qkernel
from utils.data_generator import DataGenerator
from utils.agent import TrainModel
from utils.plotter import Plotter
from utils.helper import to_python_native, gen_experiment_name

# Backend Configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def train(config):
    # Create results directory with subfolders for the dataset and experiment name
    results_dir = os.path.join(config['base_path'], config['name'], 'exp1')
    exp_name = gen_experiment_name(config)
    exp_dir = os.path.join(results_dir, exp_name)

    # Ensure the directory exists
    os.makedirs(exp_dir, exist_ok=True)

    # Initialize data generator
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

    features, target = data_generator.generate_dataset()
    training_data, testing_data, training_labels, testing_labels = train_test_split(features, target, test_size=0.50, random_state=42)
    training_data = torch.tensor(training_data.to_numpy(), dtype=torch.float32, requires_grad=True)
    testing_data = torch.tensor(testing_data.to_numpy(), dtype=torch.float32, requires_grad=True)
    training_labels = torch.tensor(training_labels.to_numpy(), dtype=torch.int)
    testing_labels = torch.tensor(testing_labels.to_numpy(), dtype=torch.int)

    # Initialize plotter
    plotter = Plotter(
        style='seaborn-v0_8',
        final_color='#ffa07a',
        initial_color='#4682b4',
        plot_dir=exp_dir
    )

    # Initialize kernel
    kernel = Qkernel(
        device=config['device'],
        n_qubits=config['n_qubits'],
        trainable=config['trainable'],
        input_scaling=config['input_scaling'],
        data_reuploading=config['data_reuploading'],
        ansatz=config['ansatz'],
        ansatz_layers=config['ansatz_layers']
    )

    # Initialize training agent
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
        base_path=exp_dir,
        lambda_kao=config['lambda_kao'],
        lambda_co=config['lambda_co'],
        clusters=config['clusters']
    )

    # Evaluate before training
    before_metrics = agent.evaluate(testing_data, testing_labels)
    print("Before Training Metrics:", before_metrics)

    # Train the model
    agent.fit_kernel(training_data, training_labels)
    print('Training Complete')

    # Evaluate after training
    after_metrics = agent.evaluate(testing_data, testing_labels)
    print("After Training Metrics:", after_metrics)

    # Prepare metrics for saving
    metrics = {
        "num_layers": config['ansatz_layers'],
        "accuracy_train_init": before_metrics['training_accuracy'],
        "accuracy_test_init": before_metrics['testing_accuracy'],
        "alignment_train_init": before_metrics['alignment'],
        "accuracy_train_final": after_metrics['training_accuracy'],
        "accuracy_test_final": after_metrics['testing_accuracy'],
        "alignment_train_epochs": after_metrics['alignment_arr'],
        "circuit_executions": after_metrics['executions'],
    }
    metrics = to_python_native(metrics)

    # Save metrics to a YAML file
    metrics_file = os.path.join(exp_dir, f"{exp_name}_metrics.yaml")
    with open(metrics_file, "w") as f:
        yaml.dump(metrics, f)
    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "This parser receives the yaml config file")
    parser.add_argument("--config", default = "configs/checkerboard.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        data = yaml.load(f, Loader = yaml.FullLoader)
    config = namedtuple("ObjectName", data.keys())(*data.values())

    methods = ['ccka', 'random', 'quack', 'full']
    epochs = [200, 500, 1000]
    clusters = [2, 4, 6, 8, 10]

    for method in methods:
        for cluster in clusters:
            for epoch in epochs:
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
                    'lr': 0.1,
                    'mclr': 0.1,
                    'cclr': 0.01,
                    'epochs': epoch,
                    'train_method': method,
                    'target_accuracy': config.agent['target_accuracy'],
                    'get_alignment_every': config.agent['get_alignment_every'],
                    'validate_every_epoch': config.agent['validate_every_epoch'],
                    'base_path': config.agent['base_path'],
                    'lambda_kao': 0.01,
                    'lambda_co': 0.01,
                    'clusters': cluster,
                    'ray_logging_path': config.ray_config['ray_logging_path']
                }

                train(search_space)