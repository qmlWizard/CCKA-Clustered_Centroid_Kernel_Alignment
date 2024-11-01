import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import math
import yaml
import json
import time
import os
import pandas as pd

import ray
from ray import tune

# Custom Libraries
from utils.model import qkernel
from archive.classification_data import plot_and_save
from utils.agent import TrainModel

# Backend Configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"MPS is available. Using device: {device}")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"Neither MPS nor CUDA is available. Using CPU: {device}")

# Read Configs
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def train_model(config, checkpoint_dir=None):
    # Dataset generation and splitting
    features, target = plot_and_save(config['dataset']['name'],
                                     config['dataset']['n_samples'],
                                     config['dataset']['noise'],
                                     save_path=f"{config['dataset']['figure_path']}/{config['dataset']['name']}.png")

    training_data, testing_data, training_labels, testing_labels = train_test_split(features, target, test_size=0.50, random_state=42)

    print(f"* Train Shape: {training_data.shape}")
    print(f"* Train Labels Shape:  {training_labels.shape}")
    print(f"* Test Shape: {testing_data.shape}")
    print(f"* Test Labels Shape:  {testing_labels.shape}")
    print(" ")

    print(config)

    # Convert each data point to a torch tensor
    training_data = torch.tensor(training_data, dtype=torch.float32, requires_grad=True)
    testing_data = torch.tensor(testing_data, dtype=torch.float32, requires_grad=True)
    training_labels = torch.tensor(training_labels, dtype=torch.int)
    testing_labels = torch.tensor(testing_labels, dtype=torch.int)

    kernel = qkernel(config)
    print(f"Sample Distance between x[0] and x[1]: ", kernel(training_data[0], training_data[1]))

    # Initialize TrainModel agent and perform training
    agent = TrainModel(
        kernel=kernel,
        training_data=training_data,
        training_labels=training_labels,
        testing_data=testing_data,
        testing_labels=testing_labels,
        optimizer=config['training']['optimizer'],
        lr=config['training']['learning_rate'],
        train_method=config['training']['method'],
        clusters=config['training']['clusters'],
        epochs=config['training']['epochs'],
        lambda_kao=config['training']['lambda_kao'],
        lambda_co=config['training']['lambda_co'],
        base_path=config['training']['base_path']
    )

    # Fit kernel and re-evaluate
    agent.fit_kernel(training_data, training_labels)
    after_metrics = agent.evaluate(testing_data, testing_labels)

    # Report final metrics to Ray Tune
    tune.report(accuracy=after_metrics['accuracy'], f1_score=after_metrics['f1_score'])


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

search_space = {
    'training': {
        'method': tune.choice(config['training']['method']),
        'optimizer': tune.choice(config['training']['optimizer']),
        'learning_rate': tune.choice(config['training']['learning_rate']),
        'clusters': tune.choice(config['training']['clusters']),
        'sampling': tune.choice(config['training']['sampling']),
        'epochs': tune.choice(config['training']['epochs']),
        'lambda_kao': tune.choice(config['training']['lambda_kao']),
        'lambda_co': tune.choice(config['training']['lambda_co']),
        'batch_size': tune.choice(config['training']['batch_size']),
        'criterion': config['training']['criterion'],
        'base_path': tune.sample_from(lambda config: f"embedded_results/{config['training']['method']}_opt-{config['training']['optimizer']}_lr-{config['training']['learning_rate']:.4f}_clusters-{config['training']['clusters']}_sampling-{config['training']['sampling']}")
    },
    'dataset': {
        'name': tune.choice(config['dataset']['name']),
        'n_samples': tune.choice(config['dataset']['n_samples']),
        'noise': tune.choice(config['dataset']['noise']),
        'figure_path': config['dataset']['figure_path'],
        'n_features': config['dataset']['n_features'],
        'n_classes': config['dataset']['n_classes'],
        'random_state': config['dataset']['random_state']
    },
    'qkernel': {
        'device': config['qkernel']['device'],
        'n_qubits': config['qkernel']['n_qubits'],
        'trainable': config['qkernel']['trainable'],
        'input_scaling': config['qkernel']['input_scaling'],
        'data_reuploading': config['qkernel']['data_reuploading'],
        'ansatz': config['qkernel']['ansatz'],
        'ansatz_layers': config['qkernel']['ansatz_layers'],
        'entangling': config['qkernel']['entangling']
    }
}

# Run Ray Tune
analysis = tune.run(
    train_model,
    config=search_space,
    num_samples=10,  # Number of hyperparameter combinations to try
    resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
    storage_path=os.path.abspath("ray_results"),
    name="qkernel_training",
    metric="accuracy",
    mode="max"
)

print("Best hyperparameters found were: ", analysis.best_config)
