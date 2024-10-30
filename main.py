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

# Custom Libraries
from utils.model import qkernel
from utils.classification_data import plot_and_save
from utils.train import train_model
from utils.train_ccka import train_ccka_model

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

# Initialize storage for metrics across all runs
all_metrics = []

# Run the CCKA training and evaluation 5 times with full re-initialization
for run in range(5):
    
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

    # Convert each data point to a torch tensor
    training_data = torch.tensor(training_data, dtype=torch.float32, requires_grad=True)
    testing_data = torch.tensor(testing_data, dtype=torch.float32, requires_grad=True)
    training_labels = torch.tensor(training_labels, dtype=torch.int)
    testing_labels = torch.tensor(testing_labels, dtype=torch.int)

    # Kernel initialization
    kernel = qkernel(config)
    print(f"Run {run + 1}: Sample Distance between x[0] and x[1]: ", kernel(training_data[0], training_data[1]))

    # Initialize agent and perform training
    if config['training']['method'] == 'ccka':
        agent = train_ccka_model(
            kernel=kernel,
            training_data=training_data,
            training_labels=training_labels,
            optimizer=config['training']['optimizer'],
            lr=config['training']['learning_rate'],
            train_method='ccka',
            clusters=config['training']['clusters'],
            epochs=config['training']['epochs']
        )
        experiment_name = f"embedding_paper/{config['dataset']['name']}_{config['training']['method']}_{config['training']['clusters']}_exp1.json"
    else:
        agent = train_model(
            kernel=kernel,
            training_data=training_data,
            training_labels=training_labels,
            optimizer=config['training']['optimizer'],
            lr=config['training']['learning_rate'],
            train_method=config['training']['method'],
            sampling_size=config['training']['sampling'],
            epochs=config['training']['epochs']
        )
        experiment_name = f"he_results_cmdloss/{config['dataset']['name']}_{config['training']['method']}_{config['training']['sampling']}_exp1.json"

    # Evaluate before fitting kernel
    init_metrics = agent.evaluate(testing_data, testing_labels)
    # Fit kernel and re-evaluate
    agent.fit_kernel(training_data, training_labels)
    after_metrics = agent.evaluate(testing_data, testing_labels)

    # Collect metrics for each run
    all_metrics.append({'init': init_metrics, 'final': after_metrics})
    print(f"Run {run + 1} metrics:")
    print("Initial metrics:", init_metrics)
    print("Final metrics:", after_metrics)

# Calculate average metrics across all runs
avg_metrics = {
    'accuracy': sum(run['final']['accuracy'] for run in all_metrics) / 5,
    'f1_score': sum(run['final']['f1_score'] for run in all_metrics) / 5
}

# Store metrics and average in JSON format

metrics_to_save = {
    'runs': all_metrics,
    'average_metrics': avg_metrics
}

# Ensure directory exists
os.makedirs("he_results_cmdloss", exist_ok=True)

# Save to JSON
#with open(experiment_name, 'w') as json_file:
#    json.dump(metrics_to_save, json_file, indent=4)

print(f"Metrics saved to {experiment_name}")
