import pennylane as qml
import torch
from torch.utils.data import DataLoader, TensorDataset
import ray
from ray import tune
from ray.air import session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yaml
import json
import time
import os
import pandas as pd


# Custom Libraries
from utils.model import Qkernel
from utils.data_generator import DataGenerator
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
with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_generator = DataGenerator(     
                                dataset_name = 'corners', 
                                file_path = '/Users/digvijaysinhajarekar/Developer/greedy_kernel_alignment/data/corners.npy',
                                n_samples = 200, 
                                noise = 0.1, 
                                num_sectors = 6, 
                                points_per_sector = 15, 
                                grid_size = 4, 
                                sampling_radius = 0.05,
                                n_pca_features=None
                              )

features, target = data_generator.generate_dataset()
training_data, testing_data, training_labels, testing_labels = train_test_split(features, target, test_size=0.50, random_state=42)
training_data = torch.tensor(training_data.to_numpy(), dtype=torch.float32, requires_grad=True)
testing_data = torch.tensor(testing_data.to_numpy(), dtype=torch.float32, requires_grad=True)
training_labels = torch.tensor(training_labels.to_numpy(), dtype=torch.int)
testing_labels = torch.tensor(testing_labels.to_numpy(), dtype=torch.int)

kernel = Qkernel(   
                    device = config['qkernel']['device'], 
                    n_qubits = 4, 
                    trainable = True, 
                    input_scaling = True, 
                    data_reuploading = True, 
                    ansatz = 'he', 
                    ansatz_layers = 5   
                )
    
agent = TrainModel(
                    kernel=kernel,
                    training_data=training_data,
                    training_labels=training_labels,
                    testing_data=testing_data,
                    testing_labels=testing_labels,
                    optimizer= 'gd',
                    lr= 0.1,
                    mclr= 0.1,
                    cclr= 0.1,
                    epochs = 400,
                    train_method= 'ccka',
                    target_accuracy=0.95,
                    get_alignment_every=1,  
                    validate_every_epoch=None, 
                    base_path='.',
                    lambda_kao=0.01,
                    lambda_co=0.01,
                    clusters=4
                )

agent.prediction_stage(testing_data, testing_labels)
intial_metrics = agent.evaluate(testing_data, testing_labels)
agent.fit_kernel(training_data, training_labels)
after_metrics = agent.evaluate(testing_data, testing_labels)
agent.prediction_stage(testing_data, testing_labels)

def tensor_to_list(data):
    if isinstance(data, torch.Tensor):
        return data.tolist()  # Convert tensor to list
    elif isinstance(data, dict):
        return {k: tensor_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [tensor_to_list(v) for v in data]
    else:
        return data
    
results = {
    'initial_metrics': tensor_to_list(intial_metrics),
    'after_metrics': tensor_to_list(after_metrics),
}

# Specify the filename
filename = "checkerboard_rodrigo_data_ccka_exp3.json"

# Write the JSON-serializable results to a file
with open(filename, "w") as file:
    json.dump(results, file, indent=4)