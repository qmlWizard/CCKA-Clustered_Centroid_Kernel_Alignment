import pennylane as qml
import torch
from torch.utils.data import DataLoader, TensorDataset
import ray
from ray import tune
from ray.air import session
from sklearn.model_selection import train_test_split
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
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data = np.load('checkerboard_dataset.npy', allow_pickle=True).item()
x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']

training_data = torch.tensor(x_train, dtype=torch.float32, requires_grad=True)
testing_data = torch.tensor(x_test, dtype=torch.float32, requires_grad=True)
training_labels = torch.tensor(y_train, dtype=torch.int)
testing_labels = torch.tensor(y_test, dtype=torch.int)
kernel = Qkernel(   
                        device = config['qkernel']['device'], 
                        n_qubits = 2, 
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
                        lr= 0.2,
                        epochs = 500,
                        train_method= 'ccka',
                        target_accuracy=0.95,
                        get_alignment_every=10,  
                        validate_every_epoch=10, 
                        base_path='.',
                        lambda_kao=0.01,
                        lambda_co=0.1,
                        clusters=4
                      )

intial_metrics = agent.evaluate(testing_data, testing_labels)
agent.fit_kernel(training_data, training_labels)
after_metrics = agent.evaluate(testing_data, testing_labels)

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
filename = "checkerboard_rodrigo_data_random.json"

# Write the JSON-serializable results to a file
with open(filename, "w") as file:
    json.dump(results, file, indent=4)