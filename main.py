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
import time
import os 

#Custom Libraries
from utils.model import qkernel
from utils.classification_data import plot_and_save
from utils.train import train_model
from utils.train_ccka import train_ccka_model


##Backend Configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"MPS is available. Using device: {device}")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"Neither MPS nor CUDA is available. Using CPU: {device}")

#Read Configs
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


#Dataset
features, target = plot_and_save(config['dataset']['name'], 
                                 config['dataset']['n_samples'], 
                                 config['dataset']['noise'], 
                                 save_path= f"{config['dataset']['figure_path']}/{config['dataset']['name']}.png")

feature_dimensions =  len(features[0]) #math.ceil(math.log2(len(feature[0])))
n_classes = len(np.unique(target))


training_data, testing_data, training_labels, testing_labels = train_test_split(features, target, test_size=0.25, random_state=42)

# Convert each data point to a torch tensor
training_data = torch.tensor(training_data, dtype=torch.float32, requires_grad=True)
testing_data = torch.tensor(testing_data, dtype=torch.float32, requires_grad=True)
training_labels = torch.tensor(training_labels, dtype=torch.int)
testing_labels = torch.tensor(testing_labels, dtype=torch.int)

print(f"* Train Shape: {training_data.shape}")
print(f"* Train Labels Shape:  {training_labels.shape}")
print(f"* Test Shape: {testing_data.shape}")
print(f"* Test Labels Shape:  {testing_labels.shape}")
print(" ")

kernel = qkernel(config)
print("Sample Distance between x[0] and x[1]: ", kernel(training_data[0], training_data[1]))

if config['training']['method'] == 'ccka':
    learning_rate = 0.01
    optimizer = optim.Adam(kernel.parameters(), lr=learning_rate)

    agent = train_ccka_model( kernel= kernel,
                        training_data = training_data,
                        training_labels = training_labels,
                        optimizer= optimizer,
                        train_method= 'ccka',
                        clusters = 4
                        )

    init_metrics = agent.evaluate(testing_data, testing_labels)
    agent.fit_kernel(training_data, training_labels)
    after_metrics = agent.evaluate(testing_data, testing_labels)


else:
    # Define optimizer
    learning_rate = 0.01
    optimizer = optim.Adam(kernel.parameters(), lr=learning_rate)

    agent = train_model( kernel= kernel,
                        training_data = training_data,
                        training_labels = training_labels,
                        optimizer= optimizer,
                        train_method= 'random',
                        sampling_size= 8
                        )

    init_metrics = agent.evaluate(testing_data, testing_labels)
    agent.fit_kernel(training_data, training_labels)
    after_metrics = agent.evaluate(testing_data, testing_labels)
