import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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
training_labels = torch.tensor(training_labels, dtype=torch.long)
testing_labels = torch.tensor(testing_labels, dtype=torch.long)

print(f"* Train Shape: {training_data.shape}")
print(f"* Train Labels Shape:  {training_labels.shape}")
print(f"* Test Shape: {testing_data.shape}")
print(f"* Test Labels Shape:  {testing_labels.shape}")
print(" ")


kernel = qkernel(config)


print("Sample Distance between x[0] and x[1]: ", kernel(training_data[0], training_data[1]))

#K_init = qml.kernels.square_kernel_matrix(training_data, kernel, assume_normalized_kernel=True)

n_samples = training_data.shape[0]
K_theta = torch.zeros((n_samples, n_samples), dtype=torch.float32)
for i in range(len(training_data)):
    for j in range(len(training_data)):
        print(" ",i, j, " ")
        K_theta[i, j] = kernel(training_data[i], training_data[j])
    print("\n")


