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


if __name__ == "__main__":

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)