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
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def train(config):
    print(config)


if __name__ == "__main__":

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    search_space = {

        'qkernel': {
                        
                   }

    }