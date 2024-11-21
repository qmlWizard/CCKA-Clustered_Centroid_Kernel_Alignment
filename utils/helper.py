import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
import yaml
import time
import os


def to_python_native(obj):
    if isinstance(obj, torch.Tensor):  # Convert tensors to Python scalars
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, np.ndarray):  # Convert numpy arrays to lists
        return obj.tolist()
    elif isinstance(obj, list):  # Handle lists of tensors or arrays
        return [to_python_native(x) for x in obj]
    elif isinstance(obj, dict):  # Handle nested dictionaries
        return {k: to_python_native(v) for k, v in obj.items()}
    return obj  # Return Python-native types as is


def tensor_to_list(data):
    if isinstance(data, torch.Tensor):
        return data.tolist()  # Convert tensor to list
    elif isinstance(data, dict):
        return {k: tensor_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [tensor_to_list(v) for v in data]
    else:
        return data