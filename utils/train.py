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
import numpy as np
import math
import yaml
import time
import os 


class train_model():

    def __init__(self, 
                 kernel,
                 training_data,
                 training_labels,
                 optimizer,
                 train_method,
                 sampling_size=8,
                 clusters=2):
        super().__init__()

        self._kernel = kernel
        self._optimizer = optimizer
        self._method = train_method
        self._epochs = 100
        self._sampling_size = sampling_size
        self._clusters = clusters
        self._kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, self._kernel)
        self._training_data = training_data
        self._training_labels = training_labels
        self._executions = 0

        if self._method == 'random':
            self._loss_function = self._loss_ta
            self._sample_function = self._sampler_random_sampling
        elif self._method == 'full':
            self._loss_function = self._loss_ta
            self._sample_function = self._full_data
        
        self._loss_arr = []
        self.alignment_arr = []
        
        _matrix = self._kernel_matrix(self._training_data, self._training_data)
        if torch.is_tensor(_matrix):
            _matrix = _matrix.detach().numpy()
        if torch.is_tensor(self._training_labels):
            self._training_labels = self._training_labels.detach().numpy()

        self._model = SVC(kernel='precomputed').fit(_matrix, self._training_labels)

    
    def _loss_ta(self, data, data_labels):
        return qml.kernels.target_alignment(data, data_labels, self._kernel, assume_normalized_kernel=True)

    def _sampler_random_sampling(self, data, data_labels):
        subset_indices = torch.randperm(len(data))[:self._sampling_size]
        return data[subset_indices], data_labels[subset_indices]

    def _full_data(self, data, data_labels):
        return data, data_labels
    
    def fit_kernel(self, training_data, training_labels):
        optimizer = self._optimizer
        epochs = self._epochs
        loss_func = self._loss_function
        samples_func = self._sample_function
        self._kernel._circuit_executions = 0
        for epoch in range(epochs):
            optimizer.zero_grad()

            sampled_data, sampled_labels = samples_func(training_data, training_labels)
            loss = -loss_func(sampled_data, sampled_labels)
            loss.backward()
            optimizer.step()

            # Store and print loss values
            self._loss_arr.append(loss.item())
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        if torch.is_tensor(training_data):
            training_data = training_data.detach().numpy()
        if torch.is_tensor(training_labels):
            training_labels = training_labels.detach().numpy()

        self._executions = self._kernel._circuit_executions
        _matrix = self._kernel_matrix(self._training_data, self._training_data)
        if torch.is_tensor(_matrix):
            _matrix = _matrix.detach().numpy()
        if torch.is_tensor(self._training_labels):
            self._training_labels = self._training_labels.detach().numpy()

        self._model = SVC(kernel='precomputed').fit(_matrix, self._training_labels)
        return self._model

    def evaluate(self, test_data, test_labels):


        matrix = self._kernel_matrix(test_data, self._training_data)
        if torch.is_tensor(matrix):
            matrix = matrix.detach().numpy()
        if torch.is_tensor(test_labels):
            test_labels = test_labels.detach().numpy()

        predictions = self._model.predict(matrix)
        # Calculate evaluation metrics
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average='weighted')
        auc = roc_auc_score(test_labels, predictions)
        # Print and return the results
        print(f"Testing Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"AUC: {auc}")
        return {
            'executions': self._executions,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc
        }
