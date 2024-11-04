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


class TrainModel():

    def __init__(self, 
                 kernel,
                 training_data,
                 training_labels,
                 testing_data,
                 testing_labels,
                 optimizer,
                 lr,
                 epochs,
                 train_method,
                 target_accuracy=None,  # New parameter for target accuracy
                 get_alignment_every=10,  # New parameter to select alignment frequency
                 validate_every_epoch=False,  # New parameter for validation accuracy
                 base_path=None,  # New parameter for base path to save plots
                 lambda_kao=0.01,
                 lambda_co=0.01,
                 clusters=4
                 ):
        super().__init__()

        self._kernel = kernel
        self._optimizer = optimizer
        self._method = train_method
        self._epochs = epochs
        self._target_accuracy = target_accuracy
        self._get_alignment_every = get_alignment_every
        self._validate_every_epoch = validate_every_epoch
        self._sampling_size = clusters * 2
        self._clusters = clusters
        self._training_data = training_data
        self._training_labels = training_labels
        self._testing_data = testing_data
        self._testing_labels = testing_labels
        self._n_classes = torch.unique(training_labels)
        self._kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, self._kernel)
        self._executions = 0
        self._lr = lr
        self.lambda_kao = lambda_kao
        self.lambda_co = lambda_co
        self._base_path = base_path

        self._main_centroids = []
        self._main_centroids_labels = []
        self._class_centroids = []
        self._class_centroid_labels = []

        self._get_centroids(self._training_data, self._training_labels)
        self._loss_arr = []
        self.alignment_arr = []
        self.validation_accuracy_arr = []
        self.initial_training_accuracy = None
        self.final_training_accuracy = None
        self.initial_testing_accuracy = None
        self.final_testing_accuracy = None

        # Flatten the list of class centroids to pass as parameters
        self._flattened_class_centroids = [centroid.clone().detach().requires_grad_() for cluster in self._class_centroids for centroid in cluster]

        if optimizer == 'adam':
            # Define optimizer with centroids as parameters
            self._kernel_optimizer = optim.Adam(self._kernel.parameters(), lr=self._lr)
            self._centroid_optimizer = optim.Adam(self._flattened_class_centroids, lr=self._lr)
        elif optimizer == 'gd':
            self._kernel_optimizer = optim.SGD(self._kernel.parameters(), lr=self._lr)
            self._centroid_optimizer = optim.SGD(self._flattened_class_centroids, lr=self._lr)
        self._centroid_minimization_opt = optim.Adam([
            {'params': self._flattened_class_centroids, 'lr': self._lr},
            {'params': self._kernel.parameters(), 'lr': self._lr},
        ])

        if self._method == 'random':
            self._loss_function = self._loss_ta
            self._sample_function = self._sampler_random_sampling
        elif self._method == 'full':
            self._loss_function = self._loss_ta
            self._sample_function = self._full_data
        elif self._method == 'ccka':
            self._loss_function = self._loss_kao
            self._centroid_loss_function = self.loss_co
            self._sample_function = self._get_all_centroids

    @property
    def _get_all_centroids(self):
        centroids = self._main_centroids + [centroid for cluster in self._class_centroids for centroid in cluster]
        return torch.stack(centroids, dim=0).requires_grad_()

    @property
    def _get_all_centroid_labels(self):
        labels = self._main_centroids_labels + [label for cluster in self._class_centroid_labels for label in cluster]
        return torch.tensor(labels, dtype=torch.int)

    def _get_centroids(self, data, data_labels):
        for c in self._n_classes:
            class_data = data[data_labels == c]
            main_centroid = torch.mean(class_data, dim=0)
            self._main_centroids.append(main_centroid.requires_grad_())
            self._main_centroids_labels.append(c)
            class_centroids = [torch.mean(cluster, dim=0) for cluster in torch.chunk(class_data, self._clusters)]
            self._class_centroids.append([centroid.requires_grad_() for centroid in class_centroids])
            self._class_centroid_labels.append([c] * self._clusters)

    def _centered_kernel_matrix(self, x):
        return qml.kernels.kernel_matrix(x, self._get_all_centroids, self._kernel)
    
    def centroid_kernel_matrix(self, X, centroid):
        kernel_matrix = [self._kernel(centroid, x_i) for x_i in X]
        return torch.stack(kernel_matrix)

    def centroid_target_alignment(self, X, Y, centroid, l=0.1, assume_normalized_kernel=False, rescale_class_labels=True):
        K = self.centroid_kernel_matrix(X, centroid)
        min_size = min(K.size(0), Y.size(0))
        K = K[:min_size]
        Y = Y[:min_size]
        numerator = l * torch.sum(Y * K)  
        denominator = torch.sqrt(torch.sum(K ** 2) * torch.sum(Y ** 2))
        TA = numerator / denominator
        return TA.requires_grad_()
    
    def _loss_kao(self, X, Y, centroid):
        TA = self.centroid_target_alignment(X, Y, centroid)
        r = self.lambda_kao * sum((param ** 2).sum() for param in self._kernel.parameters())
        return 1 - TA + r

    def loss_co(self, X, Y, centroid, cl):
        TA = self.centroid_target_alignment(X, Y, centroid)
        cl_tensor = torch.tensor(float(cl), dtype=torch.float32)
        regularization_term = torch.sum(torch.clamp(cl_tensor - 1.0, min=0.0) - torch.clamp(cl_tensor, max=0.0))
        return 1 - TA + self.lambda_co * regularization_term

    def _loss_ta(self, data, data_labels):
        return qml.kernels.target_alignment(data, data_labels, self._kernel, assume_normalized_kernel=True)

    def _sampler_random_sampling(self, data, data_labels):
        subset_indices = torch.randperm(len(data))[:self._sampling_size]
        return data[subset_indices], data_labels[subset_indices]

    def _full_data(self, data, data_labels):
        return data, data_labels
    
    def fit_kernel(self, training_data, training_labels):
        optimizer = self._kernel_optimizer
        epochs = self._epochs
        loss_func = self._loss_function
        samples_func = self._sample_function
        self._kernel._circuit_executions = 0
        for epoch in range(epochs):
            optimizer.zero_grad()

            sampled_data, sampled_labels = samples_func(training_data, training_labels)
            if self._method == 'ccka':
                _class = epoch % len(self._n_classes)
                class_centroids = self._class_centroids[_class]
                class_labels = torch.tensor(self._class_centroid_labels[_class], dtype=torch.int)
                
                # Kao loss
                loss_kao = -self._loss_kao(class_centroids, class_labels, self._main_centroids[_class])
                loss_kao.backward(retain_graph=True)
                optimizer.step() 
                
                # Co loss
                self._centroid_optimizer.zero_grad()
                loss_co = -self.loss_co(class_centroids, class_labels, self._main_centroids[_class], _class + 1)
                loss_co.backward(retain_graph=True)
                self._centroid_optimizer.step()

                print(f"Epoch {epoch + 1}th, Kernel Loss: {loss_kao} and Centroid Loss: {loss_co}" )
                
                if self._validate_every_epoch:
                    validation_accuracy = self.evaluate(training_data, training_labels)['accuracy']
                    self.validation_accuracy_arr.append(validation_accuracy)
                    print(f"Validation Accuracy at Epoch {epoch + 1}: {validation_accuracy}")

                if self._get_alignment_every and (epoch + 1) % self._get_alignment_every == 0:
                    current_alignment = qml.kernels.target_alignment(training_data, training_labels, self._kernel, assume_normalized_kernel=True)
                    self.alignment_arr.append(current_alignment.item())
                    print(f"Epoch {epoch + 1}th, Alignment : {current_alignment}")
            
            else:
                loss = -loss_func(sampled_data, sampled_labels)
                loss.backward()
                optimizer.step()

                # Store and print loss values
                self._loss_arr.append(loss.item())
                
                if self._validate_every_epoch:
                    validation_accuracy = self.evaluate(training_data, training_labels)['accuracy']
                    self.validation_accuracy_arr.append(validation_accuracy)
                    print(f"Validation Accuracy at Epoch {epoch + 1}: {validation_accuracy}")

                if self._get_alignment_every and (epoch + 1) % self._get_alignment_every == 0:
                    current_alignment = qml.kernels.target_alignment(training_data, training_labels, self._kernel, assume_normalized_kernel=True)
                    self.alignment_arr.append(current_alignment.item())
                    print(f"Epoch {epoch + 1}th, Alignment : {current_alignment}")

            if self._target_accuracy:
                validation_accuracy = self.evaluate(training_data, training_labels)['accuracy']
                if validation_accuracy >= self._target_accuracy:
                    print(f"Target accuracy of {self._target_accuracy} achieved at Epoch {epoch + 1}. Training stopped.")
                    break
        
        self._executions = self._kernel._circuit_executions
        # Store final training accuracy
        self.final_training_accuracy = accuracy_score(self._training_labels, self._model.predict(self._kernel_matrix(self._training_data, self._training_data).detach().numpy()))

    def evaluate(self, test_data, test_labels):
        current_alignment = qml.kernels.target_alignment(self._training_data, self._training_labels, self._kernel, assume_normalized_kernel=True)
        _matrix = self._kernel_matrix(self._training_data, self._training_data)
        if torch.is_tensor(_matrix):
            _matrix = _matrix.detach().numpy()
        if torch.is_tensor(self._training_labels):
            self._training_labels = self._training_labels.detach().numpy()

        self._model = SVC(kernel='precomputed').fit(_matrix, self._training_labels)
        training_accuracy = accuracy_score(test_labels, predictions)


        _matrix = self._kernel_matrix(test_data, self._training_data)
        if torch.is_tensor(_matrix):
            _matrix = _matrix.detach().numpy()
        if torch.is_tensor(test_labels):
            test_labels = test_labels.detach().numpy()
        predictions = self._model.predict(_matrix)
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average='weighted')
        auc = roc_auc_score(test_labels, predictions)
        print(f"Testing Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"AUC: {auc}")

        return {
            'alignment': current_alignment,
            'executions': self._executions,
            'training_accuracy': training_accuracy,
            'testing_accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'alignment_arr': self.alignment_arr,
            'loss_arr': self._loss_arr,
            'validation_accuracy_arr': self.validation_accuracy_arr
        }
