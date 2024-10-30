import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.metrics import hinge_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
import yaml
import time
import os


class train_ccka_model():

    def __init__(self, 
                 kernel,
                 training_data,
                 training_labels,
                 optimizer,
                 lr,
                 epochs,
                 train_method,
                 sampling_size=4,
                 clusters=4
                 ):
        super().__init__()

        self._kernel = kernel
        self._optimizer = optimizer
        self._method = train_method
        self._epochs = epochs
        self._sampling_size = sampling_size
        self._clusters = clusters
        self._training_data = training_data
        self._training_labels = training_labels
        self._n_classes = torch.unique(training_labels)
        self._kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, self._kernel)
        self._executions = 0
        self._lr = lr

        self._main_centroids = []
        self._main_centroids_labels = []
        self._class_centroids = []
        self._class_centroid_labels = []

        self._get_centroids(self._training_data, self._training_labels)
        self._loss_function = self._loss_kao
        self._centroid_loss_function = self.loss_co
        self._loss_arr = []
        self.alignment_arr = []

        # Flatten the list of class centroids to pass as parameters
        self._flattened_class_centroids = [centroid.clone().detach().requires_grad_() for cluster in self._class_centroids for centroid in cluster]

        if optimizer == 'adam':
        # Define optimizer with centroids as parameters
            self._kernel_optimizer = optim.Adam(self._kernel.parameters(), lr=self._lr)
            self._centroid_optimizer = optim.Adam(self._flattened_class_centroids, lr=self._lr)
        elif optimizer == 'gd':
            self._kernel_optimizer = optim.SGD(self._kernel.parameters(), lr=self._lr)
            self._centroid_optimizer = optim.SGD(self._flattened_class_centroids, lr=self._lr)

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
    
    def _loss_kao(self, X, Y, centroid, lambda_kao=0.01):
        TA = self.centroid_target_alignment(X, Y, centroid)
        r = lambda_kao * sum((param ** 2).sum() for param in self._kernel.parameters())
        return 1 - TA + r

    def loss_co(self, X, Y, centroid, cl, lambda_co=0.01):
        TA = self.centroid_target_alignment(X, Y, centroid)
        cl_tensor = torch.tensor(float(cl), dtype=torch.float32)
        regularization_term = torch.sum(torch.clamp(cl_tensor - 1.0, min=0.0) - torch.clamp(cl_tensor, max=0.0))
        return 1 - TA + lambda_co * regularization_term

    def fit_kernel(self, training_data, training_labels):
        self._kernel._circuit_executions = 0
        for epoch in range(self._epochs):
            _class = epoch % len(self._n_classes)
            class_centroids = self._class_centroids[_class]
            class_labels = torch.tensor(self._class_centroid_labels[_class], dtype=torch.int)
            
            # Kao loss
            loss_kao = -self._loss_kao(class_centroids, class_labels, self._main_centroids[_class])
            self._kernel_optimizer.zero_grad()
            loss_kao.backward(retain_graph=True)
            self._kernel_optimizer.step() 
            
            # Co loss
            self._centroid_optimizer.zero_grad()
            loss_co = -self.loss_co(class_centroids, class_labels, self._main_centroids[_class], _class + 1)
            loss_co.backward(retain_graph=True)
            self._centroid_optimizer.step()

            #print(f"Epoch {epoch + 1}th, Kernel Loss: {loss_kao} and Centroid Loss: {loss_co}" )
            if epoch % 50 == 0:
                current_alignment = qml.kernels.target_alignment(training_data, training_labels, self._kernel, assume_normalized_kernel=True)
                print(f"Epoch {epoch + 1}th, Alignment : {current_alignment}")

        
        self._executions = self._kernel._circuit_executions

    def evaluate(self, test_data, test_labels):
        _matrix = self._kernel_matrix(self._training_data, self._training_data)
        if torch.is_tensor(_matrix):
            _matrix = _matrix.detach().numpy()
        if torch.is_tensor(self._training_labels):
            self._training_labels = self._training_labels.detach().numpy()

        self._model = SVC(kernel='precomputed').fit(_matrix, self._training_labels)

        _matrix = self._kernel_matrix(test_data, self._training_data)
        if torch.is_tensor(_matrix):
            _matrix = _matrix.detach().numpy()
        if torch.is_tensor(test_labels):
            test_labels = test_labels.detach().numpy()
        predictions = self._model.predict(_matrix)
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average='weighted')
        #auc = roc_auc_score(test_labels, predictions)
        print(f"Testing Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        #print(f"AUC: {auc}")
        auc = 0
        return {
            'executions': self._executions,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc
        }
