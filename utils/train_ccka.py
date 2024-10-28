import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.metrics import hinge_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
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
                 train_method,
                 sampling_size=4,
                 clusters=2):
        super().__init__()

        self._kernel = kernel
        self._optimizer = optimizer
        self._method = train_method
        self._epochs = 100
        self._sampling_size = sampling_size
        self._clusters = clusters
        self._training_data = training_data
        self._training_labels = training_labels
        self._n_classes = torch.unique(training_labels)

        self._main_centroids = []
        self._main_centroids_labels = []
        self._class_centroids = []
        self._class_centroid_labels = []

        self._get_centroids(self._training_data, self._training_labels)
        self._loss_function = self.loss_kao
        self._centroid_loss_function = self.loss_co
        self._loss_arr = []
        self.alignment_arr = []
        
        self._centroid_optimizer = optim.Adam([self._get_all_centroids], lr = 0.01)        
        _matrix = self._centered_kernel_matrix(self._training_data)
        if torch.is_tensor(_matrix):
            _matrix = _matrix.detach().numpy()
        if torch.is_tensor(self._training_labels):
            self._training_labels = self._training_labels.detach().numpy()
        self._model = SVC(kernel='linear').fit(_matrix, self._training_labels)

    @property
    def _get_all_centroids(self):
        centroids = self._main_centroids + [centroid for cluster in self._class_centroids for centroid in cluster]
        centroids = np.array(centroids)  # Convert to NumPy array if not already
        return torch.tensor(centroids, dtype=torch.float32, requires_grad=True)
 
    
    @property
    def _get_all_centroid_labels(self):
        # Combine labels
        labels = self._main_centroids_labels + [label for cluster in self._class_centroid_labels for label in cluster]
        labels = np.array(labels)
        return torch.tensor(labels, dtype=torch.int)

    def _get_centroids(self, data, data_labels):
        for c in self._n_classes:
            class_data = data[np.where(data_labels == c)[0]]

            # Ensure class_data is a NumPy array
            if isinstance(class_data, torch.Tensor):
                class_data = class_data.detach().numpy()

            # Calculate centroids and convert back to torch tensor
            main_centroid = np.mean(class_data, axis=0)
            self._main_centroids.append(torch.from_numpy(main_centroid))
            self._main_centroids_labels.append(c)

            # Convert class_data to NumPy arrays if necessary
            if isinstance(class_data, torch.Tensor):
                class_data = class_data.detach().numpy()

            # Calculate centroids for clusters, and convert back to torch tensor
            class_centroids = [np.mean(cluster, axis=0) for cluster in np.array_split(class_data, self._clusters)]
            self._class_centroids.append([torch.from_numpy(centroid) for centroid in class_centroids])
            self._class_centroid_labels.extend([[c] * self._clusters])

    def _centered_kernel_matrix(self, x):
        return qml.kernels.kernel_matrix(x, self._get_all_centroids, self._kernel)
    
    def centroid_kernel_matrix(self, X, centroid):
    
        kernel_matrix = []

        for i in range(len(X)):
            kernel_matrix.append(self._kernel(centroid, X[i]))

        return torch.tensor(kernel_matrix, dtype=torch.float32, requires_grad= True)
    
    def centroid_target_alignment(self, X, Y, centroid, l = 0.1, assume_normalized_kernel=False, rescale_class_labels=True):
   
        K = self.centroid_kernel_matrix(X, centroid)
        Y = torch.tensor(Y, dtype=torch.float32, requires_grad= True)
        numerator = l * torch.sum(Y * K)  
        denominator = torch.sqrt(torch.sum(K**2) * torch.sum(Y**2))

        TA = numerator / denominator

        return TA
    
    def loss_kao(self, X, Y, centroid, lambda_kao = 0.01):
        TA = self.centroid_target_alignment(X, Y, centroid)
        params_numpy = []
        for param in self._kernel.parameters():
            # Convert the parameter to a NumPy array
            param_numpy = param.detach().cpu().numpy()
            params_numpy.append(param_numpy)
        r = lambda_kao * np.sum(np.array(param_numpy) ** 2)
        return 1 - TA + r

    def loss_co(self, X, Y, centroid, kernel, cl, lambda_kao = 0.01):
        TA = self.centroid_target_alignment(X, Y, centroid, kernel)
        r = torch.sum(torch.maximum(cl - 1, 0) - torch.minimum(cl, 0))
        return 1 - TA + r

    def _loss_svm(self):

        kernel_matrix = qml.kernels.kernel_matrix(self._get_all_centroids, self._get_all_centroids, self._kernel)
        if torch.is_tensor(kernel_matrix):
            kernel_matrix = kernel_matrix.detach().numpy()
        if torch.is_tensor(self._get_all_centroid_labels):
            test_labels = self._get_all_centroid_labels.detach().numpy()
        clf = SVC(kernel='precomputed')
        clf.fit(kernel_matrix, test_labels)
        svm_loss = hinge_loss(test_labels, clf.decision_function(kernel_matrix))

        return torch.tensor(svm_loss,  dtype=torch.float32, requires_grad= True)


    def fit_kernel(self, training_data, training_labels):
        
        optimizer = self._optimizer
        centroid_optimizer = self._centroid_optimizer
        epochs = self._epochs
        loss_func = self._loss_function
        centroid_loss_func = self._centroid_loss_function
        kao_class = 1

        for epoch in range(epochs):
            centroid_idx = kao_class - 1
            cost = -self._loss_function( torch.vstack([centroid for sublist in self._class_centroids for centroid in sublist]),  # Access current class clusters
                                         torch.tensor([label for sublist in self._class_centroid_labels for label in sublist]),  # Labels for the current class
                                         self._main_centroids[centroid_idx],   # Current centroid 
                                        )
    
            centroid_cost = lambda _centroid: -self._centroid_loss_function(
                                                                                self._class_centroids[centroid_idx],  # Access current class clusters
                                                                                self._class_centroid_labels[centroid_idx],  # Labels for the current class
                                                                                self._main_centroids[centroid_idx],        # Current centroid
                                                                                self._kernel,
                                                                                _centroid
                                                                           )

            kao_class = (kao_class % len(self._n_classes)) + 1
            kernel_loss = cost
            kernel_loss.backward()
            optimizer.step()

            centroid_loss = centroid_cost(self._main_centroids[centroid_idx])
            centroid_loss.backward()
            centroid_optimizer.step()
            

        # Update SVM model after training loop
        _matrix = self._centered_kernel_matrix(self._training_data)
        if torch.is_tensor(_matrix):
            _matrix = _matrix.detach().numpy()
        if torch.is_tensor(self._training_labels):
            self._training_labels = self._training_labels.detach().numpy()

        self._model = SVC(kernel='precomputed').fit(_matrix, self._training_labels)


    def evaluate(self, test_data, test_labels):


        matrix = self._centered_kernel_matrix(test_data)
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
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc
        }
