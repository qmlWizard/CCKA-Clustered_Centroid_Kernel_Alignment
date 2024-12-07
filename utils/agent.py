import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import hinge_loss
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
import yaml
import time
import os
from utils.helper import to_python_native
from utils.plotter import Plotter

class TrainModel():
    def __init__(self, 
                 kernel,
                 training_data,
                 training_labels,
                 testing_data,
                 testing_labels,
                 optimizer,
                 lr,
                 mclr,
                 cclr,
                 epochs,
                 train_method,
                 target_accuracy=None,  # New parameter for target accuracy
                 get_alignment_every=10,  # New parameter to select alignment frequency
                 validate_every_epoch=10,  # New parameter for validation accuracy
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
        self._sampling_size = clusters 
        self._clusters = clusters 
        self._training_data = training_data
        self._training_labels = training_labels
        self._training_labels = self._training_labels.to(torch.float32)
        self._testing_data = testing_data
        self._testing_labels = testing_labels
        self._testing_labels = self._testing_labels.to(torch.float32)
        self._n_classes = torch.unique(training_labels)
        self._kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, self._kernel)
        self._executions = 0
        self._lr = lr
        self._mclr = mclr
        self._cclr = cclr
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
        self._per_epoch_executions = None
        if self._method in ['ccka', 'quack']:
            self._epochs = int(epochs / 10)
            
        if self._method in ['random', 'full']:
            if optimizer == 'adam':
                self._kernel_optimizer = optim.Adam(self._kernel.parameters(), lr = self._lr)
            elif optimizer == 'gd':
                self._kernel_optimizer = optim.SGD(self._kernel.parameters(), lr = self._lr)
        elif self._method == 'ccka':
            if optimizer == 'adam':
                #self._kernel_optimizer = optim.Adam(self._kernel.parameters(), lr = self._lr)
                self._kernel_optimizer = optim.Adam([ {'params': self._kernel.parameters(), 'lr': self._lr}, {'params': self._class_centroids, 'lr': self._cclr}])
                self._optimizers = []
                for tensor in self._main_centroids:
                    self._optimizers.append(optim.Adam([ {'params': tensor, 'lr': self._mclr}]))
            elif optimizer == 'gd':
                #self._kernel_optimizer = optim.Adam(self._kernel.parameters(), lr = self._lr)
                self._kernel_optimizer = optim.SGD([ {'params': self._kernel.parameters(), 'lr': self._lr}, {'params': self._class_centroids, 'lr': self._cclr}])
                self._optimizers = []
                for tensor in self._main_centroids:
                    self._optimizers.append(optim.SGD([ {'params': tensor, 'lr': self._mclr}]))
        elif self._method == 'quack':
            if optimizer == 'adam':
                self._kernel_optimizer = optim.Adam(self._kernel.parameters(), lr = self._lr)
                self._optimizers = []
                for tensor in self._main_centroids:
                    self._optimizers.append(optim.Adam([ {'params': tensor, 'lr': self._mclr}]))
            elif optimizer == 'gd':
                self._kernel_optimizer = optim.SGD(self._kernel.parameters(), lr = self._lr)
                self._optimizers = []
                for tensor in self._main_centroids:
                    self._optimizers.append(optim.SGD([ {'params': tensor, 'lr': self._mclr}]))
        if self._method == 'random':
            self._loss_function = self._loss_ta
            self._sample_function = self._sampler_random_sampling
        elif self._method == 'full':
            self._loss_function = self._loss_ta
            self._sample_function = self._full_data
        elif self._method == 'ccka':
            self._loss_function = self._loss_kao
            self._centroid_loss_function = self._loss_co
            self._sample_function = self._get_centroids
        elif self._method == 'quack':
            self._loss_function = self._loss_kao
            self._centroid_loss_function = self._loss_co
            self._sample_function = self._get_centroids
            
        print("Epochs: ", self._epochs)
    def _get_centroids(self, training_data, training_labels):
        data = training_data.detach().numpy()
        data_labels = training_labels.detach().numpy()
        _class_centroids = []
        _class_centroids_labels = []
        _main_centroids = []
        for c in [1, -1]:
            cdata = data[data_labels == c]
            mc = [np.mean(cdata, axis=0)]
            _main_centroids.append(torch.tensor(np.array(mc), requires_grad= True))
            sub_centroids = [np.mean(cluster, axis=0) for cluster in np.array_split(cdata, self._clusters)]
            sub_centroids_labels = [c] * self._clusters
            class_centroids = np.array(mc + sub_centroids)
            _class_centroids.append(torch.tensor(class_centroids, requires_grad=True))
            _class_centroids_labels.append(torch.tensor(np.array(sub_centroids_labels)))
        self._class_centroids = _class_centroids
        self._class_centroid_labels = _class_centroids_labels
        self._main_centroids = _main_centroids
        
    def centroid_target_alignment(self, K, Y, l):
        num = l * torch.sum(Y * K)
        den = torch.sqrt(torch.sum(K ** 2) * torch.sum(Y ** 2))
        result = num / den       
        return result.squeeze()
    
    def _loss_kao(self, K, Y, cl):
        TA = self.centroid_target_alignment(K, Y, cl)
        r = self.lambda_kao * sum((param ** 2).sum() for param in self._kernel.parameters())
        return 1 - TA + r
        
    def _loss_co(self, K, Y, centroid, cl):
        TA = self.centroid_target_alignment(K, Y, cl)
        regularization_term = 0
        for d in centroid:
            regularization_term += torch.amax(d - 1, 0) - torch.amin(d, 0)
        return 1 - TA + self.lambda_co * regularization_term
 
    def _loss_ta(self, K, y):
        N = y.shape[0]
        assert K.shape == (N,N), "Shape of K must be (N,N)"
        yT = y.view(1,-1) #Transpose of y, shape (1,N)
        Ky = torch.matmul(K,y) # K*y, shape (N,)
        yTKy = torch.matmul(yT,Ky) #yT * Ky, shape (1,1) which is a scalar
        K2 = torch.matmul(K,K) #K^2, shape (N,N)
        trace_K2 = torch.trace(K2)
        result = yTKy / (torch.sqrt(trace_K2)* N)
        return result.squeeze()
    
    def _loss_combined(self, K, Y, cluster_centroids, main_centroid, cl):
        # Compute target alignment
        TA = self.centroid_target_alignment(K, Y, cl)
        
        regularization_term = 0
        for d in main_centroid:
            regularization_term += torch.amax(d - 1, 0) - torch.amin(d, 0)

        rterm_cl = 0
        for cluster in cluster_centroids:
            for d in cluster:
                rterm_cl += torch.amax(d - 1, 0) - torch.amin(d, 0)

        # Combined loss
        return  1 - TA + self.lambda_co * regularization_term + self.lambda_co * rterm_cl

    def _loss_hinge(self, K, y):
        if not hasattr(self, 'alpha'):
            self.alpha = torch.nn.Parameter(torch.zeros(K.shape[0], requires_grad=True))
        f = K @ self.alpha
        hinge_loss = torch.clamp(1 - y * f, min=0)
        reg_term = self.lambda_kao * (self.alpha ** 2).sum()
        return hinge_loss.mean() + reg_term

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
        self._per_epoch_executions = 0
        self.kernel_params_history = []  # Array to store kernel parameters at each alignment update
        self.best_kernel_params = None
        for epoch in range(epochs):
            if self._method in ['ccka', 'quack']:
                for nkao in range(10):
                    _class = epoch % len(self._n_classes)
                    main_centroid = self._main_centroids[_class]
                    if self._method == 'ccka':
                        class_centroids = torch.cat([tensor[1: ] for tensor in self._class_centroids]) 
                        class_centroid_labels = torch.cat(self._class_centroid_labels) 
                    else:
                        class_centroids = training_data
                        class_centroid_labels = training_labels
                    #create interleave
                    x_0 = main_centroid.repeat(class_centroids.shape[0],1)
                    x_1 = class_centroids 
                    self._per_epoch_executions += x_0.shape[0]
                    K = self._kernel(x_0, x_1).to(torch.float32)
                    optimizer.zero_grad()
                    loss_kao = self._loss_kao(K, class_centroid_labels, class_centroid_labels[0])
                    loss_kao.backward()
                    optimizer.step()
                for nco in range(10):
                    _class = epoch % len(self._n_classes)
                    main_centroid = -self._main_centroids[_class]
                    if self._method == 'ccka':
                        class_centroids = torch.cat([tensor[1: ] for tensor in self._class_centroids]) 
                        class_centroid_labels = torch.cat(self._class_centroid_labels) 
                    else:
                        class_centroids = training_data
                        class_centroid_labels = training_labels 
                    x_0 = main_centroid.repeat(class_centroids.shape[0],1)
                    x_1 = class_centroids 
                    self._optimizers[_class].zero_grad()
                    K = self._kernel(x_0, x_1).to(torch.float32)
                    loss_co = self._loss_co(K, class_centroid_labels, main_centroid, class_centroid_labels[0])
                    #loss_co = self._loss_combined(K, class_centroid_labels, class_centroids, main_centroid, class_centroid_labels[0]).mean()
                    loss_co.backward()
                    self._optimizers[_class].step()              

                if self._get_alignment_every and (epoch + 1) % self._get_alignment_every == 0:
                    x_0 = training_data.repeat(training_data.shape[0], 1)
                    x_1 = training_data.repeat_interleave(training_data.shape[0], dim=0)
                    K = self._kernel(x_0, x_1).to(torch.float32)
                    if not isinstance(self._training_labels, torch.Tensor):
                        self._training_labels = torch.tensor(self._training_labels, dtype=torch.float32)
                    current_alignment = self._loss_ta(
                        K.reshape(self._training_data.shape[0], self._training_data.shape[0]), 
                        self._training_labels
                    )
                    self.alignment_arr.append(current_alignment)
                    print(current_alignment)
            else:
                sampled_data, sampled_labels = samples_func(training_data, training_labels)
                sampled_labels = sampled_labels.to(torch.float32)
                x_0 = sampled_data.repeat(sampled_data.shape[0],1)
                x_1 = sampled_data.repeat_interleave(sampled_data.shape[0], dim=0)
                optimizer.zero_grad()   
                K = self._kernel(x_0, x_1).to(torch.float32) 
                loss = -loss_func(K.reshape(sampled_data.shape[0],sampled_data.shape[0]), sampled_labels)
                loss.backward()
                optimizer.step()
                self._loss_arr.append(loss.item())
                self._per_epoch_executions += x_0.shape[0]
                if self._get_alignment_every and (epoch + 1) % self._get_alignment_every == 0:
                    x_0 = training_data.repeat(training_data.shape[0],1)
                    x_1 = training_data.repeat_interleave(training_data.shape[0], dim=0)
                    K = self._kernel(x_0, x_1).to(torch.float32)
                    self._training_labels = torch.tensor(self._training_labels, dtype = torch.float32) 
                    current_alignment = loss_func(K.reshape(self._training_data.shape[0],self._training_data.shape[0]), self._training_labels)
                    self.alignment_arr.append(current_alignment)
                    print(current_alignment)

    def prediction_stage(self, data, labels):

        main_centroids = torch.stack([centroid.detach()[0] for centroid in self._main_centroids])
        x_0 = main_centroids.repeat(data.shape[0],1)
        x_1 = data.repeat_interleave(main_centroids.shape[0], dim=0)
        K = self._kernel(x_0, x_1).to(torch.float32).reshape(data.shape[0],main_centroids.shape[0])
        pred_labels = torch.sign(K[:, 0] - K[:, 1]) 
        correct_predictions = (pred_labels == labels).sum().item()  # Count matches
        total_predictions = len(labels)  # Total number of predictions
        accuracy = correct_predictions / total_predictions

        # Display results
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Total Predictions: {total_predictions}")
        print(f"Accuracy: {accuracy * 100:.2f}%")

     
    def evaluate(self, test_data, test_labels):
        x_0 = self._training_data.repeat(self._training_data.shape[0],1)
        x_1 = self._training_data.repeat_interleave(self._training_data.shape[0], dim=0)    
        _matrix = self._kernel(x_0, x_1).to(torch.float32).reshape(self._training_data.shape[0],self._training_data.shape[0])
        current_alignment = self._loss_ta(_matrix, self._training_labels)
        if torch.is_tensor(_matrix):
            _matrix = _matrix.detach().numpy()
        if torch.is_tensor(self._training_labels):
            self._training_labels = self._training_labels.detach().numpy()
        self._model = SVC(kernel='precomputed').fit(_matrix, self._training_labels)
        predictions = self._model.predict(_matrix)
        training_accuracy = accuracy_score(self._training_labels, predictions)
        x_0 = self._testing_data.repeat_interleave(self._training_data.shape[0],dim=0)
        x_1 = self._training_data.repeat(test_data.shape[0], 1)
        _matrix = self._kernel(x_0, x_1).to(torch.float32).reshape(test_data.shape[0],self._training_data.shape[0])
        if torch.is_tensor(_matrix):
            _matrix = _matrix.detach().numpy()
        if torch.is_tensor(test_labels):
            test_labels = test_labels.detach().numpy()
        predictions = self._model.predict(_matrix)
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average='weighted')
        metrics = {
            'alignment': current_alignment,
            'executions': self._per_epoch_executions,
            'training_accuracy': training_accuracy,
            'testing_accuracy': accuracy,
            'f1_score': f1,
            'alignment_arr': self.alignment_arr,
            'loss_arr': self._loss_arr,
            'validation_accuracy_arr': self.validation_accuracy_arr
        }
        return metrics
