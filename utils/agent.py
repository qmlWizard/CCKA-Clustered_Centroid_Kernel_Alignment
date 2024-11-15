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

        if optimizer == 'adam':
            # Define optimizer with centroids as parameters
            #self._kernel_optimizer = optim.Adam(self._kernel.parameters(), lr=self._lr)
            self._kernel_optimizer = optim.Adam([
            {'params': self._kernel.parameters(), 'lr': self._lr},
            {'params': self._main_centroids, 'lr': self._lr},
            {'params': self._class_centroids, 'lr': self._lr},
        ])
            #self._centroid_optimizer = optim.Adam(self._flattened_class_centroids, lr=self._lr)
        
        elif optimizer == 'gd':
            #self._kernel_optimizer = optim.SGD(self._kernel.parameters(), lr=self._lr)
            self._kernel_optimizer = optim.SGD([
            {'params': self._kernel.parameters(), 'lr': self._lr},
            {'params': self._main_centroids, 'lr': self._lr},
            {'params': self._class_centroids, 'lr': self._lr},
        ])
            #self._centroid_optimizer = optim.SGD(self._flattened_class_centroids, lr=self._lr)

        if self._method == 'random':
            self._loss_function = self._loss_ta
            self._sample_function = self._sampler_random_sampling
        elif self._method == 'full':
            self._loss_function = self._loss_ta
            self._sample_function = self._full_data
        elif self._method == 'ccka':
            self._loss_function = self._loss_kao
            self._centroid_loss_function = self.loss_co
            self._sample_function = self._get_centroids


    """
    def _get_centroids(self, data, data_labels):
        for c in self._n_classes:
            class_data = data[data_labels == c]
            main_centroid = torch.mean(class_data, dim=0)
            self._main_centroids.append(main_centroid.requires_grad_())
            self._main_centroids_labels.append(c)
            class_centroids = [torch.mean(cluster, dim=0) for cluster in torch.chunk(class_data, self._clusters)]
            self._class_centroids.append([centroid.requires_grad_() for centroid in class_centroids])
            self._class_centroid_labels.append([c] * self._clusters)
    """

    def _get_centroids(self, data, data_labels):
        # Initialize empty lists for main centroids and class centroids
        main_centroids = []
        class_centroids = []
        main_centroid_labels = []
        class_centroid_labels = []

        for c in self._n_classes:
            class_data = np.array(data[data_labels == c].detach())
            # Calculate the main centroid and add it to main_centroids
            main_centroid = np.mean(class_data, axis = 0)  # Shape [1, feature_dim]
            main_centroids.append(main_centroid.tolist())
            main_centroid_labels.append(c)
    
            # Calculate centroids for each cluster in the class and stack them into a single tensor
            class_centroids.append([np.mean(cluster.tolist(), axis=0).tolist() for cluster in np.array_split(class_data, self._clusters)])
            class_centroid_labels.append([c] * self._clusters)

        self._main_centroids = torch.tensor(main_centroids, requires_grad=True)
        self._main_centroids_labels = torch.tensor(main_centroid_labels)
        self._class_centroids = torch.tensor(class_centroids, requires_grad=True)
        self._class_centroid_labels = torch.tensor(class_centroid_labels)
    
    def centroid_kernel_matrix(self, X, centroid):
        kernel_matrix = [self._kernel(centroid, x_i) for x_i in X]
        return torch.stack(kernel_matrix)

    def centroid_target_alignment(self, K, Y, l=0.1):
        
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
        TA = self.centroid_target_alignment(X, Y)
        cl_tensor = torch.tensor(float(cl), dtype=torch.float32)
        regularization_term = torch.sum(torch.clamp(cl_tensor - 1.0, min=0.0) - torch.clamp(cl_tensor, max=0.0))
        return 1 - TA + self.lambda_co * regularization_term

    def _centroid_loss(self, K, Y, centroid, cl):
        
        TA = self.centroid_target_alignment(K, Y, centroid)
        r = self.lambda_kao * sum((param ** 2).sum() for param in self._kernel.parameters())
        kao_loss = 1 - TA + r

        cl_tensor = torch.tensor(float(cl), dtype=torch.float32)
        regularization_term = torch.sum(torch.clamp(cl_tensor - 1.0, min=0.0) - torch.clamp(cl_tensor, max=0.0))
        co_loss =  1 - TA + self.lambda_co * regularization_term

        return -kao_loss + co_loss
 
    def _loss_ta(self, K, y):
        
        """
        Implements the KTA as defined in https://pennylane.ai/qml/demos/tutorial_kernels_module/.

        Denominator is the square root of the trace of the kernel matrix squared times the number of training samples
        """
        # Ensure that K has the correct shape
        
        N = y.shape[0]
        assert K.shape == (N,N), "Shape of K must be (N,N)"

        yT = y.view(1,-1) #Transpose of y, shape (1,N)
        Ky = torch.matmul(K,y) # K*y, shape (N,)
        yTKy = torch.matmul(yT,Ky) #yT * Ky, shape (1,1) which is a scalar

        K2 = torch.matmul(K,K) #K^2, shape (N,N)
        trace_K2 = torch.trace(K2)

        result = yTKy / (torch.sqrt(trace_K2)* N)

        return result.squeeze()

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
        self._per_epoch_executions = 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            if self._method == 'ccka':
                
                _class = epoch % len(self._n_classes)
                class_centroids = torch.tensor(self._class_centroids[_class], requires_grad=True)
                class_labels = self._class_centroid_labels[_class]
                main_centroid = torch.tensor(self._main_centroids[_class], requires_grad=True)

                x_0 = main_centroid.repeat(self._clusters, 1)
                x_1 = class_centroids
                
                K = self._kernel(x_0, x_1).to(torch.float32)

                loss = self._centroid_loss(K = K, Y=class_labels, centroid=main_centroid, cl=_class + 1)
                loss = loss.mean()
                loss.backward()

                for param in self._kernel.parameters():
                    if param.grad is not None:
                        print(f"Kernel Gradient: {param.grad}")
                for param in [self._main_centroids[_class], K]:
                    print(f"Centroid Gradient: {param.grad}")
            
                optimizer.step()
                print(f"Epoch {epoch + 1}th, Kernel Loss: {loss}" )

                self._loss_arr.append(loss.item())
                self._per_epoch_executions += x_0.shape[0]
                """
                # Kao loss
                loss_kao = -self._loss_kao(class_centroids, class_labels, self._main_centroids[_class])
                loss_kao.backward(retain_graph=True)
                optimizer.step() 

                # Co loss
                self._centroid_optimizer.zero_grad()
                loss_co = -self.loss_co(class_centroids, class_labels, self._main_centroids[_class], _class + 1)
                loss_co.backward(retain_graph=True)
                self._centroid_optimizer.step()
                
                

                loss_kao, loss_co = self._centroid_loss(class_centroids, class_labels, self._main_centroids[_class], _class + 1)
                loss_kao.backward(retain_graph=True)
                optimizer.step() 
                
                loss_co.backward(retain_graph=True)
                self._centroid_optimizer.step()

                self._per_epoch_executions += self._kernel._circuit_executions
                print(self._per_epoch_executions)
                print(f"Epoch {epoch + 1}th, Kernel Loss: {loss}" )
                """

                if self._get_alignment_every and (epoch + 1) % self._get_alignment_every == 0:
                    x_0 = training_data.repeat(training_data.shape[0],1)
                    x_1 = training_data.repeat_interleave(training_data.shape[0], dim=0)

                    
                    K = self._kernel(x_0, x_1).to(torch.float32)
                    self._training_labels = torch.tensor(self._training_labels, dtype = torch.float32) 
                    current_alignment = self._loss_ta(K.reshape(self._training_data.shape[0],self._training_data.shape[0]), self._training_labels)
                    print(f"Epoch {epoch + 1}th, Alignment : {current_alignment}")
            
            else:
                sampled_data, sampled_labels = samples_func(training_data, training_labels)
                sampled_labels = sampled_labels.to(torch.float32)

                x_0 = sampled_data.repeat(sampled_data.shape[0],1)
                x_1 = sampled_data.repeat_interleave(sampled_data.shape[0], dim=0)

                K = self._kernel(x_0, x_1).to(torch.float32) 
                loss = -loss_func(K.reshape(sampled_data.shape[0],sampled_data.shape[0]), sampled_labels)
                loss.backward()
                optimizer.step()

                # Store and print loss values
                self._loss_arr.append(loss.item())
                self._per_epoch_executions += x_0.shape[0]
                print(self._per_epoch_executions)

                if self._get_alignment_every and (epoch + 1) % self._get_alignment_every == 0:
                    x_0 = training_data.repeat(training_data.shape[0],1)
                    x_1 = training_data.repeat_interleave(training_data.shape[0], dim=0)
                
                    K = self._kernel(x_0, x_1).to(torch.float32)
                    self._training_labels = torch.tensor(self._training_labels, dtype = torch.float32) 
                    current_alignment = loss_func(K.reshape(self._training_data.shape[0],self._training_data.shape[0]), self._training_labels)
                    print(f"Epoch {epoch + 1}th, Alignment : {current_alignment}")
        
        self._executions = self._kernel._circuit_executions
        self._kernel._circuit_executions = 0
     
        #self.final_training_accuracy = accuracy_score(self._training_labels, self._model.predict(self._kernel_matrix(self._training_data, self._training_data).detach().numpy()))

    def evaluate(self, test_data, test_labels):
        
        ##Training Accuracy
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

        ##Testing Accuracy
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
        auc = roc_auc_score(test_labels, predictions)
        print(f"Testing Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"AUC: {auc}")


        return {
            'alignment': current_alignment,
            'executions': self._per_epoch_executions,
            'training_accuracy': training_accuracy,
            'testing_accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'alignment_arr': self.alignment_arr,
            'loss_arr': self._loss_arr,
            'validation_accuracy_arr': self.validation_accuracy_arr
        }
