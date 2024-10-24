import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.ansatz import qkhe, qkcovariant, qkra
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import json
import os

torch.manual_seed(42)
np.random.seed(42)

class qnn(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self._device = config['qkernel']['device']
        self._n_qubits = config['qkernel']['n_qubits']
        self._trainable = config['qkernel']['trainable']
        self._input_scaling = config['qkernel']['input_scaling']
        self._data_reuploading = config['qkernel']['data_reuploading']
        self._ansatz = config['qkernel']['ansatz']
        self._layers = config['qkernel']['ansatz_layers']

        if self._ansatz == 'he':
            if self._input_scaling:
                self.register_parameter(name="input_scaling", param= nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=True))
            else:
                self.register_parameter(name="input_scaling", param= nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=True))

            self.register_parameter(name="variational", param= nn.Parameter(torch.ones(self._layers, self._n_qubits * 2) * 2 * torch.pi, requires_grad=True))

        if self._ansatz == 'covariant':
            if self._input_scaling:
                self.register_parameter(name="input_scaling", param= nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=True))
            else:
                self.register_parameter(name="input_scaling", param= nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=True))

            self.register_parameter(name="variational", param= nn.Parameter(torch.ones(self._layers, self._n_qubits) * 2 * torch.pi, requires_grad=True))
            self.register_parameter(name="rotational", param= nn.Parameter(torch.ones(self._layers, self._n_qubits) * 2 * torch.pi, requires_grad=True))

        dev = qml.device(self._device, wires = range(self._n_qubits))
        if self._ansatz == 'he':
            self._kernel = qml.QNode(qkhe, dev, diff_method='adjoint', interface='torch')
        if self._ansatz == 'qkra':
            self._kernel = qml.QNode(qkra, dev, diff_method='adjoint', interface='torch')
        if self._ansatz == 'qkcovariant':
            self._kernel = qml.QNode(qkhe, dev, diff_method='adjoint', interface='torch')
        
    def forward(self, x1, x2):
        pass
    
    def train(self, dataloader, optimizer, criterion, epochs, save_model_path = 'models/', test_dataloader=None, metrics_file="result/qnn_metrics.json"):
        metrics = {
            "accuracy_per_epoch": [],
            "loss_per_epoch": [],
            "eval_metrics_per_epoch": []  # For evaluation metrics like precision, recall, etc.
        }

        # Ensure the directory exists
        metrics_dir = os.path.dirname(metrics_file)
        if metrics_dir and not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        for epoch in range(epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for inputs, labels in dataloader:
                optimizer.zero_grad()

                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            epoch_loss = running_loss / len(dataloader)
            epoch_accuracy = correct_predictions / total_samples

            # Print the average loss and accuracy for the epoch
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            # Store training metrics
            metrics["loss_per_epoch"].append(epoch_loss)
            metrics["accuracy_per_epoch"].append(epoch_accuracy)
     
        # Evaluate model after each epoch if test_dataloader is provided
        if test_dataloader:
            eval_metrics = self.evaluate_model(test_dataloader, num_classes=self._n_classes)
            metrics["eval_metrics_per_epoch"].append(eval_metrics)

        # Save metrics to JSON file after each epoch
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        # Save the model
        torch.save(self.state_dict(), save_model_path)
        print(f"Model saved to {save_model_path}")

    def evaluate_model(self, dataloader, num_classes):
        # Set the model to evaluation mode
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs, dim=1)
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_preds)

        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_preds)
        else:
            auc = None

        eval_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
            "confusion_matrix": conf_matrix.tolist()  # Convert to list for JSON serialization
        }

        # Print evaluation metrics
        print(f"Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        if auc is not None:
            print(f"AUC Score: {auc:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)

        # Return evaluation metrics for storing in the JSON file
        return eval_metrics
    
class hybrid(nn.Module):
    pass

class classical(nn.Module):
    pass

class qkernel(nn.Module):
    pass

