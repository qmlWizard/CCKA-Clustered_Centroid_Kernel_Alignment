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
    
    data_generator = DataGenerator(     
                                        dataset_name = config['name'], 
                                        n_samples = config['n_samples'], 
                                        noise = config['noise'], 
                                        num_sectors = config['num_sectors'], 
                                        points_per_sector = config['points_per_sector'], 
                                        grid_size = config['grid_size'], 
                                        sampling_radius = config['sampling_radius']
                                  )
    
    features, target = data_generator.generate_dataset()
    training_data, testing_data, training_labels, testing_labels = train_test_split(features, target, test_size=0.50, random_state=42)
    training_data = torch.tensor(training_data.to_numpy(), dtype=torch.float32, requires_grad=True)
    testing_data = torch.tensor(testing_data.to_numpy(), dtype=torch.float32, requires_grad=True)
    training_labels = torch.tensor(training_labels.to_numpy(), dtype=torch.int)
    testing_labels = torch.tensor(testing_labels.to_numpy(), dtype=torch.int)

    kernel = Qkernel(   
                        device = config['device'], 
                        n_qubits = config['n_qubits'], 
                        trainable = config['trainable'], 
                        input_scaling = config['input_scaling'], 
                        data_reuploading = config['data_reuploading'], 
                        ansatz = config['ansatz'], 
                        ansatz_layers = config['ansatz_layers']
                    )
    
    agent = TrainModel(
                        kernel=kernel,
                        training_data=training_data,
                        training_labels=training_labels,
                        testing_data=testing_data,
                        testing_labels=testing_labels,
                        optimizer=config['optimizer'],
                        lr=config['lr'],
                        epochs = config['epochs'],
                        train_method=config['train_method'],
                        target_accuracy=config['target_accuracy'],
                        get_alignment_every=config['get_alignment_every'],  
                        validate_every_epoch=config['validate_every_epoch'], 
                        base_path=config['base_path'],
                        lambda_kao=config['lambda_kao'],
                        lambda_co=config['lambda_co'],
                        clusters=config['clusters']
                      )

    intial_metrics = agent.evaluate(testing_data, testing_labels)
    agent.fit_kernel(training_data, training_labels)
    after_metrics = agent.evaluate(testing_data, testing_labels)

    results = {
        'inital_metrics': intial_metrics,
        'after_metrics': after_metrics,
    }

    tune.report(accuracy=after_metrics['accuracy'], final_data = results)

if __name__ == "__main__":

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    search_space = {

        'name':  tune.grid_search(config['dataset']['name']),
        'n_samples': config['dataset']['n_samples'],
        'noise': config['dataset']['noise'],
        'num_sectors': config['dataset']['num_sectors'],
        'points_per_sector': config['dataset']['points_per_sector'],
        'grid_size': config['dataset']['grid_size'],
        'sampling_radius': config['dataset']['sampling_radius'],
        'training_size': config['dataset']['training_size'],
        'testing_size': config['dataset']['testing_size'],
        'validation_size': config['dataset']['validation_size'],
        'device': config['qkernel']['device'],
        'n_qubits': config['qkernel']['n_qubits'],
        'trainable': config['qkernel']['trainable'],
        'input_scaling': config['qkernel']['input_scaling'],
        'data_reuploading': config['qkernel']['data_reuploading'],
        'ansatz': tune.grid_search(config['qkernel']['ansatz']),
        'ansatz_layers': config['qkernel']['ansatz_layers'],
        'optimizer': tune.grid_search(config['agent']['optimizer']),
        'lr': tune.grid_search(config['agent']['lr']),
        'epochs': config['agent']['epochs'],
        'train_method': tune.grid_search(config['agent']['train_method']),
        'target_accuracy': config['agent']['target_accuracy'],
        'get_alignment_every': config['agent']['get_alignment_every'],
        'validate_every_epoch': config['agent']['validate_every_epoch'],
        'base_path': config['agent']['base_path'],
        'lambda_kao': tune.grid_search(config['agent']['lambda_kao']),
        'lambda_co': tune.grid_search(config['agent']['lambda_co']),
        'clusters': tune.grid_search(config['agent']['clusters'])

    }

    analysis = tune.run(
        tune.with_parameters(train),
        config=search_space,
        resources_per_trial={"gpu": 1},
        storage_path=os.path.abspath("ray_results"),
        name="qkernel_training",
        metric="accuracy",
        mode="max"
    )

    # Get the best hyperparameters
    best_config = analysis.get_best_config(metric="accuracy", mode="max")
    print(f"Best hyperparameters found: {best_config}")