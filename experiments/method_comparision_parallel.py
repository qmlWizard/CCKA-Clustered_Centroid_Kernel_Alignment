from ccka.models.kernel import KernelModel
from ccka.circuits.angleEmbeddingKernel import quackEmbeddingCircuit
from ccka.aligner.kta import fullKTA, centroidBasedKTA, quackKTA, randomKTA, greedyKTA
import pennylane as qml
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import os
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Run KTA method comparison experiments.')
parser.add_argument('--dataset', type=str, default='corners',
                    help='Dataset to use: corners, checkerboard, or donuts')

dataset = parser.parse_args().dataset

def load_data(dataset_path):
    data = jnp.load(dataset_path, allow_pickle=True).item()
    X = jnp.asarray(data['x_train'])
    y = jnp.asarray(data['y_train'])
    x_test = jnp.asarray(data['x_test'])
    y_test = jnp.asarray(data['y_test'])

    X_combined = jnp.concatenate([X, x_test], axis=0)
    y_combined = jnp.concatenate([y, y_test], axis=0)

    return X_combined, y_combined

def run_experiment(method, dataset, dataset_path, centroids, num_iterations):
    
    X, y = load_data(dataset_path)

    kernel = quackEmbeddingCircuit(
                                num_qubits = 5,
                                reps = 6,
                                reupload = True
                        )
    init_weights = kernel.init_weights()
    model = KernelModel(circuit = kernel)

    if method == 'fullKTA':
        aligner = fullKTA(
            kernel_model= model,
            data = X,
            labels = y,
            matrix_type='regular',
            split_size=0.5,
            learning_rate=0.1,
            optimizer= 'adam',
            epochs=num_iterations
        )
    
    elif method == 'centroidBasedKTA':
        #num_iterations = int(num_iterations / 10)
        aligner = centroidBasedKTA(
                    kernel_model= model,
                    data = X,
                    labels = y,
                    matrix_type='regular',
                    clustering='regular',
                    split_size=0.50,
                    centroids= centroids,
                    learning_rate=0.2,
                    centroid_lr=0.01,
                    sub_centroid_lr=0.01,
                    lambda_co=0.001,
                    lambda_kao=0.001,
                    epochs=num_iterations,
                    eps=0.001,
                    alpha=0.01
        )
    
    elif method == 'greedyKTA':
        aligner = greedyKTA(
                    kernel_model= model,
                    data = X,
                    labels = y,
                    matrix_type='regular',
                    split_size=0.5,
                    greedy_samples= centroids,
                    landmark_points= 10,
                    learning_rate=0.1,
                    optimizer= 'adam',
                    epochs=num_iterations
        )
    
    elif method == 'randomKTA':
        aligner = randomKTA(
                    kernel_model= model,
                    data = X,
                    labels = y,
                    matrix_type='regular',
                    split_size=0.5,
                    random_samples= centroids,
                    landmark_points=10,
                    learning_rate=0.1,
                    optimizer= 'adam',
                    epochs=num_iterations
        )
    elif method == 'quackKTA':
        aligner = quackKTA(
                    kernel_model= model,
                    data = X,
                    labels = y,
                    matrix_type='regular',
                    clustering='regular',
                    split_size=0.50,
                    centroids= centroids,
                    lambda_co=0.001,
                    lambda_kao=0.001,
                    epochs=num_iterations,
                    eps=0.001,
                    alpha=0.01
        )

    history = aligner.align()

    if method == 'centroidBasedKTA':
        num_iterations = num_iterations * 10

    result = {
        'method': method,
        'dataset': dataset,
        'centroids': centroids,
        'num_iterations': num_iterations,
        'initial_training_accuracy': history['init_train_accuracy'],
        'final_training_accuracy': history['train_accuracy_history'][-1],
        'initial_testing_accuracy': history['init_test_accuracy'],
        'final_testing_accuracy': history['test_accuracy_history'][-1],
        'f1_score': history['f1_score_history'][-1],
        'precision': history['precision_score_history'][-1],
        'recall': history['recall_score_history'][-1],
        'training_time': history['time'],
        'circuit_executions': history['circuit_executions'],
        'initial_kernel_alignment': history['alignment_history'][0],
        'final_kernel_alignment': history['alignment_history'][-1]
    }

    return result

_CENTROID_METHODS = ['centroidBasedKTA', 'randomKTA', 'greedyKTA', 'quackKTA']

datasets = ['corners', 'checkerboard', 'donuts']
dataset_paths = {
    'corners': '../data/corners.npy',
    'checkerboard': '../data/checkerboard_dataset.npy',
    'donuts': '../data/donuts.npy'
}
methods = ['fullKTA', 'randomKTA', 'quackKTA', 'centroidBasedKTA', 'greedyKTA']
centroid_values = [2, 4, 6, 8, 10]
num_iterations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]

# Run experiments
final_results = []

from tqdm import tqdm

total_tasks = 0

# pre-compute total number of runs

for num_iteration in num_iterations:
    for method in methods:
        if method in _CENTROID_METHODS:
            total_tasks += len(centroid_values)
        else:
            total_tasks += 1

pbar = tqdm(total=total_tasks, desc="Running Experiments")

final_results = []


for num_iteration in num_iterations:
    for method in methods:
        if method in _CENTROID_METHODS:
            for centroids in centroid_values:
                result = run_experiment(
                    method,
                    dataset,
                    dataset_paths[dataset],
                    centroids,
                    num_iteration
                )
                final_results.append(result)
                pbar.update(1)
        else:
            result = run_experiment(
                method,
                dataset,
                dataset_paths[dataset],
                centroids=None,
                num_iterations=num_iteration
            )
            final_results.append(result)
            pbar.update(1)

pbar.close()

df = pd.DataFrame(final_results)
df.to_csv(f'{dataset}_method_comparison_results.csv', index=False)
