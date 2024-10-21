# Step 1: Import necessary libraries
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from utils.classification_data import generate_dataset, plot_and_save
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.datasets import make_friedman1
import pandas as pd
import sys
import time
import os

dev = qml.device("default.qubit", wires=8, shots=None)
wires = dev.wires.tolist()

def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding ansatz"""
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        
        i += inc
        qml.RX(params[0, j], wires=[wire])
        qml.RY(params[1, j], wires=[wire])

    qml.broadcast(unitary=qml.CNOT, pattern="ring", wires=wires)

def ansatz(x, params, wires):
    """The embedding ansatz"""
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))


adjoint_ansatz = qml.adjoint(ansatz)


def random_params(num_wires, num_layers):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, (num_layers, 2, num_wires), requires_grad=True)

@qml.qnode(dev)
def kernel_circuit(x1, x2, params):
    global circuit_executions
    circuit_executions += 1
    ansatz(x1, params, wires=wires)
    adjoint_ansatz(x2, params, wires=wires)
    return qml.probs(wires=wires)

def kernel(x1, x2, params):
    return kernel_circuit(x1, x2, params)[0]

def centroid_kernel_matrix(X, centroid, ckernel):
    
    kernel_matrix = []

    for i in range(len(X)):
        kernel_matrix.append(ckernel(centroid, X[i]))

    return np.array(kernel_matrix)

def centroid_target_alignment(X, Y, centroid, kernel, l = 0.1, assume_normalized_kernel=False, rescale_class_labels=True):
   
    Y = np.asarray(Y)
    K = centroid_kernel_matrix(X, centroid, kernel)
    numerator = l * np.sum(Y * K)  
    denominator = np.sqrt(np.sum(K**2) * np.sum(Y**2))

    TA = numerator / denominator

    return TA

def loss_co(X, Y, centroid, kernel, cl, lambda_kao = 0.01):
    TA = centroid_target_alignment(X, Y, centroid, kernel)
    r = np.sum(np.maximum(cl - 1, 0) - np.minimum(cl, 0))
    return 1 - TA + r

def loss_kao(X, Y, centroid, kernel, params, lambda_kao = 0.01):
    TA = centroid_target_alignment(X, Y, centroid, kernel)
    r = lambda_kao * np.sum(params ** 2)
    return 1 - TA + r

def print_boxed_message(title, content):
    def format_item(item, i):
        if isinstance(item, np.ndarray):  # Check if the item is an array
            return f"Cluster {i+1}: {np.array2string(item, precision=2, floatmode='fixed')}"
        else:
            return f"Cluster {i+1}: {item}"  # If it's not an array, just print the item as-is
    
    # Ensure that we only calculate lengths for formatted strings (arrays and others)
    formatted_content = [format_item(item, i) for i, item in enumerate(content)]
    max_len = max(len(line) for line in formatted_content)
    box_width = max(len(title) + 4, max_len + 4)

    print(f"+{'-' * box_width}+")
    print(f"|  {title.center(box_width - 4)}  |")
    print(f"+{'-' * box_width}+")
    for line in formatted_content:
        print(f"|  {line.ljust(box_width - 4)}  |")
    print(f"+{'-' * box_width}+")

def plot_svm_decision_boundary(svm_model, X_train, y_train, X_test, y_test, filename = 'svm_decesion_boundary.png'):
    # Create a mesh to plot the decision boundary
    h = .2  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Plot decision boundary
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 8))
    
    # Custom colormap for decision boundary
    cmap_background = ListedColormap(['#a6cee3', '#fdbf6f'])
    plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.8)
    
    # Plot training data (filled circles)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=ListedColormap(['blue', 'orange']),
                edgecolor='k', marker='o', s=100, label='Train Data', alpha=0.9)
    
    # Plot testing data (hollow circles)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=ListedColormap(['blue', 'orange']),
                edgecolor='k', marker='o', s=100, facecolors='none', label='Test Data', alpha=0.9)
    
    # Labels and title
    plt.title('SVM Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='best')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    
    try:
        dataset = sys.argv[1]
        num_clusters = sys.argv[2]
    except:
        print("******************************************************")
        print("* python train.py <dataset> <number of clusters>     *")
        print("******************************************************")
        sys.exit()

    
    features, target = plot_and_save(dataset, 128, save_path=f'{dataset}_plot.png')

    print(target)
    print(" ")
    print(f"* Feature Shape: {features.shape}")
    print(f"* Labels Shape:  {target.shape}")
    print(" ")

    X, x_test, Y, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

    print(X[0])
    print(f"* Train Shape: {X.shape}")
    print(f"* Train Labels Shape:  {Y.shape}")
    print(f"* Test Shape: {x_test.shape}")
    print(f"* Test Labels Shape:  {y_test.shape}")
    print(" ")

    circuit_executions = 0
    init_params = random_params(num_wires=8, num_layers=6)
    kernel_value = kernel(X[0], X[1], init_params)
    print(f"*The kernel value between the first and second datapoint is {kernel_value:.3f}") 
    drawer = qml.draw_mpl(kernel_circuit)
    fig, ax = drawer(X[0], X[1], init_params)
    fig.savefig('circuit.png')
    init_kernel = lambda x1, x2: kernel(x1, x2, init_params)
    kta_init = qml.kernels.target_alignment(X, Y, init_kernel, assume_normalized_kernel=True)
    print(f"*The kernel-target alignment for {dataset} and random parameters is {kta_init:.3f}")
    print(" ")

    get_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, init_kernel)

    # Train the initial quantum SVM
    svm_model = SVC(kernel=get_kernel_matrix, max_iter=10000).fit(X, Y)

    # Train the classical SVM
    classical_model = SVC().fit(X, Y)

    # Make predictions
    y_pred_quantum = svm_model.predict(x_test)
    y_pred_classical = classical_model.predict(x_test)

    # Calculate accuracy
    accuracy_quantum = accuracy_score(y_test, y_pred_quantum)
    accuracy_classical = accuracy_score(y_test, y_pred_classical)

    print(f"\nQuantum SVM Accuracy: {accuracy_quantum:.4f}")
    print(f"Classical SVM Accuracy: {accuracy_classical:.4f}")    
        
    plot_svm_decision_boundary(svm_model, X, Y, x_test, y_test, filename= f'init_{dataset}_decesion_boundary_clustering_{num_clusters}.png')
        
    classes = np.unique(Y)
    n_clusters = int(num_clusters)  # Ensure this is an integer

    centroids = []
    class_centroids = []
    centroid_labels = []
    main_centroid_labels = []

    for c in classes:
        class_data = X[np.where(Y == c)[0]]

        if class_data.shape[0] < n_clusters:
            raise ValueError(f"Not enough data points for class {c} to form {n_clusters} clusters.")
        
        # Calculate the overall class centroid and store it
        main_centroid = np.mean(class_data, axis=0)
        centroids.append(main_centroid)
        main_centroid_labels.append(c)  # Assign the main centroid label

        # Split class data into clusters and compute centroids for each cluster
        clusters = np.array_split(class_data, n_clusters)
        class_centroids.append([np.mean(cluster, axis=0) for cluster in clusters])
        
        # Assign the class labels to the cluster centroids
        centroid_labels.extend([c] * n_clusters)

    # Print main centroids
    print_boxed_message("Main Centroids", centroids)

    # Print class centroids
    for i, class_centroid in enumerate(class_centroids):
        print_boxed_message(f"Class {classes[i]} Centroids", class_centroid)

    main_centroid = True
    opt = qml.GradientDescentOptimizer(0.2)
    circuit_executions = 0
    params = init_params

    alignments = []
    executions = []
    loss_per_epoch = []

    kao_class = 1
    n_classes = len(classes)
    params_list = []
    train_accuracy = 0
    i = 0

    for i in range(250):
        
        centroid_idx = kao_class - 1  # Index for the current class/centroid
        cost = lambda _params: loss_kao(
                np.vstack(class_centroids),  # Access current class clusters
                centroid_labels,  # Labels for the current class
                centroids[centroid_idx],        # Current centroid
                lambda x1, x2: kernel(x1, x2, params),
                _params
            )
        
        centroid_cost = lambda _centroid: -loss_co(
                np.vstack(class_centroids),  # Access current class clusters
                centroid_labels,  # Labels for the current class
                centroids[centroid_idx],        # Current centroid
                lambda x1, x2: kernel(x1, x2, params),
                _centroid
            )
        params, l = opt.step_and_cost(cost, params)
        centroids[centroid_idx] = opt.step(centroid_cost, centroids[centroid_idx])
        for sub_centroid_idx in range(len(class_centroids[centroid_idx])):
            class_centroids[centroid_idx][sub_centroid_idx] = opt.step(centroid_cost, class_centroids[centroid_idx][sub_centroid_idx])
        kao_class = (kao_class % n_classes) + 1
        loss_per_epoch.append(l)
        executions.append(circuit_executions)
        print(f'Epoch {i + 1}th, Loss: {l}')
        
        if (i + 1) % 25 == 0:
            current_alignment = qml.kernels.target_alignment(
                            X,
                            Y,
                            lambda x1, x2: kernel(x1, x2, params),
                            assume_normalized_kernel=True,
                        )
            alignments.append(current_alignment)
            params_list.append(params)

            message = f"\tLoss: {l}, Alignment = {current_alignment:.3f}"
            print_boxed_message(f"Epoch {i + 1}th:", message)
        trained_kernel = lambda x1, x2: kernel(x1, x2, params)

    alignments = np.array(alignments)
    max_alignment = np.argmax(alignments)
    params = params_list[max_alignment]

    trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)
    svm_trained = SVC(kernel=trained_kernel_matrix).fit(X, Y) 

    y_true_train = svm_trained.predict(X)
    y_true_test = svm_trained.predict(x_test)

    # Assuming Y (labels for training data) and y_test (labels for test data) are available
    train_accuracy = accuracy_score(Y, y_true_train)
    test_accuracy = accuracy_score(y_test, y_true_test)
    train_f1 = f1_score(Y, y_true_train, average='weighted')
    test_f1 = f1_score(y_test, y_true_test, average='weighted')
    train_conf_matrix = confusion_matrix(Y, y_true_train)
    test_conf_matrix = confusion_matrix(y_test, y_true_test)

    train_content = [
        f"Train Accuracy: {train_accuracy:.2f}",
        f"Train F1 Score: {train_f1:.2f}",

    ]

    test_content = [
        f"Test Accuracy: {test_accuracy:.2f}",
        f"Test F1 Score: {test_f1:.2f}",

    ]

    # Print the results in a box format
    print_boxed_message("Train Performance", train_content)
    print_boxed_message("Test Performance", test_content)

    obervations = {
        'classical_acc': [accuracy_classical],
        'init_svc_acc': [accuracy_quantum],
        'init_kta': [kta_init],
        'alignments': [alignments],
        'loss_per_epoch': [loss_per_epoch],
        'executions': [np.sum(np.array(executions))],
        'final_kta': [current_alignment],
        'train_acc': [train_accuracy],
        'train_f1': [train_f1], 
        'train_cm' : [train_conf_matrix],
        'test_acc': [test_accuracy],
        'test_f1' : [test_f1],
        'test_cm' : [test_conf_matrix]
    }

    np.save(f'{dataset}_observations_clustering_{num_clusters}.npy', obervations)

    plot_svm_decision_boundary(svm_trained, X, Y, x_test, y_test, filename= f'{dataset}_decesion_boundary_clustering_{num_clusters}.png')