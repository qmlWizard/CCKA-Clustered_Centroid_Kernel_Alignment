import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.kernel import initialize_kernel, kernel
from utils.classification_data import plot_and_save
from utils.sampling import approx_greedy_sampling
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles
from matplotlib.colors import ListedColormap


# Load data with increased sample size
#data = checkerboard_data(2)  # Increased from 10 to 100 samples
features, target = plot_and_save('double_cake', 128)

print("\nConfiguring Quantum Circuit")

n_qubits = features.shape[1]
layers = 6
ansatz = 'tutorial_ansatz'

dev = qml.device("default.qubit", wires=n_qubits)
wires = dev.wires.tolist()

print("Number of Qubits:", n_qubits)
print("Number of Variational Layers:", layers)

# Initialize the quantum kernel
wires, shape = initialize_kernel(n_qubits, ansatz, layers)
param_shape = (2,) + shape
params = np.random.uniform(-np.pi, np.pi, size=param_shape, requires_grad=True)

print("Shape for params:", param_shape)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

print("\nDividing Testing and Training dataset")
print("Train Size:", len(x_train))
print("Test Size:", len(x_test))

f_kernel = lambda x1, x2: kernel(x1, x2, params)
get_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, f_kernel)

init_kta = qml.kernels.target_alignment(
                        x_train,
                        y_train,
                        lambda x1, x2: kernel(x1, x2, params),
                        assume_normalized_kernel=True,
                    )
print(f"Initial KTA: {init_kta:.4f}")

# Train the initial quantum SVM
svm_model = SVC(kernel=get_kernel_matrix, max_iter=10000).fit(x_train, y_train)

# Train the classical SVM
classical_model = SVC().fit(x_train, y_train)

# Make predictions
y_pred_quantum = svm_model.predict(x_test)
y_pred_classical = classical_model.predict(x_test)

# Calculate accuracy
accuracy_quantum = accuracy_score(y_test, y_pred_quantum)
accuracy_classical = accuracy_score(y_test, y_pred_classical)

print(f"\nQuantum SVM Accuracy: {accuracy_quantum:.4f}")
print(f"Classical SVM Accuracy: {accuracy_classical:.4f}")

# Calculate ROC curve and AUC
fpr_quantum, tpr_quantum, _ = roc_curve(y_test, y_pred_quantum)
fpr_classical, tpr_classical, _ = roc_curve(y_test, y_pred_classical)
auc_quantum = auc(fpr_quantum, tpr_quantum)
auc_classical = auc(fpr_classical, tpr_classical)

print(f"Quantum SVM AUC: {auc_quantum:.4f}")
print(f"Classical SVM AUC: {auc_classical:.4f}")

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

def plot_svm_decision_boundary(svm_qmodel, X_train, y_train, X_test, y_test, filename = 'svm_decesion_boundary.png'):
    # Create a mesh to plot the decision boundary
    h = .2  # step size in the mesh
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    print(np.c_[xx.ravel(), yy.ravel()])
    # Plot decision boundary
    Z = svm_qmodel.predict(np.c_[xx.ravel(), yy.ravel()])
    print(Z)
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


#plot_svm_decision_boundary(svm_model, x_train, y_train, x_test, y_test, '/results/init_decesion_boundary_swiss_roll_hinge_effcientsu2_clustering.png')


####Training Loop Sttart
classes = np.unique(y_train)
n_clusters = int(4)  # Ensure this is an integer

centroids = []
class_centroids = []
centroid_labels = []
main_centroid_labels = []

for c in classes:
    class_data = x_train[np.where(y_train == c)[0]]

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
params = params

alignments = []
executions = []
loss_per_epoch = []

kao_class = 1
n_classes = len(classes)
params_list = []
train_accuracy = 0
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
                        x_train,
                        y_train,
                        lambda x1, x2: kernel(x1, x2, params),
                        assume_normalized_kernel=True,
                    )
        alignments.append(current_alignment)
        params_list.append(params)

        message = f"\tLoss: {l}, Alignment = {current_alignment:.3f}"
        print_boxed_message(f"Epoch {i + 1}th:", message)
####Training Loop End

alignments = np.array(alignments)
max_alignment = np.argmax(alignments)
params = params_list[max_alignment]

# Train the SVM with the optimized kernel
trained_kernel = lambda x1, x2: kernel(x1, x2, params)
trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)
svm_trained = SVC(kernel=trained_kernel_matrix, max_iter=10000).fit(x_train, y_train)

# Make predictions with the trained model
y_pred_train = svm_trained.predict(x_train)

# Calculate accuracy for trained model
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"\nTrained Quantum SVM Accuracy Train: {accuracy_train:.4f}")

# Make predictions with the trained model
y_pred_trained = svm_trained.predict(x_test)

# Calculate accuracy for trained model
accuracy_test = accuracy_score(y_test, y_pred_trained)
print(f"\nTrained Quantum SVM Accuracy Test: {accuracy_test:.4f}")

# Calculate ROC curve and AUC for trained model
fpr_trained, tpr_trained, _ = roc_curve(y_test, y_pred_trained)
auc_trained = auc(fpr_trained, tpr_trained)
print(f"Trained Quantum SVM AUC: {auc_trained:.4f}")

cost_data = {
    'lcost': [loss_per_epoch],
    'alignment': [alignments],
    'params': [params_list],
    'executions': [executions]
}


model_data = {
    'initial_accuracy': [accuracy_quantum],
    'initial_accuracy_classical': [accuracy_classical],
    'trained_training_accuracy': [accuracy_train],
    'trained_testing_accuracy' : [accuracy_test],
    'initial_auc': [auc_quantum],
    'initial_auc_classical': [auc_classical],
    'auc_trained': [auc_trained],
    'initial_kta': [init_kta],
    'circuit_executions': [np.sum(np.array(executions))]
    
}


file = 'swiss_roll_effcientsu2_clustering.csv'

df = pd.DataFrame(model_data)
df.to_csv(file)

file = 'swiss_roll_effcientsu2_clustering.npy'
np.save(file, cost_data)

#plot_svm_decision_boundary(svm_trained, x_train, y_train, x_test, y_test, '/results/decesion_boundary_swiss_roll_hinge_effcientsu2_clustering.png')