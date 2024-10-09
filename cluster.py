import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.kernel import initialize_kernel, kernel, get_circuit_executions
from utils.classification_data import linear_data, checkerboard_data, power_line_data, microgrid_data
from utils.sampling import approx_greedy_sampling
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles
from matplotlib.colors import ListedColormap
from sklearn.cluster import SpectralClustering, KMeans
from scipy.stats import mode


# Load data with increased sample size
#data = checkerboard_data(2)  # Increased from 10 to 100 samples
data = linear_data(2, 10)
print("Done..!")
print("Sample:\n", data.head(1))
print("Data Size:", data.shape)

# Extract features and target
features = np.asarray(data.drop(columns=['target']))
target = np.asarray(data['target'])
#target = target % 2
#target = 2 * target - 1


print("\nConfiguring Quantum Circuit")

n_qubits = features.shape[1]
layers = 5
ansatz = 'efficientsu2'

dev = qml.device("default.qubit", wires=n_qubits)
wires = dev.wires.tolist()

print("Number of Qubits:", n_qubits)
print("Number of Variational Layers:", layers)

# Initialize the quantum kernel
wires, shape = initialize_kernel(n_qubits, ansatz, layers)
param_shape = (2,) + shape
params = np.random.uniform(0, 2 * np.pi, size=param_shape, requires_grad=True)

print("Shape for params:", param_shape)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

print("\nDividing Testing and Training dataset")
print("Train Size:", len(x_train))
print("Test Size:", len(x_test))

# Define kernel functions
def kernel_function(x1, x2, _params):
    return kernel(x1, x2, _params)

f_kernel = lambda x1, x2: kernel_function(x1, x2, params)
get_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, f_kernel)


kernel_matrix = get_kernel_matrix(x_train, x_train)

print(kernel_matrix)

kmeans_clustering = KMeans(n_clusters=2, random_state=42)
cluster_labels_kmeans = kmeans_clustering.fit_predict(kernel_matrix)
 
# Get the centroids of the clusters
centroids = kmeans_clustering.cluster_centers_
centroids = centroids[:, :2]


# Store the majority class label for each centroid
centroid_labels = []

# Assign the majority class of each cluster back to the samples
for cluster in np.unique(cluster_labels_kmeans):
    mask = cluster_labels_kmeans == cluster
    if np.any(mask):  # Ensure there are elements in the cluster
        majority_class = mode(y_train[mask], axis=None).mode
        cluster_labels_kmeans[mask] = majority_class
        centroid_labels.append(majority_class)  # Store the majority class for the centroid

# Display centroids and their corresponding labels
for i, (centroid, label) in enumerate(zip(centroids, centroid_labels)):
    print(f"Centroid {i+1}: {centroid}, Majority Label: {label}")
  


print("Modified Cluster Labels: ", cluster_labels_kmeans)
print("Original y_train Labels: ", y_train)

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: X_train with two classes
ax[0].scatter(x_train[y_train == -1][:, 0], x_train[y_train == -1][:, 1], color='blue', label='Class 0')
ax[0].scatter(x_train[y_train == 1][:, 0], x_train[y_train == 1][:, 1], color='red', label='Class 1')
ax[0].set_title("X_train with 2 Classes")
ax[0].set_xlabel("Feature 1")
ax[0].set_ylabel("Feature 2")
ax[0].legend()

# Plot 2: KMeans clustering result
scatter = ax[1].scatter(x_train[:, 0], x_train[:, 1], c=cluster_labels_kmeans, cmap='viridis')
ax[1].set_title("KMeans with Class-Constrained Clusters")
ax[1].set_xlabel("Feature 1")
ax[1].set_ylabel("Feature 2")

# Add color bar for the KMeans clustering plot
fig.colorbar(scatter, ax=ax[1])

# Display both plots
plt.tight_layout()
plt.show()

current_alignment = qml.kernels.target_alignment(
            x_train,
            y_train,
            lambda x1, x2: kernel_function(x1, x2, params),
            assume_normalized_kernel=True,
        )
print(f"Initial Alignment = {current_alignment:.3f}")

opt = qml.GradientDescentOptimizer(0.2)

for i in range(500):

    cost = lambda _params: -qml.kernels.target_alignment(
                                                            x_train,
                                                            y_train,
                                                            lambda x1, x2: kernel_function(x1, x2, _params),
                                                            assume_normalized_kernel=True,
                                                        )
    
    params, curr_cost = opt.step_and_cost(cost, params)

    if (i + 1) % 50 == 0:
        current_alignment = qml.kernels.target_alignment(
            x_train,
            y_train,
            lambda x1, x2: kernel_function(x1, x2, params),
            assume_normalized_kernel=True,
        )
        print(f"Step {i+1} - Alignment = {current_alignment:.3f}")

        #kernel_matrix = get_kernel_matrix(x_train, x_train)


        #kmeans_clustering = KMeans(n_clusters=4, random_state=42)
        #cluster_labels_kmeans = kmeans_clustering.fit_predict(kernel_matrix)
        
        # Get the centroids of the clusters
        #centroids = kmeans_clustering.cluster_centers_
        #centroids = centroids[:, :2]


# Train the SVM with the optimized kernel
trained_kernel = lambda x1, x2: kernel_function(x1, x2, params)
trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)
svm_trained = SVC(kernel=trained_kernel_matrix).fit(x_train, y_train)

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