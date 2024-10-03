import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.kernel import initialize_kernel, kernel, get_circuit_executions
from utils.classification_data import linear_data
from utils.sampling import approx_greedy_sampling
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load data with increased sample size
data = linear_data(3, 100)  # Increased from 10 to 100 samples
print("Done..!")
print("Sample:\n", data.head(1))
print("Data Size:", data.shape)

# Extract features and target
features = np.asarray(data.drop(columns=['target']))
target = np.asarray(data['target'])

# Scale the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

print("\nConfiguring Quantum Circuit")

n_qubits = features.shape[1]
layers = 2
ansatz = 'basic_entangled'

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

# Train the initial quantum SVM
svm_model = SVC(kernel=get_kernel_matrix).fit(x_train, y_train)

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

# Store initial results
results = {
    'Model': ['Initial Quantum SVM', 'Classical SVM'],
    'Accuracy': [accuracy_quantum, accuracy_classical],
    'AUC': [auc_quantum, auc_classical]
}

results_df = pd.DataFrame(results)

# Initialize alpha
alpha = np.zeros_like(y_train, dtype=float)
alpha[svm_model.support_] = np.abs(svm_model.dual_coef_).flatten()

# Get initial kernel matrix
kernel_matrix = get_kernel_matrix(x_train, x_train)

# Define loss function for training
def loss(_params, x_subset, y_subset, a):
    f_kernel = lambda x1, x2: kernel_function(x1, x2, _params)
    get_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, f_kernel)
    k_sub = get_kernel_matrix(x_subset, x_subset)
    loss_value = np.sum(a[subset]) - 0.5 * np.sum(
        np.outer(a[subset], a[subset]) * np.outer(y_subset, y_subset) * k_sub
    )
    return loss_value

# Optimize the kernel parameters
opt = qml.AdamOptimizer(0.05)

lcost = []
alignments = []
for i in range(200):
    # Sample a subset
    subset = approx_greedy_sampling(kernel_matrix, 10)
    cost = lambda _params: -loss(_params, x_train[subset], y_train[subset], alpha)

    lcost.append(cost(params))
    params = opt.step(cost, params)

    # Update the kernel matrix with the new parameters
    km = get_kernel_matrix(x_train[subset], x_train[subset])
    kernel_matrix[np.ix_(subset, subset)] = km
    kernel_matrix[np.ix_(subset, subset)] += np.eye(len(subset)) * 1e-3

    # Update the SVM model and alpha with the new kernel
    f_kernel = lambda x1, x2: kernel_function(x1, x2, params)
    get_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, f_kernel)
    svm_model = SVC(kernel=get_kernel_matrix).fit(x_train, y_train)
    alpha = np.zeros_like(y_train, dtype=float)
    alpha[svm_model.support_] = np.abs(svm_model.dual_coef_).flatten()

    # Calculate and store alignment
    if (i + 1) % 50 == 0:
        current_alignment = qml.kernels.target_alignment(
            x_train,
            y_train,
            lambda x1, x2: kernel_function(x1, x2, params),
            assume_normalized_kernel=True,
        )
        alignments.append({'Step': i + 1, 'Alignment': current_alignment})
        print(f"Step {i+1} - Alignment = {current_alignment:.3f}")

    # Early stopping condition
    if len(lcost) > 1 and abs(lcost[-1] - lcost[-2]) <= 1e-07:
        break

# Train the SVM with the optimized kernel
trained_kernel = lambda x1, x2: kernel_function(x1, x2, params)
trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)
svm_trained = SVC(kernel=trained_kernel_matrix).fit(x_train, y_train)

# Make predictions with the trained model
y_pred_trained = svm_trained.predict(x_test)

# Calculate accuracy for trained model
accuracy_trained = accuracy_score(y_test, y_pred_trained)
print(f"\nTrained Quantum SVM Accuracy: {accuracy_trained:.4f}")

# Calculate ROC curve and AUC for trained model
fpr_trained, tpr_trained, _ = roc_curve(y_test, y_pred_trained)
auc_trained = auc(fpr_trained, tpr_trained)
print(f"Trained Quantum SVM AUC: {auc_trained:.4f}")

# Append trained model results
results_df = results_df.append({
    'Model': 'Trained Quantum SVM',
    'Accuracy': accuracy_trained,
    'AUC': auc_trained
}, ignore_index=True)

# Print summary of accuracies and AUCs
print("\nSummary of Accuracies and AUCs:")
print(results_df)

# Save results to CSV
results_df.to_csv('svm_results.csv', index=False)

# Plot decision boundaries (for 2D data)
def plot_decision_boundary(model, X, y, title):
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .02  # Step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Prepare grid for prediction
    if X.shape[1] > 2:
        # Fix additional features at mean values
        mean_vals = X[:, 2:].mean(axis=0)
        grid = np.c_[xx.ravel(), yy.ravel(), np.tile(mean_vals, (xx.ravel().shape[0], 1))]
    else:
        grid = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel2, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot decision boundary for trained quantum SVM
plot_decision_boundary(svm_trained, x_train, y_train, 'Decision Boundary - Trained Quantum SVM')

# Store model details and parameters
model_info = {
    'Model': ['Initial Quantum SVM', 'Classical SVM', 'Trained Quantum SVM'],
    'Support Vectors': [
        len(svm_model.support_), 
        len(classical_model.support_), 
        len(svm_trained.support_)
    ],
    'Parameters': [np.nan, np.nan, params.tolist()]  # Store parameters of the trained model
}

model_info_df = pd.DataFrame(model_info)
model_info_df.to_csv('model_info.csv', index=False)

# Save alignment values to CSV
alignment_df = pd.DataFrame(alignments)
alignment_df.to_csv('alignments.csv', index=False)
