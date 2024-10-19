import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.kernel import initialize_kernel, kernel, get_circuit_executions
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
features, target = plot_and_save(2, 128)


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
    features, target, test_size=0.25, random_state=42
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
print(y_train)
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

# Initialize alpha
#alpha = np.zeros_like(y_train, dtype=float)
#alpha[svm_model.support_] = np.abs(svm_model.dual_coef_).flatten()

# Get initial kernel matrix
#kernel_matrix = get_kernel_matrix(x_train, x_train)

# Define loss function for training
def loss(_params, x_subset, y_subset, alpha_sub):

    f_kernel = lambda x1, x2: kernel_function(x1, x2, _params)
    get_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, f_kernel)
    k_sub = np.array(get_kernel_matrix(x_subset, x_subset))

    loss_value = np.sum(alpha_sub) - 0.5 * np.sum(
        np.outer(alpha_sub, alpha_sub) * np.outer(y_subset, y_subset) * k_sub
    )
    return loss_value

# Optimize the kernel parameters
opt = qml.AdamOptimizer(0.01)

lcost = []
alignments = []
lparams = []

current_alignment = qml.kernels.target_alignment(
            x_train,
            y_train,
            lambda x1, x2: kernel_function(x1, x2, params),
            assume_normalized_kernel=True,
        )
alignments.append(current_alignment)

for i in range(200):
    
    # Sample a subset
    #subset = approx_greedy_sampling(kernel_matrix, 4, y_train, False)
    #print(y_train[subset])
    #subset = np.random.choice(list(range(len(x_train))), 4)
    
    class_1_indices = np.where(y_train == 1)[0]
    class_2_indices = np.where(y_train == -1)[0]

    # Select half of the subset size from each class
    subset_size = 16 // 2  # Assuming you want an equal split

    # Randomly select indices from each class
    subset_class_1 = np.random.choice(class_1_indices, subset_size, replace=False)
    subset_class_2 = np.random.choice(class_2_indices, subset_size, replace=False)

    # Combine the indices to form the final subset
    subset = np.concatenate((subset_class_1, subset_class_2))
    
    print(subset)
    f_kernel = lambda x1, x2: kernel_function(x1, x2, params)
    f_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, f_kernel)
    svm = SVC(kernel=f_kernel_matrix, max_iter=10000).fit(x_train[subset], y_train[subset])
    alpha_sub = np.zeros_like(y_train[subset], dtype=float)
    alpha_sub[svm.support_] = np.abs(svm.dual_coef_).flatten()
    
    print(i, "SVM Trained")
    cost = lambda _params: loss(_params, x_train[subset], y_train[subset], alpha_sub)
    #cost = lambda _params: -qml.kernels.target_alignment(
    #                                                        x_train[subset],
    #                                                        y_train[subset],
    #                                                        lambda x1, x2: kernel_function(x1, x2, _params),
    #                                                        assume_normalized_kernel=True,
    #                                                    )
    

    #lcost.append(cost(params))
    params, curr_cost = opt.step_and_cost(cost, params)
    lcost.append(curr_cost)
    lparams.append(params)
    

    
    # Update the kernel matrix with the new parameters
    #km = get_kernel_matrix(x_train[subset], x_train[subset])
    #kernel_matrix[np.ix_(subset, subset)] = km
    #kernel_matrix[np.ix_(subset, subset)] += np.eye(len(subset)) * 1

    # Calculate and store alignment
    if (i + 1) % 10 == 0:
        current_alignment = qml.kernels.target_alignment(
            x_train,
            y_train,
            lambda x1, x2: kernel_function(x1, x2, params),
            assume_normalized_kernel=True,
        )
        alignments.append(current_alignment)
        print(f"Step {i+1} - Alignment = {current_alignment:.3f}")

    # Early stopping condition
    #if len(lcost) > 1 and abs(lcost[-1] - lcost[-2]) <= 1e-09:
    #    break

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

cost_data = {
    'lcost': [lcost],
    'alignment': [alignments],
    'params': [lparams]
}


model_data = {
    'initial_accuracy': [accuracy_quantum],
    'initial_accuracy_classical': [accuracy_classical],
    'trained_training_accuracy': [accuracy_train],
    'trained_testing_accuracy' : [accuracy_test],
    'initial_auc': [auc_quantum],
    'initial_auc_classical': [auc_classical],
    'auc_trained': [auc_trained],
    'circuit_executions': [get_circuit_executions()]
    
}


file = 'linear_hinge_effcientsu2_random.csv'

df = pd.DataFrame(model_data)
df.to_csv(file)

file = 'linear_hinge_effcientsu2_random.npy'
np.save(file, cost_data)

train_color_0 = '#1f77b4'  # Blue shade
train_color_1 = '#ff7f0e'  # Orange shade
x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Predict the decision function for each point in the grid
Z = svm_trained.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting the decision boundary and training/testing data
plt.figure(figsize=(5, 5))

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap([train_color_0, train_color_1]))

# Plot training data with solid circles
plt.scatter(x_train[y_train == -1][:, 0], x_train[y_train == -1][:, 1], color=train_color_0, label='Train Class -1', s=150, marker='o', alpha=1)
plt.scatter(x_train[y_train == 1][:, 0], x_train[y_train == 1][:, 1], color=train_color_1, label='Train Class 1', s=150, marker='o', alpha=1)

# Plot testing data with hollow circles
plt.scatter(x_test[y_test == -1][:, 0], x_test[y_test == -1][:, 1], edgecolor=train_color_0, facecolor='none', label='Test Class -1', s=100, marker='o', alpha=1)
plt.scatter(x_test[y_test == 1][:, 0], x_test[y_test == 1][:, 1], edgecolor=train_color_1, facecolor='none', label='Test Class 1', s=100, marker='o', alpha=1)

# Remove extra margins for a tighter layout
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Checkerboard Data')
plt.tight_layout()
plt.savefig('decesion_plot_linear_hinge_effcientsu2_random.png', dpi=800)
plt.show()