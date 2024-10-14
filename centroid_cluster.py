from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from utils.classification_data import linear_data, checkerboard_data, power_line_data, microgrid_data, make_double_cake_data
from sklearn.model_selection import train_test_split
from pennylane import numpy as np
import pennylane as qml
from pennylane import numpy as np
import matplotlib as mpl
from pennylane import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from collections import deque
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

data = pd.read_csv('data/balanced_quantum_dataset.csv')
features = np.asarray(data.drop(columns=['label']))
target = np.asarray(data['label'])

X, x_test, Y, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)


circuit_executions = 0
def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding ansatz"""
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])

    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])

def ansatz(x, params, wires):
    """The embedding ansatz"""
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))


adjoint_ansatz = qml.adjoint(ansatz)


def random_params(num_wires, num_layers):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, (num_layers, 2, num_wires), requires_grad=True)

dev = qml.device("default.qubit", wires=5, shots=None)
wires = dev.wires.tolist()

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

def loss(X, Y, centroid, kernel, params, lambda_kao = 0.01):
    TA = centroid_target_alignment(X, Y, centroid, kernel)
    r = lambda_kao * np.sum(params ** 2)
    return 1 - TA + r

def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)

classes = np.unique(Y)
n_clusters = 8

centroids = []
class_centroids = []
centroid_labels = []
for c in classes:
    class_data = X[np.where(Y == c)[0]]
    centroids.append(np.mean(class_data, axis=0))
    class_centroids.append([np.mean(cluster, axis=0) for cluster in np.array_split(class_data, n_clusters)])
    centroid_labels.extend([[c] * n_clusters])


class1_centroids = class_centroids[0]
class2_centroids = class_centroids[1]

centroid1_labels = centroid_labels[0]
centroid2_labels = centroid_labels[1]

centroid1 = centroids[0]
centroid2 = centroids[1]

main_centroid = True
kao_class = 1
opt = qml.GradientDescentOptimizer(0.2)
circuit_executions = 0
params = random_params(num_wires=5, num_layers=6)
current_alignment = qml.kernels.target_alignment(
                X,
                Y,
                lambda x1, x2: kernel(x1, x2, params),
                assume_normalized_kernel=True,
            )
alignment = []
alignment.append(current_alignment)

for i in range(500):
    
    if main_centroid:
        if kao_class == 1:
            cost = lambda _params: loss(
                                
                                        class1_centroids, 
                                        centroid1_labels,
                                        centroid1,
                                        lambda x1, x2: kernel(x1, x2, params),
                                        _params
                                        )
            kao_class = 2

        else:

            cost = lambda _params: loss(
                                
                                        class2_centroids, 
                                        centroid2_labels,
                                        centroid2,
                                        lambda x1, x2: kernel(x1, x2, params),
                                        _params
                                        )
            kao_class = 1
            main_centroid = False

    else:

    #    cost = lambda _params: -qml.kernels.target_alignment(
    #                                                            class1_centroids + class2_centroids,
    #                                                            centroid1_labels,
    #                                                            lambda x1, x2: kernel(x1, x2, _params),
    #                                                            assume_normalized_kernel=True,
    #                                                        )
        kao_class = 1
        main_centroid = True

    #print(params)
    #print(cost(params), main_centroid, kao_class)
    params = opt.step(cost, params)
    
    
    #if (i + 1) % 50 == 0:

    print(f"Circuit Executions: {circuit_executions}") 
    current_alignment = qml.kernels.target_alignment(
                X,
                Y,
                lambda x1, x2: kernel(x1, x2, params),
                assume_normalized_kernel=True,
            )
    
    alignment.append(current_alignment)
    print(f"Alignment = {current_alignment:.3f}")


executions = circuit_executions

trained_kernel = lambda x1, x2: kernel(x1, x2, params)
trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)
svm_trained = SVC(kernel=trained_kernel_matrix).fit(X, Y) 

accuracy_trained = accuracy(svm_trained, X, Y)
print(f"The accuracy of a kernel with trained parameters is {accuracy_trained:.3f}")

accuracy_tested = accuracy(svm_trained, x_test, y_test)
print(f"The accuracy of a kernel with trained parameters is {accuracy_tested:.3f}")
y_pred_trained = svm_trained.predict(x_test)

# Calculate ROC curve and AUC for trained model
fpr_trained, tpr_trained, _ = roc_curve(y_test, y_pred_trained)
auc_trained = auc(fpr_trained, tpr_trained)
print(f"Trained Quantum SVM AUC: {auc_trained:.4f}")

model_data = {
    'trained_training_accuracy': [accuracy_trained],
    'trained_testing_accuracy' : [accuracy_tested],
    'auc_trained': [auc_trained],
    'circuit_executions': [executions]
    
}

cost_data = {
    'alignment': [alignment],
}

file = 'clusterModel_16clusters_linear.csv'

df = pd.DataFrame(model_data)
df.to_csv(file)

file = 'clusterModel_16clusters_linear.npy'
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
from matplotlib.colors import ListedColormap
# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap([train_color_0, train_color_1]))

# Plot training data with solid circles
plt.scatter(X[Y == -1][:, 0], X[Y == -1][:, 1], color=train_color_0, label='Train Class -1', s=150, marker='o', alpha=1)
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color=train_color_1, label='Train Class 1', s=150, marker='o', alpha=1)

# Plot testing data with hollow circles
plt.scatter(x_test[y_test == -1][:, 0], x_test[y_test == -1][:, 1], edgecolor=train_color_0, facecolor='none', label='Test Class -1', s=100, marker='o', alpha=1)
plt.scatter(x_test[y_test == 1][:, 0], x_test[y_test == 1][:, 1], edgecolor=train_color_1, facecolor='none', label='Test Class 1', s=100, marker='o', alpha=1)

# Remove extra margins for a tighter layout
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linear Data')
plt.tight_layout()
plt.savefig('clusterModel_16clusters_linear.png', dpi=800)
plt.show()