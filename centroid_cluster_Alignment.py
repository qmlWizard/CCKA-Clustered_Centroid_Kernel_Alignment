# Step 1: Import necessary libraries
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

data = checkerboard_data(2)

## Extract features and target
features = np.asarray(data.drop(columns=['target']))
target = np.asarray(data['target'])
target = target % 2
target = 2 * target - 1

X, x_test, Y, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)


np.random.seed(1359)
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


params = random_params(num_wires=5, num_layers=6)

kernel_value = kernel(X[0], X[1], params)
print(f"The kernel value between the first and second datapoint is {kernel_value:.3f}")

f_kernel = lambda x1, x2: kernel(x1, x2, params)
get_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, f_kernel)
class_1_indices = np.where(Y == 1)[0]
class_2_indices = np.where(Y == -1)[0]

class1 = X[class_1_indices]
class2 = X[class_2_indices]

class1_kernel_matrix = get_kernel_matrix(class1, class1)
class2_kernel_matrix = get_kernel_matrix(class2, class2)

main_cluster = KMeans(n_clusters=1, random_state=42)
cluster = KMeans(n_clusters=2, random_state=42)

class1_cent_cluster = main_cluster.fit_predict(class1_kernel_matrix)
centroid1 = main_cluster.cluster_centers_
centroid1 = centroid1[:, :len(X[0])]

class2_cent_cluster = main_cluster.fit_predict(class2_kernel_matrix)
centroid2 = main_cluster.cluster_centers_
centroid2 = centroid1[:, :len(X[0])]

class1_clusters = cluster.fit_predict(class1_kernel_matrix)
class1_centroids = cluster.cluster_centers_
class1_centroids = class1_centroids[:, :len(X[0])]
class1_clusters = np.where(class1_clusters == 1, -1, 1)


class2_clusters = cluster.fit_predict(class2_kernel_matrix)
class2_centroids = cluster.cluster_centers_
class2_centroids = class2_centroids[:, :len(X[0])]
class2_clusters = np.where(class2_clusters == 1, -1, 1)


centroid1_labels = np.ones(len(centroid1))  # Assign all labels as 1 for centroid1
centroid2_labels = -np.ones(len(centroid2))  # Assign all labels as -1 for centroid2

# Concatenate the labels for both centroids
centroid_labels = np.concatenate([centroid1_labels, centroid2_labels])

def centroid_kernel_matrix(X, centroid, kernel):
    
    kernel_matrix = []

    for i in range(len(X)):
        kernel_matrix.append(kernel(centroid, X[i]))

    return np.array(kernel_matrix)

f_kernel = lambda x1, x2: kernel(x1, x2, params)[0]
centroid_kernel_matrix(class1_centroids, centroid1, f_kernel)


def centroid_target_alignment(X, Y, centroid, kernel, assume_normalized_kernel=False, rescale_class_labels=True):
    
    K = centroid_kernel_matrix(X, centroid, kernel)
    T = np.outer(Y, Y)
    numerator = np.sum(Y * K)  
    denominator = np.sqrt(np.sum(K * K) * np.sum(Y * Y))

    TA = numerator / denominator

    return TA


get_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, f_kernel)
km = centroid_target_alignment(class1_centroids, np.array([1, 1]), centroid1, f_kernel)
km

def loss(X, Y, centroid, kernel, params, lambda_kao = 0.01):
    TA = centroid_target_alignment(X, Y, centroid, kernel)
    r = lambda_kao * np.sum(params ** 2)
    return 1 - TA + r

f_kernel = lambda x1, x2: kernel(x1, x2, params)[0]
loss(class1_centroids, np.array([1, 1], requires_grad=False), centroid1, f_kernel, params)

main_centroid = True
kao_class = 1
opt = qml.GradientDescentOptimizer(0.2)
circuit_executions = 0
params = random_params(num_wires=5, num_layers=6)
for i in range(500):
    
    if main_centroid:
        if kao_class == 1:
            cost = lambda _params: -loss(class1_centroids, centroid1_labels,centroid1,lambda x1, x2: kernel(x1, x2, params)[0],params)
            kao_class = 2
        else:

            cost = lambda _params: -loss(class2_centroids, centroid2_labels,centroid2,lambda x1, x2: kernel(x1, x2, params)[0],params)
            kao_class = 1
            main_centroid = False

    else:

        cost = lambda _params: -qml.kernels.target_alignment(class1_centroids + class2_centroids,centroid1_labels,lambda x1, x2: kernel(x1, x2, _params), assume_normalized_kernel=True)
        kao_class = 1
        main_centroid = True

    params = opt.step(cost, params)
    
    
    if (i + 1) % 50 == 0:
        current_alignment = qml.kernels.target_alignment(
            X,
            Y,
            lambda x1, x2: kernel(x1, x2, params),
            assume_normalized_kernel=True,
        )
        print(f"Step {i+1} - Alignment = {current_alignment:.3f}")
        print(f"Circuit Executions: {circuit_executions}")  
"""
        f_kernel = lambda x1, x2: kernel(x1, x2, params)
        get_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, f_kernel)
        class_1_indices = np.where(Y == 1)[0]
        class_2_indices = np.where(Y == -1)[0]

        class1 = X[class_1_indices]
        class2 = X[class_2_indices]

        class1_kernel_matrix = get_kernel_matrix(class1, class1)
        class2_kernel_matrix = get_kernel_matrix(class2, class2)

        main_cluster = KMeans(n_clusters=1, random_state=42)
        cluster = KMeans(n_clusters=4, random_state=42)

        class1_cent_cluster = main_cluster.fit_predict(class1_kernel_matrix)
        centroid1 = main_cluster.cluster_centers_
        centroid1 = centroid1[:, :len(X[0])]

        class2_cent_cluster = main_cluster.fit_predict(class2_kernel_matrix)
        centroid2 = main_cluster.cluster_centers_
        centroid2 = centroid1[:, :len(X[0])]


        class1_clusters = cluster.fit_predict(class1_kernel_matrix)
        class1_centroids = cluster.cluster_centers_
        class1_centroids = class1_centroids[:, :len(X[0])]


        class2_clusters = cluster.fit_predict(class2_kernel_matrix)
        class2_centroids = cluster.cluster_centers_
        class2_centroids = class2_centroids[:, :len(X[0])]
"""

from sklearn.svm import SVC
# First create a kernel with the trained parameter baked into it.
trained_kernel = lambda x1, x2: kernel(x1, x2, params)

# Second create a kernel matrix function using the trained kernel.
trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)

# Note that SVC expects the kernel argument to be a kernel matrix function.
svm_trained = SVC(kernel=trained_kernel_matrix).fit(X, Y)

def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)

accuracy_trained = accuracy(svm_trained, X, Y)
print(f"The accuracy of a kernel with trained parameters is {accuracy_trained:.3f}")

accuracy_trained = accuracy(svm_trained, x_test, y_test)
print(f"The accuracy of a kernel with trained parameters is {accuracy_trained:.3f}")



