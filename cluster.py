from pennylane import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from sklearn.model_selection import train_test_split
from utils.classification_data import linear_data, checkerboard_data, power_line_data, microgrid_data, make_double_cake_data
from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib as mpl

np.random.seed(1359)

train = 'random'
circuit_executions = 0

print("Algorithm: ", train)
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
    ansatz(x1, params, wires=wires)
    adjoint_ansatz(x2, params, wires=wires)
    circuit_executions += 1
    return qml.probs(wires=wires)

def kernel(x1, x2, params):
    return kernel_circuit(x1, x2, params)[0]

def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)

def plot_double_cake_data(X, Y, ax, num_sectors=None):
    """Plot double cake data and corresponding sectors."""
    x, y = X.T
    cmap = plt.colors.ListedColormap(["#FF0000", "#0000FF"])
    ax.scatter(x, y, c=Y, cmap=cmap, s=25, marker="s")

    if num_sectors is not None:
        sector_angle = 360 / num_sectors
        for i in range(num_sectors):
            color = ["#FF0000", "#0000FF"][(i % 2)]
            other_color = ["#FF0000", "#0000FF"][((i + 1) % 2)]
            ax.add_artist(
                plt.patches.Wedge(
                    (0, 0),
                    1,
                    i * sector_angle,
                    (i + 1) * sector_angle,
                    lw=0,
                    color=color,
                    alpha=0.1,
                    width=0.5,
                )
            )
            ax.add_artist(
                plt.patches.Wedge(
                    (0, 0),
                    0.5,
                    i * sector_angle,
                    (i + 1) * sector_angle,
                    lw=0,
                    color=other_color,
                    alpha=0.1,
                )
            )
            ax.set_xlim(-1, 1)

    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    return ax

def plot_decision_boundaries(classifier, ax, N_gridpoints=14):
    _xx, _yy = np.meshgrid(np.linspace(-1, 1, N_gridpoints), np.linspace(-1, 1, N_gridpoints))

    _zz = np.zeros_like(_xx)
    for idx in np.ndindex(*_xx.shape):
        _zz[idx] = classifier.predict(np.array([_xx[idx], _yy[idx]])[np.newaxis, :])

    plot_data = {"_xx": _xx, "_yy": _yy, "_zz": _zz}
    ax.contourf(
        _xx,
        _yy,
        _zz,
        cmap=mpl.colors.ListedColormap(["#FF0000", "#0000FF"]),
        alpha=0.2,
        levels=[-1, 0, 1],
    )
    plot_double_cake_data(X, Y, ax)

    return plot_data

def target_alignment(
    X,
    Y,
    kernel,
    assume_normalized_kernel=False,
    rescale_class_labels=True,
):
    """Kernel-target alignment between kernel and labels."""

    K = qml.kernels.square_kernel_matrix(
        X,
        kernel,
        assume_normalized_kernel=assume_normalized_kernel,
    )

    if rescale_class_labels:
        nplus = np.count_nonzero(np.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = np.array(Y)

    T = np.outer(_Y, _Y)
    inner_product = np.sum(K * T)
    norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
    inner_product = inner_product / norm

    return inner_product

data = linear_data(2, 150)

## Extract features and target
features = np.asarray(data.drop(columns=['target']))
target = np.asarray(data['target'])
#target = target % 2
#target = 2 * target - 1

X, x_test, Y, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

print("Train Length: ", len(X))
print("Test Length: ", len(x_test))

params = random_params(num_wires=5, num_layers=6)
init_kernel = lambda x1, x2: kernel(x1, x2, params)
kta_init = qml.kernels.target_alignment(X, Y, init_kernel, assume_normalized_kernel=True)

print(f"The kernel-target alignment for our dataset and random parameters is {kta_init:.3f}")

alignment = []
lparams = []
alignment.append(kta_init)
if train == 'random':
    opt = qml.GradientDescentOptimizer(0.2)

    for i in range(500):
        # Choose subset of datapoints to compute the KTA on.
        subset = np.random.choice(list(range(len(X))), 8)
        # Define the cost function for optimization
        cost = lambda _params: -target_alignment(
            X[subset],
            Y[subset],
            lambda x1, x2: kernel(x1, x2, _params),
            assume_normalized_kernel=True,
        )
        # Optimization step
        params = opt.step(cost, params)

        # Report the alignment on the full dataset every 50 steps.
        if (i + 1) % 50 == 0:
            current_alignment = target_alignment(
                X,
                Y,
                lambda x1, x2: kernel(x1, x2, params),
                assume_normalized_kernel=True,
            )
            alignment.append(current_alignment)

            print(f"Step {i+1} - Alignment = {current_alignment:.3f}")

elif train == 'cluster':
    f_kernel = lambda x1, x2: kernel(x1, x2, params)
    get_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, f_kernel)

    class_1_indices = np.where(Y == 1)[0]
    class_2_indices = np.where(Y == -1)[0]

    class1 = X[class_1_indices]
    class2 = X[class_2_indices]

    class1_kernel_matrix = get_kernel_matrix(class1, class1)
    class2_kernel_matrix = get_kernel_matrix(class2, class2)

    cluster = KMeans(n_clusters=4, random_state=42)

    class1_clusters = cluster.fit_predict(class1_kernel_matrix)
    class1_centroids = cluster.cluster_centers_
    class1_centroids = class1_centroids[:, :len(X[0])]


    class2_clusters = cluster.fit_predict(class2_kernel_matrix)
    class2_centroids = cluster.cluster_centers_
    class2_centroids = class2_centroids[:, :len(X[0])]

    centroids = []
    centroid_labels = []
    for c, cent in zip(np.unique(class1_clusters), class1_centroids):
        centroids.append(cent)
        centroid_labels.append(1)

    for c, cent in zip(np.unique(class2_clusters), class2_centroids):
        centroids.append(cent)
        centroid_labels.append(-1)

    opt = qml.GradientDescentOptimizer(0.2)

    for i in range(500):
        # Choose subset of datapoints to compute the KTA on.
        # Define the cost function for optimization
        cost = lambda _params: -target_alignment(
            centroids,
            centroid_labels,
            lambda x1, x2: kernel(x1, x2, _params),
            assume_normalized_kernel=True,
        )
        # Optimization step
        params = opt.step(cost, params)

        # Report the alignment on the full dataset every 50 steps.
        if (i + 1) % 50 == 0:
            current_alignment = target_alignment(
                X,
                Y,
                lambda x1, x2: kernel(x1, x2, params),
                assume_normalized_kernel=True,
            )
            alignment.append(current_alignment)
            print(f"Step {i+1} - Alignment = {current_alignment:.3f}")
            
            if alignment[len(alignment) - 2] > current_alignment:
   
                print("Updating the Centroids")
                class1_kernel_matrix = get_kernel_matrix(class1, class1)
                class2_kernel_matrix = get_kernel_matrix(class2, class2)

                class1_clusters = cluster.fit_predict(class1_kernel_matrix)
                class1_centroids = cluster.cluster_centers_
                class1_centroids = class1_centroids[:, :len(X[0])]


                class2_clusters = cluster.fit_predict(class2_kernel_matrix)
                class2_centroids = cluster.cluster_centers_
                class2_centroids = class2_centroids[:, :len(X[0])]
                
                centroids = []
                centroid_labels = []
                for c, cent in zip(np.unique(class1_clusters), class1_centroids):
                    centroids.append(cent)
                    centroid_labels.append(1)

                for c, cent in zip(np.unique(class2_clusters), class2_centroids):
                    centroids.append(cent)
                    centroid_labels.append(-1)
          

# First create a kernel with the trained parameter baked into it.
trained_kernel = lambda x1, x2: kernel(x1, x2, params)

# Second create a kernel matrix function using the trained kernel.
trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)

# Note that SVC expects the kernel argument to be a kernel matrix function.
svm_trained = SVC(kernel=trained_kernel_matrix).fit(X, Y)

accuracy_trained = accuracy(svm_trained, X, Y)
print(f"The accuracy of a kernel with trained parameters is {accuracy_trained:.3f}")

accuracy_tested= accuracy(svm_trained, x_test, y_test)
print(f"The accuracy of a kernel with trained parameters is {accuracy_tested:.3f}")

y_pred_trained = svm_trained.predict(x_test)
fpr_trained, tpr_trained, _ = roc_curve(y_test, y_pred_trained)
auc_trained = auc(fpr_trained, tpr_trained)
print(f"Trained Quantum SVM AUC: {auc_trained:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred_trained)

model_data = {
    'trained_training_accuracy': [accuracy_trained],
    'trained_testing_accuracy' : [accuracy_tested],
    'auc_trained': [auc_trained],
    'circuit_executions': [circuit_executions],
    'confusion_matrix': [conf_matrix]
    
}

np.save(f'linear_{train}.npy', model_data)

trained_plot_data = plot_decision_boundaries(svm_trained, plt.gca())
plt.savefig(f'checkerboard_{train}_decision_boundaries.png') 
plt.close()