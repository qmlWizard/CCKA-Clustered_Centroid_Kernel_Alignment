import pennylane as qml
from pennylane import numpy as np
import pandas as pd
#from utils.kernel import initialize_kernel, kernel, get_circuit_executions
from utils.classification_data import checkerboard_data, linear_data, hidden_manifold_data, power_line_data, microgrid_data
from utils.train_kernel import target_alignment
from utils.sampling import subset_sampling 
from utils.sampling import approx_greedy_sampling
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import re


try:
    dataset = sys.argv[1]
    sampling = sys.argv[2]
    subset_size = int(sys.argv[3])
    print("Reading Dataset...")
except:
    print("Error! While Execution")
    print("USAGE: python <dataset> <sampling> <ansatz> <subset_size>")

n_feat = 5
n_sam = 100

circuit_executions = 0
# Get the dataset 
if dataset == 'checkerboard':
    data = checkerboard_data(2)
elif dataset == 'linear':
    data = linear_data(n_feat, n_sam)
elif dataset == 'hidden_manifold':
    data = hidden_manifold_data(n_feat, n_sam)
elif dataset == 'powerline':
    data = power_line_data()
elif dataset == 'microgrid':
    data = microgrid_data()
    
    
print("Done..!")
print("Sample: ", data.head(1))
print("Data Size: ", data.shape)


features = np.asarray(data[[col for col in data.columns if col != 'target']].values.tolist())
target = np.asarray(data['target'].values.tolist())
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


n_qubits = len(features[0]) * 2 
layers = 6

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

def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)

dev = qml.device("default.qubit", wires= n_qubits, shots=None)
wires = dev.wires.tolist()


@qml.qnode(dev)
def kernel_circuit(x1, x2, params):
    ansatz(x1, params, wires=wires)
    adjoint_ansatz(x2, params, wires=wires)
    return qml.probs(wires=wires)

def kernel(x1, x2, params):
    return kernel_circuit(x1, x2, params)[0]



init_params = random_params(num_wires = n_qubits, num_layers=layers)

init_kernel = lambda x1, x2: kernel(x1, x2, init_params)
#kta_init = qml.kernels.target_alignment(x_train, y_train, init_kernel, assume_normalized_kernel=True)
#print(f"The kernel-target alignment for our dataset and random parameters is {kta_init:.3f}")


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


params = init_params
opt = qml.GradientDescentOptimizer(0.2)

if sampling in ['approx_greedy', 'approx_greedy_prob']:
    get_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, init_kernel)
    kernel_matrix = get_kernel_matrix(x_train, x_train)

for i in range(500):
    # Choose subset of datapoints to compute the KTA on.
    if sampling == 'random':
        subset = np.random.choice(list(range(len(x_train))), subset_size)
    elif sampling == 'approx_greedy':
        subset = approx_greedy_sampling(kernel_matrix, subset_size)
    else:
        subset = approx_greedy_sampling(kernel_matrix, subset_size, probability=True)
    print(subset)
    # Define the cost function for optimization
    
    print(type(subset))
    
    cost = lambda _params: -target_alignment(
        x_train[subset],
        y_train[subset],
        lambda x1, x2: kernel(x1, x2, _params),
        assume_normalized_kernel=True,
    )
    # Optimization step
    params = opt.step(cost, params)

    # Report the alignment on the full dataset every 50 steps.
    if (i + 1) % 50 == 0:
        current_alignment = target_alignment(
            x_train,
            y_train,
            lambda x1, x2: kernel(x1, x2, params),
            assume_normalized_kernel=True,
        )
        print(f"Step {i+1} - Alignment = {current_alignment:.3f}")
    
    if sampling == 'approx_greedy':
        km = get_kernel_matrix(x_train[subset], x_train[subset])
        kernel_matrix[np.ix_(subset, subset)] = km

# First create a kernel with the trained parameter baked into it.
trained_kernel = lambda x1, x2: kernel(x1, x2, params)
# Second create a kernel matrix function using the trained kernel.
trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)
# Note that SVC expects the kernel argument to be a kernel matrix function.
svm_trained = SVC(kernel=trained_kernel_matrix).fit(x_train, y_train)


accuracy_trained = accuracy(svm_trained, x_train, y_train)
print(f"The accuracy of a kernel with trained parameters is {accuracy_trained:.3f}")

accuracy_tested = accuracy(svm_trained, x_test, y_test)
print(f"The accuracy of a kernel with trained parameters is {accuracy_tested:.3f}")