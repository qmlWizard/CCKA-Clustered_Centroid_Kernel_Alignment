import pennylane as qml
from pennylane import numpy as np
from utils.kernel import initialize_kernel, kernel
from utils.classification_data import checkerboard_data
from utils.train_kernel import target_alignment
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

print("---------------------------------------------------------------------------------------------------------------------------------------------")
#read configurations from the command line
try:
    dataset = sys.argv[1]
    subset_size = int(sys.argv[2])
    print("Reading Dataset...")
except:
    print("Error ! While Execution")
    print("USAGE: python <dataset> <subset_size>")

print("----------------------------------------------------------------------------------------------------------------------------------------------")


##Get the dataset 
if dataset == 'checkerboard':
	data = checkerboard_data(5, 50)


print("Done..!")
print("Sample: ", data.head(1))
print("Data Size: ", data.shape)


features = np.asarray(data[[col for col in data.columns if col != 'target']].values.tolist())
target = np.asarray(data['target'].values.tolist())


print("--------------------------------------------------------------------------------------------------------------------------")

print("Configuring Quantum Circuit")

n_qubits = len(features[0])
layers = 6


print("Number of Qubits: ", n_qubits)
print("Number of Variational Layers: ", layers)

wires, shape = initialize_kernel(n_qubits, 'strong_entangled', layers)
param_shape = (2,) + shape
params = np.random.random(size = param_shape, requires_grad=True)

x1 = features[0]
x2 = features[1]

distance = kernel(x1, x2, params)

print("Distance between x1 and x2: ", distance)

print("-------------------------------------------------------------------------------------------------------------------------")
print("Dividing Testing and Training dataset")

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 42)

print("Train Size: ", len(x_train))
print("Test Size: ", len(x_test))

print("------------------------------------------------------------------------------------------------------------------------")


opt = qml.GradientDescentOptimizer(0.2)

for i in range(500):
    
    # Choose subset of datapoints to compute the KTA on.
    subset = np.random.choice(list(range(len(x_train))), 4)
    
    # Define the cost function for optimization
    cost = lambda _params: -qml.kernels.target_alignment(
        x_train[subset],
        y_train[subset],
        lambda x1, x2: kernel(x1, x2, _params),
        assume_normalized_kernel=True,
    )
    
    # Optimization step
    params = opt.step(cost, params)

    # Report the alignment on the full dataset every 50 steps.
    if (i + 1) % 10 == 0:

        current_alignment = qml.kernels.target_alignment(
            x_train,
            y_train,
            lambda x1, x2: kernel(x1, x2, params),
            assume_normalized_kernel=False,
        )
        print(current_alignment)


trained_kernel = lambda x1, x2: kernel(x1, x2, params)
trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)
svm_trained = SVC(kernel=trained_kernel_matrix).fit(x_train, y_train)

y_pred = svm_trained.predict(x_train)
train_acc = accuracy_score(y_train, y_pred)

print("Training Accuracy: ", train_acc)

y_pred = svm_trained.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("Testing Accuracy: ", accuracy)


