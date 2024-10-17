import pennylane as qml
from pennylane import numpy as np
from utils.kernel import initialize_kernel, kernel, get_circuit_executions
from utils.classification_data import checkerboard_data, linear_data, hidden_manifold_data, power_line_data
from utils.train_kernel import target_alignment, generate_origin_kernel_matrix, target_alignment_towards_origin
from utils.sampling import subset_sampling, subset_sampling_test
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

print("---------------------------------------------------------------------------------------------------------------------------------------------")
# Read configurations from the command line
try:
    dataset = sys.argv[1]
    sampling = sys.argv[2]
    subset_size = int(sys.argv[3])
    print("Reading Dataset...")
except:
    print("Error! While Execution")
    print("USAGE: python <dataset> <sampling> <subset_size>")

print("----------------------------------------------------------------------------------------------------------------------------------------------")

n_feat = 5
n_sam = 40

circuit_executions = 0
# Get the dataset 
if dataset == 'checkerboard':
    data = checkerboard_data(n_feat, n_sam)
elif dataset == 'linear':
    data = linear_data(n_feat, n_sam)
elif dataset == 'hidden_manifold':
    data = hidden_manifold_data(n_feat, n_sam)
elif dataset == 'powerline':
    data = power_line_data()

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
params = np.random.random(size=param_shape, requires_grad=True)

x1 = features[0]
x2 = features[1]

print("Before: ", get_circuit_executions())
distance = kernel(x1, x2, params)
print(qml.draw(kernel)(x1, x2, params))
print("Distance between x1 and x2: ", distance)
print("After: ", get_circuit_executions())

print("-------------------------------------------------------------------------------------------------------------------------")
print("Dividing Testing and Training dataset")

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print("Train Size: ", len(x_train))
print("Test Size: ", len(x_test))

print("------------------------------------------------------------------------------------------------------------------------")

opt = qml.GradientDescentOptimizer(0.5)

#kernel_matrix = generate_origin_kernel_matrix(x_train, kernel, params)

#print(kernel_matrix)


#Training  Loop:
for epoch in range(1000):

    cost = lambda _params: -target_alignment_towards_origin(
        x_train,
        y_train,
        lambda x1, x2: kernel(x1, x2, _params),
	_params,
        assume_normalized_kernel=True,
    )

    params = opt.step(cost, params)

    # Report the alignment on the full dataset every 50 steps.
    if (epoch + 1) % 10 == 0:
        current_alignment = qml.kernels.target_alignment(
            x_train,
            y_train,
            lambda x1, x2: kernel(x1, x2, params),
            assume_normalized_kernel=False,
        )
        print(f"Epoch: {epoch + 1} Current Kernel Alignment: {current_alignment}")

    #kernel_matrix = generate_origin_kernel_matrix(x_train, kernel, params)

#model = SVC(probability=True).fit(kernel_matrix, y_train)
kernel_matrix = generate_origin_kernel_matrix(x_train, lambda x1, x2: kernel(x1, x2, params))
model = SVC(probability=True).fit(kernel_matrix, y_train)
kernel_matrix_test = generate_origin_kernel_matrix(x_test, lambda x1, x2: kernel(x1, x2, params))
y_pred = model.predict(kernel_matrix_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set:", accuracy)
