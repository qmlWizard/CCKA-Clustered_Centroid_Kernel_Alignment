import pennylane as qml
from pennylane import numpy as np
from utils.kernel import initialize_kernel, kernel, get_circuit_executions
from utils.classification_data import checkerboard_data, linear_data, hidden_manifold_data, power_line_data
from utils.train_kernel import target_alignment
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

n_feat = 3
n_sam = 50

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

opt = qml.GradientDescentOptimizer(0.2)

f_kernel = lambda x1, x2: kernel(x1, x2, params)
get_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, f_kernel) 

if sampling in ['greedy', 'probabilistic', 'greedy_inc']:
    kernel_matrix = get_kernel_matrix(x_train, x_train)
    print("Created Kernel Matrix Training SVM now")
    svm_model = SVC(kernel='precomputed', probability=True).fit(kernel_matrix, y_train)
    print("Model trained")

for i in range(50):
    # Choose subset of datapoints to compute the KTA on.
    if sampling in ['greedy', 'probabilistic', 'greedy_inc']:
        #subset = subset_sampling_test(x_train, y_train, sampling=sampling, subset_size=subset_size)
        subset = subset_sampling(x_train, svm_model, sampling, subset_size)
    else:
        subset = subset_sampling(x_train, sampling=sampling, subset_size=subset_size)

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
        print(f"Epoch: {i + 1} Current Kernel Alignment: {current_alignment}")

    if sampling == 'greedy_inc':
        km = get_kernel_matrix(x_train[subset], x_train[subset])
        kernel_matrix[np.ix_(subset, subset)] = km 
        svm_model = SVC(kernel='precomputed', probability=True).fit(kernel_matrix, y_train)

trained_kernel = lambda x1, x2: kernel(x1, x2, params)
trained_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, trained_kernel) 
svm_trained = SVC(kernel=trained_kernel_matrix).fit(x_train, y_train)

y_pred = svm_trained.predict(x_train)
train_acc = accuracy_score(y_train, y_pred)
print("Training Accuracy: ", train_acc)


accuracy = accuracy_score(y_test, y_pred)
print("Testing Accuracy: ", accuracy)


d = {
	'algorithm': [samling],
	'subset': [subset_size],
	'dataset': [dataset],
	'Training Accuracy':[train_acc],
	'Testing Accuracy': [accuracy]

}


df = pd.DataFrame(d)
file = sampling + '_' + str(subset_size) + '_' + dataset + '.csv'
df.to_csv(f'results/{file}')

