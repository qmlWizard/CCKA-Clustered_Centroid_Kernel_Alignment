import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from utils.kernel import initialize_kernel, kernel, get_circuit_executions
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
    ansatz = sys.argv[3]
    subset_size = int(sys.argv[4])
    print("Reading Dataset...")
except:
    print("Error! While Execution")
    print("USAGE: python <dataset> <sampling> <ansatz> <subset_size>")

n_feat = 3
n_sam = 100

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
elif dataset == 'microgrid':
    data = microgrid_data()

column_mapping = {}
for col in data.columns:
    # Extract the numeric part after 'feature_'
    match = re.search(r'feature_(\d+)', col)
    if match:
        col_num = match.group(1)
        new_col = f"feature_{int(col_num) + 3}"
        column_mapping[col] = new_col
    
# Duplicate columns with new names
for original, new in column_mapping.items():
    data[new] = data[original]

print("Done..!")
print("Sample: ", data.head(1))
print("Data Size: ", data.shape)


features = np.asarray(data[[col for col in data.columns if col != 'target']].values.tolist())
target = np.asarray(data['target'].values.tolist())

print("Configuring Quantum Circuit")

n_qubits = len(features[0]) 
layers = 1

print("Number of Qubits: ", n_qubits)
print("Number of Variational Layers: ", layers)

wires, shape = initialize_kernel(n_qubits, ansatz, layers)
param_shape = (2,) + shape
params = np.random.uniform(0, 2 * np.pi, size=param_shape, requires_grad=True)

print("Shape for params: ", param_shape)
print("Dividing Testing and Training dataset")

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print("Train Size: ", len(x_train))
print("Test Size: ", len(x_test))

opt = qml.GradientDescentOptimizer(0.2)
#opt = qml.SPSAOptimizer(0.2)

f_kernel = lambda x1, x2: kernel(x1, x2, params)
get_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, f_kernel)

# Train the SVM to obtain the Lagrange multipliers 'a'
#svm_model = SVC(kernel='precomputed').fit(get_kernel_matrix(x_train, x_train), y_train)

# Retrieve the Lagrange multipliers and support vectors
#a = svm_model.dual_coef_[0]
#support_vectors_indices = svm_model.support_
alignment_per_epoch = []

if sampling in ['greedy', 'probabilistic', 'greedy_inc']:
    kernel_matrix = get_kernel_matrix(x_train, x_train)
    print("Created Kernel Matrix Training SVM now")
    svm_model = SVC(kernel = get_kernel_matrix, probability = True).fit(x_train, y_train)
    print("Model trained")

if sampling in ['approx_greedy', 'approx_greedy_prob']:
    kernel_matrix = get_kernel_matrix(x_train, x_train)

params_list = []
cost_list = []
acc = 0
epochs = 500
for i in range(epochs):
    # Choose subset of datapoints to compute the KTA on.
    if sampling in ['greedy', 'probabilistic', 'greedy_inc']:
        subset = subset_sampling(x_train, svm_model, sampling, subset_size)

    elif sampling == 'approx_greedy_prob':
        subset = approx_greedy_sampling(kernel_matrix, subset_size, probability=True)

    elif sampling == 'approx_greedy':
        subset = approx_greedy_sampling(kernel_matrix, subset_size)
    
    else:
        subset = subset_sampling(x_train, sampling=sampling, subset_size=subset_size)
        print(subset)

    # Define the cost function for optimization based on the given formula
    #def cost(_params):
    #    # Compute the first summation: sum over all a_i (restricted to support vectors)
    #    first_term = np.sum(a)
#
 #       # Compute the second summation: 0.5 * sum over all pairs (i, j)
 #       second_term = 0.5 * np.sum([
 #           a[i] * a[j] * y_train[support_vectors_indices[i]] * y_train[support_vectors_indices[j]] * 
  #          kernel(x_train[support_vectors_indices[i]], x_train[support_vectors_indices[j]], _params)
   #         for i in range(len(support_vectors_indices))
    #        for j in range(len(support_vectors_indices))
     #   ])

        # The cost function according to the formula
      #return first_term - second_term

    cost = lambda _params: -target_alignment(
                x_train[subset],
                y_train[subset],
                lambda x1, x2: kernel(x1, x2, _params),
                assume_normalized_kernel = True
            )

    # Optimization step
    params = opt.step(cost, params)
    cost_list.append(cost(params))
    params_list.append(params)

    if (i + 1) % 100 == 0:
        current_alignment = target_alignment(
            x_train,
            y_train,
            lambda x1, x2: kernel(x1, x2, params),
            assume_normalized_kernel=True,
        )
        alignment_per_epoch.append(current_alignment)
        print(f"Epoch: {i + 1} Current Kernel Alignment: {current_alignment}")

    if sampling == 'greedy_inc':
        km = get_kernel_matrix(x_train[subset], x_train[subset])
        kernel_matrix[np.ix_(subset, subset)] = km 
        svm_model = SVC(kernel=get_kernel_matrix, probability=True).fit(x_train, y_train)
    if sampling == 'approx_greedy':
        km = get_kernel_matrix(x_train[subset], x_train[subset])
        kernel_matrix[np.ix_(subset, subset)] = km

    
#min_cost_idx = cost_list.index(min(cost_list))
#print(min_cost_idx)

#params = params_list[min_cost_idx]


    

trained_kernel = lambda x1, x2: kernel(x1, x2, params)
trained_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, trained_kernel) 
svm_trained = SVC(kernel=trained_kernel_matrix).fit(x_train, y_train)

y_pred = svm_trained.predict(x_train)
train_acc = accuracy_score(y_train, y_pred)
print("Training Accuracy: ", train_acc)

y_pred = svm_trained.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Testing Accuracy: ", accuracy)

d = {
    'algorithm': [sampling],
    'subset': [subset_size],
    'dataset': [dataset],
    'ansatz': [ansatz],
    'Training Accuracy': [train_acc],
    'Testing Accuracy': [accuracy],
    'executions': [get_circuit_executions()]
}

df = pd.DataFrame(d)
file = sampling + '_' + str(subset_size) + '_' + ansatz + '_' + dataset + '.csv'
df.to_csv(f'results/{dataset}/{file}')

cost = {'cost': alignment_per_epoch}

file = f'costs/{dataset}/' + sampling + '_' + str(subset_size) + '_' + ansatz + '_' + dataset + '.npy'
np.save(file, cost)

file = f'weights/{dataset}/' + sampling + '_' + str(subset_size) + '_' + ansatz + '_' + dataset + '_weights.npy'
weights = {'params': [params]}
np.save(file, weights)
