import pennylane as qml
from pennylane import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
import scipy.stats

def subset_sampling(x, model=None, sampling='random', subset_size=4):

    if sampling == 'random':
        subset = np.random.choice(list(range(len(x))), subset_size)

    elif sampling in ['greedy', 'greedy_inc']:
        if model is None:
            print("Please provide trained SVM model")
            return None
        else:
            probs = model.predict_proba(x)
            entropy = -np.sum(probs * np.log(probs), axis=1)
            subset = np.argsort(entropy)[::-1][:subset_size]

    elif sampling == 'probabilistic':
        if model is None:
            print("Please provide trained SVM model")
            return None
        else:
            probs = model.predict_proba(x)
            entropy = -np.sum(probs * np.log(probs), axis=1)
            sorted_idx = np.argsort(entropy)
            entropy_sorted = np.sort(entropy)
            probs = np.linspace(1, 0, len(entropy))
            probs = probs / np.sum(probs)
            subset = np.random.choice(sorted_idx, size=subset_size, p=probs)

    return subset
"""
def approx_greedy_sampling(kernel_matrix, subset_size, probability = False):

    if probability:
        similarity = kernel_matrix @ kernel_matrix
        uncertinity = np.var(similarity, axis = 1)
        #print("Similarity: ", similarity)
        #print("Uncertinity: ", uncertinity)
        #subset = np.argsort(uncertinity)[::-1][:subset_size]
        uncertinity_sorted = np.sort(uncertinity)
        uncertinity_idx = np.argsort(uncertinity)[::-1][:int(len(uncertinity) / 2)]

    
        probs = np.linspace(1, 0, int(len(uncertinity)/ 2))
        probs = probs / np.sum(probs)
        subset = np.random.choice(uncertinity_idx, size = subset_size, p = probs, replace=False)
        return subset
    else:
        similarity = kernel_matrix @ kernel_matrix
        probabilities = scipy.special.softmax(similarity, axis=1)
        entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
        subset = np.argsort(entropy)[::-1][:subset_size]
        return subset

    #else:
        
        #similarity = kernel_matrix @ kernel_matrix
        #uncertinity = np.var(similarity, axis = 1)
        #subset = np.argsort(uncertinity)[:subset_size]

        #return subset
"""
def approx_greedy_sampling(kernel_matrix, subset_size, labels, probability=False):
    # Calculate the subset size for each class
    class_size = subset_size // 2

    # Get indices of each class
    class_0_indices = np.where(labels == 0)[0]
    class_1_indices = np.where(labels == 1)[0]

    if probability:
        similarity = kernel_matrix @ kernel_matrix
        uncertinity = np.var(similarity, axis=1)

        # Get uncertainty and indices for each class separately
        uncertinity_class_0 = uncertinity[class_0_indices]
        uncertinity_class_1 = uncertinity[class_1_indices]

        # Sort and select top half of uncertainty indices for each class
        uncertinity_idx_class_0 = np.argsort(uncertinity_class_0)[::-1][:len(uncertinity_class_0) // 2]
        uncertinity_idx_class_1 = np.argsort(uncertinity_class_1)[::-1][:len(uncertinity_class_1) // 2]

        # Get corresponding original indices
        uncertinity_idx_class_0 = class_0_indices[uncertinity_idx_class_0]
        uncertinity_idx_class_1 = class_1_indices[uncertinity_idx_class_1]

        # Define probabilities for each class
        probs_class_0 = np.linspace(1, 0, len(uncertinity_idx_class_0))
        probs_class_0 = probs_class_0 / np.sum(probs_class_0)
        probs_class_1 = np.linspace(1, 0, len(uncertinity_idx_class_1))
        probs_class_1 = probs_class_1 / np.sum(probs_class_1)

        # Randomly select samples from each class
        subset_class_0 = np.random.choice(uncertinity_idx_class_0, size=class_size, p=probs_class_0, replace=False)
        subset_class_1 = np.random.choice(uncertinity_idx_class_1, size=class_size, p=probs_class_1, replace=False)

        # Combine the subsets
        subset = np.concatenate([subset_class_0, subset_class_1])

    else:
        similarity = kernel_matrix @ kernel_matrix
        probabilities = scipy.special.softmax(similarity, axis=1)
        entropy = -np.sum(probabilities * np.log(probabilities), axis=1)

        # Get entropy for each class
        entropy_class_0 = entropy[class_0_indices]
        entropy_class_1 = entropy[class_1_indices]

        # Select top indices based on entropy for each class
        subset_class_0 = class_0_indices[np.argsort(entropy_class_0)[::-1][:class_size]]
        subset_class_1 = class_1_indices[np.argsort(entropy_class_1)[::-1][:class_size]]

        # Combine the subsets
        subset = np.concatenate([subset_class_0, subset_class_1])

    return subset

