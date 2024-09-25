import pennylane as qml
from pennylane import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel

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
        uncertinity = np.var(similarity, axis = 1)
        subset = np.argsort(uncertinity)[::-1][:subset_size]

        return subset
