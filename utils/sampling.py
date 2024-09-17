import pennylane as qml
from pennylane import numpy as np
from sklearn.svm  import SVC
from sklearn.metrics.pairwise import rbf_kernel


def subset_sampling_test(x_train, y_train, sampling, subset_size, classical_kernel='rbf'):
    if sampling == 'greedy':
        n_samples = len(x_train)
        B = []
        R = list(range(n_samples))  # Residual set indices

        if classical_kernel == 'rbf':
            K_classical = rbf_kernel(x_train)
        elif classical_kernel == 'linear':
            K_classical = x_train @ x_train.T
        else:
            raise ValueError("Unsupported classical kernel")

        A_current = 0.0

        for t in range(subset_size):
            max_delta_A = -np.inf
            best_candidate = None

            # Limit the number of candidates to evaluate
            candidate_indices = np.random.choice(R, size=min(50, len(R)), replace=False)

            for idx in candidate_indices:
                if not B:
                    delta_A = np.abs(K_classical[idx, idx] * y_train[idx] ** 2)
                else:
                    # Approximate alignment gain
                    y_subset = y_train[B + [idx]]
                    K_subset = K_classical[np.ix_(B + [idx], B + [idx])]
                    A_new = np.sum(K_subset * np.outer(y_subset, y_subset))
                    delta_A = A_new - A_current

                if delta_A > max_delta_A:
                    max_delta_A = delta_A
                    best_candidate = idx

            if best_candidate is not None:
                B.append(best_candidate)
                R.remove(best_candidate)
                A_current += max_delta_A
            else:
                break

        return B
    else:
        # Implement other sampling strategies or default to random sampling
        return np.random.choice(len(x_train), size=subset_size, replace=False)

def subset_sampling(x, model = None, sampling = 'random', subset_size = 4):
	
	if sampling == 'random':
		subset = np.random.choice(list(range(len(x))), subset_size)

	elif sampling in ['greedy', 'greedy_inc']:
		if model == None:
			print("Please provide trained SVM model")
			return None

		else:
			probs = model.predict_proba(x)
			entropy = -np.sum(probs * np.log(probs), axis = 1)
			subset = np.argsort(entropy)[::-1][:subset_size]

	elif sampling == 'probabilistic':
		if model == None:
			print("Please provide trained SVM model")
			return None
		else:
			probs = model.predict_proba(x)	
			entropy = -np.sum(probs * np.log(probs), axis = 1)
			subset = np.argsort(entropy)[:subset_size]
			sorted_idx = np.argsort(entropy)
			entropy_sorted = np.sort(entropy)
			probs = np.linspace(1, 0, len(entropy))
			probs = probs / np.sum(probs)
			subset = np.random.choice(sorted_idx, size = subset_size, p = probs)
			
			
	return subset

		
	
		 
