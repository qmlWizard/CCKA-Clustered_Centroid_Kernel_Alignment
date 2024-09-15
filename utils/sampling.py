import pennylane as qml
from pennylane import numpy as np
from sklearn.svm  import SVC


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

		
	
		 
