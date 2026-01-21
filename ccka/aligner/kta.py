import jax
import jax.numpy as jnp
import pennylane as qml
import optax as ox
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
from pprint import pformat

def print_box(title, lines, width=78):
    """
    Print a boxed section with a title and key-value lines.
    """
    print("┌" + "─" * (width - 2) + "┐")
    print(f"│ {title.center(width - 4)} │")
    print("├" + "─" * (width - 2) + "┤")

    for line in lines:
        wrapped = pformat(line, width=width - 6).split("\n")
        for w in wrapped:
            print(f"│ {w.ljust(width - 4)} │")

    print("└" + "─" * (width - 2) + "┘\n")

def print_training_summary(history):
    last = lambda x: x[-1]
    best = lambda x: max(x)

    print_box(
        "FULL KERNEL TARGET ALIGNMENT – TRAINING SUMMARY",
        [
            f"Epochs run           : {len(history['loss_history'])}",
            f"Total training time  : {history['time']:.2f} seconds",
        ],
    )

    print_box(
        "ACCURACY METRICS",
        [
            f"Initial train accuracy : {history['init_train_accuracy']:.4f}",
            f"Final train accuracy   : {last(history['train_accuracy_history']):.4f}",
            f"Best train accuracy    : {best(history['train_accuracy_history']):.4f}",
            f"Initial test accuracy  : {history['init_test_accuracy']:.4f}",
            f"Final test accuracy    : {last(history['test_accuracy_history']):.4f}",
            f"Best test accuracy     : {best(history['test_accuracy_history']):.4f}",
        ],
    )

    print_box(
        "CLASSIFICATION METRICS (FINAL EPOCH)",
        [
            f"F1 score    : {last(history['f1_score_history']):.4f}",
            f"Precision   : {last(history['precision_score_history']):.4f}",
            f"Recall      : {last(history['recall_score_history']):.4f}",
        ],
    )

    print_box(
        "ALIGNMENT & OPTIMIZATION",
        [
            f"Final alignment    : {last(history['alignment_history']):.6f}",
            f"Best alignment     : {max(history['alignment_history']):.6f}",
            f"Final loss         : {last(history['loss_history']):.6f}",
            f"Best loss          : {min(history['loss_history']):.6f}",
            f"Circuit Executions : {history['circuit_executions']}",
        ],
    )


class fullKTA:
    """
    Full Kernel Target Alignment loss function.
    """

    def __init__(self,
                 kernel_model,
                 data: jnp.ndarray,
                 labels: jnp.ndarray,
                 split_size: float = 0.8,
                 matrix_type: str = 'regular',
                 matrix_normalisation: bool = False,
                 landmark_points: int = 0,
                 centering: bool = False,
                 epochs: int = 100,
                 learning_rate: float = 0.01,
                 optimizer: str = 'adam',
                 **kwargs,

    ):

        self.kernel_model = kernel_model
        self.data = data
        self.labels = labels
        self.matrix_type = matrix_type
        self.matrix_normalisation = matrix_normalisation
        self.landmark_points = landmark_points
        self.centering = centering
        self.epochs = epochs
        self.split_size = split_size
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer.lower()
        self.optimizer = self._get_optimizer()
        self.opt_state = None

        # --- Initialize Weights
        self.weights = self.kernel_model.circuit.init_weights()

        # --- Initialize Optimizer State
        self.opt_state = self.optimizer.init(self.weights)

        # --- Initialize Gradient Function
        self.grad_function = jax.jit(
            jax.grad(self._loss_kta)
        )

        # --- Initialize Loss Function
        self.loss_function = jax.jit(self._loss_kta)

        # --- Data Splitting
        self.xtrain = None
        self.xtest = None
        self.ytrain = None
        self.ytest = None
        self._split_data()



    def _get_optimizer(self):
        if self.optimizer_name == 'adam':
            return ox.adam(self.learning_rate)
        elif self.optimizer_name == 'sgd':
            return ox.sgd(self.learning_rate)
        else:
            raise ValueError("Unsupported optimizer")

    def _split_data(self, seed=42):
        n = len(self.data)
        rng = jax.random.PRNGKey(seed)

        perm = jax.random.permutation(rng, n)
        split = int(n * self.split_size)

        idx_train = perm[:split]
        idx_test = perm[split:]

        self.xtrain = self.data[idx_train]
        self.xtest = self.data[idx_test]
        self.ytrain = self.labels[idx_train]
        self.ytest = self.labels[idx_test]

    def _test_kernel_matrix(self, weights, X, test_X):
        N, D = X.shape
        M, _ = test_X.shape
        x1 = jnp.repeat(test_X, N, axis=0)  
        x2 = jnp.tile(X, (M, 1)) 
        return self.kernel_model.forward(x1, x2, weights).reshape(M, N)

    def regular_kernel_matrix(self, weights, X):
        N, D = X.shape
        x1 = jnp.repeat(X, N, axis=0)  
        x2 = jnp.tile(X, (N, 1)) 
        return self.kernel_model.forward(x1, x2, weights).reshape(N, N)

    def nystrom_kernel_matrix(self, weights, X):
        if self.landmark_points > len(X) or self.landmark_points <= 0:
            raise ValueError(
                "Unacceptable number of Landmark points. "
                "Require 0 < landmark_points <= len(X)"
            )

        # --- Select landmarks
        landmarks = X[: self.landmark_points]

        N, D = X.shape
        M, _ = landmarks.shape

        x1 = jnp.repeat(X, M, axis=0)
        x2 = jnp.tile(landmarks, (N, 1))
        knm_raw = self.kernel_model.forward(x1, x2, weights)
        KNM = knm_raw.reshape(N, M)

        x1 = jnp.repeat(landmarks, M, axis=0)  
        x2 = jnp.tile(landmarks, (M, 1))
        kmm_raw = self.kernel_model.forward(x1, x2, weights)
        KMM = kmm_raw.reshape(M, M)

        reg = 1e-8 * jnp.eye(M)

        return KNM @ jnp.linalg.inv(KMM + reg) @ KNM.T
    
    def center(self, kernel_matrix):
        n = kernel_matrix.shape[0]
        H = jnp.eye(n) - jnp.ones((n, n)) / n
        return H @ kernel_matrix @ H

    def alignment(self, weights, X, y):

        if self.matrix_type == 'regular':
            kernel_matrix = self.regular_kernel_matrix(weights, X)
        elif self.matrix_type == 'nystrom':
            kernel_matrix = self.nystrom_kernel_matrix(weights, X)
        else:
            raise ValueError("Unsupported matrix type")
        

        if self.centering:
            kernel_matrix = self.center(kernel_matrix)
        
        y = y.reshape(-1, 1)
        T = y @ y.T

        kta = jnp.sum(kernel_matrix * T) / (jnp.linalg.norm(kernel_matrix, ord='fro') * jnp.linalg.norm(T, ord='fro'))

        return kta
    
    def _loss_kta(self, weights, X, y):
        return 1 - self.alignment(weights, X, y)

    def align(self):
        history = {}
        
        param_history = []
        alignment_history = []
        loss_history = []
        train_accuracy_history = []
        test_accuracy_history = []
        f1_history = []
        precision_history = []
        recall_history = []

        result = self.svm_training(self.xtrain, self.ytrain)
        init_train_accuracy = result['train_accuracy']
        init_test_accuracy = result['test_accuracy']
        init_f1 = result['f1_score']
        init_precision = result['precision_score']
        init_recall = result['recall_score']
        alignment_history.append(self.alignment(self.weights, self.xtrain, self.ytrain))
        loss_history.append(self._loss_kta(self.weights, self.xtrain, self.ytrain))

        start = time.time()

        for epoch in tqdm(range(self.epochs), desc="Aligning Kernel with Full Kernel KTA"):

            loss = self.loss_function(self.weights, self.xtrain, self.ytrain)
            loss_history.append(loss)

            alignment_history.append(self.alignment(self.weights, self.xtrain, self.ytrain))
            param_history.append(self.weights)

            grads = self.grad_function(self.weights, self.xtrain, self.ytrain)

            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.weights = ox.apply_updates(self.weights, updates) 

            result = self.svm_training(self.xtrain, self.ytrain)
            train_accuracy_history.append(result['train_accuracy'])
            test_accuracy_history.append(result['test_accuracy'])
            f1_history.append(result['f1_score'])
            precision_history.append(result['precision_score'])
            recall_history.append(result['recall_score'])

        history['weights'] = self.weights
        history['init_train_accuracy'] = init_train_accuracy
        history['init_test_accuracy'] = init_test_accuracy
        history['alignment_history'] = alignment_history        
        history['loss_history'] = loss_history
        history['train_accuracy_history'] = train_accuracy_history
        history['test_accuracy_history'] = test_accuracy_history
        history['f1_score_history'] = f1_history
        history['precision_score_history'] = precision_history
        history['recall_score_history'] = recall_history
        history['time'] = time.time() - start
        history['circuit_executions'] = self.kernel_model.circuit_executions

        print_training_summary(history)

        return history
    
    def svm_training(self, X, y):
        result = {}

        if self.matrix_type == 'regular':
            kernel_matrix = self.regular_kernel_matrix(self.weights, X)
        elif self.matrix_type == 'nystrom':
            kernel_matrix = self.nystrom_kernel_matrix(self.weights, X)
        else:
            raise ValueError("Unsupported matrix type")

        if self.centering:
            kernel_matrix = self.center(kernel_matrix)

        svm = SVC(kernel='precomputed', C=1.0, gamma='scale', probability=True, max_iter=10000)
        svm.fit(kernel_matrix, y)

        result['svm'] = svm
        result['kernel_matrix'] = kernel_matrix
        result['y'] = y

        test_kernel_matrix = self._test_kernel_matrix(self.weights, self.xtrain, self.xtest)

        if self.centering:
            test_kernel_matrix = self.center(test_kernel_matrix)


        train_accuracy = accuracy_score(self.ytrain, svm.predict(kernel_matrix))
        test_accuracy = accuracy_score(self.ytest, svm.predict(test_kernel_matrix))

        f1 = f1_score(self.ytest, svm.predict(test_kernel_matrix))
        precision = precision_score(self.ytest, svm.predict(test_kernel_matrix))
        recall = recall_score(self.ytest, svm.predict(test_kernel_matrix))

        result['train_accuracy'] = train_accuracy
        result['test_accuracy'] = test_accuracy
        result['f1_score'] = f1
        result['precision_score'] = precision
        result['recall_score'] = recall

        return result


class randomKTA:

    def __init__(self,
                 kernel_model,
                 data: jnp.ndarray,
                 labels: jnp.ndarray,
                 split_size: float = 0.8,
                 random_samples: int = 4,
                 matrix_type: str = 'regular',
                 matrix_normalisation: bool = False,
                 landmark_points: int = 0,
                 centering: bool = False,
                 epochs: int = 100,
                 learning_rate: float = 0.01,
                 optimizer: str = 'adam',
                 **kwargs,

    ):

        self.kernel_model = kernel_model
        self.data = data
        self.labels = labels
        self.matrix_type = matrix_type
        self.matrix_normalisation = matrix_normalisation
        self.landmark_points = landmark_points
        self.centering = centering
        self.epochs = epochs
        self.split_size = split_size
        self.random_samples = random_samples
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer.lower()
        self.optimizer = self._get_optimizer()
        self.opt_state = None
        self.rng = jax.random.PRNGKey(42)

        self.num_samples = self.data.shape[0]
        self.permutation = None
        self.perm_ptr = 0

        # --- Initialize Weights
        self.weights = self.kernel_model.circuit.init_weights()

        # --- Initialize Optimizer State
        self.opt_state = self.optimizer.init(self.weights)

        # --- Initialize Gradient Function
        self.grad_function = jax.jit(
            jax.grad(self._loss_kta)
        )

        # --- Initialize Loss Function
        self.loss_function = jax.jit(self._loss_kta)

        # --- Data Splitting
        self.xtrain = None
        self.xtest = None
        self.ytrain = None
        self.ytest = None
        self._split_data()

    
        # --- Initialize Random Samples
        self._reshuffle()

    def _get_optimizer(self):
        if self.optimizer_name == 'adam':
            return ox.adam(self.learning_rate)
        elif self.optimizer_name == 'sgd':
            return ox.sgd(self.learning_rate)
        else:
            raise ValueError("Unsupported optimizer")

    def _split_data(self, seed=42):
        n = len(self.data)
        rng = jax.random.PRNGKey(seed)

        perm = jax.random.permutation(rng, n)
        split = int(n * self.split_size)

        idx_train = perm[:split]
        idx_test = perm[split:]

        self.xtrain = self.data[idx_train]
        self.xtest = self.data[idx_test]
        self.ytrain = self.labels[idx_train]
        self.ytest = self.labels[idx_test]

    def _reshuffle(self):
        self.rng, subkey = jax.random.split(self.rng)
        self.permutation = jax.random.permutation(subkey, self.num_samples)
        self.perm_ptr = 0
    
    def _get_random_samples(self, X, y):
        if self.permutation is None:
            self._reshuffle()
        if self.perm_ptr + self.random_samples > self.num_samples:
            self._reshuffle()
        idxs = self.permutation[self.perm_ptr : self.perm_ptr + self.random_samples]
        self.perm_ptr += self.random_samples
        return X[idxs], y[idxs]


    def _test_kernel_matrix(self, weights, X, test_X):
        N, D = X.shape
        M, _ = test_X.shape
        x1 = jnp.repeat(test_X, N, axis=0)  
        x2 = jnp.tile(X, (M, 1)) 
        return self.kernel_model.forward(x1, x2, weights).reshape(M, N)

    def regular_kernel_matrix(self, weights, X):
        N, D = X.shape
        x1 = jnp.repeat(X, N, axis=0)  
        x2 = jnp.tile(X, (N, 1)) 
        return self.kernel_model.forward(x1, x2, weights).reshape(N, N)

    def nystrom_kernel_matrix(self, weights, X):
        if self.landmark_points > len(X) or self.landmark_points <= 0:
            raise ValueError(
                "Unacceptable number of Landmark points. "
                "Require 0 < landmark_points <= len(X)"
            )

        # --- Select landmarks
        landmarks = X[: self.landmark_points]

        N, D = X.shape
        M, _ = landmarks.shape

        x1 = jnp.repeat(X, M, axis=0)
        x2 = jnp.tile(landmarks, (N, 1))
        knm_raw = self.kernel_model.forward(x1, x2, weights)
        KNM = knm_raw.reshape(N, M)

        x1 = jnp.repeat(landmarks, M, axis=0)  
        x2 = jnp.tile(landmarks, (M, 1))
        kmm_raw = self.kernel_model.forward(x1, x2, weights)
        KMM = kmm_raw.reshape(M, M)

        reg = 1e-8 * jnp.eye(M)

        return KNM @ jnp.linalg.inv(KMM + reg) @ KNM.T
    
    def center(self, kernel_matrix):
        n = kernel_matrix.shape[0]
        H = jnp.eye(n) - jnp.ones((n, n)) / n
        return H @ kernel_matrix @ H

    def alignment(self, weights, X, y):

        if self.matrix_type == 'regular':
            kernel_matrix = self.regular_kernel_matrix(weights, X)
        elif self.matrix_type == 'nystrom':
            kernel_matrix = self.nystrom_kernel_matrix(weights, X)
        else:
            raise ValueError("Unsupported matrix type")
        
        if self.centering:
            kernel_matrix = self.center(kernel_matrix)
        
        y = y.reshape(-1, 1)
        T = y @ y.T

        kta = jnp.sum(kernel_matrix * T) / (jnp.linalg.norm(kernel_matrix, ord='fro') * jnp.linalg.norm(T, ord='fro'))

        return kta
    
    def _loss_kta(self, weights, X, y):
        return 1 - self.alignment(weights, X, y)

    def align(self):
        history = {}
        
        param_history = []
        alignment_history = []
        loss_history = []
        train_accuracy_history = []
        test_accuracy_history = []
        f1_history = []
        precision_history = []
        recall_history = []

        result = self.svm_training(self.xtrain, self.ytrain)
        init_train_accuracy = result['train_accuracy']
        init_test_accuracy = result['test_accuracy']
        init_f1 = result['f1_score']
        init_precision = result['precision_score']
        init_recall = result['recall_score']
        alignment_history.append(self.alignment(self.weights, self.xtrain, self.ytrain))
        loss_history.append(self._loss_kta(self.weights, self.xtrain, self.ytrain))

        start = time.time()

        for epoch in tqdm(range(self.epochs), desc="Aligning Kernel with Full Kernel KTA"):

            x, y = self._get_random_samples(self.xtrain, self.ytrain)

            loss = self.loss_function(self.weights, x, y)
            loss_history.append(loss)

            alignment_history.append(self.alignment(self.weights, self.xtrain, self.ytrain))
            param_history.append(self.weights)

            grads = self.grad_function(self.weights, x, y)

            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.weights = ox.apply_updates(self.weights, updates) 

            result = self.svm_training(self.xtrain, self.ytrain)
            train_accuracy_history.append(result['train_accuracy'])
            test_accuracy_history.append(result['test_accuracy'])
            f1_history.append(result['f1_score'])
            precision_history.append(result['precision_score'])
            recall_history.append(result['recall_score'])

        history['weights'] = self.weights
        history['init_train_accuracy'] = init_train_accuracy
        history['init_test_accuracy'] = init_test_accuracy
        history['alignment_history'] = alignment_history        
        history['loss_history'] = loss_history
        history['train_accuracy_history'] = train_accuracy_history
        history['test_accuracy_history'] = test_accuracy_history
        history['f1_score_history'] = f1_history
        history['precision_score_history'] = precision_history
        history['recall_score_history'] = recall_history
        history['time'] = time.time() - start
        history['circuit_executions'] = self.kernel_model.circuit_executions

        print_training_summary(history)

        return history
    
    def svm_training(self, X, y):
        result = {}

        if self.matrix_type == 'regular':
            kernel_matrix = self.regular_kernel_matrix(self.weights, X)
        elif self.matrix_type == 'nystrom':
            kernel_matrix = self.nystrom_kernel_matrix(self.weights, X)
        else:
            raise ValueError("Unsupported matrix type")

        if self.centering:
            kernel_matrix = self.center(kernel_matrix)

        svm = SVC(kernel='precomputed', C=1.0, gamma='scale', probability=True, max_iter=10000)
        svm.fit(kernel_matrix, y)

        result['svm'] = svm
        result['kernel_matrix'] = kernel_matrix
        result['y'] = y

        test_kernel_matrix = self._test_kernel_matrix(self.weights, self.xtrain, self.xtest)

        if self.centering:
            test_kernel_matrix = self.center(test_kernel_matrix)


        train_accuracy = accuracy_score(self.ytrain, svm.predict(kernel_matrix))
        test_accuracy = accuracy_score(self.ytest, svm.predict(test_kernel_matrix))

        f1 = f1_score(self.ytest, svm.predict(test_kernel_matrix))
        precision = precision_score(self.ytest, svm.predict(test_kernel_matrix))
        recall = recall_score(self.ytest, svm.predict(test_kernel_matrix))

        result['train_accuracy'] = train_accuracy
        result['test_accuracy'] = test_accuracy
        result['f1_score'] = f1
        result['precision_score'] = precision
        result['recall_score'] = recall

        return result
    

class greedyKTA:

    def __init__(self,
                 kernel_model,
                 data: jnp.ndarray,
                 labels: jnp.ndarray,
                 split_size: float = 0.8,
                 greedy_samples: int = 4,
                 matrix_type: str = 'regular',
                 matrix_normalisation: bool = False,
                 landmark_points: int = 0,
                 centering: bool = False,
                 epochs: int = 100,
                 learning_rate: float = 0.01,
                 optimizer: str = 'adam',
                 **kwargs,

    ):

        self.kernel_model = kernel_model
        self.data = data
        self.labels = labels
        self.matrix_type = matrix_type
        self.matrix_normalisation = matrix_normalisation
        self.landmark_points = landmark_points
        self.centering = centering
        self.epochs = epochs
        self.split_size = split_size
        self.greedy_samples = greedy_samples
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer.lower()
        self.optimizer = self._get_optimizer()
        self.opt_state = None
        self.rng = jax.random.PRNGKey(42)

        self.num_samples = self.data.shape[0]
        self.permutation = None
        self.perm_ptr = 0

        # --- Initialize Weights
        self.weights = self.kernel_model.circuit.init_weights()

        # --- Initialize Optimizer State
        self.opt_state = self.optimizer.init(self.weights)

        # --- Initialize Gradient Function
        self.grad_function = jax.jit(
            jax.grad(self._loss_kta)
        )

        # --- Initialize Loss Function
        self.loss_function = jax.jit(self._loss_kta)

        # --- Data Splitting
        self.xtrain = None
        self.xtest = None
        self.ytrain = None
        self.ytest = None
        self._split_data()

    def _get_optimizer(self):
        if self.optimizer_name == 'adam':
            return ox.adam(self.learning_rate)
        elif self.optimizer_name == 'sgd':
            return ox.sgd(self.learning_rate)
        else:
            raise ValueError("Unsupported optimizer")

    def _split_data(self, seed=42):
        n = len(self.data)
        rng = jax.random.PRNGKey(seed)

        perm = jax.random.permutation(rng, n)
        split = int(n * self.split_size)

        idx_train = perm[:split]
        idx_test = perm[split:]

        self.xtrain = self.data[idx_train]
        self.xtest = self.data[idx_test]
        self.ytrain = self.labels[idx_train]
        self.ytest = self.labels[idx_test]

    def _compute_uncertainty(self, svm, kernel_matrix):
        probs = svm.predict_proba(kernel_matrix)[:, 1]  # P(y=1)
        return 1.0 - jnp.abs(2.0 * probs - 1.0)
    
    def _get_greedy_samples(self, X, y, weights):
        kernel_matrix = self.nystrom_kernel_matrix(weights, X)
        svm, _ = self.svm_training(X, y) 
        # Compute uncertainity
        uncertainty = self._compute_uncertainty(svm, kernel_matrix)
        # Select top-k most uncertain samples
        k = min(self.greedy_samples, X.shape[0])
        topk = jnp.argsort(-uncertainty)[:k]

        return X[topk], y[topk] 

    def _test_kernel_matrix(self, weights, X, test_X):
        N, D = X.shape
        M, _ = test_X.shape
        x1 = jnp.repeat(test_X, N, axis=0)  
        x2 = jnp.tile(X, (M, 1)) 
        return self.kernel_model.forward(x1, x2, weights).reshape(M, N)

    def regular_kernel_matrix(self, weights, X):
        N, D = X.shape
        x1 = jnp.repeat(X, N, axis=0)  
        x2 = jnp.tile(X, (N, 1)) 
        return self.kernel_model.forward(x1, x2, weights).reshape(N, N)

    def nystrom_kernel_matrix(self, weights, X):
        if self.landmark_points > len(X) or self.landmark_points <= 0:
            raise ValueError(
                "Unacceptable number of Landmark points. "
                "Require 0 < landmark_points <= len(X)"
            )

        # --- Select landmarks
        landmarks = X[: self.landmark_points]

        N, D = X.shape
        M, _ = landmarks.shape

        x1 = jnp.repeat(X, M, axis=0)
        x2 = jnp.tile(landmarks, (N, 1))
        knm_raw = self.kernel_model.forward(x1, x2, weights)
        KNM = knm_raw.reshape(N, M)

        x1 = jnp.repeat(landmarks, M, axis=0)  
        x2 = jnp.tile(landmarks, (M, 1))
        kmm_raw = self.kernel_model.forward(x1, x2, weights)
        KMM = kmm_raw.reshape(M, M)

        reg = 1e-8 * jnp.eye(M)

        return KNM @ jnp.linalg.inv(KMM + reg) @ KNM.T
    
    def center(self, kernel_matrix):
        n = kernel_matrix.shape[0]
        H = jnp.eye(n) - jnp.ones((n, n)) / n
        return H @ kernel_matrix @ H

    def alignment(self, weights, X, y):

        if self.matrix_type == 'regular':
            kernel_matrix = self.regular_kernel_matrix(weights, X)
        elif self.matrix_type == 'nystrom':
            kernel_matrix = self.nystrom_kernel_matrix(weights, X)
        else:
            raise ValueError("Unsupported matrix type")
        
        if self.centering:
            kernel_matrix = self.center(kernel_matrix)
        
        y = y.reshape(-1, 1)
        T = y @ y.T

        kta = jnp.sum(kernel_matrix * T) / (jnp.linalg.norm(kernel_matrix, ord='fro') * jnp.linalg.norm(T, ord='fro'))

        return kta
    
    def _loss_kta(self, weights, X, y):
        return 1 - self.alignment(weights, X, y)

    def align(self):
        history = {}
        
        param_history = []
        alignment_history = []
        loss_history = []
        train_accuracy_history = []
        test_accuracy_history = []
        f1_history = []
        precision_history = []
        recall_history = []

        _, result = self.svm_training(self.xtrain, self.ytrain)
        init_train_accuracy = result['train_accuracy']
        init_test_accuracy = result['test_accuracy']
        init_f1 = result['f1_score']
        init_precision = result['precision_score']
        init_recall = result['recall_score']
        alignment_history.append(self.alignment(self.weights, self.xtrain, self.ytrain))
        loss_history.append(self._loss_kta(self.weights, self.xtrain, self.ytrain))

        start = time.time()

        for epoch in tqdm(range(self.epochs), desc="Aligning Kernel with Full Kernel KTA"):

            x, y = self._get_greedy_samples(self.xtrain, self.ytrain, self.weights)

            loss = self.loss_function(self.weights, x, y)
            loss_history.append(loss)

            alignment_history.append(self.alignment(self.weights, self.xtrain, self.ytrain))
            param_history.append(self.weights)

            grads = self.grad_function(self.weights, x, y)

            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.weights = ox.apply_updates(self.weights, updates) 

            _, result = self.svm_training(self.xtrain, self.ytrain)
            train_accuracy_history.append(result['train_accuracy'])
            test_accuracy_history.append(result['test_accuracy'])
            f1_history.append(result['f1_score'])
            precision_history.append(result['precision_score'])
            recall_history.append(result['recall_score'])

        history['weights'] = self.weights
        history['init_train_accuracy'] = init_train_accuracy
        history['init_test_accuracy'] = init_test_accuracy
        history['alignment_history'] = alignment_history        
        history['loss_history'] = loss_history
        history['train_accuracy_history'] = train_accuracy_history
        history['test_accuracy_history'] = test_accuracy_history
        history['f1_score_history'] = f1_history
        history['precision_score_history'] = precision_history
        history['recall_score_history'] = recall_history
        history['time'] = time.time() - start
        history['circuit_executions'] = self.kernel_model.circuit_executions

        print_training_summary(history)

        return history
    
    def svm_training(self, X, y):
        result = {}

        if self.matrix_type == 'regular':
            kernel_matrix = self.regular_kernel_matrix(self.weights, X)
        elif self.matrix_type == 'nystrom':
            kernel_matrix = self.nystrom_kernel_matrix(self.weights, X)
        else:
            raise ValueError("Unsupported matrix type")

        if self.centering:
            kernel_matrix = self.center(kernel_matrix)

        svm = SVC(kernel='precomputed', C=1.0, gamma='scale', probability=True, max_iter=10000)
        svm.fit(kernel_matrix, y)

        result['svm'] = svm
        result['kernel_matrix'] = kernel_matrix
        result['y'] = y

        test_kernel_matrix = self._test_kernel_matrix(self.weights, self.xtrain, self.xtest)

        if self.centering:
            test_kernel_matrix = self.center(test_kernel_matrix)


        train_accuracy = accuracy_score(self.ytrain, svm.predict(kernel_matrix))
        test_accuracy = accuracy_score(self.ytest, svm.predict(test_kernel_matrix))

        f1 = f1_score(self.ytest, svm.predict(test_kernel_matrix))
        precision = precision_score(self.ytest, svm.predict(test_kernel_matrix))
        recall = recall_score(self.ytest, svm.predict(test_kernel_matrix))

        result['train_accuracy'] = train_accuracy
        result['test_accuracy'] = test_accuracy
        result['f1_score'] = f1
        result['precision_score'] = precision
        result['recall_score'] = recall

        return svm, result


