import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_moons, make_swiss_roll, make_gaussian_quantiles, load_iris, fetch_openml, load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from sklearn.decomposition import PCA


class DataGenerator:

    def __init__(self, 
                 dataset_name=None, 
                 file_path=None, 
                 n_samples=1000, 
                 noise=0.1, 
                 num_sectors=3, 
                 points_per_sector=10,
                 grid_size=4, 
                 sampling_radius=0.05, 
                 n_pca_features=None,
                 test_size=0.2
        ):
        
        self.dataset_name = dataset_name
        self.file_path = file_path
        self.n_samples = n_samples
        self.noise = noise
        self.num_sectors = num_sectors
        self.points_per_sector = points_per_sector
        self.grid_size = grid_size
        self.sampling_radius = sampling_radius
        self.n_pca_features = n_pca_features
        self.dmin, self.dmax = 0, 1
        self._test_size = test_size

    from sklearn.datasets import load_wine  # Add this import if not already present

    def generate_dataset(self):
        if self.file_path:
            return self.load_from_file()
        elif self.dataset_name == 'moons':
            X, y = make_moons(n_samples=self.n_samples, noise=0.1, random_state=42)
            y = np.where(y == 0, -1, 1)
        elif self.dataset_name == 'xor':
            X, y = self.create_xor()
        elif self.dataset_name == 'swiss_roll':
            X, y = make_swiss_roll(n_samples=self.n_samples, noise=self.noise, random_state=0)
            X = np.hstack((X, np.random.randn(self.n_samples, 2)))
            y = np.where(y > np.median(y), 1, -1)
        elif self.dataset_name == 'gaussian':
            X, y = make_gaussian_quantiles(n_samples=self.n_samples, n_features=2, n_classes=2, random_state=0)
            y = np.where(y == 0, -1, 1)
        elif self.dataset_name == 'double_cake':
            X, y = self.make_double_cake_data()
        elif self.dataset_name == 'iris':
            iris = load_iris()
            X = iris.data
            y = iris.target + 1
        elif self.dataset_name == 'wine':
            wine = load_wine()
            X = wine.data
            y = wine.target + 1  # Shift to 1, 2, 3 to match your convention (if needed)
        elif self.dataset_name == 'mnist_fashion':
            X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True)
            y = y.astype(int)
            binary_classes = [1, 6]
            mask = np.isin(y, binary_classes)
            X = X[mask]
            y = y[mask]
            y = np.where(y == 1, 1, -1)
            X_class_1 = X[y == 1]
            X_class_neg_1 = X[y == -1]
            y_class_1 = y[y == 1]
            y_class_neg_1 = y[y == -1]
            samples_per_class = min(len(X_class_1), len(X_class_neg_1), self.n_samples // 2)
            indices_1 = np.random.choice(len(X_class_1), samples_per_class, replace=False)
            indices_neg_1 = np.random.choice(len(X_class_neg_1), samples_per_class, replace=False)
            X_class_1 = X_class_1[indices_1]
            y_class_1 = y_class_1[indices_1]
            X_class_neg_1 = X_class_neg_1[indices_neg_1]
            y_class_neg_1 = y_class_neg_1[indices_neg_1]
            X = np.vstack((X_class_1, X_class_neg_1))
            y = np.hstack((y_class_1, y_class_neg_1))
        elif self.dataset_name == 'checkerboard':
            X, y = self.create_checkerboard_data()
        else:
            raise ValueError(
                "Dataset not supported. Choose from 'moons', 'xor', 'swiss_roll', 'gaussian', 'double_cake', 'iris', 'wine', 'mnist_fashion', 'checkerboard'.")

        # Apply MinMax scaling to the range [-π, π]
        scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        X_scaled = scaler.fit_transform(X)

        # Apply PCA if specified
        if self.n_pca_features:
            X_scaled = self.apply_pca(X_scaled)

        x_train_scaled, x_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        return (
            pd.DataFrame(x_train_scaled, columns=[f'Feature {i + 1}' for i in range(x_train_scaled.shape[1])]),
            pd.Series(y_train, name='Label'),
            pd.DataFrame(x_test_scaled, columns=[f'Feature {i + 1}' for i in range(x_test_scaled.shape[1])]),
            pd.Series(y_test, name='Label')
        )

    def apply_pca(self, X):
        """Apply PCA to reduce features."""
        if not self.n_pca_features:
            raise ValueError("Number of PCA features not specified.")
        pca = PCA(n_components=self.n_pca_features)
        X_pca = pca.fit_transform(X)
        return X_pca

    def load_from_file(self):
        """Load a dataset from a file and return a merged pandas DataFrame and Series."""
        data = np.load(self.file_path, allow_pickle=True).item()
        if 'checkerboard' in self.file_path or 'corners' in self.file_path or 'adult' in self.file_path or 'covtype' in self.file_path or 'donuts' in self.file_path \
                or 'one_vs_nonone' in self.file_path or 'zero_vs_nonzero' in self.file_path:
            x_train, x_test = data['x_train'], data['x_test']
            y_train, y_test = data['y_train'], data['y_test']

            # Apply Min-Max Scaling to the range [0, π]
            scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)

            return (
                pd.DataFrame(x_train_scaled, columns=[f'Feature {i+1}' for i in range(x_train_scaled.shape[1])]),
                pd.Series(y_train, name='Label'),
                pd.DataFrame(x_test_scaled, columns=[f'Feature {i+1}' for i in range(x_test_scaled.shape[1])]),
                pd.Series(y_test, name='Label')
            )

        else:
            x = data['features']
            y = data['labels']
            # Apply Min-Max Scaling to the range [0, π]
            scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
            X_scaled = scaler.fit_transform(x)
            
            # Apply PCA if specified
            if self.n_pca_features:
                X_scaled = self.apply_pca(X_scaled)

            x_train_scaled, x_test_scaled, y_train, y_test = train_test_split(X, y, test_size=self._test_size, random_state=42)

            return (
                    pd.DataFrame(x_train_scaled, columns=[f'Feature {i+1}' for i in range(x_train_scaled.shape[1])]),
                    pd.Series(y_train, name='Label'),
                    pd.DataFrame(x_test_scaled, columns=[f'Feature {i+1}' for i in range(x_test_scaled.shape[1])]),
                    pd.Series(y_test, name='Label')
                )

    
    def create_xor(self):
        np.random.seed(0)
        X = np.random.rand(self.n_samples, 2) * 2 - 1
        y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)
        X += self.noise * np.random.randn(self.n_samples, 2)
        y = np.where(y == 0, -1, 1)  # Replace 0 with -1
        return X, y

    def _make_circular_data(self):
        """Generate datapoints arranged in an even circle."""
        center_indices = np.repeat(np.array(range(0, self.num_sectors)), self.points_per_sector)
        sector_angle = 2 * np.pi / self.num_sectors
        angles = (center_indices + np.random.rand(center_indices.shape[0])) * sector_angle  # Add randomness for more points
        
        x = 0.7 * np.cos(angles)
        y = 0.7 * np.sin(angles)
        labels = 2 * np.remainder(np.floor_divide(center_indices, 1), 2) - 1

        return x, y, labels

    def make_double_cake_data(self):
        x1, y1, labels1 = self._make_circular_data()
        x2, y2, labels2 = self._make_circular_data()

        # x and y coordinates of the datapoints
        x = np.hstack([x1, 0.5 * x2])
        y = np.hstack([y1, 0.5 * y2])

        # Canonical form of dataset
        X = np.vstack([x, y]).T

        labels = np.hstack([labels1, -1 * labels2])

        # Canonical form of labels
        Y = labels.astype(int)

        return X, Y

    def create_checkerboard_data(self):
        cords = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = (2 * i + 1) / (2 * self.grid_size)
                y = (2 * j + 1) / (2 * self.grid_size)
                cords.append((x, y))

        points = []
        labels = []
        cluster = 0
        for (cx, cy) in cords:
            label = 1 if cluster else -1
            for _ in range(random.randint(1, 10)):
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, self.sampling_radius)
                x = cx + radius * np.cos(angle)
                y = cy + radius * np.sin(angle)
                # Ensure points stay within the domain boundaries
                x = np.clip(x, self.dmin, self.dmax)
                y = np.clip(y, self.dmin, self.dmax)
                points.append((x, y))
                labels.append(label)
            cluster = 1 - cluster

        X = np.array(points)
        y = np.array(labels)
        return X, y



    def _to_numpy(self, x):
        # Accept torch, pandas, numpy
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        import pandas as pd
        if isinstance(x, (pd.DataFrame, pd.Series)):
            return x.to_numpy()
        return np.asarray(x)

    def plot_dataset(self, train_features, train_labels, test_features, test_labels, classifier=None):
        # --- Normalize types ---
        Xtr = self._to_numpy(train_features)
        Xte = self._to_numpy(test_features)
        ytr = self._to_numpy(train_labels).reshape(-1)
        yte = self._to_numpy(test_labels).reshape(-1)

        # Ensure 2D feature arrays
        if Xtr.ndim == 1: Xtr = Xtr.reshape(-1, 1)
        if Xte.ndim == 1: Xte = Xte.reshape(-1, 1)

        assert Xtr.shape[1] == Xte.shape[1], "Train/Test feature dims differ"
        d = Xtr.shape[1]

        # Style
        fig = None
        plt.style.use('seaborn-v0_8')

        # Colors for labels +1 / -1 (fallback otherwise)
        pos_color = '#ff7f0f'
        neg_color = '#1f77b4'

        # --- 1D ---
        if d == 1:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.set_facecolor("#eaeaf2")

            # Light vertical jitter so points don’t overlap
            rng = np.random.default_rng(42)
            jitter_tr = (rng.random(len(ytr)) - 0.5) * 0.06
            jitter_te = (rng.random(len(yte)) - 0.5) * 0.06

            # Train
            ax.scatter(Xtr[ytr == 1, 0], 0.2 + jitter_tr[ytr == 1],
                    c=pos_color, s=60, edgecolor=pos_color, alpha=0.9, label='train +1')
            ax.scatter(Xtr[ytr == -1, 0], -0.2 + jitter_tr[ytr == -1],
                    c=neg_color, s=60, edgecolor=neg_color, alpha=0.9, label='train -1')

            # Test (hollow)
            ax.scatter(Xte[yte == 1, 0], 0.2 + jitter_te[yte == 1],
                    facecolors='none', edgecolor=pos_color, s=70, linewidth=1.5, label='test +1')
            ax.scatter(Xte[yte == -1, 0], -0.2 + jitter_te[yte == -1],
                    facecolors='none', edgecolor=neg_color, s=70, linewidth=1.5, label='test -1')

            ax.set_yticks([-0.2, 0.2])
            ax.set_yticklabels(['−1', '+1'])
            ax.set_xlabel('Feature 1')
            ax.set_title(f'{self.dataset_name} Dataset (1D)', fontsize=13, fontweight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.legend(loc='upper right', frameon=False)

        # --- 2D ---
        elif d == 2:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_facecolor("#eaeaf2")

            # Train solid
            ax.scatter(Xtr[ytr == 1][:, 0], Xtr[ytr == 1][:, 1],
                    c=pos_color, s=160, edgecolor=pos_color, alpha=0.8, label='train +1')
            ax.scatter(Xtr[ytr == -1][:, 0], Xtr[ytr == -1][:, 1],
                    c=neg_color, s=160, edgecolor=neg_color, alpha=0.8, label='train -1')

            # Test hollow
            ax.scatter(Xte[yte == 1][:, 0], Xte[yte == 1][:, 1],
                    edgecolor=pos_color, facecolors='none', s=160, alpha=1, linewidth=1.5, label='test +1')
            ax.scatter(Xte[yte == -1][:, 0], Xte[yte == -1][:, 1],
                    edgecolor=neg_color, facecolors='none', s=160, alpha=1, linewidth=1.5, label='test -1')

            # Optional decision boundary for classifiers with predict/predict_proba/decision_function
            if classifier is not None and callable(getattr(classifier, "predict", None)):
                # Build a grid over the joint train+test span
                allX = np.vstack([Xtr, Xte])
                x_min, x_max = allX[:, 0].min() - 0.5, allX[:, 0].max() + 0.5
                y_min, y_max = allX[:, 1].min() - 0.5, allX[:, 1].max() + 0.5
                xx, yy = np.meshgrid(
                    np.linspace(x_min, x_max, 300),
                    np.linspace(y_min, y_max, 300)
                )
                grid = np.c_[xx.ravel(), yy.ravel()]
                # Prefer decision_function > predict_proba > predict
                if callable(getattr(classifier, "decision_function", None)):
                    zz = classifier.decision_function(grid)
                    levels = [0.0]
                elif callable(getattr(classifier, "predict_proba", None)):
                    proba = classifier.predict_proba(grid)
                    # Two-class: take prob of positive class
                    zz = proba[:, 1] - 0.5
                    levels = [0.0]
                else:
                    pred = classifier.predict(grid)
                    zz = pred.reshape(-1)
                    levels = [0.0]
                zz = zz.reshape(xx.shape)
                ax.contour(xx, yy, zz, levels=levels, linewidths=1.2, linestyles='--')

            ax.set_xlabel('Feature 1', fontsize=12)
            ax.set_ylabel('Feature 2', fontsize=12)
            ax.set_title(f'{self.dataset_name} Dataset (2D)', fontsize=14, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.legend(loc='best', frameon=False)

        # --- 3D ---
        elif d == 3:
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor("#eaeaf2")

            # Train solid
            ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], Xtr[ytr == 1, 2],
                    s=60, edgecolor=pos_color, c=pos_color, alpha=0.85, label='train +1')
            ax.scatter(Xtr[ytr == -1, 0], Xtr[ytr == -1, 1], Xtr[ytr == -1, 2],
                    s=60, edgecolor=neg_color, c=neg_color, alpha=0.85, label='train -1')

            # Test hollow
            ax.scatter(Xte[yte == 1, 0], Xte[yte == 1, 1], Xte[yte == 1, 2],
                    s=70, edgecolor=pos_color, facecolors='none', linewidth=1.2, label='test +1')
            ax.scatter(Xte[yte == -1, 0], Xte[yte == -1, 1], Xte[yte == -1, 2],
                    s=70, edgecolor=neg_color, facecolors='none', linewidth=1.2, label='test -1')

            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')
            ax.set_title(f'{self.dataset_name} Dataset (3D)', fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', frameon=False)

        # --- >3D → PCA(2) ---
        else:
            # Fit PCA on combined data so train/test share the same projection
            X_all = np.vstack([Xtr, Xte])
            pca = PCA(n_components=2, random_state=42)
            X_all_2d = pca.fit_transform(X_all)
            Xtr_2d = X_all_2d[:len(Xtr)]
            Xte_2d = X_all_2d[len(Xtr):]

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_facecolor("#eaeaf2")

            ax.scatter(Xtr_2d[ytr == 1, 0], Xtr_2d[ytr == 1, 1],
                    c=pos_color, s=160, edgecolor=pos_color, alpha=0.8, label='train +1')
            ax.scatter(Xtr_2d[ytr == -1, 0], Xtr_2d[ytr == -1, 1],
                    c=neg_color, s=160, edgecolor=neg_color, alpha=0.8, label='train -1')

            ax.scatter(Xte_2d[yte == 1, 0], Xte_2d[yte == 1, 1],
                    edgecolor=pos_color, facecolors='none', s=160, alpha=1, linewidth=1.5, label='test +1')
            ax.scatter(Xte_2d[yte == -1, 0], Xte_2d[yte == -1, 1],
                    edgecolor=neg_color, facecolors='none', s=160, alpha=1, linewidth=1.5, label='test -1')

            ax.set_xlabel('PCA1')
            ax.set_ylabel('PCA2')
            ax.set_title(f'{self.dataset_name} Dataset (>3D → PCA2)', fontsize=14, fontweight='bold')
            ax.grid(axis='both', linestyle='--', alpha=0.6)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.legend(loc='best', frameon=False)

        plt.tight_layout()
        return fig



