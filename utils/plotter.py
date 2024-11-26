import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import torch
import os

class Plotter:
    """
    Class Plotter:
                    Provides methods for static plots such as accuracy comparison, alignment over epochs, 
                    circuit execution comparison, decision boundary visualization, etc., using Seaborn and Matplotlib.
    """
    def __init__(self, style="whitegrid", final_color="steelblue", initial_color="salmon", plot_dir="plots"):
        """
        Initialize the Plotter class.

        Args:
            style (str): Plotting style.
            final_color (str): Color for final plots.
            initial_color (str): Color for initial plots.
            plot_dir (str): Directory to save plots.
        """
        self._fclr = final_color
        self._iclr = initial_color
        self._plot_style = style
        self._plot_dir = plot_dir
        os.makedirs(self._plot_dir, exist_ok=True)

    def compare_accuracy(self, init_train_accuracy, init_test_accuracy, final_train_accuracy, final_test_accuracy, plot_name, dataset):
        """
        Plot comparison of initial and final training and testing accuracies.

        Args:
            init_train_accuracy (float): Initial training accuracy.
            init_test_accuracy (float): Initial testing accuracy.
            final_train_accuracy (float): Final training accuracy.
            final_test_accuracy (float): Final testing accuracy.
            plot_name (str): Name of the file to save the plot.
            dataset (str): Dataset name.
        """
        plt.style.use(self._plot_style)
        fig, ax = plt.subplots(figsize=(8, 6))
        x_labels = ['Initial', 'Final']
        x_pos = range(len(x_labels))
        ax.bar(x_pos, [final_train_accuracy, final_test_accuracy], width=0.2, color=self._fclr, label='Final Accuracy')
        ax.bar([pos - 0.2 for pos in x_pos], [init_train_accuracy, init_test_accuracy], width=0.2, color=self._iclr, label='Initial Accuracy')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Accuracy: {lr}" for lr in x_labels], fontsize=10, fontweight='bold')
        ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
        ax.set_title(f"{dataset.capitalize()} Dataset: Test Accuracies", fontsize=14, fontweight='bold')
        for x, value in zip(x_pos, [final_train_accuracy, final_test_accuracy]):
            ax.text(x, value + 0.01, f'{value:.2f}', ha='center', fontsize=10)
        for x, value in zip([pos - 0.2 for pos in x_pos], [init_train_accuracy, init_test_accuracy]):
            ax.text(x, value + 0.01, f'{value:.2f}', ha='center', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self._plot_dir, plot_name), dpi=300, bbox_inches='tight')

    def plot_alignment(self, alignments, init_alignment, dataset, plot_name):
        """
        Plot alignment values over epochs.

        Args:
            alignments (list): List of alignment values for epochs.
            init_alignment (float): Initial alignment value.
            dataset (str): Dataset name.
            plot_name (str): Name of the file to save the plot.
        """
        plt.style.use(self._plot_style)
        fig, ax = plt.subplots(figsize=(8, 6))
        markers = ['o', 's', 'D', '^', 'v']
        
        # Ensure alignments are NumPy-compatible
        alignments = [init_alignment] + alignments
        alignments = [
            alignment.detach().numpy() if isinstance(alignment, torch.Tensor) else alignment
            for alignment in alignments
        ]
        
        ax.plot(
            range(len(alignments)),
            alignments,
            alpha=0.8,
            marker=markers[0],  # Cycle through markers
            markersize=6  # Adjust marker size
        )
        ax.set_xlabel("Epoch", fontsize=10, fontweight='bold')
        ax.set_ylabel("Alignment", fontsize=10, fontweight='bold')
        ax.set_title(
            f"{dataset.capitalize()} Dataset: Alignment Over Epochs",
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(self._plot_dir, plot_name), dpi=300, bbox_inches='tight')

    def decision_boundary(self, svm_model, X, y, plot_name):
        """
        Plot the decision boundary for a trained SVM model.

        Args:
            svm_model: Trained SVM model.
            X (np.ndarray): Input features (2D for visualization).
            y (np.ndarray): Labels.
            plot_name (str): Name of the file to save the plot.
        """
        plt.style.use(self._plot_style)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create a meshgrid for decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        
        # Predict over the meshgrid
        Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
        
        ax.set_title("Decision Boundary", fontsize=14, fontweight='bold')
        ax.set_xlabel("Feature 1", fontsize=10, fontweight='bold')
        ax.set_ylabel("Feature 2", fontsize=10, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label="Class")
        plt.tight_layout()
        plt.savefig(os.path.join(self._plot_dir, plot_name), dpi=300, bbox_inches='tight')
