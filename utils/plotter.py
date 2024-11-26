import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import json
import pennylane
import torch
import sklearn
import time
import os

class Plotter():
    """
    Class Plotter:
                    accuracy comparisom, plot circuit executions, plot gradients, plot error line chart, plot decesision boundaries
                    use seabornv8 style, steal blue and salmon orange.
    """
    def __init__(self, style, final_color, initial_color, plot_dir):
        
        self._fclr = final_color
        self._iclr = initial_color
        self._plot_style = style
        self._plot_dir = plot_dir

    #Static Plots
    def compare_accuracy(self, init_train_accuracy, init_test_accuracy, final_train_accuracy, final_test_accuracy, plot_name, dataset):
        plt.style.use(self._plot_style)
        fig, ax = plt.subplots(figsize=(8, 6))
        x_labels = ['Initial', 'Final']
        x_pos = range(len(x_labels))
        ax.bar(x_pos, [final_train_accuracy ,final_test_accuracy], width=0.2, color=self._fclr, label= 'Final Accuracy')
        ax.bar([pos - 0.2 for pos in x_pos], [init_train_accuracy, init_test_accuracy], width=0.2, color= self._iclr, label='Initial Accuracy')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Accuracy: {lr}" for lr in x_labels], fontsize=10, fontweight='bold')
        ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
        ax.set_title(f"{dataset.capitalize()} Dataset: Test Accuracies", fontsize=14, fontweight='bold')
        for x, value in zip(x_pos, [final_train_accuracy ,final_test_accuracy]):
            ax.text(x, value + 0.01, f'{value:.2f}', ha='center', fontsize=10)
        for x, value in zip([pos - 0.2 for pos in x_pos],  [init_train_accuracy, init_test_accuracy]):
            ax.text(x, value + 0.01, f'{value:.2f}', ha='center', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self._plot_dir, plot_name), dpi=300, bbox_inches='tight')

    def plot_alignment(self, alignments, init_alignment, dataset, plot_name):
        plt.style.use(self._plot_style)
        fig, ax = plt.subplots(figsize=(8, 6))
        markers = ['o', 's', 'D', '^', 'v']
        alignments = [init_alignment] + alignments
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
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self._plot_dir, plot_name), dpi=300, bbox_inches='tight')

    def decesion_boundary(self, svm_model, plot_name):
        pass


    #Comparision Plots

    def compare_accuracies():
        pass

    def compare_circuit_exec():
        #plot in log axis
        pass

