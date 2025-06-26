import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor
import os
import torch
import pandas as pd

# Set consistent style and color palette
plt.style.use('seaborn-v0_8')
cmap_background = ListedColormap(["#ffcccb", "#add8e6"])
color_pos = '#f6932a'
color_neg = '#3cb0ea'
background_color = "#eaeaf2"


def to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else x

def kernel_heatmap(K, path, title="Kernel Heatmap"):
    """Plot a kernel matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor(background_color)
    sns.heatmap(K, cmap='viridis', ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Samples", fontsize=12)
    ax.set_ylabel("Samples", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{path}/{title}.png")
    plt.close(fig)

def alignment_progress_over_iterations(alignment_scores, path, title="Alignment Progress Over Iterations"):
    
    if torch.is_tensor(alignment_scores):
        alignment_scores = alignment_scores.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor(background_color)
    ax.plot(alignment_scores, marker='o', color=color_pos)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Alignment Score", fontsize=12)
    ax.grid(axis='both', linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{path}/{title}.png")
    plt.close(fig)

def execution_reduction_vs_performance_trade_off(executions, performances, path, title="Execution Reduction vs Performance Trade-off"):
    full_exec = max(executions)
    reduction = [(full_exec - e) / full_exec * 100 for e in executions]
    performance_retention = [p / performances[0] * 100 for p in performances]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor(background_color)
    ax.plot(reduction, performance_retention, 'o-', color=color_neg, markersize=8)
    for i in range(len(reduction)):
        ax.text(reduction[i]+1, performance_retention[i], f"Run {i+1}", fontsize=9)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Execution Reduction (%)", fontsize=12)
    ax.set_ylabel("Performance Retention (%)", fontsize=12)
    ax.grid(axis='both', linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{path}/{title}.png")
    plt.close(fig)


def plot_initial_final_accuracies(initial_train_acc, initial_test_acc,
                                  final_train_acc, final_test_acc,
                                  path, title="Initial vs Final Accuracies"):
    """
    Plot initial and final training/testing accuracy in a grouped bar chart.
    """
    os.makedirs(path, exist_ok=True)

    labels = ['Training Accuracy', 'Testing Accuracy']
    initial_scores = [initial_train_acc * 100, initial_test_acc * 100]
    final_scores = [final_train_acc * 100, final_test_acc * 100]

    x = np.arange(len(labels))  # Label locations
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor(background_color)
    bars1 = ax.bar(x - width / 2, initial_scores, width, label='Initial', color=color_neg)
    bars2 = ax.bar(x + width / 2, final_scores, width, label='Final', color=color_pos)

    # Add labels and titles
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend()
    ax.set_ylim(0, 100)

    # Add values on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Offset above bar
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{path}/{title}.png")
    plt.close(fig)

def decision_boundary(model, training_data, training_labels, test_data, test_labels,
                      kernel, compute_test_kernel_row, path, title="decision_boundary"):

    training_data = to_numpy(training_data)
    training_labels = to_numpy(training_labels)
    test_data = to_numpy(test_data)
    test_labels = to_numpy(test_labels)

    # Create a high-res mesh grid for decision surface
    x_min, x_max = np.vstack([training_data, test_data])[:, 0].min() - 1, np.vstack([training_data, test_data])[:, 0].max() + 1
    y_min, y_max = np.vstack([training_data, test_data])[:, 1].min() - 1, np.vstack([training_data, test_data])[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Compute kernel values for mesh grid (as test points vs training)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_test_kernel_row, kernel, grid_points, training_data, i)
                   for i in range(len(grid_points))]
        kernel_rows = [f.result() for f in futures]

    kernel_matrix = np.stack(kernel_rows)
    Z = model.predict(kernel_matrix).reshape(xx.shape)

    # Compute predictions on test set
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_test_kernel_row, kernel, test_data, training_data, i)
                   for i in range(test_data.shape[0])]
        test_rows = [f.result() for f in futures]

    test_kernel_matrix = np.stack(test_rows)
    test_predictions = model.predict(test_kernel_matrix)

    decision_table = pd.DataFrame({
        "x": test_data[:, 0],
        "y": test_data[:, 1],
        "true_label": test_labels,
        "predicted_label": test_predictions
    })

    # Plot decision boundary with soft contours
    fig, ax = plt.subplots(figsize=(6, 6))

    # Background with soft contourf
    contour = ax.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=["#eba152", "#7ec7ec"], alpha=0.4)

    # Training points
    ax.scatter(training_data[training_labels == 1][:, 0], training_data[training_labels == 1][:, 1],
               c="#3cb0ea", s=60, label="Train +1")
    ax.scatter(training_data[training_labels == -1][:, 0], training_data[training_labels == -1][:, 1],
               c="#f6932a", s=60, label="Train -1")

    # Test points with outlined circle markers (no fill)
    ax.scatter(test_data[test_labels == 1][:, 0], test_data[test_labels == 1][:, 1],
            facecolors='none', edgecolors='#3cb0ea', s=60, linewidths=1.2, marker='o', label="Test +1")

    ax.scatter(test_data[test_labels == -1][:, 0], test_data[test_labels == -1][:, 1],
            facecolors='none', edgecolors="#f6932a", s=60, linewidths=1.2, marker='o', label="Test -1")


    # Style adjustments to mimic paper
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, fontsize=8, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=8, frameon=False)

    os.makedirs(path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{path}/{title}.png", dpi=800, bbox_inches='tight')
    plt.close()

    return decision_table


def decision_boundary_pennylane(model, training_data, training_labels, test_data, test_labels,
                      kernel_fn, path, title="decision_boundary"):

    # Convert tensors to numpy arrays for plotting and model prediction
    if torch.is_tensor(training_data):
        training_data = training_data.detach()
    if torch.is_tensor(training_labels):
        training_labels = training_labels.detach()
    if torch.is_tensor(test_data):
        test_data = test_data.detach()
    if torch.is_tensor(test_labels):
        test_labels = test_labels.detach()

    training_data_np = training_data
    training_labels_np = training_labels
    test_data_np = test_data
    test_labels_np = test_labels

    # Create mesh grid
    x_min, x_max = np.vstack([training_data_np, test_data_np])[:, 0].min() - 1, np.vstack([training_data_np, test_data_np])[:, 0].max() + 1
    y_min, y_max = np.vstack([training_data_np, test_data_np])[:, 1].min() - 1, np.vstack([training_data_np, test_data_np])[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=training_data.dtype)

    # Compute kernel between mesh points and training data
    x_0 = grid_tensor.repeat_interleave(training_data.shape[0], dim=0)
    x_1 = training_data.repeat(grid_tensor.shape[0], 1)
    kernel_matrix = kernel_fn(x_0, x_1).to(torch.float32).reshape(grid_tensor.shape[0], training_data.shape[0])
    kernel_matrix_np = kernel_matrix.detach().numpy()

    # Predict using SVC model trained on precomputed kernel
    Z = model.predict(kernel_matrix_np).reshape(xx.shape)

    # Compute test predictions for overlay
    x_0_test = test_data.repeat_interleave(training_data.shape[0], dim=0)
    x_1_test = training_data.repeat(test_data.shape[0], 1)
    test_kernel_matrix = kernel_fn(x_0_test, x_1_test).to(torch.float32).reshape(test_data.shape[0], training_data.shape[0])
    test_kernel_matrix_np = test_kernel_matrix.detach().numpy()
    test_predictions = model.predict(test_kernel_matrix_np)

    decision_table = pd.DataFrame({
        "x": test_data_np[:, 0],
        "y": test_data_np[:, 1],
        "true_label": test_labels_np,
        "predicted_label": test_predictions
    })

    # Plot decision surface and points
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=["#eba152", "#7ec7ec"], alpha=0.4)

    ax.scatter(training_data_np[training_labels_np == 1][:, 0], training_data_np[training_labels_np == 1][:, 1],
               c="#3cb0ea", s=60, label="Train +1")
    ax.scatter(training_data_np[training_labels_np == -1][:, 0], training_data_np[training_labels_np == -1][:, 1],
               c="#f6932a", s=60, label="Train -1")

    ax.scatter(test_data_np[test_labels_np == 1][:, 0], test_data_np[test_labels_np == 1][:, 1],
               facecolors='none', edgecolors='#3cb0ea', s=60, linewidths=1.2, marker='o', label="Test +1")
    ax.scatter(test_data_np[test_labels_np == -1][:, 0], test_data_np[test_labels_np == -1][:, 1],
               facecolors='none', edgecolors="#f6932a", s=60, linewidths=1.2, marker='o', label="Test -1")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, fontsize=8, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=8, frameon=False)

    os.makedirs(path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{path}/{title}.png", dpi=800, bbox_inches='tight')
    plt.close()

    return decision_table