import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

# Set consistent style and color palette
plt.style.use('seaborn-v0_8')
cmap_background = ListedColormap(["#ffcccb", "#add8e6"])
color_pos = '#ff7f0f'
color_neg = '#1f77b4'
background_color = "#eaeaf2"

def kernel_heatmap(K, path, title="Kernel Heatmap"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor(background_color)
    sns.heatmap(K, cmap='viridis', ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Samples", fontsize=12)
    ax.set_ylabel("Samples", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{path}/kernel_heatmap.png")
    plt.close()

def alignment_progress_over_iterations(alignment_scores, path, title="Alignment Progress Over Iterations"):
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
    plt.savefig(f"{path}/alignment_progress.png")
    plt.close()

def decision_boundary(model, X, y, path, title="Decision Boundary"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor(background_color)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)

    ax.scatter(X_pca[y == 1][:, 0], X_pca[y == 1][:, 1], c=color_pos, s=100, edgecolor=color_pos, alpha=0.8)
    ax.scatter(X_pca[y == -1][:, 0], X_pca[y == -1][:, 1], c=color_neg, s=100, edgecolor=color_neg, alpha=0.8)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel("PC 2", fontsize=12)
    ax.grid(axis='both', linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{path}/decision_boundary.png")
    plt.close()

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
    plt.savefig(f"{path}/execution_vs_performance.png")
    plt.close()
