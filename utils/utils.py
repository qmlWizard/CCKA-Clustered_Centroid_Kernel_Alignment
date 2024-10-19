import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
import pandas as pd



def plot_svm_decision_boundary(svm_model, X_train, y_train, X_test, y_test, filename = 'svm_decesion_boundary.png'):
    # Create a mesh to plot the decision boundary
    h = .2  # step size in the mesh
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Plot decision boundary
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 8))
    
    # Custom colormap for decision boundary
    cmap_background = ListedColormap(['#a6cee3', '#fdbf6f'])
    plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.8)
    
    # Plot training data (filled circles)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=ListedColormap(['blue', 'orange']),
                edgecolor='k', marker='o', s=100, label='Train Data', alpha=0.9)
    
    # Plot testing data (hollow circles)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=ListedColormap(['blue', 'orange']),
                edgecolor='k', marker='o', s=100, facecolors='none', label='Test Data', alpha=0.9)
    
    # Labels and title
    plt.title('SVM Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='best')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def print_boxed_message(title, content):
    def format_item(item, i):
        if isinstance(item, np.ndarray):  # Check if the item is an array
            return f"Cluster {i+1}: {np.array2string(item, precision=2, floatmode='fixed')}"
        else:
            return f"Cluster {i+1}: {item}"  # If it's not an array, just print the item as-is
    
    # Ensure that we only calculate lengths for formatted strings (arrays and others)
    formatted_content = [format_item(item, i) for i, item in enumerate(content)]
    max_len = max(len(line) for line in formatted_content)
    box_width = max(len(title) + 4, max_len + 4)

    print(f"+{'-' * box_width}+")
    print(f"|  {title.center(box_width - 4)}  |")
    print(f"+{'-' * box_width}+")
    for line in formatted_content:
        print(f"|  {line.ljust(box_width - 4)}  |")
    print(f"+{'-' * box_width}+")