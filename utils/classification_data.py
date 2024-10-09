import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time

def checkerboard_data(n_features):
    data_path = 'data/checkerboard_data_' + str(n_features) + '.csv'

    # Check if the data file already exists
    
    df = pd.read_csv(data_path)
    return df


def linear_data(n_features, n_samples):
    data_path = 'data/linear_data_' + str(n_features) + '.csv'
    if data_path in os.listdir('data/'):
        df = pd.read_csv(data_path)
        return df
    else:
        X = np.random.uniform(-500, 500, size=(n_samples, n_features))

        # Define the linear decision boundary based on the first two features
        # We will create a linear combination of the features to define the boundary
        coefficients = np.random.uniform(-10, 10, size=n_features)
        intercept = np.random.uniform(-50, 50)
        
        # Calculate the linear decision boundary
        linear_combination = np.dot(X, coefficients) + intercept
        
        # Define the target variable y as +1 or -1 based on the linear combination
        y = np.where(linear_combination > 0, 1, -1)

        # Create a DataFrame for the dataset
        linear_data = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
        linear_data['target'] = y

        # Save the DataFrame to a CSV file
        linear_data.to_csv(data_path, index=False)
	
        return linear_data

def hidden_manifold_data(n_features, n_samples):

    # Define the path for the data file
    data_path = 'data/hidden_manifold_data_' + str(n_features) + '.csv'

    # Check if the data file already exists
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df
    
    else:

        # Generate points along a hidden manifold
        t = np.random.uniform(-100, 100, n_samples)  # Parameter t for the manifold
        X = np.zeros((n_samples, n_features))

        # Map t to higher dimensions
        X[:, 0] = t  # First feature is t
        X[:, 1] = np.sin(t) + np.random.normal(0, 0.1, n_samples)  # Second feature
        X[:, 2] = np.cos(t) + np.random.normal(0, 0.1, n_samples)  # Third feature

        if n_features > 3:
            X[:, 3:] = np.random.normal(0, 1, (n_samples, n_features - 3))

        # Assign binary classes based on the parameter t
        y = (t > 0).astype(int) * 2 - 1  # Convert to -1 and 1

        # Create a DataFrame for the dataset
        hidden_manifold_data = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
        hidden_manifold_data['target'] = y

        # Save the DataFrame to a CSV file
        hidden_manifold_data.to_csv(data_path, index=False)

        return hidden_manifold_data	

def power_line_data():
    data_path = 'data/powerline.csv'
    data = pd.read_csv(data_path)
    return data

def microgrid_data():
    df = pd.read_csv('data/microgrid.csv')
    curr_cols = df.columns[1:]
    cols  = ['f2', 'f3', 'f4', 'f5', 'target']
    df = df[curr_cols]
    df.columns = cols
    return df

def ionosphere_data():
    pass


def _make_circular_data(num_sectors, points_per_sector):
    """Generate datapoints arranged in an even circle."""
    center_indices = np.repeat(np.array(range(0, num_sectors)), points_per_sector)
    sector_angle = 2 * np.pi / num_sectors
    angles = (center_indices + np.random.rand(center_indices.shape[0])) * sector_angle  # Add randomness for more points
    
    x = 0.7 * np.cos(angles)
    y = 0.7 * np.sin(angles)
    labels = 2 * np.remainder(np.floor_divide(center_indices, 1), 2) - 1

    return x, y, labels

def make_double_cake_data(num_sectors, points_per_sector=1):
    x1, y1, labels1 = _make_circular_data(num_sectors, points_per_sector)
    x2, y2, labels2 = _make_circular_data(num_sectors, points_per_sector)

    # x and y coordinates of the datapoints
    x = np.hstack([x1, 0.5 * x2])
    y = np.hstack([y1, 0.5 * y2])

    # Canonical form of dataset
    X = np.vstack([x, y]).T

    labels = np.hstack([labels1, -1 * labels2])

    # Canonical form of labels
    Y = labels.astype(int)

    return X, Y