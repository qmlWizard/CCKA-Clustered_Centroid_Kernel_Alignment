import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time

def checkerboard_data(n_features, n_samples):
    data_path = 'data/checkerboard_data_' + str(n_features) + '.csv'

    # Check if the data file already exists
    if data_path in os.listdir('data/'):
        df = pd.read_csv(data_path)
        return df
    else:
        # Initialize random data points for n_features features
        X = np.random.uniform(-10, 10, size=(n_samples, n_features))

        # Define the checkerboard pattern based on the first two features
        y = np.floor(X[:, 0]) + np.floor(X[:, 1])
        y = (y % 2 == 0).astype(int)  # Convert to binary classes (0 and 1)
        y = 2 * y - 1  # Convert 0 to -1 and 1 remains 1

        # Create a DataFrame for the dataset
        checkerboard_data = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
        checkerboard_data['target'] = y

        # Save the DataFrame to a CSV file
        checkerboard_data.to_csv(data_path, index=False)

        return checkerboard_data

def linear_data(n_features, n_samples):
    data_path = 'data/linear_data_' + str(n_features) + '.csv'
    if data_path in os.listdir('data/'):
        df = pd.read_csv(data_path)
        return df
    else:
        X = np.random.uniform(-10, 10, size=(n_samples, n_features))

        # Define the linear decision boundary based on the first two features
        # We will create a linear combination of the features to define the boundary
        coefficients = np.random.uniform(-1, 1, size=n_features)
        intercept = np.random.uniform(-5, 5)
        
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
        t = np.random.uniform(-10, 10, n_samples)  # Parameter t for the manifold
        X = np.zeros((n_samples, n_features))

        # Map t to higher dimensions
        X[:, 0] = t  # First feature is t
        X[:, 1] = np.sin(t) + np.random.normal(0, 0.1, n_samples)  # Second feature
        X[:, 2] = np.cos(t) + np.random.normal(0, 0.1, n_samples)  # Third feature
        # Fill remaining features with random noise
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
    pass

def ionosphere_data():
    pass
