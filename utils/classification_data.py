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

def microgrid_data():
    pass

def power_line_data():
    pass

def ionosphere_data():
    pass

