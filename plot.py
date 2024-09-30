import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

datasets = ['checkerboard', 'linear', 'powerline', 'microgrid']
results_path = 'results/'

reuslts = []
alignments = []
for d in datasets:
    csv_files = [file for file in os.listdir(results_path + d) if file.endswith('.csv')]
    npy_files = [file for file in os.listdir(results_path + d) if file.endswith('.npy')]

    for i in range(len(csv_files)):
        df = pd.read_csv(results_path + d + '/' + csv_files[i])
        reuslts.append(df)

    
    for i in range(len(npy_files)):
        np_data = np.load(results_path + d + '/' + npy_files[i],  allow_pickle=True)
        alignments.append(np_data)



df_alltg = pd.concat(reuslts)

df_alltg.to_csv('all_results.csv')