import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# Specify datasets and results path
datasets = ['checkerboard', 'linear', 'microgrid']
results_path = 'results/'

reuslts = []
alignments = []

# Function to recursively extract all numeric values from a nested structure
def extract_numeric_values(data):
    numeric_values = []
    # Handle 0-dimensional arrays (scalar-like)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        if np.issubdtype(data.dtype, np.number):
            numeric_values.append(data.item())  # Extract the single numeric value
    elif isinstance(data, dict):
        for key, value in data.items():
            numeric_values.extend(extract_numeric_values(value))
    elif isinstance(data, (list, np.ndarray)):
        for item in data:
            numeric_values.extend(extract_numeric_values(item))
    elif isinstance(data, (int, float)):
        numeric_values.append(data)
    return numeric_values

# Iterate over datasets to load CSV and NPY files
for d in datasets:
    csv_files = [file for file in os.listdir(results_path + d) if file.endswith('.csv')]
    npy_files = [file for file in os.listdir(results_path + d) if file.endswith('.npy')]

    # Append CSV data to results list
    for csv_file in csv_files:
        # Extract algorithm, subset, and ansatz from the filename
        file_parts = csv_file.replace('.csv', '').split('_')
        algorithm = file_parts[0]
        
        # Find the numeric subset using regex
        subset_match = re.search(r'_(\d+)_', csv_file)
        if subset_match:
            subset = int(subset_match.group(1))
        else:
            print(f"Skipping file with unexpected format: {csv_file}")
            continue

        # Extract ansatz by joining parts between subset and dataset
        ansatz = '_'.join(file_parts[2:-1])  # Assuming ansatz is the part between subset and dataset

        df = pd.read_csv(results_path + d + '/' + csv_file)
        df['algorithm'] = algorithm
        df['subset'] = subset
        df['ansatz'] = ansatz
        df['dataset'] = d
        
        reuslts.append(df)
    
    # Append NPY data to alignments list
    for npy_file in npy_files:
        np_data = np.load(results_path + d + '/' + npy_file, allow_pickle=True)
        alignments.append({
            'file_name': npy_file,
            'dataset': d,
            'alignment_data': np_data
        })

# Concatenate all CSV results into a single DataFrame
df_alltg = pd.concat(reuslts)

# Save concatenated DataFrame as a CSV
df_alltg.to_csv('all_results.csv')

# Select relevant columns for analysis, including ansatz
df_cleaned = df_alltg[['algorithm', 'subset', 'dataset', 'ansatz', 'Training Accuracy', 'Testing Accuracy', 'executions']]

# Extract unique algorithms from the cleaned DataFrame for later use
algorithms = df_cleaned['algorithm'].unique()

# Plotting 1: Average Training and Testing Accuracy by Algorithm
accuracy_by_algorithm = df_cleaned.groupby('algorithm')[['Training Accuracy', 'Testing Accuracy']].mean()

plt.figure(figsize=(10, 6))
accuracy_by_algorithm.plot(kind='bar')
plt.title('Average Training and Testing Accuracy by Algorithm')
plt.ylabel('Accuracy')
plt.xlabel('Algorithm')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.savefig('Average_Accuracy_by_Algorithm.png')
plt.close()

# Plotting 2: Distribution of Executions by Algorithm
plt.figure(figsize=(10, 6))
df_cleaned.boxplot(column='executions', by='algorithm')
plt.title('Distribution of Executions by Algorithm')
plt.ylabel('Executions')
plt.xlabel('Algorithm')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.savefig('Executions_Distribution_by_Algorithm.png')
plt.close()

# Plotting 3: Training and Testing Accuracy across Different Datasets
accuracy_by_dataset = df_cleaned.groupby('dataset')[['Training Accuracy', 'Testing Accuracy']].mean()

plt.figure(figsize=(10, 6))
accuracy_by_dataset.plot(kind='bar')
plt.title('Average Training and Testing Accuracy by Dataset')
plt.ylabel('Accuracy')
plt.xlabel('Dataset')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.savefig('Average_Accuracy_by_Dataset.png')
plt.close()

# Plotting 4: Accuracy per Ansatz, Dataset, and Subset
subset_ansatz_analysis = df_cleaned.groupby(['ansatz', 'dataset', 'subset']).agg(
    avg_training_accuracy=('Training Accuracy', 'mean'),
    avg_testing_accuracy=('Testing Accuracy', 'mean')
).reset_index()

# Create directory to save the figures
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# Plotting accuracy per subset and ansatz for each dataset
for dataset in datasets:
    data = subset_ansatz_analysis[subset_ansatz_analysis['dataset'] == dataset]
    
    if not data.empty:
        # Plot Training Accuracy per Ansatz
        plt.figure(figsize=(10, 6))
        
        for ansatz in data['ansatz'].unique():
            subset_data = data[data['ansatz'] == ansatz]
            plt.plot(subset_data['subset'], subset_data['avg_training_accuracy'], marker='o', label=f'{ansatz}')
        
        plt.title(f'Training Accuracy per Subset for Dataset: {dataset}')
        plt.xlabel('Subset')
        plt.ylabel('Training Accuracy')
        plt.legend()
        plt.grid(axis='y')
        
        # Save training accuracy figure to file
        file_name = f"{dataset}_training_accuracy_by_ansatz.png"
        file_path = os.path.join(output_dir, file_name)
        plt.savefig(file_path)
        plt.close()
        
        # Plot Testing Accuracy per Ansatz
        plt.figure(figsize=(10, 6))
        
        for ansatz in data['ansatz'].unique():
            subset_data = data[data['ansatz'] == ansatz]
            plt.plot(subset_data['subset'], subset_data['avg_testing_accuracy'], marker='x', label=f'{ansatz}')
        
        plt.title(f'Testing Accuracy per Subset for Dataset: {dataset}')
        plt.xlabel('Subset')
        plt.ylabel('Testing Accuracy')
        plt.legend()
        plt.grid(axis='y')
        
        # Save testing accuracy figure to file
        file_name = f"{dataset}_testing_accuracy_by_ansatz.png"
        file_path = os.path.join(output_dir, file_name)
        plt.savefig(file_path)
        plt.close()

# Parse alignment data and plot by algorithm, ansatz, and subset
for dataset in datasets:
    alignment_data = [a for a in alignments if a['dataset'] == dataset]
    
    if alignment_data:
        plt.figure(figsize=(10, 6))
        
        for algorithm in algorithms:
            # Extract subset and alignment information
            subset_data = [a for a in alignment_data if a['file_name'].startswith(algorithm)]
            
            if subset_data:
                subsets = []
                alignment_means = []

                for a in subset_data:
                    # Use regex to extract numeric subset if possible
                    match = re.search(r'_(\d+)_', a['file_name'])
                    if match:
                        subset = int(match.group(1))
                        
                        # Extract numeric values from alignment_data using the recursive function
                        numeric_values = extract_numeric_values(a['alignment_data'])
                        
                        # Calculate mean of all numeric values if they exist
                        if numeric_values:
                            alignment_means.append(np.mean(numeric_values))
                            subsets.append(subset)
                    else:
                        print(f"Skipping file with unexpected format: {a['file_name']}")
                
                # Plot alignment for each algorithm
                plt.plot(subsets, alignment_means, marker='o', label=f'{algorithm}')
        
        plt.title(f'Alignment per Subset for Dataset: {dataset}')
        plt.xlabel('Subset')
        plt.ylabel('Alignment')
        plt.legend()
        plt.grid(axis='y')
        
        # Save alignment figure to file
        file_name = f"{dataset}_alignment.png"
        file_path = os.path.join(output_dir, file_name)
        plt.savefig(file_path)
        plt.close()


# Calculate the mean accuracy per dataset, subset, and ansatz
ansatz_accuracy = df_cleaned.groupby(['dataset', 'subset', 'ansatz']).agg(
    avg_training_accuracy=('Training Accuracy', 'mean'),
    avg_testing_accuracy=('Testing Accuracy', 'mean')
).reset_index()

# Generate bar charts for each subset within each dataset
for dataset in df_cleaned['dataset'].unique():
    data_by_dataset = ansatz_accuracy[ansatz_accuracy['dataset'] == dataset]
    
    # Generate separate plots for each subset
    for subset in data_by_dataset['subset'].unique():
        data = data_by_dataset[data_by_dataset['subset'] == subset]

        # Plot training and testing accuracy for each ansatz within the dataset and subset
        plt.figure(figsize=(10, 6))
        bar_width = 0.35
        index = range(len(data['ansatz']))

        # Training accuracy bars
        plt.bar(index, data['avg_training_accuracy'], bar_width, label='Training Accuracy')

        # Testing accuracy bars (placed next to training bars)
        plt.bar([i + bar_width for i in index], data['avg_testing_accuracy'], bar_width, label='Testing Accuracy')

        plt.xlabel('Ansatz')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy by Ansatz for {dataset} Dataset, Subset {subset}')
        plt.xticks([i + bar_width / 2 for i in index], data['ansatz'], rotation=45)
        plt.legend()
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"{output_dir}/{dataset}_subset_{subset}_accuracy_by_ansatz.png")
        plt.close()