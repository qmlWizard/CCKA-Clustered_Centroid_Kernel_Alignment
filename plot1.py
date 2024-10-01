import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to the CSV file (modify as needed)
file_path = 'all_results.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Create directory to save the figures
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# Calculate the mean accuracy per dataset, subset, and ansatz
ansatz_accuracy = df.groupby(['dataset', 'subset', 'ansatz']).agg(
    avg_training_accuracy=('Training Accuracy', 'mean'),
    avg_testing_accuracy=('Testing Accuracy', 'mean')
).reset_index()

# Generate bar charts for each subset within each dataset
for dataset in df['dataset'].unique():
    data_by_dataset = ansatz_accuracy[ansatz_accuracy['dataset'] == dataset]
    
    # Generate separate plots for each subset
    for subset in [1, 2, 4, 6, 8, 12]:
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
        plt.title(f'Accuracy by Ansatz for {dataset.capitalize()} Dataset, Subset {subset}')
        plt.xticks([i + bar_width / 2 for i in index], data['ansatz'], rotation=45)
        plt.legend()
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"{output_dir}/{dataset}_subset_{subset}_accuracy_by_ansatz.png")
        plt.close()
