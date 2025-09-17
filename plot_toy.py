import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# --- Load both datasets ---
df_main = pd.read_csv("results/double_cake/per_run_summary_1.csv")
df_noise = pd.read_csv("noise_results/checkerboard/per_run_summary_1.csv")
# clean alignment col in noise df
df_noise['alignment'] = df_noise['alignment'].astype(str).str.extract(r'([\d\.]+)').astype(float)

for df in [df_main, df_noise]:
    for c in ['test_accuracy','train_time_sec','subcentroids','noise_level','n_samples','alignment']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

# --- Helper: mean ± 95% CI ---
def agg_ci(data, group, value):
    g = (data.groupby(group)[value]
         .agg(['mean','std','count'])
         .reset_index())
    g['sem'] = g['std'] / np.sqrt(g['count'])
    g['ci95'] = 1.96 * g['sem']
    g['lower'] = g['mean'] - g['ci95']
    g['upper'] = g['mean'] + g['ci95']
    return g

# --------------------------
#  PART 1: G–L (df_main)
# --------------------------
out_paths_main = {
    "G": "fig_G_accuracy_vs_clusters.png",
    "H": "fig_H_accuracy_vs_dsetsize.png",
    "I": "fig_I_dsetsize_vs_time.png",
    "J": "fig_J_clusters_vs_time.png",
    "K": "fig_K_heatmap.png",
    "L": "fig_L_alignment_vs_clusters.png",
}

# G
g = agg_ci(df_main, ['subcentroids'], 'test_accuracy')
plt.figure()
plt.errorbar(g['subcentroids'], g['mean'], 
             yerr=g['ci95'], fmt='-o', capsize=4)
plt.xlabel("Clusters (subcentroids)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Clusters")
plt.xscale("log", base=2)
plt.savefig(out_paths_main["G"], bbox_inches="tight")
plt.close()

# H
g = agg_ci(df_main, ['n_samples'], 'test_accuracy')
plt.figure()
plt.errorbar(g['n_samples'], g['mean'], 
             yerr=g['ci95'], fmt='-o', capsize=4)
plt.xlabel("Dataset size (n_samples)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Dataset size")
plt.xscale("log")
plt.savefig(out_paths_main["H"], bbox_inches="tight")
plt.close()

# I
g = agg_ci(df_main, ['n_samples'], 'train_time_sec')
plt.figure()
plt.errorbar(g['n_samples'], g['mean'], 
             yerr=g['ci95'], fmt='-o', capsize=4)
plt.xlabel("Dataset size (n_samples)")
plt.ylabel("Training time (sec)")
plt.title("Dataset size vs Training time")
plt.xscale("log")
plt.savefig(out_paths_main["I"], bbox_inches="tight")
plt.close()

# J
g = agg_ci(df_main, ['subcentroids'], 'train_time_sec')
plt.figure()
plt.errorbar(g['subcentroids'], g['mean'], 
             yerr=g['ci95'], fmt='-o', capsize=4)
plt.xlabel("Clusters (subcentroids)")
plt.ylabel("Training time (sec)")
plt.title("Clusters vs Training time")
plt.xscale("log", base=2)
plt.savefig(out_paths_main["J"], bbox_inches="tight")
plt.close()

# K (heatmap: keep as before, no CIs)
pivot = df_main.pivot_table(index='n_samples', columns='subcentroids',
                            values='test_accuracy', aggfunc='mean')
plt.figure()
plt.imshow(pivot, aspect='auto', origin='lower',
           extent=[pivot.columns.min(), pivot.columns.max(),
                   pivot.index.min(), pivot.index.max()])
plt.colorbar(label='Accuracy')
plt.xlabel("Clusters (subcentroids)")
plt.ylabel("Dataset size (n_samples)")
plt.title("Clusters vs Dataset size vs Accuracy")
plt.xscale("log")
plt.yscale("log")
plt.savefig(out_paths_main["K"], bbox_inches="tight")
plt.close()

# L
g = agg_ci(df_main, ['subcentroids'], 'alignment')
plt.figure()
plt.errorbar(g['subcentroids'], g['mean'], 
             yerr=g['ci95'], fmt='-o', capsize=4)
plt.xlabel("Clusters (subcentroids)")
plt.ylabel("Alignment score")
plt.title("Alignment score vs Clusters")
plt.xscale("log", base=2)
plt.savefig(out_paths_main["L"], bbox_inches="tight")
plt.close()

# --------------------------
#  PART 2: Noise plots (df_noise)
# --------------------------
out_paths_noise = {}

# A
plt.figure(figsize=(8,6))
for sc, sdf in df_noise.groupby('subcentroids'):
    g = agg_ci(sdf, ['noise_level'], 'test_accuracy')
    plt.errorbar(g['noise_level'], g['mean'], yerr=g['ci95'], 
                 fmt='-o', capsize=4, label=f"Clusters={sc}")
plt.xlabel("Noise level")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Noise Level")
plt.legend()
plt.grid(alpha=0.3)
out_paths_noise["A"] = "noise_acc_vs_noise.png"
plt.savefig(out_paths_noise["A"], bbox_inches="tight")
plt.close()

# B
plt.figure(figsize=(8,6))
for nl, sdf in df_noise.groupby('noise_level'):
    g = agg_ci(sdf, ['subcentroids'], 'test_accuracy')
    plt.errorbar(g['subcentroids'], g['mean'], yerr=g['ci95'],
                 fmt='-o', capsize=4, label=f"Noise={nl}")
plt.xscale("log", base=2)
plt.xlabel("Clusters (subcentroids)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Clusters under Noise")
plt.legend()
plt.grid(alpha=0.3)
out_paths_noise["B"] = "noise_acc_vs_clusters.png"
plt.savefig(out_paths_noise["B"], bbox_inches="tight")
plt.close()

import matplotlib.pyplot as plt
import pandas as pd

# your results dictionary
results = {
  "checkerboard": {"initial": 80,"ccka": 96.7,"random": 100,"quack": 100,"rbf": 57},
  "corners": {"initial": 89,"ccka": 93,"random": 94,"quack": 96,"rbf": 65},
  "double_cake": {"initial": 83.3,"ccka": 96.7,"random": 73.3,"quack": 86.7,"rbf": 91.1},
  "moons": {"initial": 82.9,"ccka": 96.7,"random": 96.7,"quack": 86.7,"rbf": 93},
  "donuts": {"initial": 78.9,"ccka": 85,"random": 80,"quack": 86.7,"rbf": 80},
  "zero vs non zero": {"initial": 89,"ccka": 93,"random": 76,"quack": 87,"rbf": 98},
  "one vs non one": {"initial": 88,"ccka": 92,"random": 88,"quack": 95,"rbf": 98},
  "adult": {"initial": 68,"ccka": 74,"random": 68,"quack": 82,"rbf": 74}
}

# convert to DataFrame
df = pd.DataFrame(results).T

# define custom colors
colors = {
    "initial": "0.7",      # grey
    "ccka": "#1f77b4",     # scientific blue (highlight)
    "random": "0.5",       # darker grey
    "quack": "0.3",        # even darker grey
    "rbf": "0.85"          # light grey
}

# plot
ax = df.plot(kind="bar", figsize=(12,6), width=0.8, color=[colors[c] for c in df.columns])

ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("Dataset")
ax.set_title("Model Accuracy Across Datasets")
plt.xticks(rotation=45, ha="right")

# move legend outside plot
plt.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

#plt.tight_layout()
#plt.show()

plt.savefig("fig_M_model_comparison.png", bbox_inches="tight", dpi=600)