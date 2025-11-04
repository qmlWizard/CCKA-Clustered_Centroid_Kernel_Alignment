import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.4)
palette = {"ccka": "#1f77b4", "random": "#ff7f0e"}
markers = {"ccka": "o", "random": "s"}

df = pd.read_csv("results/double_cake/per_run_summary_1.csv")

df = df[df['method'].isin(['ccka','random'])]

def agg_ci(data, group, value):
    g = (data.groupby(group)[value]
         .agg(['mean','std','count'])
         .reset_index())
    g['sem'] = g['std'] / np.sqrt(g['count'])
    g['ci95'] = 1.96 * g['sem']
    return g

fig, axes = plt.subplots(2, 2, figsize=(14,10))
axes = axes.ravel()

# (a) Accuracy vs Dataset Size
for m, sdf in df.groupby('method'):
    g = agg_ci(sdf, ['n_samples'], 'test_accuracy')
    axes[0].plot(g['n_samples'], g['mean'], marker=markers[m], color=palette[m], label=m)
    axes[0].fill_between(g['n_samples'], g['mean']-g['ci95'], g['mean']+g['ci95'],
                         color=palette[m], alpha=0.2)
axes[0].set_xscale("log")
axes[0].set_xlabel("Dataset size ($n$)")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("(a) Accuracy vs Dataset Size")

# (b) Circuit Executions vs Dataset Size (fixed k=4)
df_fixed = df[df['subcentroids']==4]
for m, sdf in df_fixed.groupby('method'):
    g = agg_ci(sdf, ['n_samples'], 'circuits_total')
    axes[1].plot(g['n_samples'], g['mean'], marker=markers[m], color=palette[m], label=m)
    axes[1].fill_between(g['n_samples'], g['mean']-g['ci95'], g['mean']+g['ci95'],
                         color=palette[m], alpha=0.2)
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_xlabel("Dataset size ($n$)")
axes[1].set_ylabel("Circuit Executions")
axes[1].set_title("(b) Circuit Executions vs Dataset Size ($k=4$)")

# (c) Accuracy vs Circuit Executions (Trade-off)
for m, sdf in df.groupby('method'):
    g = agg_ci(sdf, ['circuits_total'], 'test_accuracy')
    axes[2].plot(g['circuits_total'], g['mean'], marker=markers[m], color=palette[m], label=m)
    axes[2].fill_between(g['circuits_total'], g['mean']-g['ci95'], g['mean']+g['ci95'],
                         color=palette[m], alpha=0.2)
axes[2].set_xscale("log")
axes[2].set_xlabel("Circuit Executions (log scale)")
axes[2].set_ylabel("Accuracy")
axes[2].set_title("(c) Accuracy vs Circuit Executions\n(CCKA achieves accuracy with fewer circuits)")

# (d) Training Time vs Dataset Size (fixed k=4)
df_fixed = df[df['subcentroids']==4]
for m, sdf in df_fixed.groupby('method'):
    g = agg_ci(sdf, ['n_samples'], 'train_time_sec')
    axes[3].plot(g['n_samples'], g['mean'], marker=markers[m], color=palette[m], label=m)
    axes[3].fill_between(g['n_samples'], g['mean']-g['ci95'], g['mean']+g['ci95'],
                         color=palette[m], alpha=0.2)
axes[3].set_xscale("log"); axes[3].set_yscale("log")
axes[3].set_xlabel("Dataset size ($n$)")
axes[3].set_ylabel("Training time (s)")
axes[3].set_title("(d) Training Time vs Dataset Size ($k=4$)")

# unified legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)

plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("ccka_vs_random_tradeoff_panel.png", dpi=600, bbox_inches="tight")
plt.close()
