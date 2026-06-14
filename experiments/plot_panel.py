from ccka.models.kernel import KernelModel
from ccka.circuits.angleEmbeddingKernel import quackEmbeddingCircuit
from ccka.aligner.kta import fullKTA, centroidBasedKTA, quackKTA, randomKTA, greedyKTA
import pennylane as qml
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import os
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── global matplotlib style (publication-ready) ───────────────────────────
import matplotlib as mpl
mpl.rcParams.update({
    'font.family':           'serif',
    'font.serif':            ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset':      'stix',
    'axes.labelsize':        16,
    'axes.titlesize':        16,
    'xtick.labelsize':       14,
    'ytick.labelsize':       14,
    'legend.fontsize':       13,
    'legend.title_fontsize': 13,
    'figure.dpi':            1200,
    'savefig.dpi':           1200,
    'axes.linewidth':        0.8,
    'xtick.major.width':     0.8,
    'ytick.major.width':     0.8,
    'xtick.minor.width':     0.5,
    'ytick.minor.width':     0.5,
    'lines.linewidth':       1.6,
    'patch.linewidth':       0.6,
})

# ── publication color palette (colorblind-safe, print-safe) ───────────────
METHOD_STYLES = {
    'fullKTA': {
        'color':     '#0077BB',
        'marker':    'o',
        'linestyle': '-',
        'bar_color': '#0077BB',
        'hatch':     '',
    },
    'centroidBasedKTA': {
        'color':     '#EE7733',
        'marker':    's',
        'linestyle': '--',
        'bar_color': '#EE7733',
        'hatch':     '//',
    },
    'randomKTA': {
        'color':     '#009988',
        'marker':    '^',
        'linestyle': '-.',
        'bar_color': '#009988',
        'hatch':     'xx',
    },
    'greedyKTA': {
        'color':     '#CC3311',
        'marker':    'D',
        'linestyle': ':',
        'bar_color': '#CC3311',
        'hatch':     '\\\\',
    },
    'quackKTA': {
        'color':     '#AA3377',
        'marker':    'v',
        'linestyle': (0, (3, 1, 1, 1)),
        'bar_color': '#AA3377',
        'hatch':     '..',
    },
}

_EXTRA_STYLES = [
    {'color': '#BBBBBB', 'marker': 'p', 'linestyle': '-',  'bar_color': '#BBBBBB', 'hatch': ''},
    {'color': '#44BB99', 'marker': 'h', 'linestyle': '--', 'bar_color': '#44BB99', 'hatch': '//'},
    {'color': '#AAAA00', 'marker': '*', 'linestyle': ':',  'bar_color': '#AAAA00', 'hatch': 'xx'},
]

INITIAL_BAR_COLOR = '#4477AA'
FINAL_BAR_COLOR   = '#EE6677'

_ALL_METHODS = ['fullKTA', 'randomKTA', 'quackKTA', 'centroidBasedKTA', 'greedyKTA']

def _get_style(method, methods):
    if method in METHOD_STYLES:
        return METHOD_STYLES[method]
    idx = [m for m in methods if m not in METHOD_STYLES].index(method)
    return _EXTRA_STYLES[idx % len(_EXTRA_STYLES)]


def plot_experiment_panel(final_results, datasets, methods, centroids, figsize=(20, 12)):
    """
    Plot a 3-row x 3-column publication-quality panel of KTA experiment results.
    """
    df = final_results

    group_keys = ['method', 'dataset', 'num_iterations']
    agg_cols = [
        'final_testing_accuracy',   'final_training_accuracy',
        'initial_training_accuracy', 'initial_testing_accuracy',
        'f1_score',
    ]
    for col in agg_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_agg = df.groupby(group_keys)[agg_cols].max().reset_index()

    num_datasets = len(datasets)
    num_rows     = 2

    row_titles = [
        'Initial vs. Final Test Accuracy',
        'Final Test Accuracy'
    ]
    metric_cols = [
        None,
        'final_testing_accuracy'
    ]

    # ── figure & grid ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, facecolor='white')
    fig.suptitle('Kernel Target Alignment Experiment Results',
                 fontsize=17, fontweight='bold', y=0.998)

    gs = gridspec.GridSpec(
        num_rows, num_datasets,
        figure=fig,
        hspace=0.28,       # reduced from 0.42 — much tighter vertical spacing
        wspace=0.22,       # reduced from 0.32 — much tighter horizontal spacing
        top=0.96, bottom=0.05,
        left=0.07, right=0.97,
    )

    axes = [[fig.add_subplot(gs[r, c])
             for c in range(num_datasets)]
            for r in range(num_rows)]

    # ── row 0 : initial vs final test accuracy (grouped bars) ─────────────
    bar_w = 0.30
    for col, dataset in enumerate(datasets):
        ax  = axes[0][col]
        sub = df_agg[df_agg['dataset'] == dataset]
        x   = np.arange(len(methods))

        init_vals  = [sub[sub['method'] == m]['initial_testing_accuracy'].mean()
                      for m in methods]
        final_vals = [sub[sub['method'] == m]['final_testing_accuracy'].mean()
                      for m in methods]

        bars_init = ax.bar(
            x - bar_w / 2, init_vals, bar_w,
            label='Initial', color=INITIAL_BAR_COLOR,
            edgecolor='white', alpha=0.88, zorder=3,
        )
        bars_final = ax.bar(
            x + bar_w / 2, final_vals, bar_w,
            label='Final', color=FINAL_BAR_COLOR,
            hatch='//', edgecolor='white', alpha=0.88, zorder=3,
        )

        for bar in list(bars_init) + list(bars_final):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.006,
                f'{h:.2f}', ha='center', va='bottom', fontsize=10.5,
            )

        short_labels = [m.replace('centroidBased', 'Centroid-\n')
                         .replace('quack', 'Quack-\n') for m in methods]

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_title(dataset, fontsize=15, fontweight='bold')
        ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', length=0)

        if col == 0:
            ax.legend(fontsize=11, loc='upper left',
                      framealpha=0.85, edgecolor='#cccccc', frameon=True)

    axes[0][0].annotate(
        row_titles[0],
        xy=(-0.18, 0.5), xycoords='axes fraction',
        fontsize=12, fontweight='bold', rotation=90,
        va='center', ha='right',
    )

    # ── rows 1-2 : line plots vs num_iterations ───────────────────────────
    for row in range(1, num_rows):
        metric = metric_cols[row]

        for col, dataset in enumerate(datasets):
            ax  = axes[row][col]
            sub = df_agg[df_agg['dataset'] == dataset]

            for method in methods:
                msub = sub[sub['method'] == method].sort_values('num_iterations')
                if msub.empty:
                    continue
                sty = _get_style(method, methods)
                if method in ['centroidBasedKTA']:
                    num_iter = msub['num_iterations'] / 10
                else:
                    num_iter = msub['num_iterations']
                ax.plot(
                    num_iter, msub[metric],
                    label=method,
                    color=sty['color'],
                    marker=sty['marker'],
                    linestyle=sty['linestyle'],
                    markersize=6,
                    linewidth=1.8,
                    alpha=0.9,
                    markerfacecolor='white',
                    markeredgewidth=1.5,
                )

            ax.set_ylim(0, 1.05)
            ax.set_xlabel('Iterations', fontsize=14)
            ax.set_ylabel(row_titles[row], fontsize=14)

            if row == 1:
                ax.set_title(dataset, fontsize=14, fontweight='bold')

            ax.grid(linestyle=':', linewidth=0.6, alpha=0.6)
            ax.spines[['top', 'right']].set_visible(False)
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
            ax.tick_params(which='minor', length=2)

            if col == num_datasets - 1 and row == 1:
                handles = [
                    plt.Line2D(
                        [0], [0],
                        color=_get_style(m, methods)['color'],
                        marker=_get_style(m, methods)['marker'],
                        linestyle=_get_style(m, methods)['linestyle'],
                        markersize=6,
                        markerfacecolor='white',
                        markeredgewidth=1.5,
                        label=m,
                    )
                    for m in methods
                ]
                ax.legend(
                    handles=handles,
                    fontsize=11,
                    loc='lower right',
                    framealpha=0.85,
                    edgecolor='#cccccc',
                    frameon=True,
                    title='Method',
                    title_fontsize=11,
                )

        axes[row][0].annotate(
            row_titles[row],
            xy=(-0.18, 0.5), xycoords='axes fraction',
            fontsize=12, fontweight='bold', rotation=90,
            va='center', ha='right',
        )

    # ── save ──────────────────────────────────────────────────────────────
    for fmt in ('png', 'pdf'):
        fig.savefig(f'kta_results_panel.{fmt}',
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Saved: kta_results_panel.png  /  kta_results_panel.pdf")

# ── centroid sweep panel ───────────────────────────────────────────────────

_CENTROID_METHODS = ['centroidBasedKTA', 'randomKTA', 'greedyKTA', 'quackKTA']


def plot_centroid_sweep_panel(final_results, datasets, figsize=(18, 6)):
    """
    Plot a 1-row x 3-column panel: accuracy vs. number of centroids.
    """
    df = final_results

    df = df[df['method'].isin(_CENTROID_METHODS)].copy()

    df_agg = (
        df.groupby(['method', 'dataset', 'centroids'])['final_testing_accuracy']
        .mean()
        .reset_index()
    )

    # ── figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=figsize,
        facecolor='white',
        sharey=True,
    )
    fig.suptitle('Final Test Accuracy vs. Number of Samples/Centroids',
                 fontsize=15, fontweight='bold', y=1.02)

    # Tighter spacing between the 3 tiles
    fig.subplots_adjust(wspace=0.12)

    for col, (ax, dataset) in enumerate(zip(axes, datasets)):
        sub = df_agg[df_agg['dataset'] == dataset]

        for method in _CENTROID_METHODS:
            msub = sub[sub['method'] == method].sort_values('centroids')
            if msub.empty:
                continue

            sty = METHOD_STYLES.get(method, {})
            ax.plot(
                msub['centroids'],
                msub['final_testing_accuracy'],
                label=method,
                color=sty.get('color', '#888888'),
                marker=sty.get('marker', 'o'),
                linestyle=sty.get('linestyle', '-'),
                markersize=6,
                linewidth=1.8,
                alpha=0.9,
                markerfacecolor='white',
                markeredgewidth=1.5,
            )

        centroid_vals = sorted(df_agg['centroids'].dropna().unique().astype(int))
        ax.set_xticks(centroid_vals)
        ax.set_xticklabels(centroid_vals, fontsize=13)

        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Number of Centroids', fontsize=14)
        ax.set_title(dataset, fontsize=14, fontweight='bold')
        ax.grid(linestyle=':', linewidth=0.6, alpha=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax.tick_params(which='minor', length=2)

        if col == 0:
            ax.set_ylabel('Final Test Accuracy', fontsize=14)

    handles = [
        plt.Line2D(
            [0], [0],
            color=METHOD_STYLES.get(m, {}).get('color', '#888888'),
            marker=METHOD_STYLES.get(m, {}).get('marker', 'o'),
            linestyle=METHOD_STYLES.get(m, {}).get('linestyle', '-'),
            markersize=6,
            markerfacecolor='white',
            markeredgewidth=1.5,
            label=m,
        )
        for m in _CENTROID_METHODS
    ]
    axes[-1].legend(
        handles=handles,
        fontsize=11,
        loc='lower right',
        framealpha=0.85,
        edgecolor='#cccccc',
        frameon=True,
        title='Method',
        title_fontsize=11,
    )

    for fmt in ('png', 'pdf'):
        fig.savefig(f'kta_centroid_sweep.{fmt}',
                    dpi=1200, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Saved: kta_centroid_sweep.png  /  kta_centroid_sweep.pdf")


corners_df = pd.read_csv('corners_method_comparison_results.csv')
checkerboard_df = pd.read_csv('checkerboard_method_comparison_results.csv')
donuts_df = pd.read_csv('donuts_method_comparison_results.csv')

final_results = pd.concat([corners_df, checkerboard_df, donuts_df], ignore_index=True)
datasets = ['corners', 'checkerboard', 'donuts']
dataset_paths = {
    'corners': '../data/corners.npy',
    'checkerboard': '../data/checkerboard_dataset.npy',
    'donuts': '../data/donuts.npy'
}
methods = ['fullKTA', 'randomKTA', 'quackKTA', 'centroidBasedKTA', 'greedyKTA']
centroid_values = [2, 4, 6, 8, 10]
num_iterations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
plot_experiment_panel(final_results, datasets, methods, centroid_values)
plot_centroid_sweep_panel(final_results, datasets)