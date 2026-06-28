import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── global matplotlib style (publication-ready) ───────────────────────────
mpl.rcParams.update({
    'font.family':           'serif',
    'font.serif':            ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset':      'stix',
    'axes.labelsize':        14,
    'axes.titlesize':        14,
    'xtick.labelsize':       12,
    'ytick.labelsize':       12,
    'legend.fontsize':       11,
    'legend.title_fontsize': 11,
    'figure.dpi':            150,
    'savefig.dpi':           1200,
    'axes.linewidth':        0.8,
    'xtick.major.width':     0.8,
    'ytick.major.width':     0.8,
    'lines.linewidth':       1.8,
})

# ── publication color palette ─────────────────────────────────────────────
METHOD_STYLES = {
    'fullKTA': {
        'color':     '#0077BB',
        'marker':    'o',
        'linestyle': '-',
    },
    'centroidBasedKTA': {
        'color':     '#EE7733',
        'marker':    's',
        'linestyle': '--',
    },
    'randomKTA': {
        'color':     '#009988',
        'marker':    '^',
        'linestyle': '-.',
    },
    'greedyKTA': {
        'color':     '#CC3311',
        'marker':    'D',
        'linestyle': ':',
    },
    'quackKTA': {
        'color':     '#AA3377',
        'marker':    'v',
        'linestyle': (0, (3, 1, 1, 1)),
    },
}

METRICS = [
    ('final_kernel_alignment',  'KTA',              (0.0, 0.75)),
    ('final_training_accuracy', 'Train Accuracy',   (0.4, 1.05)),
    ('final_testing_accuracy',  'Test Accuracy',    (0.4, 1.05)),
]

SUBPLOT_TITLES = ['KTA Score', 'Train Accuracy', 'Test Accuracy']


def plot_variance_panel(final_results, datasets, methods, figsize=(20, 5)):
    """
    For each dataset: plot a 1x3 panel of (KTA | Train Acc | Test Acc)
    vs. num_iterations, with shaded min–max variance bands.
    Each dataset gets its own figure.
    """
    df = final_results.copy()

    # coerce numeric
    for col in ['kta_score', 'final_training_accuracy', 'final_testing_accuracy',
                'initial_training_accuracy', 'initial_testing_accuracy', 'num_iterations']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for dataset in datasets:
        fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor='white')
        fig.suptitle(f'KTA Alignment Results — {dataset.capitalize()}',
                     fontsize=15, fontweight='bold', y=1.02)
        fig.subplots_adjust(wspace=0.28)

        sub_ds = df[df['dataset'] == dataset]

        for ax, (metric, ylabel, ylim) in zip(axes, METRICS):

            # skip if metric column missing
            if metric not in df.columns:
                ax.text(0.5, 0.5, f'"{metric}"\nnot in data',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=11, color='grey')
                ax.set_title(ylabel, fontsize=13, fontweight='bold')
                continue

            for method in methods:
                msub = sub_ds[sub_ds['method'] == method].dropna(
                    subset=['num_iterations', metric]
                )
                if msub.empty:
                    continue

                # For centroidBasedKTA scale iterations /10 to match other methods
                scale = 10 if method == 'centroidBasedKTA' else 1
                tmp = msub.copy()
                tmp['_iter'] = tmp['num_iterations'] / scale

                # Average across centroid values per iteration first,
                # then compute mean/min/max variance across those averages
                grp = (
                    tmp.groupby('_iter')[metric]
                    .agg(mean='mean', min='min', max='max')
                    .reset_index()
                    .sort_values('_iter')
                )

                sty = METHOD_STYLES.get(method, {'color': '#888888',
                                                  'marker': 'o',
                                                  'linestyle': '-'})

                ax.plot(
                    grp['_iter'], grp['mean'],
                    label=method,
                    color=sty['color'],
                    marker=sty['marker'],
                    linestyle=sty['linestyle'],
                    markersize=5,
                    linewidth=1.8,
                    alpha=0.95,
                    markerfacecolor='white',
                    markeredgewidth=1.5,
                    markevery=max(1, len(grp) // 10),
                )
                ax.fill_between(
                    grp['_iter'], grp['min'], grp['max'],
                    color=sty['color'], alpha=0.15,
                )

            ax.set_ylim(*ylim)
            ax.set_xlabel('Epoch', fontsize=13)
            ax.set_ylabel(ylabel, fontsize=13)
            ax.set_title(ylabel, fontsize=13, fontweight='bold')
            ax.grid(linestyle=':', linewidth=0.6, alpha=0.6)
            ax.spines[['top', 'right']].set_visible(False)
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
            ax.tick_params(which='minor', length=2)

        # shared legend on last axis
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
            for m in methods if m in METHOD_STYLES
        ]
        axes[-1].legend(
            handles=handles,
            fontsize=10,
            loc='lower right',
            framealpha=0.88,
            edgecolor='#cccccc',
            frameon=True,
            title='Method',
            title_fontsize=10,
        )

        for fmt in ('png', 'pdf'):
            fig.savefig(f'kta_variance_panel_{dataset}.{fmt}',
                        dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"Saved: kta_variance_panel_{dataset}.png / .pdf")


# ── load data ─────────────────────────────────────────────────────────────
corners_df      = pd.read_csv('corners_method_comparison_results.csv')
checkerboard_df = pd.read_csv('checkerboard_method_comparison_results.csv')
donuts_df       = pd.read_csv('donuts_method_comparison_results.csv')

final_results = pd.concat([corners_df, checkerboard_df, donuts_df], ignore_index=True)

datasets = ['corners', 'checkerboard', 'donuts']
methods  = ['fullKTA', 'randomKTA', 'quackKTA', 'centroidBasedKTA', 'greedyKTA']

plot_variance_panel(final_results, datasets, methods)