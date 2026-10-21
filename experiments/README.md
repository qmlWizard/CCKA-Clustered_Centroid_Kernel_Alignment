# Reproducible experiments

`experiments.py` is the single-process entry point for method comparisons, CCKA
ablations, initialization comparisons, and noisy/noiseless runs. All model
selection uses a validation split; the original dataset test split is evaluated
only once, after validation-based checkpoint restoration. Input scaling is fit
on training data only and maps features to `[0, 1]`, matching the current CCKA
main-centroid box constraint. Kernel centering is disabled. CCKA keeps its
centroid-to-sub-centroid alignment objective unchanged and evaluates its SVM
using a Nyström approximation.

## Smoke tests

Run each included method on checkerboard and corners, in noiseless, depolarizing, and
coherent-noise modes. The smoke configuration also exercises main-only,
sub-only, KAO/CO ablations, and class-wise chunk-mean, random, and K-means
sub-centroid initializations:

```bash
python experiments/experiments.py --config experiments/configs/smoke.yaml
```

Or select the smoke config with `--smoke`:

```bash
python experiments/experiments.py --smoke
```

For Ray execution across a cluster, use the repository-level runner:

```bash
python main.py --config experiments/configs/smoke.yaml
```

Start a Ray head and join workers before using the default `--address auto`.
Use `--address local` to launch Ray on one machine. Each completed trial is
saved separately, with a cumulative summary, under `results/<run-name>/`.

Noisy smoke runs use 8 shots to keep execution short. Finite-shot kernel
evaluations are executed pairwise rather than through JAX vectorization.
The runner disables optional alignment, block-ratio, and kPCA diagnostics by
default so recorded circuit counts focus on training, model selection, and
final SVM evaluation; set `training.collect_diagnostics: true` to collect them.
The smoke config reduces QUACK to one inner update; the default config retains
the method's ten inner updates.

## Full experiment suite

Run all available packaged real-world datasets and synthetic datasets with ten
seeds, the K sweep, all methods/ablations/initializations, two noise levels,
and both depolarizing and coherent noise:

```bash
python experiments/experiments.py --config experiments/configs/default.yaml
```

The default config sweeps 256 and 1024 shots for noisy runs. To run fewer
datasets, seeds, methods, K values, noise levels, or shots, edit a copy of the
YAML config. Results are written as CSV to the configured `output_csv` path;
failed runs are logged with their settings and stop the batch instead of being
silently skipped.

Packaged NumPy data are loaded from `data/`; `moons` and `double_cake` are
generated deterministically. The two named smoke datasets are subsampled for
speed, so smoke results are checks of execution paths, not reportable
benchmarks. The runner selects JAX/Optax if available, otherwise PyTorch;
install one with `pip install .[jax]` or `pip install .[torch]`.
