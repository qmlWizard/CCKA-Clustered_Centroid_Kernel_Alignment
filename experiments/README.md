# Reproducible experiments

`experiments.py` is the single-process entry point for method comparisons, CCKA
ablations, initialization comparisons, and noisy/noiseless runs. The
`training.test_fraction` setting controls the held-out test fraction (default
`0.5`); `training.validation_fraction` selects a validation subset from the
remaining training partition (default `0.2`). Test examples are kept out of
validation-based model selection and evaluated only after restoring the best
validation checkpoint. Input scaling is fit on the training subset only and
maps features to `[0, 1]`, matching the current CCKA main-centroid box
constraint. Kernel centering is disabled. CCKA evaluates its SVM using a
Nyström approximation.
For comparisons that intentionally use the test partition as validation, pass
`--use-test-as-validation`; in that mode, the same held-out examples guide
checkpoint selection and final scoring, so the test accuracy is optimistic and
not an unbiased generalization estimate.

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
Use `--address local` to launch Ray on one machine. When Ray has one live node,
the runner executes experiments sequentially and reserves that node's CPUs for
each experiment; on multi-node clusters, experiments remain scheduled in
parallel. Each completed trial is saved separately, with a cumulative summary,
under `results/<run-name>/`.

Noisy smoke runs use 8 shots to keep execution short. Finite-shot kernel
evaluations are executed pairwise rather than through JAX vectorization.
The runner disables optional alignment, block-ratio, and kPCA diagnostics by
default so recorded circuit counts focus on training, model selection, and
final SVM evaluation; set `training.collect_diagnostics: true` to collect them.
The smoke config reduces QUACK to one inner update; the default config retains
the method's ten inner updates.
CCKA's CO update uses a softplus margin loss between the mean kernel similarity
to same-class and other-class sub-centroids. `training.co_margin` and
`training.co_temperature` control the separation margin and softplus
temperature; CO updates main centroids only, while KAO retains its
kernel-parameter and sub-centroid alignment update.
Inference is selectable in the training config: `inference_method` sets the
default (`svm` or `kernel_distance`), and method-specific keys such as
`fullKTA_inference_method`, `randomKTA_inference_method`,
`quackKTA_inference_method`, and `ccka_inference_method` override it. The
accuracy-versus-executions command accepts `--inference-method` to select one
method for every model, or a method-specific option to override an individual
model. `inference_neighbors` sets k for kernel-distance k-NN.
The comparison runner also accepts `--ccka-learning-rate` to override only
CCKA's kernel-weight learning rate, leaving QUACK and other methods unchanged.
Use `--learning-rate` to apply one rate to all model weight optimizers and
CCKA/QUACK centroid optimizers.
Use `--run-full-iterations` to keep training to the iteration cap after the
accuracy target is reached. For each model, it records and restores the
checkpoint with the highest validation accuracy, then evaluates that restored
checkpoint once on the held-out test set.
CCKA's Nyström inference size is swept using
`experiments.ccka_inference_landmarks`. Positive integers select that many
landmarks (capped at the training-set size); `all` builds exact training and
inference kernel matrices instead of using Nyström. This setting changes
inference/model-selection kernel matrices, not the centroid-alignment losses
used for KAO and CO.

Run the dedicated checkerboard noise sweep with
`python -m experiments.noise`. It compares QUACK and K-means-initialized CCKA
under five coherent and five depolarizing noise levels and writes one combined
chart plus a CSV. The defaults use K=6, SVM inference, a shared learning rate
of 1.5, 256 shots, and the test partition as validation. Override the five
levels with `--noise-levels P1 P2 P3 P4 P5`.

## Accuracy versus circuit executions

Run the separate checkerboard comparison until each method reaches 90%
validation accuracy, with the configured epoch count as a per-method safety cap:

```bash
python -m experiments.accuracy_vs_executions
```

Use `--dataset corners` to run the same comparison on the corners dataset.
Pass `--methods quackKTA ccka` to restrict the comparison to QUACK and CCKA.
The script compares Full KTA, Random-Pairwise KTA, QUACK, and CCKA using the
first configured seed and centroid count. Use `--centroids 2 4 6 8 10` to sweep
multiple centroid counts, and `--ccka-initialization kmeans` to select only
K-means initialization. CCKA uses
`training.inference_landmarks` from the config (default `32`), overridable with
`--inference-landmarks` (a positive count or `all`). CCKA is run with regular, k-means,
and random centroid initialization; each initialization gets a separate curve.
For CCKA and QUACK, `training.epochs` and the maximum-iteration option count
total inner iterations; with 10 inner iterations, 100 iterations run as 10
outer epochs. Each CCKA inner iteration runs KAO followed by CO when both are
enabled, while QUACK performs one update per inner iteration and alternates
KAO/CO by outer epoch. Other methods perform one optimizer update per epoch. It records the
initial model and each subsequent optimization step, stopping a method as soon
as its validation accuracy reaches the target. The plot uses a distinct color
per method and CCKA initialization; the CSV records initialization, accuracy,
outer epoch, inner step, total inner iterations, optimizer updates, and
cumulative training kernel-pair count. Initial and per-step SVM evaluation kernels are
excluded from that count. Results are written under
`results/{dataset}_accuracy_vs_executions` with `.png`, `.csv`, and
`_summary.csv` suffixes. Override the dataset, target, epoch cap, seed,
centroid count, inference method, or output prefix with command-line options.
`--inference-method` selects SVM or kernel distance for every model;
`--full-kta-inference-method`, `--random-kta-inference-method`,
`--quack-inference-method`, and `--ccka-inference-method` override individual
models. `--max-epochs` is retained as an alias for `--max-iterations`. Counts
are logical
kernel-pair evaluations from the optimizer's loss/gradient calls; they do not
include SVM evaluation or backward-pass gate/shot overhead.

For a single optimization update, the current JAX implementations have these
pair-count costs, where `N` is training samples, `b` is the random batch size,
`C` is classes, `K` is sub-centroids per class, and `R(n)=n(n+1)/2`:

| Method | Pair evaluations per update | Complexity |
| --- | ---: | --- |
| Full KTA | `R(N)` | `O(N^2)` |
| Random KTA | `R(b)` | `O(b^2)` |
| QUACK | `N` per inner update | `O(N)`; a complete sweep of one update per class is `O(C*N)` |
| CCKA | `2*C*K` per paired inner iteration | `O(C*K)` |

The QUACK loop alternates KAO and CO outer epochs, performs
`quack_inner_steps` updates of the selected type, and selects one class per
outer epoch. Thus its actual work per outer epoch is
`quack_inner_steps * N`, not `C*N` unless the number of class updates is
included in the measurement interval. The CCKA loop selects one class per
outer epoch and performs `ccka_inner_steps` paired iterations (defaulting to
`quack_inner_steps`, then 10). Each iteration runs enabled KAO and CO updates
in that order. Each loss compares the selected main centroid against all
`C*K` sub-centroids, so the paired cost is `2*C*K` when both updates are
enabled. When only one operation is enabled, that update runs on every inner
iteration.

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

For Ray runs, plot a completed run's CSVs after it finishes:

```bash
python experiments/plot_results.py \
  --input results/default-YYYYMMDD-HHMMSS \
  --output results/default-YYYYMMDD-HHMMSS/plots
```

The plotting script writes runtime and circuit-count figures, validation
accuracy histories (mean across seeds with min/max bands), CCKA centroid
sweeps, coherent/depolarizing-noise comparisons, and ablation figures.
Noisy plots use CCKA with regular initialization at K=2 by default; override
that choice with `--noise-centroids K`. Epoch histories are recorded in the
summary CSV, so use results produced by the updated runner. Ablation runs use
K=1 and include a matching full-CCKA baseline.

Packaged NumPy data are loaded from `data/`; `moons` and `double_cake` are
generated deterministically. The two named smoke datasets are subsampled for
speed, so smoke results are checks of execution paths, not reportable
benchmarks. The runner selects JAX/Optax if available, otherwise PyTorch;
install one with `pip install .[jax]` or `pip install .[torch]`.
