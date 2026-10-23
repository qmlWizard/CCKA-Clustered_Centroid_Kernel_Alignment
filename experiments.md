# Experimental methodology

## Protocol

Run the YAML-defined sweep with `python main.py --config
experiments/configs/default.yaml` on a Ray cluster. Start a Ray head and join
worker nodes before launching the command. Each independent
dataset/method/seed/noise/K/initialization/ablation/shot trial is one Ray task.
The driver writes an individual CSV per completed trial and a cumulative
`summary.csv` under `results/<config>-<timestamp>/`. Use `--address local` for
a single-machine run, or `--smoke` for the reduced checkerboard/corners checks.

The full dataset training partition is used for fitting. The dataset test
partition is used for validation-based checkpoint selection and is also
evaluated after restoring the selected checkpoint. PCA and feature scaling are
fit using training data only. Because the same test partition is used for model
selection and final reporting, the reported test metrics are not an unbiased
estimate of generalization performance. The default study uses ten seeds, a
CCKA K sweep, all available packaged datasets, and noiseless, depolarizing, and
coherent-noise conditions at the configured shot counts.

Compare all-pairwise Full KTA, Random-Pairwise KTA, QUACK, and CCKA under the
same data split, seed, circuit, and noise settings. CCKA ablations switch KAO
and CO updates on/off and compare class-wise chunk-mean, random, and K-means
sub-centroid initialization. CCKA aligns main centroids with sub-centroids;
its SVM kernel evaluation uses a training-set Nyström approximation. Kernel
centering is disabled. Record test accuracy and macro-averaged F1, precision,
and recall alongside validation accuracy, circuit executions, and wall time.
Summarize repeated runs with mean and standard deviation over seeds.

## Backend and reproducibility

The runner selects JAX/Optax when both are importable; otherwise it selects the
PyTorch/PennyLane implementation. Install either optional dependency group:
`pip install .[jax]` or `pip install .[torch]`. All Ray nodes must have the
project and selected backend installed. The driver sends prepared train,
validation, and test arrays through Ray's object store; worker nodes therefore
do not need local copies of the dataset files.

Keep the YAML config, source revision, and output directory with the results.
Trial CSV filenames encode dataset, method, noise type/level, shots, seed, K,
initialization, and ablation; the summary contains the same trial records in
completion order.

## Figure labels and filenames

Use consistent method labels in legends and captions:

| Config method | Figure label |
|---|---|
| `ccka` | CCKA |
| `quackKTA` | QUACK |
| `fullKTA` | All-Pairwise KTA (Full KTA) |
| `randomKTA` | Random-Pairwise KTA |

Use filenames of the form
`fig_<dataset>_<comparison>_<metric>.<ext>`, for example
`fig_checkerboard_method-comparison_test-accuracy.pdf` and
`fig_checkerboard_circuit-budget_test-accuracy.png`. Spell out the objective
and metric in axis labels (for example, “Circuit executions” and “Test
accuracy”); avoid internal class/config identifiers in the rendered figure.
