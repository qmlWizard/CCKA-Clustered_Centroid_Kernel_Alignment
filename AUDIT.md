# CCKA Implementation and Experiment Audit

This is a read-only audit of the checked-in implementation, experiment scripts,
archived YAML configurations, notebooks, saved results, and README. The repository
does not contain a paper PDF or a manifest that maps every reported paper figure
to a particular run, so figure provenance is identified only where the source
does so explicitly.

## 1. What the CCKA loss evaluates; per-epoch count

Let `N` be the training-set size, `C` the number of classes, and `K` the number
of sub-centroids per class. CCKA stores `C` main centroids and `C*K`
sub-centroids. Its KAO and CO loss functions both call
`_centroid_kernel_vec(weights, main_centroid, self.sub_centroids)`. This makes a
vector of kernel values between **one selected main centroid and every
sub-centroid in the collection**. The input to this vector is not the training
data: the CCKA loss does not compute data-to-centroid kernels.

In `CentroidBasedKTA.align()`, each outer epoch calls `_kao_joint_update` once
and `_co_main_update` once. Each loss call evaluates `C*K` pairs, so the active
aligner performs:

> **CCKA objective: `2*C*K` kernel evaluations per epoch** (`C*K` for KAO plus
> `C*K` for CO).

For binary classification (`C=2`), that is `4*K`, independent of `N`. The
class selected for the update cycles through the labels; it is not a loop over
all classes. The pair list does include sub-centroids from every class.

There is a documentation/experiment discrepancy: the docstring in
`CentroidBasedKTA.align()` says “10x” KAO and “10x” CO, but the method itself
does one update of each per epoch. The separate manual loop in
`experiments/method_comparision_parallel.py` really does use ten of each, so
that script's **objective-only** count is `20*C*K` per epoch. Its cost
estimator does not calculate that formula correctly (see section 3).
`experiments/computational_Cost.py` also assumes ten of each.

The aligner's full **reported `circuit_executions`** includes much more than
the loss. With `n=N` training points, `t` test points, and
`R(x)=x*(x+1)/2`, the active CCKA loop makes, per epoch, the following logical
forward-pair requests:

| Work | Pair requests |
|---|---:|
| KAO and CO loss/gradient calls | `2*C*K` |
| Full-training global alignment | `R(n)` |
| CCKA centroid-KTA diagnostic (each of C main centroids against C*K sub-centroids) | `C*C*K` |
| Full-training block-ratio diagnostic | `R(n)` |
| SVM training kernel | `R(n)` |
| SVM test/inference kernel | `n*t` |
| kPCA diagnostic, over training points + C main centroids + C*K sub-centroids | `R(n+C+C*K)` |

Thus the per-epoch logical total is
`2*C*K + C*C*K + 3*R(n) + n*t + R(n+C+C*K)`, plus an initial SVM evaluation
and a final restored-checkpoint SVM evaluation (`R(n)+n*t` each). These
diagnostics, SVM kernels, and the kPCA kernel matrix are not part of the CCKA
loss. This large diagnostic/evaluation overhead means the actual aligner
counter is not the same as the loss-only complexity often quoted for CCKA.

CCKA's sub-centroid count is not restricted to powers of two. The checked-in
configs explicitly use values such as 2, 3, 4, 5, 6, 8, and 10. Regular
initialization shuffles each class's training points, splits them into `K`
chunks, and averages each chunk; k-means initialization uses `KMeans` with
`n_clusters=K` when selected and enough points are available.

The CO box constraint is applied to **main centroids only**: the CO loss has a
penalty outside `[0,1]`, and `_co_main_update()` then clips the resulting main
centroids to `[0,1]`. Sub-centroids are not clipped. This is at odds with the
`[-pi/2, pi/2]` feature range attributed to the paper: the current package does
not normalize input features to that range, and the archived data generator
uses `MinMaxScaler(feature_range=(-pi, pi))`. The feature map applies trainable
`scale*x`; its weight initializer uses `[-pi, pi]`, which is a weight range,
not feature preprocessing. QUACK instead clips main centroids to each
training feature's observed minimum/maximum.

## 2. Checkpoint and K selection

`CentroidBasedKTA.align()` updates `best_test_acc` using
`result["test_accuracy"]`, stores weights and both centroid arrays, and restores
that checkpoint. `QuackKTA.align()` does the same, including checks after its
inner updates. **Both use the held-out test set to choose a checkpoint.** The
restored model is then evaluated on the same test set again. This is test-set
leakage, not validation-based checkpointing.

There is no automatic K search in the aligner itself: K is a constructor/config
argument. The comparison script does select K using test accuracy:
`_best_centroid_for_method()` takes the maximum test score for each
`(K, seed)`, averages those maxima by K, then selects the K with the highest
mean. `_per_seed_best_stats()` then selects the best test-score epoch per seed
at that K. Consequently both checkpoint and comparison-script K selection
optimistically reuse the test set.

The base `FullKTA`, `RandomKTA`, and `GreedyKTA` loops do not restore a best
checkpoint; their reported histories are trajectories and their final state
is the last update. The legacy notebook `experiments/method_comparision.ipynb`
records its last test score, while other result scripts report best-per-epoch
scores. These are different quantities.

## 3. Circuit accounting and SVM costs

`KernelModel.forward()` increments `circuit_executions` by the number of
input pairs passed to that call. One pair is one kernel circuit (the feature
map/ansatz for the first input followed by the inverse for the second, with
one measurement); the counter does not multiply by shots, gates, or circuit
depth. `regular_kernel_matrix()` evaluates only the upper triangle including
the diagonal and mirrors it, so a symmetric `n*n` matrix costs `R(n)`, not
`n*n`. The rectangular test/train matrix costs `n*t`. The Nyström path costs
`n*m + m*m`: `K_NM` is rectangular, and `K_MM` is evaluated as all `m*m`
ordered pairs rather than using triangular symmetry.

The counter is a **forward-pair-call counter**, not a reliable count of all
physical circuit executions needed to differentiate on hardware. In the
base aligner, both loss evaluation and gradient are wrapped with `jax.jit`,
while the counter is a Python-side increment in `KernelModel.forward()`.
Consequently a call during JIT tracing does not necessarily increment the
counter again for every compiled execution. The counter also does not account
for parameter-shift/shot overhead. PennyLane simulation uses backprop by
default; the Qiskit hardware runners count submitted pair circuits in their
forward methods, but the JAX gradient path is not a hardware-gradient
accounting mechanism.

For the base aligner, let `b` be the objective batch size (`n` for FullKTA and
the random mini-batch size for RandomKTA). Each epoch logically requests two
objective matrices (the recorded loss and the gradient), plus five full
training matrices (global KTA, centroid-KTA fallback, block ratio, SVM fit,
and kPCA) and one rectangular test/train matrix: `2*R(b) + 5*R(n) + n*t`.
The SVM fit and its inference use the same SVM and do not themselves execute
quantum circuits once their kernel matrices have been built.

| Method | Update/selection kernel calls per epoch | Other circuit work and SVMs |
|---|---|---|
| FullKTA | Full-batch objective and gradient; each logical evaluation uses `R(n)` pairs. | Shared base diagnostics and one SVM fit/test evaluation per epoch. |
| RandomKTA | Batch objective and gradient; each logical evaluation uses `R(k)` pairs, where `k=random_samples`. | Shared base diagnostics and one SVM fit/test evaluation per epoch. |
| GreedyKTA | Nyström uncertainty kernel `n*m + m*m`, followed by objective and gradient on `g=greedy_samples` selected points (`R(g)` each). | Fits an additional `SVC` for selection and calls `predict_proba` on the approximate training kernel each epoch; also incurs shared base diagnostics/SVM. |
| CCKA | Active aligner: `2*C*K`; manual comparison loop: `20*C*K`. | CCKA diagnostics and SVM/kPCA work as itemized above. |
| QUACK | The active aligner runs ten KAO **or** ten CO updates per epoch; each is one main-centroid-to-all-training-points vector: `10*n`. It alternates update type by epoch. | Centroid-KTA diagnostic costs `C*n`; global KTA, block ratio, kPCA, and an SVM fit/test are also run. `svm_training()` is additionally called after each of the ten inner updates, so there are 11 SVM training and inference kernel matrices per outer epoch, plus initial/final SVM evaluations. The passed `centroids`/K value is ignored by QUACK's objective. |

All SVMs in `ccka/aligner/kta.py` use `SVC(kernel="precomputed", C=1.0)`.
The regular SVM routine also enables `probability=True`. The additional
GreedyKTA selection SVM also uses `C=1.0` and `probability=True`. Fitting and
prediction are classical operations on already computed kernels; the quantum
circuit costs are the matrices supplied to those operations. In
`method_comparision_parallel.py` the custom SVM uses `C=1.0` and omits
probability calibration.

The comparison runner's `circuit_executions` output is a formula-generated
estimate, not `KernelModel.circuit_executions`. Its estimator uses `n*n` for
the full matrix (ignoring upper-triangle symmetry), omits test/train kernels
and repeated SVM/kPCA work, and assigns updates using `K*LANDMARK_POINTS`
even for methods whose objective uses a different pair set. Its CCKA estimator
is `10*(K*m+K)`, not the manual loop's `20*C*K`; its QUACK estimator is
`10*K*m`, although QUACK's update vector is over the training data. Treat those
reported estimates as incomparable to the package counter.

## 4. K, epochs, and representative-sample settings

There is no single source of truth; the checked-in scripts/configurations
represent separate experiment generations.

| Experiment/config source | Dataset(s) | K / sample-size settings | Epochs and split |
|---|---|---|---|
| `experiments/method_comparision_parallel.py` | corners, checkerboard, donuts, concentric circles | `K=[2,4,6,8,10]`; Random uses K as `random_samples`; Greedy uses K as `greedy_samples` and `m=10`; CCKA uses K sub-centroids/class; QUACK receives K but ignores it; FullKTA has no K. | 100 epochs, ten seeds, 50/50 split. |
| `experiments/real_world_datasets_multiseed.py` | banknote, breast cancer, German credit, Pima diabetes, QSAR | K=20 for centroid methods; Random/Greedy use 20 samples; Greedy uses `m=20`. | 100 iterations, seeds 1,2,3, 70/30 split. |
| `archive/configs/all_datasets/synthetic/*.yaml` | checkerboard, corners | K=4; 200 epochs; methods listed as Random/QUACK. | Legacy YAML; 50/50 dataset sizes. |
| Same legacy config group | double cake | K=8; 200 epochs; Random/QUACK. | Legacy YAML; 50/50. |
| Same legacy config group | donuts | K=8; 700 epochs; Random/QUACK/Full. | Legacy YAML; 50/50. |
| Same legacy config group | moons | K=10; 100 epochs; Random/QUACK. | Legacy YAML; 50/50. |
| `archive/configs/extented/` and `cross_validation/` | checkerboard, corners, donuts, moons | K sweeps vary: commonly `[2,4,6,8,10]`; corners/donuts/moons include `[2,3,4,5,8,10]`; separate QUACK YAMLs pass K=2. | Extended runs use 100 epochs; cross-validation configs use 500. |
| `archive/configs/extented/double_cake.yaml` | double cake | Random K sweep `[2,4,6,8,10]`; separate QUACK file passes K=2. | 100 epochs. |
| `archive/configs/all_datasets/real_world/*.yaml` | adult, covtype, one-vs-nonone, zero-vs-nonzero | K=4; power-fault uses `[2,4,6,8,10]`. | Adult/power fault 500 epochs; others 200. |
| `archive/configs/noise.yaml` | checkerboard | CCKA K sweep `[2,4,6,8,10]`. | 20 epochs; noise settings 0.01 through 0.10. |

In the current aligner, Random's `random_samples` and Greedy's `greedy_samples`
default to 4 if not supplied. Greedy additionally needs `landmark_points` for
its Nyström selection kernel; the comparison scripts set this to 10 (or 20 in
the real-world multi-seed script). The YAML's `clusters` value was a shared
legacy parameter and is not itself a complete method-specific specification.

## 5. Seeds and split variation

`BaseKTA._split_data()` uses `jax.random.permutation(PRNGKey(seed), n)` and
therefore changes the train/test split when the seed changes. The same seed is
also passed to circuit weight initialization. It also seeds RandomKTA's batch
permutation, Greedy's SVM, and centroid initialization/KMeans where applicable.
Thus, in the multi-seed comparisons, seeds vary **both data splits and initial
weights**, not just initial weights. Some legacy YAML-driven runs load already
split data and may use their own runner-specific split behavior; do not assume
those runs share the same seed semantics as the current package.

## 6. Are CCKA results/configurations consistent? Figures and README

No. The current parallel comparison, the older notebook, the archived YAMLs,
the saved result tables, and the README do not define one common reproducible
configuration. They differ in K, epochs, train/test split, number of seeds,
whether updates are manually repeated ten times, whether a score is last-epoch
or best-epoch, and whether the score is an average or a single run.

The parallel script explicitly labels its Fig. 4 as a **global-KTA vs test
accuracy scatter** pooling epoch/seed observations. Its Fig. 5 is **best test
accuracy per method**, after selecting K by mean per-seed best test accuracy
and selecting a best test-score epoch per seed. These are not the same
statistic. The saved `results/checkerboard/method_comparison_results.csv`
contains all CCKA epoch observations (mean test accuracy across those rows is
about 0.76, with a maximum of 1.0), while the paired
`results/checkerboard/summary_table.csv` reports CCKA's mean per-seed best
score as 0.9633 at K=8. That difference is consistent with plotting a typical
or final trajectory point near 0.8 versus a selected peak near 0.95; it does
not establish that the user's exact two plotted figures used those files.

There are also materially different result tables in the repository:
`results/checkerboard_summary_table.csv` reports a 5-seed CCKA best mean of
1.0 at K=4, whereas `results/checkerboard/summary_table.csv` reports a
10-seed mean of 0.9633 at K=8. The corresponding saved method-comparison rows
are different as well. These are evidence of separate configurations/data
snapshots, not a single controlled result.

The README's checkerboard CCKA 96.7% and QUACK 100% are presented as
unattributed point results. With the legacy 60-point, 50/50 checkerboard
configuration, 96.7% corresponds to 29 correct out of 30 test points, but the
README does not specify the run, seed, K, checkpoint rule, or result file.
The current ten-seed summary instead gives CCKA best mean 96.33% and QUACK
best mean 93.0%; another five-seed table gives different values. Therefore
the README values cannot be confirmed as outputs of the same configs as the
figure results. QUACK's 100% is plausible as a single run's score but is not
the saved ten-seed mean.

The repository-defined Fig. 6 in `method_comparision_parallel.py` is a
**cost-efficiency plot**, not a noise-robustness plot. Its x-axis is the
categorical method list and its y-axis is approximate accuracy per 1,000
circuit executions. There is no committed paper-to-script figure manifest, so
a “paper Fig. 6” referring to noise cannot be mapped uniquely from this
repository.

Two separate noise artifacts are present:

* `experiments/test_accuracy_vs_depolarising_noise.ipynb` loads the
  **checkerboard** dataset and sets `noisy=True` with
  `noise_type="depolarising"`. The same noisy kernel model is used to train,
  compute SVM training kernels, and compute test/inference kernels, so noise
  affects both training and evaluation. Although the notebook has code for a
  noise sweep, its checked-in invocation sets `noise_levels=[0]`; the
  nonzero values `[0.002, 0.004, 0.008, 0.01]` are commented out. As committed,
  its x-axis has only the value **0** and cannot support a robustness curve.
* `experiments/Noise Simulation Kernel Degradation.ipynb` also loads
  **checkerboard**, but its later kernel-similarity experiment constructs
  noisy kernels at `[0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`
  and compares them to a normal kernel. That is a kernel-evaluation
  degradation experiment, not noisy training followed by a separate clean
  test. Its noise type is `"coherent"` despite the other notebook's
  depolarising setup.

## Source map

* [CCKA aligner](./ccka/aligner/kta.py)
* [Kernel model and circuit counter](./ccka/models/kernel.py)
* [CCKA kernel circuit and feature map](./ccka/circuits/angleEmbeddingKernel.py)
* [Parallel comparison and figure generation](./experiments/method_comparision_parallel.py)
* [Cost-estimator plot](./experiments/computational_Cost.py)
* [Real-world multi-seed experiment](./experiments/real_world_datasets_multiseed.py)
* [Legacy configs](./archive/configs/)
* [README result snapshot](./README.md)
* [Saved checkerboard epoch results](./results/checkerboard/method_comparison_results.csv)
* [Saved checkerboard summary](./results/checkerboard/summary_table.csv)
