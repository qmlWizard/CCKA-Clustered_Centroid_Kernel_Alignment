# CCKA: Clustered Centroid Kernel Alignment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A scalable quantum kernel learning framework that **minimizes circuit executions** by aligning **clustered class centroids**. Developed as part of the MSc Thesis in Applied Computer Science at Deggendorf Institute of Technology.

---

## 🧠 Overview

This repository contains the official implementation of the **Clustered Centroid Kernel Alignment (CCKA)** method – a quantum kernel alignment technique that reduces the number of circuit executions by aligning only between class-representative centroids and data points.

CCKA is designed to be efficient on both synthetic and real-world datasets, supporting classical and quantum backends such as [PennyLane](https://pennylane.ai/) and [Qiskit](https://qiskit.org/).

> 📘 For full details, refer to the thesis:  
> *“Minimizing Circuit Execution Overhead with Clustered Centroid Kernel Alignment”*  
> [Digvijaysinh Ajarekar, 2025]

---

## 📁 Repository Structure

```
CCKA-Clustered_Centroid_Kernel_Alignment/
├── configs/                  # YAML configuration files for experiments
├── data/                     # (Empty) directory for input and generated datasets
├── plots/                    # Scripts and outputs for result visualizations
├── utils/                    # Utility modules for kernels, metrics, alignment loss
├── comparision.json          # Accuracy results for synthetic datasets
├── main.py                   # Entry point for training and evaluation
├── requirements.txt          # Python dependencies
├── test_ray.py               # Ray parallelization testing
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- pip
- (Optional) Virtual environment tools like `venv` or `conda`

### Setup
```bash
# Clone the repository
git clone https://github.com/qmlWizard/CCKA-Clustered_Centroid_Kernel_Alignment.git
cd CCKA-Clustered_Centroid_Kernel_Alignment

# Create virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate  # on Windows use `.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the Model (CCKA / QUACK / Random)
```bash
python main.py --backend pennylane --config configs/all_datasets/synthetic/checkerboard.yaml
```

Available methods in config:
- `ccka`: Clustered Centroid Kernel Alignment
- `quack`: Quantum Aligned Centroid Kernel
- `random`: Random representative subset
- `full`: Full kernel alignment (baseline)

### 2. Run Classical Baseline (RBF)
```bash
jupyter notebook classical_rbf.ipynb
```

---

## 🧪 Supported Datasets

The following datasets are supported via configuration files:

### 🧪 Synthetic:
- Moons
- Donuts
- Double Cake
- Checkerboard
- Corners

### 📊 Real-World:
- Adult Income
- MNIST (Zero vs Non Zero)
- MNIST (One vs None One)

Each dataset has a dedicated YAML config under `configs/`.

---

## 🧩 Methodology Summary

CCKA optimizes a variational quantum kernel using **class-representative centroids** rather than all pairwise data, resulting in:

- 💡 **Reduced Quantum Circuit Executions:** Linear in N × K (where K ≪ N)
- 🔬 **Faster Convergence:** Better performance with fewer iterations
- 📉 **Memory Efficient:** Smaller kernel matrices, ideal for large datasets
- 🧪 **Configurable Backends:** Supports PennyLane, Qiskit, and classical simulations

---

## 📈 Results Snapshot

| Dataset       | Initial Accuracy | CCKA Accuracy | QUACK Accuracy | Full Kernel |
|---------------|------------------|---------------|----------------|-------------|
| Checkerboard  | 80.0%            | **96.7%**     | 100%           | 100%        |
| Corners       | 89.0%            | **93.0%**     | 96.0%          | 94.0%       |
| Double Cake   | 83.3%            | **96.7%**     | 91.1%          | 73.3%       |
| Moons         | 82.9%            | **96.7%**     | 86.7%          | 96.7%       |
| Donuts        | 78.9%            | **85.0%**     | 86.7%          | 80.0%       |

---

## 📄 Citation

If you use this code or the CCKA methodology in your work, please cite:

```
@misc{ajarekar2025ccka,
  author       = {Digvijaysinh Ajarekar},
  title        = {CCKA: Clustered Centroid Kernel Alignment},
  year         = {2025},
  howpublished = {\url{https://github.com/qmlWizard/CCKA-Clustered_Centroid_Kernel_Alignment}},
  note         = {MSc Thesis, Deggendorf Institute of Technology}
}
```

---

## 🛠️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This work was developed as part of the master's thesis at **Deggendorf Institute of Technology**, under the supervision of **Prof. Dr. Helena Liebelt** and **Mr. Rodrigo Coelho**, AI Modelling, Fraunhofer IISB, Erlangen. Special thanks to supporting colleagues, research groups, and the open-source community.
---

## Distributed experiments

Run the configured sweep across a Ray cluster with:

```bash
python main.py --config configs/default.yaml
```

The default address is `auto`; when running under Slurm, the Python
distributor starts a persistent Ray head on the first allocated node and
connects the remaining allocated nodes itself. Without Slurm or an existing
`RAY_ADDRESS`, it starts a local single-node Ray runtime. Use
`--address local` for an explicit single-machine Ray run.
Results are written per trial and combined into a summary under
`results/<run-name>/`.
The runner uses JAX/Optax when available and falls back to PyTorch otherwise;
install a backend with `pip install .[jax]` or `pip install .[torch]`.

## Experiment suite

Run an individual configuration with
`sbatch --export=ALL,CONFIG=configs/exp06_k_sweep.yaml jobs/run_all.slurm`.
The distributor uses every available Ray node and falls back to one local
node when no cluster is available.

Configs live in `configs/` (`base.yaml` plus `exp01…exp15`, each inheriting with
`base: base.yaml`). Submit everything with `bash jobs/submit_all.sh`, or one
experiment with
`sbatch --nodes=8 --export=ALL,CONFIG=configs/exp06_k_sweep.yaml jobs/run_experiments.slurm`
(output: `results/<config-name>/`). Single-node analyses:
`sbatch --export=ALL,ANALYSIS=<budget|scalability|kernel|ansatz|stats|plots> jobs/run_analysis.slurm`.

| Plan section | Config / script | Plots |
|---|---|---|
| 0, 1, 11 main benchmark, 10 seeds, val-based checkpoint selection | `exp01_main_benchmark`, `experiments/stats_report.py` | grouped bars per ansatz, ansatz×method / ansatz×dataset heatmaps, Pareto, critical-difference, summary tables with Wilcoxon p-values (bold only if significant) |
| 1 multiclass (C>2) | `exp02_multiclass` (`training.multiclass_strategy: ovr\|ovo`) | strategy bars |
| 2 accuracy vs circuit budget | `experiments/accuracy_vs_executions.py` (job `ANALYSIS=budget`) | accuracy vs evaluations, cost-to-target |
| 3 scalability, Nyström | `experiments/scalability.py` | evals/epoch and time vs N (log-log), time vs qubits/layers, stacked pipeline cost, accuracy vs landmarks |
| 4 ablations (**corners, checkerboard only**) | `exp03` structure/clustering/fixed centroids, `exp04` λ_KAO×λ_CO, `exp05` MCLR×CCLR×LR | Δ-accuracy bars, learning curves, heatmaps |
| 5 K sweep, multimodality | `exp06_k_sweep` | accuracy vs K (+cost axis), by modes, optimal K vs modes |
| 6 data regimes | `exp07` N_train, `exp08` label noise, `exp09` imbalance, `exp10` PCA dim | curves per dataset |
| 7 kernel analysis (**corners, checkerboard only**) | `experiments/kernel_analysis.py` | kernel heatmaps, eigenspectra, violins, CKA, KTA vs accuracy, effective rank, decision boundaries |
| 8 noise | `exp11_noise` (depolarising, amplitude/phase damping, coherent, shots-only), `exp12_shots` | accuracy vs level, noise×method heatmap, accuracy vs shots |
| 9 ansätze | `exp13_ansatz_depth`, `experiments/ansatz_analysis.py` | accuracy vs depth/params, gradient variance, expressibility, Meyer-Wallach, rank stability (Kendall τ) |
| 10 optimiser / init | `exp14_optimizer_init` (Adam/SGD, init scale 0/0.1/1, epochs) | init sensitivity, epochs-to-plateau |

**Ansätze** (`ansatze:` in a config, `--ansatz` in scripts): `crz_ring`
(default, identical to the original circuit), `hardware_efficient`, `iqp`,
`zz_feature_map`, `strongly_entangling`, `separable` (no-entanglement control).

**Scope rule:** ablations run only on `corners` and `checkerboard`
(`experiments.ablation_datasets`); all other datasets only appear in accuracy
experiments, and kernel analysis rejects other datasets.

**Synthetic datasets and the CCKA behaviour each one tests**
(`experiments/datasets.py::SYNTHETIC_CLAIMS`, evaluated by `stats_report.py` →
`claims_report.md`): `rings` (equal class means; main centroid uninformative),
`xor_grid` (accuracy rises with K until each cell has a representative),
`gmm_modes_1/2/4` (gap to QUACK grows with modes per class; 1 mode is the
control), `rare_mode` (sub-centroids cover a rare mode; macro-F1),
`spirals` (accuracy per circuit budget), `moons` (low-data generalisation),
`overlapping_blobs` (label-noise robustness), `circles`/`double_cake`
(sanity), `multiclass_modes`/`three_rings` (C>2, OvO vs OvR).

**Not implemented (need new algorithm or hardware work):** bisecting k-means /
k-medoids, joint KAO+CO updates and KAO:CO ratios, centred-TA / cross-entropy
losses, one-hot/pairwise targets, sub↔sub terms, box penalty, >2 hierarchy
levels, COBYLA/SPSA/L-BFGS, train-noisy/test-clean evaluation, real hardware /
fake backends, AUC, Fashion-MNIST, centroid-trajectory PCA/t-SNE, per-epoch
effective rank, per-method gradient variance. Multiclass is JAX-backend only;
the Qiskit hardware circuit supports only `crz_ring`. FullKTA at N=10⁴ needs
~10⁸ circuits per epoch, so `scalability.py` caps it with `--max-full-n`.
