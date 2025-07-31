# CCKA: Clustered Centroid Kernel Alignment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A scalable quantum kernel learning framework that minimizes quantum circuit executions by leveraging clustered centroids for kernel alignment. Developed as part of the MSc Thesis in Applied Computer Science at Deggendorf Institute of Technology.

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
│
├── classical_rbf.ipynb       # Baseline classical kernel experiments (e.g., RBF)
├── comparision.json          # Accuracy results for synthetic datasets
├── create_datasets.ipynb     # Notebook to generate synthetic datasets
├── cross_validate.py         # Double-cake and cross-validation runner
├── main.py                   # Entry point for training and evaluation
├── requirements.txt          # Python dependencies
├── results.ipynb             # Notebook to visualize and analyze results
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
python main.py --backend pennylane --config configs/all_datasets/synthetic/donuts.yaml
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

### 3. Generate Synthetic Datasets
```bash
jupyter notebook create_datasets.ipynb
```

### 4. Visualize Results
```bash
jupyter notebook results.ipynb
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
- Microgrid Fault Detection
- Network Intrusion (KDD’99, DoH)
- Forest Covertype

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
| Moons         | 82.2%            | **96.7%**     | 86.7%          | 93%         |
| Donuts        | 78.9%            | **96.7%**     | 100%           | 100%        |
| Checkerboard  | 82.5%            | **96.7%**     | 100%           | 100%        |
| Double Cake   | 80%              | **96.7%**     | 86.7%          | 73.3%       |

See [results.ipynb](notebooks/results.ipynb) for full plots.

---

## 📄 Citation

If you use this code or the CCKA methodology in your work, please cite:

```
@misc{ajarekar2025ccka,
  author       = {Digvijaysinh Ajarekar},
  title        = {CCKA: Clustered Centroid Kernel Alignment - Thesis Code},
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

This work was developed as part of the master's thesis at **Deggendorf Institute of Technology**, under the supervision of **Prof. Dr. Helena Liebelt**. Special thanks to supporting colleagues, research groups, and the open-source community.
