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
