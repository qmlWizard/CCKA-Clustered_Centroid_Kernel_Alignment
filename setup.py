from setuptools import setup, find_packages

setup(
    name="ccka",
    version="0.1.0",
    description="Clustered Centroid Kernel Alignment for Quantum and Classical Kernels",
    packages=find_packages(),
    python_requires=">=3.9,<3.12",
    install_requires=[
        # Quantum
        "pennylane>=0.38,<0.41",
        "PennyLane-Lightning>=0.38,<0.41",
        "pennylane-qiskit>=0.38,<0.41",

        # JAX (Python-version specific)
        "jax==0.4.30; python_version<'3.11'",
        "jaxlib==0.4.30; python_version<'3.11'",

        "jax>=0.4.30,<0.5; python_version>='3.11'",
        "jaxlib>=0.4.30,<0.5; python_version>='3.11'",

        # Scientific Computing
        "numpy>=1.24,<3",
        "scipy>=1.11,<2",
        "scikit-learn>=1.4,<2",
        "pandas>=2.0,<3",

        # Visualization
        "matplotlib>=3.8,<4",
        "seaborn>=0.13,<1",
        "plotly>=6,<7",

        # Utilities
        "ray>=2.40,<3",
        "requests>=2.32,<3",
        "PyYAML>=6,<7",
        "tqdm>=4.66",
    ],
)