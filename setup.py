from setuptools import setup, find_packages

setup(
    name="ccka",
    version="0.1.0",
    description="Clustered Centroid Kernel Alignment for Quantum and Classical Kernels",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pennylane==0.38.0",
        "PennyLane-Lightning==0.38.0",
        "pennylane-qiskit==0.38.0",

        "jax==0.4.17",
        "jaxlib==0.4.17",

        "scikit-learn==1.6.1",
        "scipy==1.11.4",
        "pandas==2.3.1",

        "matplotlib==3.9.4",
        "seaborn==0.13.2",
        "plotly==6.2.0",

        "ray==2.48.0",
        "requests==2.32.4",
        "PyYAML==6.0.2",
        "tqdm"
    ],
)
