import torch
from sklearn.model_selection import train_test_split
import yaml
import argparse
import shutil
from collections import namedtuple
import os
import datetime
import sys
from pathlib import Path
from qiskit.providers.aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.test.mock import FakeLima

# Custom Libraries
from utils.data_generator import DataGenerator
from utils.agent import TrainModel
from utils.helper import to_python_native, gen_experiment_name, set_seed, save_model_state, _now_iso, _safe_bool_str, _ensure_dir, _append_csv_row
from utils.qiskit.model import Qkernel
from utils.logger import Logger

# === Backend Configuration ===
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print("Select device: ", device)
set_seed(42)



def test_noisy_accuracy(config):
    logger = Logger(dataset_name=config['name'], log_dir=config.get('base_path'), mirror_json=False)

    repeat_idx = int(config.get('repeat', 1))

    data_generator = DataGenerator(
        dataset_name=config['name'],
        file_path=config['file'],
        n_samples=config['n_samples'],
        noise=config['noise'],
        num_sectors=config['num_sectors'],
        points_per_sector=config['points_per_sector'],
        grid_size=config['grid_size'],
        sampling_radius=config['sampling_radius']
    )

    training_data, training_labels, testing_data, testing_labels = data_generator.generate_dataset()
    training_data = torch.tensor(training_data.to_numpy(), dtype=torch.float32, requires_grad=True)
    testing_data = torch.tensor(testing_data.to_numpy(), dtype=torch.float32, requires_grad=True)
    training_labels = torch.tensor(training_labels.to_numpy(), dtype=torch.int)
    testing_labels = torch.tensor(testing_labels.to_numpy(), dtype=torch.int)
    multi_class = True if int(training_labels.unique().numel()) == 2 else False

    # === NEW/UPDATED: run identifiers and static fields ===
    exp_name = gen_experiment_name(config)
    run_id = f"{exp_name}_rep{repeat_idx}"
    dataset_name = config['name']
    method = str(config.get('train_method', 'CCKA')) or "CCKA"
    subcentroids = config.get('clusters', "")
    noise_level = config.get('noise', "")
    mitigation = _safe_bool_str(config.get('mitigation', ""))  # not in your search_space yet; remains blank
    n_samples = config.get('n_samples', "")

    # Setup fake backend with noise model
    fake_backend = FakeLima()
    noise_model = NoiseModel.from_backend(fake_backend)
    simulator = AerSimulator(noise_model=noise_model,
                             basis_gates=noise_model.basis_gates,
                             coupling_map=fake_backend.configuration().coupling_map)

    # === Kernel/model ===
    kernel = Qkernel(
        n_qubits=2,
        trainable=config.getboolean("trainable", False),
        input_scaling=config.getboolean('input_scaling', True),
        data_reuploading=config.getboolean('data_reuploading', False),
        ansatz=config.get('ansatz', {'he'}),
        ansatz_layers=config.get('ansatz_layers', []),
        use_noisy_backend=False,
        simulator=simulator,
    )

    kernel_noisy = Qkernel(
        n_qubits=2,
        trainable=config.getboolean("trainable", False),
        input_scaling=config.getboolean('input_scaling', True),
        data_reuploading=config.getboolean('data_reuploading', False),
        ansatz=config.get('ansatz', {'he'}),
        ansatz_layers= config.get('ansatz_layers', []),
        use_noisy_backend=True,
        simulator= simulator,
    )

    # --- Fitting Kernel without Noise ---
    kernel.fit(training_data, training_labels, testing_data, testing_labels)

    # --- Fitting kernel with Noise ---
    kernel_noisy.fit(training_data, training_labels, testing_data, testing_labels)


