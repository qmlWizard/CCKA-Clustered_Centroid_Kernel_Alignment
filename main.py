import torch
import ray
from ray import tune
from ray.air import session
from sklearn.model_selection import train_test_split
import yaml
import argparse
import shutil
from collections import namedtuple
import os
import datetime


# Custom Libraries
from utils.data_generator import DataGenerator
from utils.agent import TrainModel
from utils.helper import to_python_native, gen_experiment_name, set_seed, save_model_state
from utils.plotter import alignment_progress_over_iterations, plot_initial_final_accuracies

# === Backend Configuration ===
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

set_seed(42)

def train(config):
    name_str = f"_{config['name']}_{config['n_qubits']}_{config['ansatz']}_{config['ansatz_layers']}_{config['optimizer']}_{config['lr']}_{config['mclr']}_{config['cclr']}_{config['train_method']}_{config['lambda_kao']}_{config['lambda_co']}_{config['clusters']}_Kmeans_{config['use_kmeans']}"

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


    kernel = Qkernel(
        device=config['device'],
        n_qubits=config['n_qubits'],
        trainable=config['trainable'],
        input_scaling=config['input_scaling'],
        data_reuploading=config['data_reuploading'],
        ansatz=config['ansatz'],
        ansatz_layers=config['ansatz_layers']
    )

    agent = TrainModel(
        kernel=kernel,
        training_data=training_data,
        training_labels=training_labels,
        testing_data=testing_data,
        testing_labels=testing_labels,
        optimizer=config['optimizer'],
        lr=config['lr'],
        mclr=config['mclr'],
        cclr=config['cclr'],
        epochs=config['epochs'],
        train_method=config['train_method'],
        target_accuracy=config['target_accuracy'],
        get_alignment_every=config['get_alignment_every'],
        validate_every_epoch=config['validate_every_epoch'],
        base_path=config['base_path'],
        lambda_kao=config['lambda_kao'],
        lambda_co=config['lambda_co'],
        clusters=config['clusters'],
        get_decesion_boundary = config['decesion_boundary'],
        use_kmeans= config['use_kmeans']
    )

    if args.backend == 'qiskit':
        before_metrics = agent.evaluate_parallel(testing_data, testing_labels, 'before')
    else:
        before_metrics = agent.evaluate_test(testing_data, testing_labels, 'before')

    print(before_metrics)

    if multi_class == True:
        kernel, params, main_centroid, sub_centroid = agent.fit_multiclass(training_data, training_labels)
    else:
        kernel, params, main_centroid, sub_centroid = agent.fit_kernel(training_data, training_labels)
    print('Training Complete')
    
    if args.backend == 'qiskit':
        after_metrics = agent.evaluate_parallel(testing_data, testing_labels, 'after')
    else:
        after_metrics = agent.evaluate_test(testing_data, testing_labels, 'after')
    print(after_metrics)3 
    
    metrics = {
        "num_layers": config['ansatz_layers'],
        "accuracy_train_init": before_metrics['training_accuracy'],
        "accuracy_test_init": before_metrics['testing_accuracy'],
        "alignment_train_init": before_metrics['alignment'],
        "accuracy_train_final": after_metrics['training_accuracy'],
        "accuracy_test_final": after_metrics['testing_accuracy'],
        "alignment_train_epochs": after_metrics['alignment_arr'],
        "circuit_executions": after_metrics['executions'],
    }

    metrics = to_python_native(metrics)
    exp_name = gen_experiment_name(config)

    alignment_arr = after_metrics["alignment_arr"]
    alignment_progress_over_iterations(
                                       alignment_arr,
                                       path = config['base_path'],
                                       title = f"alignment_{name_str}"
                                      )

    plot_initial_final_accuracies(
                                  before_metrics['training_accuracy'], 
                                  before_metrics['testing_accuracy'], 
                                  after_metrics['training_accuracy'], 
                                  after_metrics['testing_accuracy'],
                                  path = config['base_path'],
                                  title= f"accuracy_{name_str}"
                                )

    save_model_state(kernel, params, main_centroid, sub_centroid, f"{config['base_path']}/model/model_{name_str}.pth")
    
    session.report(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This parser receives the yaml config file")
    parser.add_argument("--backend", default="qiskit")
    parser.add_argument("--config", default="configs/checkerboard.yaml")
    args = parser.parse_args()

    if args.backend == 'qiskit':
        from utils.qiskit.model import Qkernel
    elif args.backend == 'pennylane':
        from utils.pennylane.model import Qkernel

    with open(args.config) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    config = namedtuple("ObjectName", data.keys())(*data.values())

    ray.init(log_to_driver=False)
    search_space = {
        'name': config.dataset['name'],
        'file': None if config.dataset['file'] == 'None' else config.dataset['file'],
        'n_samples': config.dataset['n_samples'],
        'noise': config.dataset['noise'],
        'num_sectors': config.dataset['num_sectors'],
        'points_per_sector': config.dataset['points_per_sector'],
        'grid_size': config.dataset['grid_size'],
        'sampling_radius': config.dataset['sampling_radius'],
        'training_size': config.dataset['training_size'],
        'testing_size': config.dataset['testing_size'],
        'validation_size': config.dataset['validation_size'],
        'device': config.qkernel['device'],
        'n_qubits': tune.grid_search(config.qkernel['n_qubits']),
        'trainable': config.qkernel['trainable'],
        'input_scaling': config.qkernel['input_scaling'],
        'data_reuploading': config.qkernel['data_reuploading'],
        'ansatz': tune.grid_search(config.qkernel['ansatz']),
        'ansatz_layers': tune.grid_search(config.qkernel['ansatz_layers']),
        'optimizer': tune.grid_search(config.agent['optimizer']),
        'lr': tune.grid_search(config.agent['lr']),
        'mclr': tune.grid_search(config.agent['mclr']),
        'cclr': tune.grid_search(config.agent['cclr']),
        'epochs': tune.grid_search(config.agent['epochs']),
        'train_method': tune.grid_search(config.agent['train_method']),
        'target_accuracy': config.agent['target_accuracy'],
        'get_alignment_every': config.agent['get_alignment_every'],
        'validate_every_epoch': config.agent['validate_every_epoch'],
        'base_path': config.agent['base_path'],
        'lambda_kao': tune.grid_search(config.agent['lambda_kao']),
        'lambda_co': tune.grid_search(config.agent['lambda_co']),
        'clusters': tune.grid_search(config.agent['clusters']),
        'decesion_boundary': config.agent['decesion_boundary'],
        'use_kmeans': tune.grid_search(config.agent['use_kmeans']),
        'ray_logging_path': config.ray_config['ray_logging_path']
    }

    def trial_name_creator(trial):
        return trial.__str__() + '_' + trial.experiment_tag + ','

    tuner = tune.Tuner(
                        tune.with_resources(train, 
                                            resources={"cpu": config.ray_config['num_cpus'], 
                                                    "gpu": config.ray_config['num_gpus']}
                                            ),
                        
                        tune_config=tune.TuneConfig(
                                                    num_samples=config.ray_config['ray_num_trial_samples'],
                                                    trial_dirname_creator=trial_name_creator,
                                                ),

                        param_space=search_space,
                    )

    tuner.fit()
    ray.shutdown()
