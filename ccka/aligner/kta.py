"""Backend-selected public entry points for kernel-alignment optimizers."""

from ccka.backend import BACKEND

if BACKEND == "jax":
    from .kta_jax import (
        BaseKTA,
        CentroidBasedKTA,
        FullKTA,
        GreedyKTA,
        QuackKTA,
        RandomKTA,
        centroidBasedKTA,
        fullKTA,
        greedyKTA,
        print_training_summary,
        quackKTA,
        randomKTA,
    )
else:
    from .torch_kta import (
        TorchCentroidBasedKTA as CentroidBasedKTA,
        TorchExperimentKTA as BaseKTA,
        TorchFullKTA as FullKTA,
        TorchQuackKTA as QuackKTA,
        TorchRandomKTA as RandomKTA,
    )

    centroidBasedKTA = CentroidBasedKTA
    fullKTA = FullKTA
    quackKTA = QuackKTA
    randomKTA = RandomKTA

    def print_training_summary(history, width=78):
        del width
        print(
            f"Validation accuracy: {history['best_validation_accuracy']:.4f}; "
            f"test accuracy: {history['test_accuracy_history'][-1]:.4f}; "
            f"circuit executions: {history['circuit_executions']}"
        )


__all__ = [
    "BaseKTA",
    "CentroidBasedKTA",
    "FullKTA",
    "QuackKTA",
    "RandomKTA",
    "print_training_summary",
    "centroidBasedKTA",
    "fullKTA",
    "randomKTA",
    "quackKTA",
]

if BACKEND == "jax":
    __all__.extend(["GreedyKTA", "greedyKTA"])
