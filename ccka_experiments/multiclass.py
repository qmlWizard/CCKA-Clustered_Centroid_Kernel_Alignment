"""Multiclass training by reduction to binary kernel-alignment problems.

The alignment objectives (KTA, centroid TA) are defined for ±1 targets, so a
multiclass task is split into binary problems, each with its own trainable
kernel:

* ``ovr`` (one-vs-all): one problem per class, ``class`` vs. the rest;
  prediction is the arg-max of the per-class SVM decision values.
* ``ovo`` (one-vs-one): one problem per class pair trained on that pair only;
  prediction is a majority vote (ties broken by the summed decision values).
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC

STRATEGY_ALIASES = {
    "ovr": "ovr",
    "ova": "ovr",
    "one_vs_all": "ovr",
    "one_vs_rest": "ovr",
    "ovo": "ovo",
    "one_vs_one": "ovo",
}


def normalise_strategy(value: str) -> str:
    try:
        return STRATEGY_ALIASES[str(value).lower()]
    except KeyError as exc:
        raise ValueError(
            f"multiclass_strategy must be one of {sorted(STRATEGY_ALIASES)}, "
            f"got {value!r}."
        ) from exc


def binary_problems(
    classes: np.ndarray, strategy: str
) -> list[tuple[Any, ...]]:
    if strategy == "ovr":
        return [(label,) for label in classes]
    return list(itertools.combinations(classes, 2))


def _relabel(y: np.ndarray, problem: tuple[Any, ...]) -> np.ndarray:
    if len(problem) == 1:
        return np.where(y == problem[0], 1, -1).astype(np.int32)
    return np.where(y == problem[1], 1, -1).astype(np.int32)


def _subset(problem: tuple[Any, ...], *arrays: tuple[np.ndarray, np.ndarray]):
    """Restrict (X, y) pairs to the classes of a pairwise problem."""
    if len(problem) == 1:
        return arrays
    restricted = []
    for X, y in arrays:
        mask = np.isin(y, problem)
        restricted.append((X[mask], y[mask]))
    return tuple(restricted)


class MulticlassAligner:
    """Wrap a binary aligner factory into a C-class classifier.

    ``factory(data) -> (aligner, kernel_model)`` receives the usual
    ``(X_train, X_validation, X_test, y_train, y_validation, y_test)`` tuple
    with ±1 labels and must return an object exposing ``align()``,
    ``weights``, ``xtrain`` and ``_evaluation_kernel_matrices``.
    """

    inference_method = "svm"

    def __init__(
        self,
        factory: Callable[[tuple[np.ndarray, ...]], tuple[Any, Any]],
        data: tuple[np.ndarray, ...],
        *,
        strategy: str,
        svm_c: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.strategy = normalise_strategy(strategy)
        self.factory = factory
        self.data = data
        self.svm_c = svm_c
        self.seed = seed
        self.classes = np.unique(data[3])
        if len(self.classes) < 3:
            raise ValueError("MulticlassAligner needs at least three classes.")
        self.problems = binary_problems(self.classes, self.strategy)

    def _binary_data(self, problem: tuple[Any, ...]):
        X_tr, X_va, X_te, y_tr, y_va, y_te = self.data
        (train, validation, test) = _subset(
            problem, (X_tr, y_tr), (X_va, y_va), (X_te, y_te)
        )
        for name, (_, labels) in (("training", train), ("validation", validation)):
            if len(np.unique(_relabel(labels, problem))) < 2:
                raise ValueError(
                    f"The {name} split has a single class for problem "
                    f"{problem}; use more samples."
                )
        return (
            train[0],
            validation[0],
            test[0],
            _relabel(train[1], problem),
            _relabel(validation[1], problem),
            _relabel(test[1], problem),
        )

    def _decision_values(self, aligner, y_binary, queries):
        K_train, K_validation, K_test = aligner._evaluation_kernel_matrices(
            aligner.weights, aligner.xtrain, queries[0], queries[1]
        )
        svm = SVC(
            kernel="precomputed",
            C=self.svm_c,
            max_iter=10_000,
            random_state=self.seed,
        ).fit(np.asarray(K_train), y_binary)
        return (
            svm.decision_function(np.asarray(K_validation)),
            svm.decision_function(np.asarray(K_test)),
        )

    def _predict(self, scores: list[np.ndarray]) -> np.ndarray:
        stacked = np.stack(scores, axis=1)
        if self.strategy == "ovr":
            return self.classes[np.argmax(stacked, axis=1)]
        votes = np.zeros((stacked.shape[0], len(self.classes)))
        confidence = np.zeros_like(votes)
        index = {label: k for k, label in enumerate(self.classes)}
        for column, (first, second) in enumerate(self.problems):
            positive = stacked[:, column] > 0
            winner = np.where(positive, index[second], index[first])
            votes[np.arange(len(winner)), winner] += 1.0
            confidence[:, index[second]] += stacked[:, column]
            confidence[:, index[first]] -= stacked[:, column]
        return self.classes[np.argmax(votes + 1e-3 * confidence, axis=1)]

    def align(self) -> dict[str, Any]:
        started = time.perf_counter()
        X_validation, X_test = self.data[1], self.data[2]
        y_validation, y_test = self.data[4], self.data[5]
        validation_scores, test_scores = [], []
        histories, models = [], []
        for problem in self.problems:
            binary = self._binary_data(problem)
            aligner, model = self.factory(binary)
            histories.append(aligner.align())
            models.append(model)
            val_scores, te_scores = self._decision_values(
                aligner, binary[3], (X_validation, X_test)
            )
            validation_scores.append(val_scores)
            test_scores.append(te_scores)

        validation_pred = self._predict(validation_scores)
        test_pred = self._predict(test_scores)
        metrics = {
            "validation_accuracy": float(
                accuracy_score(y_validation, validation_pred)
            ),
            "test_accuracy": float(accuracy_score(y_test, test_pred)),
            "f1_score": float(
                f1_score(y_test, test_pred, average="macro", zero_division=0)
            ),
            "precision_score": float(
                precision_score(
                    y_test, test_pred, average="macro", zero_division=0
                )
            ),
            "recall_score": float(
                recall_score(y_test, test_pred, average="macro", zero_division=0)
            ),
        }
        epochs = min(
            len(h.get("validation_accuracy_history", [])) for h in histories
        )
        # Mean binary validation accuracy per epoch (not the C-class accuracy).
        mean_history = (
            np.mean(
                [h["validation_accuracy_history"][:epochs] for h in histories],
                axis=0,
            ).tolist()
            if epochs
            else []
        )
        return {
            "final_svm_metrics": metrics,
            "validation_accuracy_history": mean_history,
            "best_validation_accuracy": metrics["validation_accuracy"],
            "circuit_executions": int(
                sum(model.circuit_executions for model in models)
            ),
            "n_binary_problems": len(self.problems),
            "time": time.perf_counter() - started,
        }
