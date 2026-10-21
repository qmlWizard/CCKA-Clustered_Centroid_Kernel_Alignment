"""PyTorch implementations for running experiments without JAX/Optax."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC


class TorchExperimentKTA:
    """Torch-backed Full/Random KTA, QUACK, and CCKA experiment aligner."""

    def __init__(
        self,
        kernel_model,
        data,
        labels,
        *,
        method: str,
        validation_data,
        validation_labels,
        test_data,
        test_labels,
        epochs: int = 100,
        learning_rate: float = 0.01,
        random_samples: int = 4,
        centroids: int = 4,
        clustering: str = "regular",
        centroid_lr: float | None = None,
        sub_centroid_lr: float | None = None,
        lambda_kao: float = 0.001,
        lambda_co: float = 0.001,
        svm_c: float = 1.0,
        inference_landmarks: int = 32,
        inner_steps: int = 10,
        update_kao: bool = True,
        update_co: bool = True,
        seed: int = 0,
        **_: Any,
    ) -> None:
        if method not in {"fullKTA", "randomKTA", "quackKTA", "ccka"}:
            raise ValueError(f"Unsupported PyTorch experiment method {method!r}.")
        if method == "ccka" and clustering not in {"regular", "random", "kmeans"}:
            raise ValueError("clustering must be regular, random, or kmeans.")
        self.kernel_model = kernel_model
        self.method = method
        self.xtrain = self._tensor(data)
        self.ytrain = self._tensor(labels).reshape(-1)
        self.xvalidation = self._tensor(validation_data)
        self.yvalidation = self._tensor(validation_labels).reshape(-1)
        self.xtest = self._tensor(test_data)
        self.ytest = self._tensor(test_labels).reshape(-1)
        self.epochs = epochs
        self.random_samples = random_samples
        self.lambda_kao = lambda_kao
        self.lambda_co = lambda_co
        self.svm_c = svm_c
        self.inference_landmarks = inference_landmarks
        self.inner_steps = inner_steps
        self.update_kao = update_kao
        self.update_co = update_co
        self.seed = seed
        self.weights = torch.nn.Parameter(
            torch.as_tensor(
                np.array(
                    kernel_model.circuit.init_weights(seed=seed), copy=True
                ),
                dtype=torch.float32,
            )
        )
        self.weight_optimizer = torch.optim.Adam([self.weights], lr=learning_rate)

        self.main_centroids: torch.nn.Parameter | None = None
        self.sub_centroids: torch.nn.Parameter | None = None
        if method in {"ccka", "quackKTA"}:
            main, main_labels, sub, sub_labels = self._initialize_centroids(
                centroids, clustering
            )
            self.main_centroids = torch.nn.Parameter(main)
            self.main_centroid_labels = main_labels
            self.sub_centroids = torch.nn.Parameter(sub) if method == "ccka" else None
            self.sub_centroid_labels = sub_labels
            self.main_optimizer = torch.optim.Adam(
                [self.main_centroids],
                lr=centroid_lr if centroid_lr is not None else learning_rate,
            )
            if self.sub_centroids is not None:
                self.sub_optimizer = torch.optim.Adam(
                    [self.sub_centroids],
                    lr=sub_centroid_lr
                    if sub_centroid_lr is not None
                    else centroid_lr
                    if centroid_lr is not None
                    else learning_rate,
                )

    @staticmethod
    def _tensor(value) -> torch.Tensor:
        return torch.as_tensor(np.asarray(value), dtype=torch.float32)

    def _initialize_centroids(self, count: int, clustering: str):
        X = self.xtrain.detach().cpu().numpy()
        y = self.ytrain.detach().cpu().numpy()
        classes = np.unique(y)
        mains, main_labels, subs, sub_labels = [], [], [], []
        for class_index, label in enumerate(classes):
            class_data = X[y == label]
            mains.append(class_data.mean(axis=0))
            main_labels.append(label)
            if clustering == "kmeans" and len(class_data) >= count:
                selected = KMeans(
                    n_clusters=count, n_init="auto", random_state=self.seed
                ).fit(class_data).cluster_centers_
            elif clustering == "random":
                rng = np.random.default_rng(self.seed + class_index)
                selected = class_data[
                    rng.choice(len(class_data), size=count, replace=len(class_data) < count)
                ]
            else:
                shuffled = class_data[
                    np.random.default_rng(self.seed).permutation(len(class_data))
                ]
                chunks = np.array_split(shuffled, count)
                selected = np.stack(
                    [
                        chunk.mean(axis=0)
                        if len(chunk)
                        else shuffled[index % len(shuffled)]
                        for index, chunk in enumerate(chunks)
                    ]
                )
            subs.extend(selected)
            sub_labels.extend([label] * count)
        return (
            torch.tensor(np.asarray(mains), dtype=torch.float32),
            torch.tensor(main_labels, dtype=torch.float32),
            torch.tensor(np.asarray(subs), dtype=torch.float32),
            torch.tensor(sub_labels, dtype=torch.float32),
        )

    def _kernel_vector(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.kernel_model.forward(left, right, self.weights).reshape(-1)

    def _matrix(self, X: torch.Tensor) -> torch.Tensor:
        n = len(X)
        rows, cols = torch.triu_indices(n, n)
        values = self._kernel_vector(X[rows], X[cols])
        matrix = torch.zeros((n, n), dtype=values.dtype)
        matrix[rows, cols] = values
        matrix[cols, rows] = values
        return matrix

    def _nystrom_matrices(self, query: torch.Tensor | None = None):
        n = len(self.xtrain)
        count = min(self.inference_landmarks, n)
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(n, generator=generator)[:count]
        landmarks = self.xtrain[indices]
        train_landmark = self._kernel_vector(
            self.xtrain.repeat_interleave(count, dim=0),
            landmarks.repeat(n, 1),
        ).reshape(n, count)
        landmark_kernel = self._kernel_vector(
            landmarks.repeat_interleave(count, dim=0),
            landmarks.repeat(count, 1),
        ).reshape(count, count)
        landmark_kernel = (landmark_kernel + landmark_kernel.T) * 0.5
        landmark_inverse = torch.linalg.pinv(landmark_kernel)
        train_kernel = train_landmark @ landmark_inverse @ train_landmark.T
        query_kernel = None
        if query is not None:
            query_count = len(query)
            query_landmark = self._kernel_vector(
                query.repeat_interleave(count, dim=0),
                landmarks.repeat(query_count, 1),
            ).reshape(query_count, count)
            query_kernel = query_landmark @ landmark_inverse @ train_landmark.T
        return train_kernel.detach().numpy(), (
            None if query_kernel is None else query_kernel.detach().numpy()
        )

    def _svm_metrics(self, *, evaluate_test: bool = False):
        train_kernel, validation_kernel = self._nystrom_matrices(self.xvalidation)
        svm = SVC(kernel="precomputed", C=self.svm_c, random_state=self.seed)
        y_train = self.ytrain.detach().numpy()
        y_validation = self.yvalidation.detach().numpy()
        svm.fit(train_kernel, y_train)
        train_prediction = svm.predict(train_kernel)
        validation_prediction = svm.predict(validation_kernel)
        metrics = {
            "train_accuracy": float(accuracy_score(y_train, train_prediction)),
            "validation_accuracy": float(
                accuracy_score(y_validation, validation_prediction)
            ),
        }
        if evaluate_test:
            _, test_kernel = self._nystrom_matrices(self.xtest)
            y_test = self.ytest.detach().numpy()
            prediction = svm.predict(test_kernel)
            metrics.update(
                {
                    "test_accuracy": float(accuracy_score(y_test, prediction)),
                    "f1_score": float(f1_score(y_test, prediction, average="macro")),
                    "precision_score": float(
                        precision_score(y_test, prediction, average="macro")
                    ),
                    "recall_score": float(
                        recall_score(y_test, prediction, average="macro")
                    ),
                }
            )
        return metrics

    @staticmethod
    def _alignment(kernel: torch.Tensor, target: torch.Tensor, sign: float = 1.0):
        target = target.to(dtype=kernel.dtype)
        return sign * torch.dot(kernel, target) / (
            torch.linalg.vector_norm(kernel) * torch.linalg.vector_norm(target) + 1e-10
        )

    def _centroid_loss(self, main: torch.Tensor, points: torch.Tensor,
                       labels: torch.Tensor, sign: float, regularizer: float,
                       *, weight_penalty: bool = False):
        repeated = main.unsqueeze(0).expand(len(points), -1)
        kernel = self._kernel_vector(repeated, points)
        penalty = torch.zeros((), dtype=kernel.dtype)
        if regularizer and weight_penalty:
            penalty = self.weights.square().mean()
        elif regularizer and self.method == "ccka":
            penalty = torch.relu(main - 1.0).sum() + torch.relu(-main).sum()
        elif regularizer:
            penalty = torch.relu(main - self.xtrain.max(dim=0).values).sum()
            penalty = penalty + torch.relu(self.xtrain.min(dim=0).values - main).sum()
        return 1.0 - self._alignment(kernel, labels, sign) + regularizer * penalty

    def _step(self, epoch: int):
        if self.method in {"fullKTA", "randomKTA"}:
            if self.method == "randomKTA":
                count = min(self.random_samples, len(self.xtrain))
                indices = torch.randperm(
                    len(self.xtrain), generator=torch.Generator().manual_seed(self.seed + epoch)
                )[:count]
                points, labels = self.xtrain[indices], self.ytrain[indices]
            else:
                points, labels = self.xtrain, self.ytrain
            kernel = self._matrix(points)
            target = labels[:, None] * labels[None, :]
            loss = 1.0 - (kernel * target).sum() / (
                torch.linalg.matrix_norm(kernel) * torch.linalg.matrix_norm(target)
                + 1e-10
            )
            self.weight_optimizer.zero_grad()
            loss.backward()
            self.weight_optimizer.step()
            return float(loss.detach())

        classes = self.main_centroid_labels
        class_index = epoch % len(classes)
        label = float(classes[class_index])
        main = self.main_centroids[class_index]
        if self.method == "ccka":
            if self.update_kao:
                self.weight_optimizer.zero_grad()
                self.sub_optimizer.zero_grad()
                loss = self._centroid_loss(
                    main, self.sub_centroids, self.sub_centroid_labels,
                    label, self.lambda_kao, weight_penalty=True
                )
                loss.backward()
                self.weight_optimizer.step()
                self.sub_optimizer.step()
            if self.update_co:
                self.main_optimizer.zero_grad()
                loss = self._centroid_loss(
                    self.main_centroids[class_index],
                    self.sub_centroids,
                    self.sub_centroid_labels,
                    -label,
                    self.lambda_co,
                )
                loss.backward()
                self.main_optimizer.step()
                with torch.no_grad():
                    self.main_centroids.clamp_(0.0, 1.0)
        else:
            labels = torch.where(self.ytrain == label, 1.0, -1.0)
            for _ in range(self.inner_steps):
                self.weight_optimizer.zero_grad()
                loss = self._centroid_loss(
                    main,
                    self.xtrain,
                    labels,
                    label,
                    self.lambda_kao,
                    weight_penalty=True,
                )
                loss.backward()
                self.weight_optimizer.step()
            for _ in range(self.inner_steps):
                self.main_optimizer.zero_grad()
                loss = self._centroid_loss(
                    self.main_centroids[class_index], self.xtrain, labels,
                    -label, self.lambda_co
                )
                loss.backward()
                self.main_optimizer.step()
                with torch.no_grad():
                    self.main_centroids.clamp_(
                        self.xtrain.min(dim=0).values,
                        self.xtrain.max(dim=0).values,
                    )
        return float(loss.detach()) if "loss" in locals() else 0.0

    def align(self):
        start = time.perf_counter()
        initial = self._svm_metrics()
        validation_history, train_history, losses = [], [], []
        best_accuracy = -float("inf")
        best_weights = self.weights.detach().clone()
        best_main = (
            None if self.main_centroids is None else self.main_centroids.detach().clone()
        )
        best_sub = (
            None if self.sub_centroids is None else self.sub_centroids.detach().clone()
        )
        for epoch in range(self.epochs):
            losses.append(self._step(epoch))
            metrics = self._svm_metrics()
            train_history.append(metrics["train_accuracy"])
            validation_history.append(metrics["validation_accuracy"])
            if metrics["validation_accuracy"] > best_accuracy:
                best_accuracy = metrics["validation_accuracy"]
                best_weights = self.weights.detach().clone()
                if self.main_centroids is not None:
                    best_main = self.main_centroids.detach().clone()
                if self.sub_centroids is not None:
                    best_sub = self.sub_centroids.detach().clone()
        with torch.no_grad():
            self.weights.copy_(best_weights)
            if self.main_centroids is not None:
                self.main_centroids.copy_(best_main)
            if self.sub_centroids is not None:
                self.sub_centroids.copy_(best_sub)
        final = self._svm_metrics(evaluate_test=True)
        return {
            "final_svm_metrics": final,
            "test_accuracy_history": [final["test_accuracy"]],
            "validation_accuracy_history": validation_history,
            "train_accuracy_history": train_history,
            "loss_history": losses,
            "best_validation_accuracy": best_accuracy,
            "init_train_accuracy": initial["train_accuracy"],
            "init_validation_accuracy": initial["validation_accuracy"],
            "init_test_accuracy": float("nan"),
            "weights": self.weights.detach(),
            "time": time.perf_counter() - start,
            "circuit_executions": self.kernel_model.circuit_executions,
        }


class TorchFullKTA(TorchExperimentKTA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, method="fullKTA", **kwargs)


class TorchRandomKTA(TorchExperimentKTA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, method="randomKTA", **kwargs)


class TorchCentroidBasedKTA(TorchExperimentKTA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, method="ccka", **kwargs)


class TorchQuackKTA(TorchExperimentKTA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, method="quackKTA", **kwargs)
