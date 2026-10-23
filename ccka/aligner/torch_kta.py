"""PyTorch implementations for running experiments without JAX/Optax."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
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
        co_margin: float = 0.1,
        co_temperature: float = 0.1,
        svm_c: float = 1.0,
        inference_method: str = "svm",
        inference_neighbors: int = 1,
        inference_landmarks: int | str = 32,
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
        if inner_steps <= 0:
            raise ValueError("inner_steps must be > 0")
        if co_margin < 0:
            raise ValueError("co_margin must be non-negative")
        if co_temperature <= 0:
            raise ValueError("co_temperature must be positive")
        if inference_method not in {"svm", "kernel_distance", "centroid"}:
            raise ValueError(
                "inference_method must be 'svm', 'kernel_distance', or 'centroid'"
            )
        if inference_neighbors <= 0:
            raise ValueError("inference_neighbors must be positive")
        if inference_landmarks != "all" and (
            isinstance(inference_landmarks, bool)
            or not isinstance(inference_landmarks, int)
            or inference_landmarks <= 0
        ):
            raise ValueError("inference_landmarks must be > 0 or 'all'")
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
        self.co_margin = co_margin
        self.co_temperature = co_temperature
        self.svm_c = svm_c
        self.inference_method = inference_method
        self.inference_neighbors = inference_neighbors
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
        if self.inference_landmarks == "all":
            train_kernel = self._matrix(self.xtrain)
            query_kernel = None
            if query is not None:
                query_count = len(query)
                query_kernel = self._kernel_vector(
                    query.repeat_interleave(n, dim=0),
                    self.xtrain.repeat(query_count, 1),
                ).reshape(query_count, n)
            return train_kernel.detach().numpy(), (
                None if query_kernel is None else query_kernel.detach().numpy()
            )
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
        y_train = self.ytrain.detach().numpy()
        y_validation = self.yvalidation.detach().numpy()
        predictor = None
        if self.inference_method == "centroid":
            train_prediction = self._centroid_predict(self.xtrain)
            validation_prediction = self._centroid_predict(self.xvalidation)
            test_prediction = (
                self._centroid_predict(self.xtest) if evaluate_test else None
            )
        else:
            train_kernel, validation_kernel = self._nystrom_matrices(
                self.xvalidation
            )
            if self.inference_method == "svm":
                predictor = SVC(
                    kernel="precomputed", C=self.svm_c, random_state=self.seed
                )
                predictor.fit(train_kernel, y_train)
                train_prediction = predictor.predict(train_kernel)
                validation_prediction = predictor.predict(validation_kernel)
            else:
                train_diagonal = np.diag(train_kernel)
                validation_diagonal = self._kernel_vector(
                    self.xvalidation, self.xvalidation
                ).detach().numpy()
                train_distances = np.maximum(
                    train_diagonal[:, None]
                    + train_diagonal[None, :]
                    - 2.0 * train_kernel,
                    0.0,
                )
                np.fill_diagonal(train_distances, 0.0)
                validation_distances = np.maximum(
                    validation_diagonal[:, None]
                    + train_diagonal[None, :]
                    - 2.0 * validation_kernel,
                    0.0,
                )
                predictor = KNeighborsClassifier(
                    n_neighbors=min(self.inference_neighbors, len(y_train)),
                    metric="precomputed",
                )
                predictor.fit(train_distances, y_train)
                train_prediction = predictor.predict(train_distances)
                validation_prediction = predictor.predict(validation_distances)
            if evaluate_test:
                _, test_kernel = self._nystrom_matrices(self.xtest)
                if self.inference_method == "svm":
                    test_prediction = predictor.predict(test_kernel)
                else:
                    test_diagonal = self._kernel_vector(
                        self.xtest, self.xtest
                    ).detach().numpy()
                    test_distances = np.maximum(
                        test_diagonal[:, None]
                        + train_diagonal[None, :]
                        - 2.0 * test_kernel,
                        0.0,
                    )
                    test_prediction = predictor.predict(test_distances)
        metrics = {
            "train_accuracy": float(accuracy_score(y_train, train_prediction)),
            "validation_accuracy": float(
                accuracy_score(y_validation, validation_prediction)
            ),
        }
        if evaluate_test:
            y_test = self.ytest.detach().numpy()
            metrics.update(
                {
                    "test_accuracy": float(accuracy_score(y_test, test_prediction)),
                    "f1_score": float(f1_score(y_test, test_prediction, average="macro")),
                    "precision_score": float(
                        precision_score(y_test, test_prediction, average="macro")
                    ),
                    "recall_score": float(
                        recall_score(y_test, test_prediction, average="macro")
                    ),
                }
            )
        return metrics

    def _centroid_predict(self, X: torch.Tensor) -> np.ndarray:
        n_samples = len(X)
        if self.method == "quackKTA":
            centers = self.main_centroids
            scores = self._kernel_vector(
                X.repeat_interleave(len(centers), dim=0),
                centers.repeat(n_samples, 1),
            ).reshape(n_samples, len(centers))
            return self.main_centroid_labels[scores.argmax(dim=1)].numpy()
        if self.method == "ccka":
            main_scores = self._kernel_vector(
                X.repeat_interleave(len(self.main_centroids), dim=0),
                self.main_centroids.repeat(n_samples, 1),
            ).reshape(n_samples, len(self.main_centroids))
            sub_scores = self._kernel_vector(
                X.repeat_interleave(len(self.sub_centroids), dim=0),
                self.sub_centroids.repeat(n_samples, 1),
            ).reshape(n_samples, len(self.sub_centroids))
            class_scores = []
            for label in self.main_centroid_labels:
                main_index = int(
                    torch.nonzero(self.main_centroid_labels == label)[0]
                )
                sub_mask = self.sub_centroid_labels == label
                class_scores.append(
                    0.5
                    * (
                        main_scores[:, main_index]
                        + sub_scores[:, sub_mask].mean(dim=1)
                    )
                )
            stacked = torch.stack(class_scores, dim=1)
            return self.main_centroid_labels[stacked.argmax(dim=1)].numpy()
        raise ValueError(
            f"inference_method='centroid' is unsupported for {self.method!r}"
        )

    @staticmethod
    def _alignment(kernel: torch.Tensor, target: torch.Tensor, sign: float = 1.0):
        target = target.to(dtype=kernel.dtype)
        return sign * torch.dot(kernel, target) / (
            torch.linalg.vector_norm(kernel) * torch.linalg.vector_norm(target) + 1e-10
        )

    def _centroid_loss(self, main: torch.Tensor, points: torch.Tensor,
                       labels: torch.Tensor, sign: float, regularizer: float,
                       *, weight_penalty: bool = False,
                       class_label: float | None = None):
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
        if self.method == "ccka" and not weight_penalty:
            if class_label is None:
                raise ValueError("class_label is required for the CCKA CO loss")
            same_class = labels == class_label
            other_class = ~same_class
            same_similarity = (
                kernel[same_class].sum() / same_class.sum().clamp_min(1)
            )
            other_similarity = (
                kernel[other_class].sum() / other_class.sum().clamp_min(1)
            )
            similarity_gap = same_similarity - other_similarity
            margin_loss = torch.nn.functional.softplus(
                (self.co_margin - similarity_gap) / self.co_temperature
            )
            return margin_loss + regularizer * penalty
        return 1.0 - self._alignment(kernel, labels, sign) + regularizer * penalty

    def _step(self, epoch: int, inner_steps: int | None = None):
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
        n_inner_steps = self.inner_steps if inner_steps is None else inner_steps
        if self.method == "ccka":
            for _ in range(n_inner_steps):
                if self.update_kao:
                    main = self.main_centroids[class_index]
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
                        class_label=label,
                    )
                    loss.backward()
                    self.main_optimizer.step()
                    with torch.no_grad():
                        self.main_centroids.clamp_(0.0, 1.0)
            return float(loss.detach())
        else:
            labels = torch.where(self.ytrain == label, 1.0, -1.0)
            if epoch % 2 == 0:
                for _ in range(n_inner_steps):
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
            else:
                for _ in range(n_inner_steps):
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
        best_accuracy = float(initial["validation_accuracy"])
        best_weights = self.weights.detach().clone()
        best_main = (
            None if self.main_centroids is None else self.main_centroids.detach().clone()
        )
        best_sub = (
            None if self.sub_centroids is None else self.sub_centroids.detach().clone()
        )
        if self.method in {"quackKTA", "ccka"}:
            outer_epochs = (self.epochs + self.inner_steps - 1) // self.inner_steps
        else:
            outer_epochs = self.epochs
        for epoch in range(outer_epochs):
            steps_this_epoch = min(
                self.inner_steps, self.epochs - epoch * self.inner_steps
            ) if self.method in {"quackKTA", "ccka"} else None
            losses.append(self._step(epoch, steps_this_epoch))
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
