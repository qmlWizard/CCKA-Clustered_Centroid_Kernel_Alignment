import torch

class CentroidBasedClassifier:
    def __init__(self, kernel_function, centroids, subcentroids=None, use_subcentroids=False, temperature=1.0):
        """
        Args:
            kernel_function: Callable kernel function (x1, x2) -> similarity scores
            centroids: Dict[int, Tensor] — one centroid per class [d]
            subcentroids: Dict[int, List[Tensor]] or Dict[int, Tensor] — subcentroids per class
            use_subcentroids: Whether to use subcentroids (True) or centroids (False)
            temperature: Temperature for softmax probability
        """
        self.kernel = kernel_function
        self.centroids = centroids
        self.subcentroids = subcentroids
        self.use_subcentroids = use_subcentroids
        self.temperature = temperature

        print(self.subcentroids)
        print(self.centroids)

    def _prepare_prototypes(self):
        """
        Flatten all (sub)centroids into a tensor and collect class labels.
        Returns:
            all_prototypes: Tensor [n_prototypes, d]
            proto_labels: Tensor [n_prototypes]
        """
        prototypes = []
        labels = []
        for cls, centroid in self.centroids.items():
            if self.use_subcentroids and self.subcentroids:
                class_prototypes = self.subcentroids[cls]
                if isinstance(class_prototypes, torch.Tensor):
                    p = class_prototypes
                else:
                    p = torch.stack(class_prototypes)
                prototypes.append(p)
                labels.extend([cls] * p.shape[0])
            else:
                prototypes.append(centroid.unsqueeze(0))
                labels.append(cls)

        all_prototypes = torch.cat(prototypes, dim=0)
        proto_labels = torch.tensor(labels)
        return all_prototypes, proto_labels

    def predict_batch(self, X):
        subcentroids = []
        proto_labels = []

        for cls, sub_list in self.subcentroids.items():
            subcentroids.extend(sub_list)
            proto_labels.extend([cls] * len(sub_list))

        subcentroids = torch.stack(subcentroids)
        proto_labels = torch.tensor(proto_labels)

        # Kernel similarity between test points and all subcentroids
        X_0 = X.repeat_interleave(subcentroids.shape[0], dim=0)
        X_1 = subcentroids.repeat(X.shape[0], 1)
        sims = self.kernel(X_0, X_1).reshape(X.shape[0], subcentroids.shape[0])

        # Get class scores
        unique_classes = torch.tensor(sorted(set(proto_labels.tolist())))
        scores = torch.stack([
            torch.max(sims[:, proto_labels == cls], dim=1).values
            for cls in unique_classes
        ], dim=1)  # shape [m, num_classes]

        pred_indices = torch.argmax(scores, dim=1)
        preds = unique_classes[pred_indices]  # now in {-1, 1}

        print("Sims mean per class:")
        for cls in unique_classes:
            sims_cls = sims[:, proto_labels == cls]
            print(f"Class {cls}: mean sim = {sims_cls.mean().item():.4f}, std = {sims_cls.std().item():.4f}")

        return preds

    def predict_proba_batch(self, X):
        """
        Predict class probabilities for a batch using softmax over max kernel similarity.
        Args:
            X: Tensor [m, d]
        Returns:
            List of dicts with class probabilities
        """
        prototypes, proto_labels = self._prepare_prototypes()
        m, d = X.shape
        n_total = prototypes.shape[0]
        unique_classes = sorted(set(proto_labels.tolist()))

        # Compute kernel matrix [m, n_total]
        x_0 = X.repeat_interleave(n_total, dim=0)
        x_1 = prototypes.repeat(X.shape[0], 1)
        print("X0:", x_0.shape, "X1:", x_1.shape)
        sims = self.kernel(x_0, x_1).to(torch.float32).reshape(m, n_total)

        # Max over each class's subcentroid group
        scores = torch.stack([
            torch.max(sims[:, proto_labels == cls], dim=1).values
            for cls in unique_classes
        ], dim=1)  # [m, n_classes]

        logits = scores / self.temperature
        probs = torch.nn.functional.softmax(logits, dim=1)  # [m, n_classes]

        # Convert to list of dicts
        return [
            {cls: prob for cls, prob in zip(unique_classes, row)}
            for row in probs.tolist()
        ]
