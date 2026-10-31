"""Generated and sklearn-bundled datasets used by the CCKA experiments.

Every synthetic dataset is built to expose one specific behaviour of CCKA, so a
result on it supports (or refutes) a concrete claim. :data:`SYNTHETIC_CLAIMS`
records the claim and the experiment that tests it; the plot/statistics
scripts and the README read the same registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Claim:
    behaviour: str
    expected: str
    experiment: str


def _ring(rng, n, radius, width=0.05):
    angle = rng.uniform(0.0, 2.0 * np.pi, n)
    r = radius + rng.normal(0.0, width, n)
    return np.column_stack((r * np.cos(angle), r * np.sin(angle)))


def _split_counts(n_samples: int, parts: int) -> list[int]:
    base = [n_samples // parts] * parts
    for index in range(n_samples - sum(base)):
        base[index] += 1
    return base


def _moons(rng, n, seed):
    from sklearn.datasets import make_moons

    return make_moons(n_samples=n, noise=0.12, random_state=seed)


def _circles(rng, n, seed):
    from sklearn.datasets import make_circles

    return make_circles(n_samples=n, noise=0.05, factor=0.5, random_state=seed)


def _double_cake(rng, n, seed):
    angles = rng.uniform(0.0, 2.0 * np.pi, n)
    labels = rng.integers(0, 2, n)
    radii = np.where(labels == 0, 0.35, 0.9) + rng.normal(0, 0.06, n)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles))), labels


def _rings(rng, n, seed):
    """Two classes, each made of two concentric rings (same mean: the origin)."""
    counts = _split_counts(n, 4)
    radii = (0.3, 0.6, 0.9, 1.2)
    X = np.concatenate([_ring(rng, c, r) for c, r in zip(counts, radii)])
    y = np.concatenate([np.full(c, k % 2) for k, c in enumerate(counts)])
    return X, y


def _three_rings(rng, n, seed):
    counts = _split_counts(n, 3)
    X = np.concatenate(
        [_ring(rng, c, r) for c, r in zip(counts, (0.3, 0.75, 1.2))]
    )
    y = np.concatenate([np.full(c, k) for k, c in enumerate(counts)])
    return X, y


def _xor_grid(rng, n, seed, cells: int = 3):
    X = rng.uniform(0.0, float(cells), size=(n, 2))
    y = (np.floor(X[:, 0]).astype(int) + np.floor(X[:, 1]).astype(int)) % 2
    return X + rng.normal(0.0, 0.02, X.shape), y


def _spirals(rng, n, seed):
    counts = _split_counts(n, 2)
    arms = []
    for arm, count in enumerate(counts):
        t = np.sqrt(rng.uniform(0.04, 1.0, count)) * 3.0 * np.pi
        radius = t / (3.0 * np.pi)
        sign = 1.0 if arm == 0 else -1.0
        arms.append(
            np.column_stack(
                (sign * radius * np.cos(t), sign * radius * np.sin(t))
            )
            + rng.normal(0.0, 0.03, (count, 2))
        )
    return np.concatenate(arms), np.concatenate(
        [np.full(c, k) for k, c in enumerate(counts)]
    )


def _gmm(rng, n, modes: int, classes: int = 2, spread: float = 0.12):
    """Every class is a union of ``modes`` well-separated Gaussians on a circle."""
    slots = classes * modes
    angles = 2.0 * np.pi * np.arange(slots) / slots
    centres = np.column_stack((np.cos(angles), np.sin(angles)))
    owner = np.arange(slots) % classes
    counts = _split_counts(n, slots)
    X = np.concatenate(
        [rng.normal(centres[k], spread, (counts[k], 2)) for k in range(slots)]
    )
    y = np.concatenate([np.full(counts[k], owner[k]) for k in range(slots)])
    return X, y


def _rare_mode(rng, n, seed):
    """Class 1 has a dominant mode and a distant mode holding 10% of its mass."""
    n_class = _split_counts(n, 2)
    class_zero = rng.normal([-0.6, 0.0], 0.15, (n_class[0], 2))
    rare = max(1, int(0.1 * n_class[1]))
    dominant = rng.normal([0.6, 0.0], 0.15, (n_class[1] - rare, 2))
    far = rng.normal([-0.6, 1.1], 0.1, (rare, 2))
    X = np.concatenate((class_zero, dominant, far))
    y = np.concatenate(
        (np.zeros(n_class[0], int), np.ones(n_class[1], int))
    )
    return X, y


def _overlapping_blobs(rng, n, seed):
    counts = _split_counts(n, 2)
    X = np.concatenate(
        (
            rng.normal([-0.3, 0.0], 0.45, (counts[0], 2)),
            rng.normal([0.3, 0.0], 0.45, (counts[1], 2)),
        )
    )
    y = np.concatenate((np.zeros(counts[0], int), np.ones(counts[1], int)))
    return X, y


GeneratorFn = Callable[[np.random.Generator, int, int], tuple]

GENERATORS: dict[str, GeneratorFn] = {
    "moons": _moons,
    "circles": _circles,
    "double_cake": _double_cake,
    "rings": _rings,
    "three_rings": _three_rings,
    "xor_grid": _xor_grid,
    "spirals": _spirals,
    "gmm_modes_1": lambda rng, n, seed: _gmm(rng, n, 1),
    "gmm_modes_2": lambda rng, n, seed: _gmm(rng, n, 2),
    "gmm_modes_4": lambda rng, n, seed: _gmm(rng, n, 4),
    "multiclass_modes": lambda rng, n, seed: _gmm(rng, n, 2, classes=3),
    "rare_mode": _rare_mode,
    "overlapping_blobs": _overlapping_blobs,
}

SYNTHETIC_CLAIMS: dict[str, Claim] = {
    "gmm_modes_1": Claim(
        "Control: unimodal classes",
        "CCKA is on par with QUACK (a single centroid is already enough).",
        "Method comparison; gap CCKA-QUACK should be ~0.",
    ),
    "gmm_modes_2": Claim(
        "Sub-centroids capture multimodal classes",
        "CCKA - QUACK gap > 0 and larger than on gmm_modes_1.",
        "Method comparison + K sweep (K=1 degenerates to QUACK).",
    ),
    "gmm_modes_4": Claim(
        "Gap grows with the number of modes per class",
        "CCKA - QUACK gap grows with modes; accuracy saturates near K=4.",
        "K sweep; gap versus modes (gmm_modes_1/2/4).",
    ),
    "rings": Claim(
        "Equal class means: a main centroid carries no label information",
        "QUACK (one main centroid per class) stays near chance; CCKA learns "
        "ring-wise representative points.",
        "Method comparison (QUACK vs CCKA); ablations are not run on rings.",
    ),
    "three_rings": Claim(
        "Multiclass with identical class means",
        "CCKA works with both one-vs-one and one-vs-all reduction.",
        "Multiclass strategy comparison.",
    ),
    "xor_grid": Claim(
        "Needs one representative point per cell (K >= cells/2 per class)",
        "Accuracy increases with K and saturates once K covers the cells.",
        "K sweep (representative-point saturation).",
    ),
    "spirals": Claim(
        "Highly non-linear boundary learned from few representative points",
        "CCKA reaches FullKTA accuracy at far fewer circuit evaluations.",
        "Accuracy vs circuit budget.",
    ),
    "moons": Claim(
        "Generalisation in the low-data regime",
        "CCKA test accuracy degrades more slowly than FullKTA as N_train shrinks.",
        "N_train sweep.",
    ),
    "circles": Claim(
        "Non-linearly separable with distinct class means that coincide",
        "Kernel must learn a radial boundary; all KTA variants should succeed.",
        "Method comparison (sanity check).",
    ),
    "double_cake": Claim(
        "Radially separable classes",
        "Sanity check for QUACK-style single centroids.",
        "Method comparison.",
    ),
    "rare_mode": Claim(
        "Sub-centroids cover rare modes of a class",
        "Macro-F1 / recall of CCKA exceeds QUACK, which ignores the rare mode.",
        "Method comparison reporting macro-F1.",
    ),
    "overlapping_blobs": Claim(
        "Robustness to label noise / class overlap",
        "Centroid objectives average noisy labels; CCKA degrades less than "
        "FullKTA as label noise grows.",
        "Label-noise sweep.",
    ),
    "multiclass_modes": Claim(
        "Multiclass, multimodal classes",
        "Sub-centroids help for C > 2; one-vs-one and one-vs-all are comparable.",
        "Multiclass strategy comparison.",
    ),
}

SKLEARN_LOADERS = ("iris", "wine", "digits_c3", "digits_c5", "digits_c10")

GENERATED_DATASETS = tuple(GENERATORS) + SKLEARN_LOADERS


def generate_dataset(
    name: str, seed: int, n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    if name in GENERATORS:
        rng = np.random.default_rng(seed)
        X, y = GENERATORS[name](rng, n_samples, seed)
        return np.asarray(X, dtype=np.float32), np.asarray(y).astype(int)
    if name in SKLEARN_LOADERS:
        from sklearn import datasets

        if name == "iris":
            bunch = datasets.load_iris()
        elif name == "wine":
            bunch = datasets.load_wine()
        else:
            bunch = datasets.load_digits(n_class=int(name.split("_c")[1]))
        return np.asarray(bunch.data, dtype=np.float32), np.asarray(bunch.target)
    raise ValueError(f"No generator is defined for dataset {name!r}.")
