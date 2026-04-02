import pennylane as qml
import numpy as np
import jax
import jax.numpy as jnp


class quackEmbeddingCircuit:
    """
        JAX-safe angle-embedding quantum kernel.
        All inputs MUST already have shape (num_qubits,).
        No shape logic is allowed inside the circuit.

        Args:
            num_qubits  : number of qubits
            reps        : number of circuit repetitions
            reupload    : whether to re-upload features each rep
            noisy       : if True, apply depolarising noise after every gate
                        on BOTH the forward (x1) and manual-adjoint (x2) passes
            noise_level : depolarising error probability p ∈ [0, 1]

        Device requirement
        ------------------
        noisy=False  →  default.qubit  (state-vector)
        noisy=True   →  default.mixed  (density-matrix)
    """
    def __init__(self, num_qubits, reps=1, reupload=True, noisy=False, noise_level=0.01):
        self.num_qubits  = num_qubits
        self.reps        = reps
        self.reupload    = reupload
        self.noisy       = noisy
        self.noise_level = noise_level
        self.wires       = list(range(num_qubits))

    def _depolarise(self, wire, apply_noise):
        if self.noisy and apply_noise:
            qml.DepolarizingChannel(self.noise_level, wires=wire)

    def feature_map(self, x, scale, apply_noise=True):
        scale = qml.math.asarray(scale)   # ← backend-agnostic
        for i, wire in enumerate(self.wires):
            qml.Hadamard(wires=wire)
            self._depolarise(wire, apply_noise)
            qml.RZ(scale[i] * x[i], wires=wire)
            self._depolarise(wire, apply_noise)

    def _feature_map_dagger(self, x, scale, apply_noise=True):
        scale = qml.math.asarray(scale)
        for i in reversed(range(self.num_qubits)):
            wire = self.wires[i]
            qml.RZ(-scale[i] * x[i], wires=wire)
            self._depolarise(wire, apply_noise)
            qml.Hadamard(wires=wire)
            self._depolarise(wire, apply_noise)

    def ansatz(self, var, rot, apply_noise=True):
        var = qml.math.asarray(var)
        rot = qml.math.asarray(rot)
        for i, wire in enumerate(self.wires):
            qml.RY(var[i], wires=wire)
            self._depolarise(wire, apply_noise)
        for i in range(self.num_qubits):
            target = self.wires[(i + 1) % self.num_qubits]
            qml.CRZ(rot[i], wires=[self.wires[i], target])
            self._depolarise(self.wires[i], apply_noise)
            self._depolarise(target, apply_noise)

    def _ansatz_dagger(self, var, rot, apply_noise=True):
        var = qml.math.asarray(var)
        rot = qml.math.asarray(rot)
        for i in reversed(range(self.num_qubits)):
            target = self.wires[(i + 1) % self.num_qubits]
            qml.CRZ(-rot[i], wires=[self.wires[i], target])
            self._depolarise(self.wires[i], apply_noise)
            self._depolarise(target, apply_noise)
        for i in reversed(range(self.num_qubits)):
            wire = self.wires[i]
            qml.RY(-var[i], wires=wire)
            self._depolarise(wire, apply_noise)

    def _build_circuit(self, x, weights, apply_noise=True):
        idx = 0
        for rep in range(self.reps):
            scale = weights[idx: idx + self.num_qubits]; idx += self.num_qubits
            var   = weights[idx: idx + self.num_qubits]; idx += self.num_qubits
            rot   = weights[idx: idx + self.num_qubits]; idx += self.num_qubits
            if rep == 0 or self.reupload:
                self.feature_map(x, scale, apply_noise)
            self.ansatz(var, rot, apply_noise)

    def _build_circuit_dagger(self, x, weights, apply_noise=True):
        params, idx = [], 0
        for _ in range(self.reps):
            scale = weights[idx: idx + self.num_qubits]; idx += self.num_qubits
            var   = weights[idx: idx + self.num_qubits]; idx += self.num_qubits
            rot   = weights[idx: idx + self.num_qubits]; idx += self.num_qubits
            params.append((scale, var, rot))
        for rep in reversed(range(self.reps)):
            scale, var, rot = params[rep]
            self._ansatz_dagger(var, rot, apply_noise)
            if rep == 0 or self.reupload:
                self._feature_map_dagger(x, scale, apply_noise)

    def kernel_circuit(self, x1, x2, weights):
        self._build_circuit(x1, weights, apply_noise=True)
        self._build_circuit_dagger(x2, weights, apply_noise=True)
        return qml.expval(qml.Projector([0] * self.num_qubits, wires=self.wires))

    def init_weights(self, seed=0, minval=-np.pi, maxval=np.pi):
        rng = np.random.default_rng(seed)
        total = self.reps * 3 * self.num_qubits
        return rng.uniform(minval, maxval, size=(total,)).astype(np.float64)

    def init_weights_jax(self, seed=0, minval=-jnp.pi, maxval=jnp.pi):
        """Use this variant only in sim mode."""
        key = jax.random.PRNGKey(seed)
        total = self.reps * 3 * self.num_qubits
        return jax.random.uniform(key, shape=(total,), minval=minval, maxval=maxval)


import numpy as np
from qiskit import QuantumCircuit


class QuackEmbeddingQiskitCircuit:
    """
    Qiskit + NumPy quantum kernel circuit for hardware execution.

    Circuit structure (per rep):
        feature_map : H → RZ(scale·x)  per qubit
        ansatz      : RY(var) per qubit → CRZ(rot) ring entangler
    
    Kernel circuit:
        forward(x1, weights) ⊕ dagger(x2, weights)
        measured in the computational basis → K(x1,x2) = P(|0…0⟩)

    Args:
        num_qubits : number of qubits
        reps       : number of circuit repetitions
        reupload   : re-upload features every rep
    """

    def __init__(self, num_qubits: int, reps: int = 1, reupload: bool = True):
        self.num_qubits = num_qubits
        self.reps       = reps
        self.reupload   = reupload

    # ------------------------------------------------------------------
    # Sub-circuits
    # ------------------------------------------------------------------

    def _feature_map(self, qc: QuantumCircuit, x: np.ndarray, scale: np.ndarray):
        for i in range(self.num_qubits):
            qc.h(i)
            qc.rz(float(scale[i] * x[i]), i)

    def _feature_map_dagger(self, qc: QuantumCircuit, x: np.ndarray, scale: np.ndarray):
        for i in reversed(range(self.num_qubits)):
            qc.rz(float(-scale[i] * x[i]), i)
            qc.h(i)

    def _ansatz(self, qc: QuantumCircuit, var: np.ndarray, rot: np.ndarray):
        for i in range(self.num_qubits):
            qc.ry(float(var[i]), i)
        for i in range(self.num_qubits):
            target = (i + 1) % self.num_qubits
            qc.crz(float(rot[i]), i, target)

    def _ansatz_dagger(self, qc: QuantumCircuit, var: np.ndarray, rot: np.ndarray):
        for i in reversed(range(self.num_qubits)):
            target = (i + 1) % self.num_qubits
            qc.crz(float(-rot[i]), i, target)
        for i in reversed(range(self.num_qubits)):
            qc.ry(float(-var[i]), i)

    # ------------------------------------------------------------------
    # Forward / dagger passes
    # ------------------------------------------------------------------

    def _build_forward(self, qc: QuantumCircuit, x: np.ndarray, weights: np.ndarray):
        idx = 0
        for rep in range(self.reps):
            scale = weights[idx : idx + self.num_qubits]; idx += self.num_qubits
            var   = weights[idx : idx + self.num_qubits]; idx += self.num_qubits
            rot   = weights[idx : idx + self.num_qubits]; idx += self.num_qubits
            if rep == 0 or self.reupload:
                self._feature_map(qc, x, scale)
            self._ansatz(qc, var, rot)

    def _build_dagger(self, qc: QuantumCircuit, x: np.ndarray, weights: np.ndarray):
        params, idx = [], 0
        for _ in range(self.reps):
            scale = weights[idx : idx + self.num_qubits]; idx += self.num_qubits
            var   = weights[idx : idx + self.num_qubits]; idx += self.num_qubits
            rot   = weights[idx : idx + self.num_qubits]; idx += self.num_qubits
            params.append((scale, var, rot))

        for rep in reversed(range(self.reps)):
            scale, var, rot = params[rep]
            self._ansatz_dagger(qc, var, rot)
            if rep == 0 or self.reupload:
                self._feature_map_dagger(qc, x, scale)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_kernel_circuit(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        weights: np.ndarray,
        measure: bool = True,
    ) -> QuantumCircuit:
        """
        Returns the full kernel circuit: forward(x1) ⊕ dagger(x2).
        
        Args:
            x1, x2   : input vectors of shape (num_qubits,)
            weights   : weight vector of shape (reps * 3 * num_qubits,)
            measure   : if True, appends measurements on all qubits
                        (required for hardware / sampler execution)
        """
        qc = QuantumCircuit(self.num_qubits, self.num_qubits if measure else 0)
        self._build_forward(qc, x1, weights)
        self._build_dagger(qc, x2, weights)
        if measure:
            qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    def kernel_value_from_counts(
        self,
        counts: dict,
        shots: int,
    ) -> float:
        """
        Estimates K(x1, x2) = P(|0…0⟩) from hardware measurement counts.

        Args:
            counts : dict of bitstring → count, e.g. {'0000': 812, '0101': 12, ...}
            shots  : total number of shots
        
        Returns:
            Estimated kernel value in [0, 1]
        """
        all_zeros = "0" * self.num_qubits
        return counts.get(all_zeros, 0) / shots

    def kernel_matrix_from_circuits(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[list[QuantumCircuit], list[tuple[int, int]]]:
        """
        Returns all (i, j) kernel circuits as a flat list ready for
        batch submission to a hardware backend or sampler.

        Usage:
            circuits, indices = qke.kernel_matrix_from_circuits(X1, X2, weights)
            # submit `circuits` in one batch, then:
            K = np.zeros((len(X1), len(X2)))
            for (i, j), counts in zip(indices, all_counts):
                K[i, j] = qke.kernel_value_from_counts(counts, shots)

        Returns:
            circuits : list of QuantumCircuit
            indices  : matching list of (i, j) index pairs
        """
        circuits, indices = [], []
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                circuits.append(self.build_kernel_circuit(x1, x2, weights))
                indices.append((i, j))
        return circuits, indices

    def init_weights(
        self,
        seed: int = 0,
        minval: float = -np.pi,
        maxval: float = np.pi,
    ) -> np.ndarray:
        """Returns a weight vector of shape (reps × 3 × num_qubits,)."""
        rng   = np.random.default_rng(seed)
        total = self.reps * 3 * self.num_qubits
        return rng.uniform(minval, maxval, size=(total,)).astype(np.float64)