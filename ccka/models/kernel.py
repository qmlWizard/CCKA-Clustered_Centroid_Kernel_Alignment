# kernelModel.py

import pennylane as qml
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Batch
from qiskit_ibm_runtime.options import SamplerOptions
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import numpy as np
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

class KernelModel:

    def __init__(self, 
                 circuit = None,
                 device_name: str = 'default.qubit',
                 interface: str = 'jax',
                 diff_method: str = 'backprop',
                 backend = None,
                 matrix_type: str = 'regular',
                 matrix_normalisation: bool = False,
                 landmark_points: int = 0,
                 noisy: bool = False,
                 seed: int = 42
    ):

        self.matrix_type = matrix_type
        self.matrix_normalisation = matrix_normalisation
        self.interface = interface
        self.diff_method = diff_method
        self.landmark_points = landmark_points
        self.noisy = noisy
        self.seed = seed
        self.backend = backend 
        self.circuit_executions = 0
        if circuit is None:
            raise ValueError("A circuit must be provided to the KernelModel.")

        self.circuit = circuit
    
        # ----- device
        if noisy:
            self._device_name = 'default.mixed'
            self._shots = 1024
        else:
            self._device_name = device_name
            self._shots = None

        dev = qml.device(self._device_name, wires = self.circuit.num_qubits, shots = self._shots)

        # ----- circuit

        self.circuit_instance = qml.QNode(self.circuit.kernel_circuit, dev, interface= self.interface, diff_method= self.diff_method)
        self._kernel = jax.jit(self.circuit_instance)
        self._vectorized_kernel = jax.vmap(
            lambda a, b, w: self._kernel(a, b, w),
            in_axes=(0, 0, None),
        )

    def forward(self, x1, x2, weights):
        self.circuit_executions += len(x1)
        return self._vectorized_kernel(x1, x2, weights)  


class HardwareKernelRunner:
    """
    Kernel matrix computation on real IBM Quantum hardware.

    Uses:
        SamplerV2       : IBM Runtime primitive for shot-based execution
        Batch mode      : submits all circuits in one session to minimise
                          queue wait time and preserve backend calibration
        Error mitigation: optional readout error mitigation via SamplerOptions

    Args:
        circuit          : QuackEmbeddingCircuit instance
        backend          : IBM backend from QiskitRuntimeService
        shots            : shots per circuit
        batch_size       : circuits per Batch job submission
        optimization_level : transpiler optimization level (1–3)
        mitigation_level : 0 = none, 1 = readout error mitigation
    """

    def __init__(
        self,
        circuit,
        backend,
        shots: int = 1024,
        batch_size: int = 100,
        optimization_level: int = 3,
        mitigation_level: int = 0,
    ):
        if circuit is None:
            raise ValueError("A circuit must be provided.")
        if backend is None:
            raise ValueError("backend must be provided.")

        self.circuit          = circuit
        self.backend          = backend
        self.shots            = shots
        self.batch_size       = batch_size
        self.num_qubits       = circuit.num_qubits
        self.circuit_executions = 0

        # Transpiler pass manager
        self._pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            backend=backend,
        )

        # SamplerV2 options — readout error mitigation
        options = SamplerOptions()
        options.dynamical_decoupling.enable = True          # reduce idle noise
        if mitigation_level >= 1:
            options.resilience_level = 1                    # readout mitigation

        self._sampler_options = options

    # ------------------------------------------------------------------
    # Transpile
    # ------------------------------------------------------------------

    def _transpile(self, circuits: list) -> list:
        return [self._pm.run(qc) for qc in circuits]

    # ------------------------------------------------------------------
    # P(|0…0⟩) from a SamplerV2 pub result
    # ------------------------------------------------------------------

    def _prob_zero(self, pub_result) -> float:
        data = pub_result.data
        
        # ✅ Case 1: BitArray (your current case)
        if hasattr(data, "c"):
            bit_array = data.c  # BitArray object

            shots = bit_array.num_shots
            bits  = bit_array.num_bits

            # Convert to numpy: shape (shots, bits)
            arr = bit_array.array

            # Count all-zero rows
            zero_count = np.sum(np.all(arr == 0, axis=1))

            return zero_count / shots

        # ✅ Case 2: counts (older / different config)
        elif hasattr(data, "meas"):
            counts = data.meas.get_counts()
            total = sum(counts.values())
            if total == 0:
                return 0.0
            zero = "0" * len(next(iter(counts)))
            return counts.get(zero, 0) / total

        # ✅ Case 3: quasi distributions
        elif hasattr(data, "quasi_dists"):
            quasi = data.quasi_dists[0]
            return quasi.get(0, 0.0)

        else:
            raise ValueError(f"Unknown result format: {data}")
    # ------------------------------------------------------------------
    # Build all circuits for a paired (x1, x2) array
    # ------------------------------------------------------------------

    def _build_circuits(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        weights: np.ndarray,
    ) -> list:
        circuits = [
            self.circuit.build_kernel_circuit(a, b, weights, measure=True)
            for a, b in zip(x1, x2)
        ]
        return self._transpile(circuits)     # transpile once before hardware

    # ------------------------------------------------------------------
    # Forward — pairwise kernel values using IBM Runtime Batch
    # ------------------------------------------------------------------

    def forward(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluates K(x1[i], x2[i]) for all i using Batch mode.

        Batch mode:
          - Opens a single session with the backend.
          - Submits all batch jobs within that session.
          - Guarantees all jobs run on the same backend calibration snapshot.
          - Minimises queue re-entry between batches.
        """
        x1         = np.array(x1, dtype=np.float64)
        x2         = np.array(x2, dtype=np.float64)
        weights    = np.array(weights, dtype=np.float64)
        N          = len(x1)
        results    = []

        self.circuit_executions += N

        # Open a Batch session — all jobs share one queue slot
        with Batch(backend=self.backend) as batch:
            sampler  = SamplerV2(mode=batch, options=self._sampler_options)
            job_list = []

            for start in tqdm(range(0, N, self.batch_size), desc="Submitting batches"):
                end      = min(start + self.batch_size, N)
                circuits = self._build_circuits(x1[start:end], x2[start:end], weights)

                # Each sampler.run() call = one job inside the batch
                job = sampler.run(
                    [(qc,) for qc in circuits],   # list of PUBs
                    shots=self.shots,
                )
                job_list.append((job, len(circuits)))

            # Collect results after all jobs are submitted
            print(f"\nAll {len(job_list)} job(s) submitted. Waiting for results...")
            for job, n_circuits in tqdm(job_list, desc="Collecting results"):
                pub_results = job.result()
                for k in range(n_circuits):
                    results.append(self._prob_zero(pub_results[k]))

        return np.array(results, dtype=np.float64)

    # ------------------------------------------------------------------
    # Symmetric kernel matrix
    # ------------------------------------------------------------------

    def kernel_matrix(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        N = len(X)

        pairs_a, pairs_b = [], []
        for i in range(N):
            for j in range(i, N):
                pairs_a.append(X[i])
                pairs_b.append(X[j])

        vals = self.forward(np.array(pairs_a), np.array(pairs_b), weights)

        K, idx = np.zeros((N, N), dtype=np.float64), 0
        for i in range(N):
            for j in range(i, N):
                K[i, j] = K[j, i] = float(vals[idx])
                idx += 1
        return K

    # ------------------------------------------------------------------
    # Rectangular kernel matrix (test vs train)
    # ------------------------------------------------------------------

    def rectangular_kernel_matrix(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        X_train = np.array(X_train, dtype=np.float64)
        X_test  = np.array(X_test,  dtype=np.float64)
        M, N    = len(X_test), len(X_train)

        pairs_a = np.repeat(X_test, N, axis=0)
        pairs_b = np.tile(X_train, (M, 1))

        vals = self.forward(pairs_a, pairs_b, weights)
        return vals.reshape(M, N)


class HardwareKernelModel:
    """
    Adapter that makes HardwareKernelRunner compatible with BaseKTA's
    kernel_model interface:
        - kernel_model.circuit.init_weights()
        - kernel_model.forward(x1, x2, weights)
        - kernel_model.circuit_executions
    """

    def __init__(self, circuit, runner):
        self.circuit = circuit          # exposes .init_weights()
        self._runner = runner

    def forward(self, x1, x2, weights):
        # KTA optimizers pass JAX arrays — convert to numpy for Qiskit
        return self._runner.forward(
            np.array(x1, dtype=np.float64),
            np.array(x2, dtype=np.float64),
            np.array(weights, dtype=np.float64),
        )

    @property
    def circuit_executions(self):
        return self._runner.circuit_executions

    @circuit_executions.setter
    def circuit_executions(self, value):
        self._runner.circuit_executions = value