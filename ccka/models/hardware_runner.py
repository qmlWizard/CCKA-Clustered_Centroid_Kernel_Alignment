# hardware_runner.py

import numpy as np
import pennylane as qml
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2, Session
from tqdm.auto import tqdm


class HardwareKernelRunner:
    """
    Drop-in hardware replacement for KernelModel.

    Mirrors KernelModel's interface exactly:
        .forward(x1, x2, weights)       → np.ndarray (N,)
        .kernel_matrix(X, weights)      → np.ndarray (N, N)
        .rectangular_kernel_matrix(...) → np.ndarray (M, N)
        .circuit_executions             → int

    Uses Qiskit 1.x SamplerV2 directly — no pennylane-qiskit needed.

    Args:
        circuit      : quackEmbeddingCircuit instance
        ibm_backend  : AerSimulator() for local testing,
                       or QiskitRuntimeService().backend(...) for real hardware
        shots        : number of shots per circuit execution
        batch_size   : circuits per job submission (IBM limit: 300)
    """

    def __init__(
        self,
        circuit,
        ibm_backend,
        shots: int = 1024,
        batch_size: int = 50,
    ):
        if circuit is None:
            raise ValueError("A circuit must be provided.")
        if ibm_backend is None:
            raise ValueError("ibm_backend must be provided.")

        self.circuit            = circuit
        self.backend            = ibm_backend
        self.shots              = shots
        self.batch_size         = batch_size
        self.num_qubits         = circuit.num_qubits
        self.circuit_executions = 0

        # PennyLane device — used ONLY for tape building, never executed
        self._pl_dev = qml.device('default.qubit', wires=self.num_qubits)

        # Transpilation pass manager for the target backend
        self._pm = generate_preset_pass_manager(
            optimization_level=1,
            backend=ibm_backend,
        )

    # ------------------------------------------------------------------
    # Internal: build one transpiled Qiskit circuit for a single pair
    # ------------------------------------------------------------------
    def _build_qiskit_circuit(self, x1, x2, weights):
        with qml.tape.QuantumTape() as tape:
            self.circuit.kernel_circuit(
                np.array(x1, dtype=np.float64),
                np.array(x2, dtype=np.float64),
                np.array(weights, dtype=np.float64),
            )

        # Convert PennyLane tape → Qiskit circuit (adds measurements)
        from pennylane_qiskit.converter import tape_to_qiskit
        qc = tape_to_qiskit(tape)
        qc.measure_all()

        return self._pm.run(qc)

    # ------------------------------------------------------------------
    # Internal: prob(|00...0>) from a single SamplerV2 pub result
    # ------------------------------------------------------------------
    @staticmethod
    def _prob_zero(pub_result, num_qubits):
        counts    = pub_result.data.meas.get_counts()
        total     = sum(counts.values())
        zero_state = '0' * num_qubits
        return counts.get(zero_state, 0) / total

    # ------------------------------------------------------------------
    # forward  — identical signature to KernelModel.forward
    # ------------------------------------------------------------------
    def forward(self, x1, x2, weights):
        """
        x1, x2   : array-like (N, num_qubits)
        weights  : flat array (reps * 3 * num_qubits,)
        returns  : np.ndarray of shape (N,)
        """
        x1         = np.array(x1, dtype=np.float64)
        x2         = np.array(x2, dtype=np.float64)
        weights_np = np.array(weights, dtype=np.float64)
        N          = len(x1)

        self.circuit_executions += N
        results = []

        with Session(backend=self.backend) as session:
            sampler = SamplerV2(mode=session)
            sampler.options.default_shots = self.shots

            for start in tqdm(range(0, N, self.batch_size), desc='HW batch'):
                batch_x1 = x1[start : start + self.batch_size]
                batch_x2 = x2[start : start + self.batch_size]

                circuits = [
                    self._build_qiskit_circuit(a, b, weights_np)
                    for a, b in zip(batch_x1, batch_x2)
                ]

                job        = sampler.run(circuits)
                pub_result = job.result()

                for i in range(len(circuits)):
                    results.append(
                        self._prob_zero(pub_result[i], self.num_qubits)
                    )

        return np.array(results, dtype=np.float64)   # shape (N,)

    # ------------------------------------------------------------------
    # kernel_matrix  — identical to KernelModel.kernel_matrix
    # ------------------------------------------------------------------
    def kernel_matrix(self, X, weights):
        """
        Full symmetric N×N kernel matrix.
        Only computes N(N+1)/2 pairs (upper triangle), mirrors the rest.
        """
        X  = np.array(X, dtype=np.float64)
        N  = len(X)

        pairs_a, pairs_b = [], []
        for i in range(N):
            for j in range(i, N):
                pairs_a.append(X[i])
                pairs_b.append(X[j])

        vals = self.forward(np.array(pairs_a), np.array(pairs_b), weights)

        K   = np.zeros((N, N), dtype=np.float64)
        idx = 0
        for i in range(N):
            for j in range(i, N):
                K[i, j] = K[j, i] = float(vals[idx])
                idx += 1
        return K

    # ------------------------------------------------------------------
    # rectangular_kernel_matrix  — identical to KernelModel equivalent
    # ------------------------------------------------------------------
    def rectangular_kernel_matrix(self, X_train, X_test, weights):
        """
        M×N matrix for prediction (test rows × train cols).
        """
        X_train = np.array(X_train, dtype=np.float64)
        X_test  = np.array(X_test,  dtype=np.float64)
        M, N    = len(X_test), len(X_train)

        pairs_a = np.repeat(X_test,  N, axis=0)   # M*N rows
        pairs_b = np.tile(X_train, (M, 1))         # M*N rows

        vals = self.forward(pairs_a, pairs_b, weights)
        return vals.reshape(M, N)