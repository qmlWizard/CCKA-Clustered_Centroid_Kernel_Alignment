import pennylane as qml
import jax
import jax.numpy as jnp


class quackEmbeddingCircuit:
    """
    JAX-safe angle-embedding quantum kernel.
    All inputs MUST already have shape (num_qubits,).
    No shape logic is allowed inside the circuit.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        reupload: bool = True,
    ):
        self.num_qubits = num_qubits
        self.reps = reps
        self.reupload = reupload
        self.wires = list(range(num_qubits))

    # --------------------------------------------------
    # Feature Map
    # --------------------------------------------------
    def feature_map(self, x, scale):
        scale = jnp.asarray(scale)

        for i, wire in enumerate(self.wires):
            qml.Hadamard(wires=wire)
            qml.RZ(scale[i] * x[i], wires=wire)

    # --------------------------------------------------
    # Ansatz
    # --------------------------------------------------
    def ansatz(self, var, rot):
        var = jnp.asarray(var)
        rot = jnp.asarray(rot)

        for i, wire in enumerate(self.wires):
            qml.RY(var[i], wires=wire)

        for i in range(self.num_qubits):
            qml.CRZ(
                rot[i],
                wires=[self.wires[i], self.wires[(i + 1) % self.num_qubits]],
            )

    # --------------------------------------------------
    # Circuit Builder
    # --------------------------------------------------
    def _build_circuit(self, x, weights):
        """
        weights[rep] = (
            scaling_params: (num_qubits,),
            (variational_params: (num_qubits,),
             rotational_params: (num_qubits,))
        )
        """

        idx = 0
        for rep in range(self.reps):
            scale = weights[idx : idx + self.num_qubits]
            idx += self.num_qubits

            var = weights[idx : idx + self.num_qubits]
            idx += self.num_qubits

            rot = weights[idx : idx + self.num_qubits]
            idx += self.num_qubits

            if rep == 0 or self.reupload:
                self.feature_map(x, scale)

            self.ansatz(var, rot)

    # --------------------------------------------------
    # Kernel Circuit (QNode entry)
    # --------------------------------------------------
    def kernel_circuit(self, x1, x2, weights):
        """
        x1, x2 must have shape (num_qubits,)
        """
        self._build_circuit(x1, weights)
        qml.adjoint(self._build_circuit)(x2, weights)

        return qml.expval(qml.Projector([0] * self.num_qubits, wires=self.wires))

    # --------------------------------------------------
    # Parameter Shape
    # --------------------------------------------------
    def parameter_shape(self):
        return (
            self.reps,
            (
                (self.num_qubits,),                 # scaling
                (
                    (self.num_qubits,),             # variational
                    (self.num_qubits,)              # rotational
                ),
            ),
        )

    # --------------------------------------------------
    # JAX Weight Initializer
    # --------------------------------------------------
    def init_weights(self, seed=0, minval=-jnp.pi, maxval=jnp.pi):
        key = jax.random.PRNGKey(seed)

        total_params = self.reps * 3 * self.num_qubits
        flat = jax.random.uniform(
            key,
            shape=(total_params,),
            minval=minval,
            maxval=maxval,
        )   
        return flat