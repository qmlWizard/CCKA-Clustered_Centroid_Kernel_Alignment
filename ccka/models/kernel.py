import pennylane as qml
import jax

from ccka.circuits.angleEmbeddingKernel import quackEmbeddingCircuit


class KernelModel:

    def __init__(self, 
                 circuit = None,
                 device_name: str = 'default.qubit',
                 interface: str = 'jax',
                 diff_method: str = 'backprop',
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
        self.circuit_executions = 0
        if circuit is None:
            raise ValueError("A circuit must be provided to the KernelModel.")

        self.circuit = circuit
    
        # ----- device
        if noisy:
            self._device_name = 'default.mixed'
            self._shots = 1000
        else:
            self._device_name = device_name
            self._shots = None

        dev = qml.device(self._device_name, wires = self.circuit.num_qubits, shots = self._shots, )

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

