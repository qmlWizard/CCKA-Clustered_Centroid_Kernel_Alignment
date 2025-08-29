import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from utils.pennylane.ansatz_noisy import qkhe, qkcovariant, qkembedding_paper, he_state, covariant_state, embedding_paper_state
from utils.pennylane.ansatz import qkhe, qkcovariant, qkembedding_paper

torch.manual_seed(42)
np.random.seed(42)

class Qkernel(nn.Module):
    def __init__(self, device, n_qubits, trainable, input_scaling,
                 data_reuploading, ansatz, ansatz_layers, noise_prob, diff_method, shots = None, noisy=False):
        super().__init__()
        self._device_name = device
        self._n = n_qubits
        self._layers = ansatz_layers
        self._data_reuploading = data_reuploading
        self._ansatz = ansatz
        self._noisy = noisy
        self._noise_p = float(noise_prob)
        self._wires = list(range(self._n))
        self._shots = shots
        self._projector = torch.zeros((2**self._n,2**self._n))
        self._projector[0,0] = 1
        self._circuit_executions = 0

        # ----- params
        self.input_scaling = nn.Parameter(torch.ones(self._layers, self._n), requires_grad=True)
        if ansatz == "he":
            self.variational = nn.Parameter((torch.rand(self._layers, 2*self._n)*2*math.pi)-math.pi, requires_grad=True)
        elif ansatz == "embedding_paper":
            self.variational = nn.Parameter((torch.rand(self._layers, self._n)*2*math.pi)-math.pi, requires_grad=True)
            self.rotational  = nn.Parameter((torch.rand(self._layers, self._n)*2*math.pi)-math.pi, requires_grad=True)
        elif ansatz == "covariant":
            self.variational = nn.Parameter((torch.rand(self._layers, 2*self._n)*2*math.pi)-math.pi, requires_grad=True)
        else:
            raise ValueError("Unknown ansatz")

        # ----- device
        dev = qml.device(self._device_name, wires=self._wires, shots = self._shots)

        # ----- pick state function
        if self._noisy:
            if ansatz == "he":
                def _state(x, weights, wires, layers, data_reuploading, noise_p):
                    return he_state(x, weights, wires, layers, data_reuploading, noise_p)
            elif ansatz == "embedding_paper":
                def _state(x, weights, wires, layers, data_reuploading, noise_p):
                    return embedding_paper_state(x, weights, wires, layers, data_reuploading, noise_p)
            else:  # covariant
                def _state(x, weights, wires, layers, data_reuploading, noise_p):
                    return covariant_state(x, weights, wires, layers, data_reuploading, entanglement=None, noise_p=noise_p)

            # QNode returning density matrix
            self._rho = qml.QNode(_state, dev, interface="torch", diff_method=diff_method)

        else:
            if self._ansatz == 'he':
                self._kernel = qml.QNode(qkhe, dev, diff_method= diff_method, interface='torch')
            elif self._ansatz == 'embedding_paper':
                self._kernel = qml.QNode(qkembedding_paper, dev, diff_method= diff_method, interface='torch')
            elif self._ansatz == 'covariant':
                self._kernel = qml.QNode(qkhe, dev, diff_method= diff_method, interface='torch')
            else:
                #self._kernel = qml.QNode(qkhe, dev, diff_method='adjoint', interface='torch')
                print("No Kernel Ansatz selected!")

    @property
    def weights(self):
        W = {"input_scaling": self.input_scaling, "variational": self.variational}
        if hasattr(self, "rotational"):
            W["rotational"] = self.rotational
        return W

    def forward(self, x1, x2):
        if self._noisy:
            # ensure 2D batch: (B, n) or (1, n)
            if x1.ndim == 1: x1 = x1.unsqueeze(0)
            if x2.ndim == 1: x2 = x2.unsqueeze(0)

            # build noisy states; broadcasting over batch handled by PL
            rho1 = self._rho(x1, self.weights, self._wires, self._layers, self._data_reuploading, self._noise_p)
            rho2 = self._rho(x2, self.weights, self._wires, self._layers, self._data_reuploading, self._noise_p)

            # HS kernel with purity normalization
            # density_matrix returns complex dtype; keep real parts
            hs   = torch.real(torch.einsum("...ij,...ji->...", rho1, rho2))          # Tr[rho1 rho2]
            pur1 = torch.real(torch.einsum("...ij,...ji->...", rho1, rho1)) + 1e-12  # Tr[rho1^2]
            pur2 = torch.real(torch.einsum("...ij,...ji->...", rho2, rho2)) + 1e-12
            K    = hs / torch.sqrt(pur1 * pur2)
            return K  # in [0,1]
        else:
            all_zero_state = self._kernel(x1, x2, self._parameters, self._wires, self._layers, self._projector, self._data_reuploading, self._noise_p)
            self._circuit_executions += 1
            return torch.abs(all_zero_state)
