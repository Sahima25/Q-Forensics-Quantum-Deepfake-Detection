import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

# -------------------------------
# 4-Qubit Quantum Circuit (Multi-output)
# -------------------------------
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit_multi(inputs, weights):
    """
    inputs: 4 classical features (after reduction) in [0, pi]
    weights: trainable parameters of shape (n_layers, n_qubits, 3)
    Returns: expectation values of PauliZ on all 4 qubits
    """
    # Angle encoding
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    n_layers = weights.shape[0]
    for l in range(n_layers):
        # Variational layer: RZ, RY, RZ on each qubit
        for i in range(n_qubits):
            qml.RZ(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)

        # Entanglement layer (ring of CNOTs)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i+1) % n_qubits])

    # Return expectation values on all qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayerMulti(nn.Module):
    """
    Wrapper for the quantum circuit that handles batch processing.
    """
    def __init__(self, n_layers=6):
        super().__init__()
        self.n_layers = n_layers
        # Initialize weights with small random values
        self.weights = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits, 3, dtype=torch.float32))

    def forward(self, x):
        """
        x: tensor of shape (batch_size, 4) – each row is a set of 4 features in [0, pi]
        Returns: tensor of shape (batch_size, 4) – quantum outputs for all qubits
        """
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            out = quantum_circuit_multi(x[i], self.weights)
            out = torch.stack(out)          # list -> tensor (4,)
            outputs.append(out.float())
        return torch.stack(outputs)          # (batch, 4)

# -------------------------------
# Hybrid Model (Concatenation)
# -------------------------------
class HybridConcatModel(nn.Module):
    """
    Hybrid quantum-classical model for deepfake detection.
    Steps:
      1. Reduce input features (578-dim) to 4-dim via a small neural network.
      2. Feed reduced features to quantum circuit (clamped to [0, pi]).
      3. Concatenate reduced features and quantum outputs (4 + 4 = 8-dim).
      4. Final classifier (8 -> hidden -> 1) produces logit.
    """
    def __init__(self, input_dim, reduced_dim=4, n_layers=6, hidden=64):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, reduced_dim)
        )
        self.quantum = QuantumLayerMulti(n_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(reduced_dim + 4, hidden),   # concatenated: reduced + quantum_out
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        """
        x: input features of shape (batch, input_dim)
        Returns: logits of shape (batch,)
        """
        reduced = self.reducer(x)
        # Clamp to [0, pi] for angle embedding
        reduced_clamped = torch.clamp(reduced, 0, np.pi)
        quantum_out = self.quantum(reduced_clamped)   # (batch, 4)
        combined = torch.cat([reduced, quantum_out], dim=1)  # (batch, 8)
        logits = self.classifier(combined).squeeze(1)
        return logits
