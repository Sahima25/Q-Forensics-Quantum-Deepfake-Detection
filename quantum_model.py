import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

n_qubits = 4
# Lightning qubit is fine, but we remove the broadcasting setting
dev = qml.device("lightning.qubit", wires=n_qubits)

# REMOVE diff_method="parameter-shift" from the decorator for now
# 'backprop' or 'best' is safer when using Torch + Loops
@qml.qnode(dev, interface="torch")
def q_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumGenerator(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.pre_q = nn.Linear(latent_dim, n_qubits)
        self.q_weights = nn.Parameter(0.01 * torch.randn(3, n_qubits, 3))
        self.post_q = nn.Linear(n_qubits, 578) 
        self.lateral_v = nn.Parameter(torch.randn(1, 578) * 0.05)

    def forward(self, z):
        z_q = torch.tanh(self.pre_q(z)) * np.pi
        
        # --- FIXED: MANUAL BATCHING FOR GRADIENT SAFETY ---
        # PennyLane's parameter-shift hates broadcasted gradients. 
        # We loop here but keep it as a torch.stack to preserve the graph.
        q_results = []
        for i in range(z_q.shape[0]):
            res = q_circuit(z_q[i], self.q_weights)
            q_results.append(torch.stack(res))
        
        q_out = torch.stack(q_results)
        return torch.tanh(self.post_q(q_out.float()) + self.lateral_v)

class HybridConcatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.reducer = nn.Linear(578, n_qubits)
        self.q_weights = nn.Parameter(0.01 * torch.randn(3, n_qubits, 3))
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits + n_qubits, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        red = torch.tanh(self.reducer(x)) * np.pi
        
        # --- FIXED: MANUAL BATCHING FOR GRADIENT SAFETY ---
        q_results = []
        for i in range(red.shape[0]):
            res = q_circuit(red[i], self.q_weights)
            q_results.append(torch.stack(res))
            
        q_out = torch.stack(q_results)
        combined = torch.cat([red, q_out.float()], dim=1)
        return self.classifier(combined)
