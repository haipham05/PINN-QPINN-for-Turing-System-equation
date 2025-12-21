"""
Thomas 2D QPINN - Inference Script
Evaluates 2D models on dense 3D domain grid (t, x, y)

Author: QPINN Research
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Configuration for inference evaluation grid"""
    # Inference Domain Parameters (choose number of evaluation points)
    T_EVAL_POINTS = 11      # Number of time points for evaluation
    X_EVAL_POINTS = 11      # Number of x points for evaluation
    Y_EVAL_POINTS = 11      # Number of y points for evaluation
    T_MIN = 0.0
    T_MAX = 1.0
    X_MIN = 0.0
    X_MAX = 1.0
    Y_MIN = 0.0
    Y_MAX = 1.0


# ============================================================
# NEURAL NETWORK MODELS
# ============================================================

class FNNBasisNet(nn.Module):
    """FNN for basis generation (3D input)"""
    
    def __init__(self, n_hidden_layers, width, output_dim, input_dim=3):
        super().__init__()
        layers = [nn.Linear(input_dim, width)]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, output_dim))
        self.layers = nn.ModuleList(layers)
        self.n_hidden_layers = n_hidden_layers
    
    def forward(self, x):
        for i in range(self.n_hidden_layers):
            x = torch.tanh(self.layers[i](x))
        return self.layers[-1](x)


class QNNEmbedding(nn.Module):
    """Quantum embedding for 3D"""
    
    def __init__(self, n_wires, n_layers_embed, output_dim, input_dim=3):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers_embed = n_layers_embed
        self.output_dim = output_dim
        
        self.weights_embed = nn.Parameter(
            torch.randn(n_layers_embed, n_wires, 3, requires_grad=True)
        )
        
        self.dev = qml.device("default.qubit", wires=n_wires)
        self.qnode_embed = qml.QNode(self._circuit_embed, self.dev, interface="torch", diff_method="best")
    
    def _circuit_embed(self, x, weights):
        for layer in range(self.n_layers_embed):
            for i in range(self.n_wires):
                qml.RX(x[i % len(x)], wires=i)
            for i in range(self.n_wires):
                qml.RX(weights[layer, i, 0], wires=i)
                qml.RY(weights[layer, i, 1], wires=i)
                qml.RZ(weights[layer, i, 2], wires=i)
            if self.n_wires > 1:
                for i in range(self.n_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.output_dim)]
    
    def forward(self, x):
        x_t = x.T
        basis_t = self.qnode_embed(x_t, self.weights_embed)
        if isinstance(basis_t, list):
            basis_t = torch.stack(basis_t) * torch.pi
        else:
            basis_t = basis_t * torch.pi
        return basis_t.T


class Thomas2DInferenceEngine:
    """Inference engine for 2D Thomas equation"""
    
    def __init__(self, device):
        self.device = device
        self.T_MIN, self.T_MAX = 0.0, 1.0
        self.X_MIN, self.X_MAX = 0.0, 1.0
        self.Y_MIN, self.Y_MAX = 0.0, 1.0
        self.domain_min = torch.tensor([self.T_MIN, self.X_MIN, self.Y_MIN], device=device)
        self.domain_max = torch.tensor([self.T_MAX, self.X_MAX, self.Y_MAX], device=device)
        self._load_reference_solution()
    
    def _load_reference_solution(self):
        """Load 2D reference solution"""
        ref_path = os.path.join("result", "thomas_reference_solution_2d.npy")
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference solution not found at {ref_path}")
        
        loaded = np.load(ref_path, allow_pickle=True)[()]
        self.interp_u = loaded['u']
        self.interp_v = loaded['v']
        self.ref_t = loaded['t']
        self.ref_x = loaded['x']
        self.ref_y = loaded['y']
    
    def _create_circuit(self, n_wires, n_layers):
        """Create quantum circuit for 3D"""
        dev = qml.device("default.qubit", wires=n_wires)
        
        @qml.qnode(dev, interface="torch")
        def circuit(x, theta, basis):
            for i in range(n_wires):
                qml.RY(basis[i] * x[i % 3], wires=i)
            
            for layer in range(n_layers):
                for qubit in range(n_wires):
                    qml.RX(theta[layer, qubit, 0], wires=qubit)
                    qml.RY(theta[layer, qubit, 1], wires=qubit)
                    qml.RZ(theta[layer, qubit, 2], wires=qubit)
                for qubit in range(n_wires - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
        
        return circuit
    
    def infer_pinn(self, points):
        """Inference for PINN"""
        x_rescaled = 2.0 * (points - self.domain_min) / (self.domain_max - self.domain_min) - 1.0
        return self.pinn(x_rescaled).detach().cpu().numpy()
    
    def infer_fnn_basis(self, points):
        """Inference for FNN-TE-QPINN"""
        x_rescaled = 2.0 * (points - self.domain_min) / (self.domain_max - self.domain_min) - 1.0
        basis = self.basis_net(x_rescaled)
        circuit = self._create_circuit(4, 5)
        raw = circuit(x_rescaled.T, torch.tensor(self.theta), basis.T)
        return torch.stack(raw).T.detach().cpu().numpy()
    
    def infer_qnn(self, points):
        """Inference for QNN-TE-QPINN"""
        x_rescaled = 2.0 * (points - self.domain_min) / (self.domain_max - self.domain_min) - 1.0
        basis = self.qnn_embedding(x_rescaled)
        circuit = self._create_circuit(4, 5)
        raw = circuit(x_rescaled.T, torch.tensor(self.theta), basis.T)
        return torch.stack(raw).T.detach().cpu().numpy()
    
    def evaluate(self, t_eval, x_eval, y_eval, model_type):
        """Evaluate model on 3D grid"""
        # Create 3D evaluation grid
        T, X, Y = np.meshgrid(t_eval, x_eval, y_eval, indexing='ij')
        points = np.column_stack([T.ravel(), X.ravel(), Y.ravel()])
        points_torch = torch.tensor(points, dtype=torch.float32, device=self.device)
        
        # Load and evaluate model
        if model_type == "PINN":
            self.pinn = FNNBasisNet(2, 20, 2, input_dim=3).to(self.device)
            self.pinn.load_state_dict(torch.load("result/thomas_2d_pinn.pt"))
            self.pinn.eval()
            with torch.no_grad():
                pred = self.infer_pinn(points_torch)
        elif model_type == "FNN":
            data = np.load("result/thomas_2d_fnn_basis.npy", allow_pickle=True)[()]
            self.theta = data['theta']
            self.basis_net = FNNBasisNet(2, 20, 4, input_dim=3).to(self.device)
            self.basis_net.load_state_dict(data['basis_net'])
            self.basis_net.eval()
            with torch.no_grad():
                pred = self.infer_fnn_basis(points_torch)
        else:
            data = np.load("result/thomas_2d_qnn.npy", allow_pickle=True)[()]
            self.theta = data['theta']
            self.qnn_embedding = QNNEmbedding(4, 2, 4, input_dim=3).to(self.device)
            self.qnn_embedding.load_state_dict(data['qnn_embedding'])
            self.qnn_embedding.eval()
            with torch.no_grad():
                pred = self.infer_qnn(points_torch)
        
        # Reshape predictions
        u_pred = pred[:, 0].reshape(T.shape)
        v_pred = pred[:, 1].reshape(T.shape)
        
        # Get reference solution
        u_ref = np.array([[[self.interp_u([t, x, y]).squeeze() for y in y_eval] for x in x_eval] for t in t_eval])
        v_ref = np.array([[[self.interp_v([t, x, y]).squeeze() for y in y_eval] for x in x_eval] for t in t_eval])
        
        # Compute metrics
        mse_u = np.mean((u_pred - u_ref) ** 2)
        mse_v = np.mean((v_pred - v_ref) ** 2)
        linf_u = np.max(np.abs(u_pred - u_ref))
        linf_v = np.max(np.abs(v_pred - v_ref))
        
        print(f"{model_type:12} | MSE_u: {mse_u:.4e} | MSE_v: {mse_v:.4e} | L∞_u: {linf_u:.4e} | L∞_v: {linf_v:.4e}")
        
        return u_pred, v_pred, u_ref, v_ref


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    engine = Thomas2DInferenceEngine(device)
    
    # Dense evaluation grid
    t_eval = np.linspace(0.0, 1.0, 11)
    x_eval = np.linspace(0.0, 1.0, 32)
    y_eval = np.linspace(0.0, 1.0, 32)
    
    print("=" * 90)
    print("THOMAS 2D INFERENCE RESULTS")
    print("=" * 90)
    
    for model_type in ["PINN", "FNN", "QNN"]:
        try:
            u, v, u_ref, v_ref = engine.evaluate(t_eval, x_eval, y_eval, model_type)
        except Exception as e:
            print(f"{model_type:12} | Error: {str(e)}")
