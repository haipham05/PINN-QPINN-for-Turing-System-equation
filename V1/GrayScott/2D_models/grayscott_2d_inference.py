"""
Gray-Scott 2D QPINN - Inference Script
Evaluation of trained 2D Gray-Scott QPINN models

This script loads trained models and evaluates them on the domain.
Computes metrics (MSE, L2, L∞) against reference solution.

Author: QPINN Research
Date: 2024-2025
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Configuration for Gray-Scott 2D inference"""
    
    SEED = 42
    
    # Quantum Circuit Parameters
    N_LAYERS = 4
    N_WIRES = 4
    
    # FNN Basis Parameters
    HIDDEN_LAYERS_FNN = 2
    NEURONS_FNN = 16
    
    # QNN Embedding Parameters
    N_LAYERS_EMBED = 2
    
    # Domain Parameters
    T_MIN = 0.0
    T_MAX = 1.0
    X_MIN = 0.0
    X_MAX = 1.0
    Y_MIN = 0.0
    Y_MAX = 1.0
    
    # Inference grid
    N_T = 11
    N_X = 32
    N_Y = 32
    
    # Model directory
    MODEL_DIR = "result"
    OUTPUT_DIR = "result"


# ============================================================
# NEURAL NETWORK MODELS
# ============================================================

class FNNBasisNet(nn.Module):
    """FNN network for basis generation"""
    
    def __init__(self, n_hidden_layers, width, output_dim, input_dim=3):
        super().__init__()
        self.n_hidden_layers = n_hidden_layers
        layers = [nn.Linear(input_dim, width)]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, output_dim))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for i in range(self.n_hidden_layers):
            x = torch.tanh(self.layers[i](x))
        return self.layers[-1](x)


class QNNEmbedding(nn.Module):
    """Quantum embedding network"""
    
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
                if i % 2 == 0:
                    qml.RX(x[0], wires=i)
                else:
                    qml.RY(x[1], wires=i)
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


# ============================================================
# INFERENCE ENGINE
# ============================================================

class GrayScott2DQPINNInference:
    """Inference engine for Gray-Scott 2D QPINN"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.domain_min = torch.tensor([config.T_MIN, config.X_MIN, config.Y_MIN], device=device)
        self.domain_max = torch.tensor([config.T_MAX, config.X_MAX, config.Y_MAX], device=device)
        
        # Load reference solution
        self._load_reference_solution()
        
        # Create evaluation grid
        self._create_eval_grid()
    
    def _load_reference_solution(self):
        """Load reference solution"""
        ref_path = os.path.join(self.config.MODEL_DIR, "grayscott_3d_reference_solution.npy")
        
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference solution not found at {ref_path}")
        
        loaded = np.load(ref_path, allow_pickle=True)[()]
        self.interp_u = loaded['u']
        self.interp_v = loaded['v']
        self.t_ref = loaded['t']
        self.x_ref = loaded['x']
        self.y_ref = loaded['y']
        
        # Get bounds from solution arrays
        u_sol = loaded['u_sol']
        v_sol = loaded['v_sol']
        
        print(f"✓ Reference solution loaded: u∈[{u_sol.min():.4f}, {u_sol.max():.4f}], "
              f"v∈[{v_sol.min():.4f}, {v_sol.max():.4f}]")
    
    def _create_eval_grid(self):
        """Create evaluation grid"""
        t_eval = np.linspace(self.config.T_MIN, self.config.T_MAX, self.config.N_T)
        x_eval = np.linspace(self.config.X_MIN, self.config.X_MAX, self.config.N_X)
        y_eval = np.linspace(self.config.Y_MIN, self.config.Y_MAX, self.config.N_Y)
        
        self.T_eval = t_eval
        self.X_eval = x_eval
        self.Y_eval = y_eval
        
        # Store unique values
        self.T_unique = t_eval
        self.X_unique = x_eval
        self.Y_unique = y_eval
        
        # Create full evaluation domain
        eval_points = []
        for t in t_eval:
            for x in x_eval:
                for y in y_eval:
                    eval_points.append([t, x, y])
        
        self.eval_domain = torch.tensor(eval_points, dtype=torch.float32, device=self.device)
        
        print(f"✓ Evaluation grid created: {len(self.eval_domain)} points")
    
    def _create_circuit(self):
        """Create quantum circuit"""
        dev = qml.device("default.qubit", wires=self.config.N_WIRES)
        
        @qml.qnode(dev, interface="torch")
        def circuit(x, theta, basis):
            for i in range(self.config.N_WIRES):
                qubit_idx = i % 3
                if qubit_idx == 0:
                    qml.RY(basis[i] * x[0], wires=i)  # t
                elif qubit_idx == 1:
                    qml.RY(basis[i] * x[1], wires=i)  # x
                else:
                    qml.RY(basis[i] * x[2], wires=i)  # y
            
            for layer in range(self.config.N_LAYERS):
                for qubit in range(self.config.N_WIRES):
                    qml.RX(theta[layer, qubit, 0], wires=qubit)
                    qml.RY(theta[layer, qubit, 1], wires=qubit)
                    qml.RZ(theta[layer, qubit, 2], wires=qubit)
                for qubit in range(self.config.N_WIRES - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
        
        return circuit
    
    def _postprocess_output(self, raw_output):
        """No postprocessing - output is already in physical range"""
        return raw_output
    
    def model(self, x, embedding_type, theta, basis_net=None, qnn_embedding=None):
        """Forward pass"""
        x_rescaled = 2.0 * (x - self.domain_min) / (self.domain_max - self.domain_min) - 1.0
        
        if embedding_type == "FNN_BASIS":
            with torch.no_grad():
                basis = basis_net(x_rescaled)
                raw = self.circuit(x_rescaled.T, theta, basis.T)
                raw_stacked = torch.stack(raw).T
            return self._postprocess_output(raw_stacked)
        elif embedding_type == "QNN":
            with torch.no_grad():
                basis = qnn_embedding(x_rescaled)
                raw = self.circuit(x_rescaled.T, theta, basis.T)
                raw_stacked = torch.stack(raw).T
            return self._postprocess_output(raw_stacked)
        else:  # PINN
            with torch.no_grad():
                return basis_net(x_rescaled)
    
    def inference(self, embedding_type):
        """Perform inference and compute metrics"""
        method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
        
        print(f"\n{'='*70}")
        print(f"INFERENCE: {method_name}")
        print(f"{'='*70}")
        
        # Load model
        if embedding_type == "FNN_BASIS":
            model_path = os.path.join(self.config.MODEL_DIR, "grayscott_2d_fnn_basis.npy")
            model_data = np.load(model_path, allow_pickle=True)[()]
            theta = torch.tensor(model_data['theta'], dtype=torch.float32, device=self.device)
            basis_net = FNNBasisNet(
                self.config.HIDDEN_LAYERS_FNN,
                self.config.NEURONS_FNN,
                self.config.N_WIRES,
                input_dim=3
            ).to(self.device)
            basis_net.load_state_dict(model_data['basis_net'])
            basis_net.eval()
            
            self.circuit = self._create_circuit()
            pred = self.model(self.eval_domain, embedding_type, theta, basis_net=basis_net)
            
        elif embedding_type == "QNN":
            model_path = os.path.join(self.config.MODEL_DIR, "grayscott_2d_qnn.npy")
            model_data = np.load(model_path, allow_pickle=True)[()]
            theta = torch.tensor(model_data['theta'], dtype=torch.float32, device=self.device)
            qnn_embedding = QNNEmbedding(
                self.config.N_WIRES,
                self.config.N_LAYERS_EMBED,
                self.config.N_WIRES,
                input_dim=3
            ).to(self.device)
            qnn_embedding.load_state_dict(model_data['qnn_embedding'])
            qnn_embedding.eval()
            
            self.circuit = self._create_circuit()
            pred = self.model(self.eval_domain, embedding_type, theta, qnn_embedding=qnn_embedding)
            
        else:  # PINN
            model_path = os.path.join(self.config.MODEL_DIR, "grayscott_2d_pinn.pt")
            pinn = FNNBasisNet(
                self.config.HIDDEN_LAYERS_FNN,
                self.config.NEURONS_FNN,
                2,
                input_dim=3
            ).to(self.device)
            pinn.load_state_dict(torch.load(model_path, map_location=self.device))
            pinn.eval()
            
            pred = self.model(self.eval_domain, embedding_type, None, basis_net=pinn)
        
        # Get reference
        eval_np = self.eval_domain.detach().cpu().numpy()
        ref_u = np.array([self.interp_u([pt[1], pt[2], pt[0]]) for pt in eval_np])
        ref_v = np.array([self.interp_v([pt[1], pt[2], pt[0]]) for pt in eval_np])
        ref = torch.tensor(np.stack([ref_u, ref_v], axis=1), device=self.device, dtype=torch.float32)
        
        # Compute metrics
        pred_u = pred[:, 0]
        pred_v = pred[:, 1]
        ref_u_t = ref[:, 0]
        ref_v_t = ref[:, 1]
        
        mse_u = torch.mean((pred_u - ref_u_t) ** 2).item()
        mse_v = torch.mean((pred_v - ref_v_t) ** 2).item()
        
        l2_u = torch.sqrt(torch.mean((pred_u - ref_u_t) ** 2)) / (torch.max(ref_u_t) - torch.min(ref_u_t))
        l2_v = torch.sqrt(torch.mean((pred_v - ref_v_t) ** 2)) / (torch.max(ref_v_t) - torch.min(ref_v_t))
        
        linf_u = torch.max(torch.abs(pred_u - ref_u_t)).item()
        linf_v = torch.max(torch.abs(pred_v - ref_v_t)).item()
        
        metrics = {
            'mse_u': mse_u, 'mse_v': mse_v,
            'l2_u': l2_u.item(), 'l2_v': l2_v.item(),
            'linf_u': linf_u, 'linf_v': linf_v
        }
        
        print(f"\nMetrics for {method_name}:")
        print(f"  u: MSE={mse_u:.4e}, L2_rel={l2_u:.4e}, L∞={linf_u:.4e}")
        print(f"  v: MSE={mse_v:.4e}, L2_rel={l2_v:.4e}, L∞={linf_v:.4e}")
        
        return metrics, pred


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = Config()
    inference = GrayScott2DQPINNInference(config, device)
    
    # Run inference
    results = {}
    results['PINN'], _ = inference.inference("NONE")
    results['FNN-TE-QPINN'], _ = inference.inference("FNN_BASIS")
    results['QNN-TE-QPINN'], _ = inference.inference("QNN")
    
    # Save results
    results_path = os.path.join(config.OUTPUT_DIR, "grayscott_2d_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
