"""
Gray-Scott 1D QPINN - Inference Script
Load trained models and perform inference with evaluation

This script loads the trained models and generates performance analysis

Author: QPINN Research
Date: 2024-2025
"""

import os
import json
import pickle
from itertools import product

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Configuration class for Gray-Scott 1D QPINN"""
    
    # Quantum Circuit Parameters
    N_LAYERS = 5
    N_WIRES = 4
    
    # FNN Basis Parameters
    HIDDEN_LAYERS_FNN = 2
    NEURONS_FNN = 20
    
    # QNN Embedding Parameters
    N_LAYERS_EMBED = 2
    
    # Physics Parameters
    D_U = 2.0e-5
    D_V = 1.0e-5
    F = 0.04
    K = 0.06
    
    # Boundary conditions
    U_BOUNDARY = 1.0
    V_BOUNDARY = 0.0
    
    # Domain
    T_MIN = 0.0
    T_MAX = 1.0
    X_MIN = 0.0
    X_MAX = 1.0
    
    # Inference Domain Parameters (choose number of evaluation points)
    T_EVAL_POINTS = 11      # Number of time points for evaluation
    X_EVAL_POINTS = 11      # Number of x points for evaluation
    
    # Directory configuration
    INPUT_DIR = "result"
    OUTPUT_DIR = "result"


# ============================================================
# NEURAL NETWORK MODELS
# ============================================================

class FNNBasisNet(nn.Module):
    """FNN network to generate basis for quantum circuit encoding"""
    
    def __init__(self, n_hidden_layers, width, output_dim, input_dim=2):
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
    
    def __init__(self, n_wires, n_layers_embed, output_dim, input_dim=2):
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

class GrayScott1DQPINNInference:
    """Inference engine for Gray-Scott 1D QPINN"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.domain_min = torch.tensor([config.T_MIN, config.X_MIN], device=device)
        self.domain_max = torch.tensor([config.T_MAX, config.X_MAX], device=device)
        
        # Reference solution
        self.interp_u = None
        self.interp_v = None
        
        # Grid for evaluation
        self.T_unique = None
        self.X_unique = None
    
    def load_reference(self, ref_path):
        """Load reference solution interpolators and setup evaluation grid"""
        if os.path.exists(ref_path):
            loaded = np.load(ref_path, allow_pickle=True)[()]
            self.interp_u = loaded['u']
            self.interp_v = loaded['v']
            
            # Get bounds from solution arrays
            u_sol = loaded['u_sol']
            v_sol = loaded['v_sol']
            
            print(f"✓ Reference solution loaded: u∈[{u_sol.min():.4f}, {u_sol.max():.4f}], "
                  f"v∈[{v_sol.min():.4f}, {v_sol.max():.4f}]")
        else:
            raise FileNotFoundError(f"Reference solution not found: {ref_path}")
        
        # Create evaluation grid from config parameters (independent of reference)
        self.T_unique = np.linspace(self.config.T_MIN, self.config.T_MAX, self.config.T_EVAL_POINTS)
        self.X_unique = np.linspace(self.config.X_MIN, self.config.X_MAX, self.config.X_EVAL_POINTS)
        
        print(f"   Evaluation grid: t={self.config.T_EVAL_POINTS} points, x={self.config.X_EVAL_POINTS} points")
    
    def _create_circuit(self):
        """Create quantum circuit"""
        dev = qml.device("default.qubit", wires=self.config.N_WIRES)
        
        @qml.qnode(dev, interface="torch")
        def circuit(x, theta, basis):
            for i in range(self.config.N_WIRES):
                if i % 2 == 0:
                    qml.RY(basis[i] * x[0], wires=i)
                else:
                    qml.RY(basis[i] * x[1], wires=i)
            
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
    
    def model(self, x):
        """Forward pass through model"""
        x_rescaled = 2.0 * (x - self.domain_min) / (self.domain_max - self.domain_min) - 1.0
        
        if self.embedding_type == "FNN_BASIS":
            basis = self.basis_net(x_rescaled)
            raw = self.circuit(x_rescaled.T, self.theta, basis.T)
            raw_stacked = torch.stack(raw).T
            return self._postprocess_output(raw_stacked)
        elif self.embedding_type == "QNN":
            basis = self.qnn_embedding(x_rescaled)
            raw = self.circuit(x_rescaled.T, self.theta, basis.T)
            raw_stacked = torch.stack(raw).T
            return self._postprocess_output(raw_stacked)
        else:  # NONE (PINN)
            return self.pinn(x_rescaled)
    
    def inference(self, model_path, embedding_type):
        """Load model and perform inference"""
        self.embedding_type = embedding_type
        method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
        
        print(f"\n{'='*60}")
        print(f"Running Inference: {method_name}")
        print(f"{'='*60}")
        
        # Load model
        if embedding_type == "FNN_BASIS":
            model_state = np.load(model_path, allow_pickle=True)[()]
            self.theta = torch.tensor(model_state['theta'], device=self.device)
            self.basis_net = FNNBasisNet(
                self.config.HIDDEN_LAYERS_FNN,
                self.config.NEURONS_FNN,
                self.config.N_WIRES,
                input_dim=2
            ).to(self.device)
            self.basis_net.load_state_dict(model_state['basis_net'])
            self.basis_net.eval()
            self.circuit = self._create_circuit()
            
        elif embedding_type == "QNN":
            model_state = np.load(model_path, allow_pickle=True)[()]
            self.theta = torch.tensor(model_state['theta'], device=self.device)
            self.qnn_embedding = QNNEmbedding(
                self.config.N_WIRES,
                self.config.N_LAYERS_EMBED,
                self.config.N_WIRES,
                input_dim=2
            ).to(self.device)
            self.qnn_embedding.load_state_dict(model_state['qnn_embedding'])
            self.qnn_embedding.eval()
            self.circuit = self._create_circuit()
            
        else:  # NONE (PINN)
            self.pinn = FNNBasisNet(
                self.config.HIDDEN_LAYERS_FNN,
                self.config.NEURONS_FNN,
                2,
                input_dim=2
            ).to(self.device)
            self.pinn.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            self.pinn.eval()
        
        # Create evaluation grid
        T_eval = torch.tensor(self.T_unique, dtype=torch.float32)
        X_eval = torch.tensor(self.X_unique, dtype=torch.float32)
        
        grid_points = torch.tensor(
            list(product(T_eval, X_eval)), dtype=torch.float32, device=self.device
        )
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(grid_points)
        
        pred_u = predictions[:, 0].cpu().numpy()
        pred_v = predictions[:, 1].cpu().numpy()
        
        # Get reference
        grid_np = grid_points.cpu().numpy()
        ref_u = np.array([self.interp_u([pt[0], pt[1]]).squeeze() for pt in grid_np])
        ref_v = np.array([self.interp_v([pt[0], pt[1]]).squeeze() for pt in grid_np])
        
        # Compute metrics
        mse_u = np.mean((pred_u - ref_u) ** 2)
        mse_v = np.mean((pred_v - ref_v) ** 2)
        
        eps = 1e-10
        l2_u = np.sqrt(np.mean((pred_u - ref_u) ** 2)) / (np.sqrt(np.mean(ref_u ** 2)) + eps)
        l2_v = np.sqrt(np.mean((pred_v - ref_v) ** 2)) / (np.sqrt(np.mean(ref_v ** 2)) + eps)
        
        lmax_u = np.max(np.abs(pred_u - ref_u)) / (np.max(np.abs(ref_u)) + eps)
        lmax_v = np.max(np.abs(pred_v - ref_v)) / (np.max(np.abs(ref_v)) + eps)
        
        print(f"✅ Inference Complete: {method_name}")
        print(f"   MSE_u: {mse_u:.6E}, MSE_v: {mse_v:.6E}")
        print(f"   L2_u: {l2_u:.6E}, L2_v: {l2_v:.6E}")
        print(f"   Lmax_u: {lmax_u:.6E}, Lmax_v: {lmax_v:.6E}")
        
        return {
            'predictions_u': pred_u,
            'predictions_v': pred_v,
            'reference_u': ref_u,
            'reference_v': ref_v,
            'mse_u': mse_u, 'mse_v': mse_v,
            'l2_u': l2_u, 'l2_v': l2_v,
            'lmax_u': lmax_u, 'lmax_v': lmax_v
        }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = Config()
    inference_engine = GrayScott1DQPINNInference(config, device)
    
    # Load reference solution
    ref_path = os.path.join(config.INPUT_DIR, "grayscott_reference_solution.npy")
    inference_engine.load_reference(ref_path)
    
    # Run inference for each model type
    results = {}
    for embedding_type, model_name in [("NONE", "PINN"), ("FNN_BASIS", "FNN-TE-QPINN"), ("QNN", "QNN-TE-QPINN")]:
        if embedding_type == "NONE":
            model_path = os.path.join(config.INPUT_DIR, "grayscott_1d_pinn.pt")
        elif embedding_type == "FNN_BASIS":
            model_path = os.path.join(config.INPUT_DIR, "grayscott_1d_fnn_basis.npy")
        else:
            model_path = os.path.join(config.INPUT_DIR, "grayscott_1d_qnn.npy")
        
        if os.path.exists(model_path):
            results[embedding_type] = inference_engine.inference(model_path, embedding_type)
        else:
            print(f"⚠ Model not found: {model_path}")
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
