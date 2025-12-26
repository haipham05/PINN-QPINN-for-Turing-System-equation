"""
Lengyel-Epstein 2D QPINN - Inference Script
Load trained models and perform inference with visualization

This script loads the trained models and generates:
1. Plot 2: Performance analysis at t=0.5 (solution, L2 error, L∞ error)
2. Plot 4: Time evolution comparison (t ∈ [0, 1])

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
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def count_parameters(model_components):
    """Count total trainable parameters in model components"""
    param_dict = {}
    total_params = 0
    
    for name, component in model_components.items():
        if isinstance(component, torch.Tensor):
            n_params = component.numel()
            param_dict[name] = n_params
            total_params += n_params
        elif isinstance(component, nn.Module):
            n_params = sum(p.numel() for p in component.parameters() if p.requires_grad)
            param_dict[name] = n_params
            total_params += n_params
    
    return total_params, param_dict


def plot_quantum_circuit(circuit_func, embedding_type, config, save_dir="result"):
    """Plot and save quantum circuit visualization"""
    os.makedirs(save_dir, exist_ok=True)
    
    method_name = {"FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
    
    x_dummy = np.random.rand(3)  # (t, x, y)
    theta_dummy = np.random.rand(config.N_LAYERS, config.N_WIRES, 3)
    basis_dummy = np.random.rand(config.N_WIRES)
    
    print(f"\n📊 Generating quantum circuit diagram for {method_name}...")
    
    try:
        fig, ax = qml.draw_mpl(circuit_func)(x_dummy, theta_dummy, basis_dummy)
        ax.set_title(f'{method_name} Quantum Circuit Architecture\n'
                    f'({config.N_LAYERS} layers, {config.N_WIRES} qubits)', 
                    fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        filename = f"plot3_quantum_circuit_{embedding_type.lower()}_inference.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Main circuit diagram saved: {save_path}")
    except Exception as e:
        print(f"   ⚠ Warning: Could not generate main circuit diagram: {e}")


def plot_qnn_embedding_circuit(qnn_embedding, config, save_dir="result"):
    """Plot and save QNN embedding circuit visualization"""
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n📊 Generating QNN embedding circuit diagram...")
    try:
        x_dummy = np.random.rand(3)
        fig, ax = qml.draw_mpl(qnn_embedding.qnode_embed)(x_dummy, qnn_embedding.weights_embed)
        ax.set_title(f'QNN Embedding Circuit Architecture\n'
                    f'({config.N_LAYERS_EMBED} layers, {config.N_WIRES} qubits)', 
                    fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        filename = "plot3_quantum_circuit_qnn_embedding_inference.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ QNN embedding circuit diagram saved: {save_path}")
    except Exception as e:
        print(f"   ⚠ Warning: Could not generate QNN embedding circuit diagram: {e}")


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Configuration class for Lengyel-Epstein 2D QPINN"""
    
    # Quantum Circuit Parameters
    N_LAYERS = 4
    N_WIRES = 4
    
    # FNN Basis Parameters
    HIDDEN_LAYERS_FNN = 2
    NEURONS_FNN = 16
    
    # QNN Embedding Parameters
    N_LAYERS_EMBED = 2
    
    # Physics Parameters (Lengyel-Epstein)
    D_U = 1.0
    D_V = 10.0
    A = 10.0
    B = 1.5
    
    # Boundary conditions (Dirichlet)
    U_BOUNDARY = 2.0
    V_BOUNDARY = 5.0
    
    # Domain
    T_MIN = 0.0
    T_MAX = 1.0
    X_MIN = 0.0
    X_MAX = 1.0
    Y_MIN = 0.0
    Y_MAX = 1.0
    
    # Inference Domain Parameters
    T_EVAL_POINTS = 11
    X_EVAL_POINTS = 11
    Y_EVAL_POINTS = 11
    
    # Directory configuration
    INPUT_DIR = "result"
    OUTPUT_DIR = "result"


# ============================================================
# NEURAL NETWORK MODELS
# ============================================================

class FNNBasisNet(nn.Module):
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
    def __init__(self, n_wires, n_layers_embed, output_dim, input_dim=3):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers_embed = n_layers_embed
        self.output_dim = output_dim
        self.weights_embed = nn.Parameter(torch.randn(n_layers_embed, n_wires, 3, requires_grad=True))
        self.dev = qml.device("default.qubit", wires=n_wires)
        self.qnode_embed = qml.QNode(self._circuit_embed, self.dev, interface="torch", diff_method="best")
    
    def _circuit_embed(self, x, weights):
        for layer in range(self.n_layers_embed):
            for i in range(self.n_wires):
                qubit_idx = i % 3
                if qubit_idx == 0: qml.RX(x[0], wires=i)
                elif qubit_idx == 1: qml.RY(x[1], wires=i)
                else: qml.RZ(x[2], wires=i)
            for i in range(self.n_wires):
                qml.RX(weights[layer, i, 0], wires=i)
                qml.RY(weights[layer, i, 1], wires=i)
                qml.RZ(weights[layer, i, 2], wires=i)
            if self.n_wires > 1:
                for i in range(self.n_wires - 1): qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.output_dim)]
    
    def forward(self, x):
        basis_t = self.qnode_embed(x.T, self.weights_embed)
        basis_t = torch.stack(basis_t) if isinstance(basis_t, list) else basis_t
        return (basis_t * torch.pi).T


# ============================================================
# INFERENCE ENGINE
# ============================================================

class LengyelEpstein2DQPINNInference:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.domain_min = torch.tensor([config.T_MIN, config.X_MIN, config.Y_MIN], device=device)
        self.domain_max = torch.tensor([config.T_MAX, config.X_MAX, config.Y_MAX], device=device)
        self.interp_u, self.interp_v = None, None
        self.T_unique, self.X_unique, self.Y_unique = None, None, None
    
    def load_reference(self, ref_path):
        if os.path.exists(ref_path):
            loaded = np.load(ref_path, allow_pickle=True)[()]
            self.interp_u, self.interp_v = loaded['u'], loaded['v']
            print("✓ 2D Reference solution loaded successfully")
        else:
            raise FileNotFoundError(f"Reference solution not found: {ref_path}")
        
        self.T_unique = np.linspace(self.config.T_MIN, self.config.T_MAX, self.config.T_EVAL_POINTS)
        self.X_unique = np.linspace(self.config.X_MIN, self.config.X_MAX, self.config.X_EVAL_POINTS)
        self.Y_unique = np.linspace(self.config.Y_MIN, self.config.Y_MAX, self.config.Y_EVAL_POINTS)
    
    def _create_circuit(self):
        dev = qml.device("default.qubit", wires=self.config.N_WIRES)
        @qml.qnode(dev, interface="torch")
        def circuit(x, theta, basis):
            for i in range(self.config.N_WIRES):
                qubit_idx = i % 3
                if qubit_idx == 0: qml.RY(basis[i] * x[0], wires=i)
                elif qubit_idx == 1: qml.RY(basis[i] * x[1], wires=i)
                else: qml.RY(basis[i] * x[2], wires=i)
            for layer in range(self.config.N_LAYERS):
                for qubit in range(self.config.N_WIRES):
                    qml.RX(theta[layer, qubit, 0], wires=qubit)
                    qml.RY(theta[layer, qubit, 1], wires=qubit)
                    qml.RZ(theta[layer, qubit, 2], wires=qubit)
                for qubit in range(self.config.N_WIRES - 1): qml.CNOT(wires=[qubit, qubit + 1])
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
        return circuit
    
    def _postprocess_output(self, raw_output):
        return raw_output * 5.0 + 5.0
    
    def model(self, x):
        x_rescaled = 2.0 * (x - self.domain_min) / (self.domain_max - self.domain_min) - 1.0
        if self.embedding_type == "FNN_BASIS":
            basis = self.basis_net(x_rescaled)
            raw = torch.stack(self.circuit(x_rescaled.T, self.theta, basis.T)).T
            return self._postprocess_output(raw)
        elif self.embedding_type == "QNN":
            basis = self.qnn_embedding(x_rescaled)
            raw = torch.stack(self.circuit(x_rescaled.T, self.theta, basis.T)).T
            return self._postprocess_output(raw)
        else:
            return self.pinn(x_rescaled)
    
    def inference(self, model_path, embedding_type):
        self.embedding_type = embedding_type
        method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
        print(f"\nRunning Inference: {method_name}")
        
        if embedding_type == "FNN_BASIS":
            model_state = np.load(model_path, allow_pickle=True)[()]
            self.theta = torch.tensor(model_state['theta'], device=self.device)
            self.basis_net = FNNBasisNet(self.config.HIDDEN_LAYERS_FNN, self.config.NEURONS_FNN, self.config.N_WIRES).to(self.device)
            self.basis_net.load_state_dict(model_state['basis_net'])
            self.basis_net.eval(); self.circuit = self._create_circuit()
        elif embedding_type == "QNN":
            model_state = np.load(model_path, allow_pickle=True)[()]
            self.theta = torch.tensor(model_state['theta'], device=self.device)
            self.qnn_embedding = QNNEmbedding(self.config.N_WIRES, self.config.N_LAYERS_EMBED, self.config.N_WIRES).to(self.device)
            self.qnn_embedding.load_state_dict(model_state['qnn_embedding'])
            self.qnn_embedding.eval(); self.circuit = self._create_circuit()
        else:
            self.pinn = FNNBasisNet(self.config.HIDDEN_LAYERS_FNN, self.config.NEURONS_FNN, 2).to(self.device)
            self.pinn.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            self.pinn.eval()
        
        grid_points = torch.tensor(list(product(self.T_unique, self.X_unique, self.Y_unique)), dtype=torch.float32, device=self.device)
        with torch.no_grad(): predictions = self.model(grid_points)
        pred_u, pred_v = predictions[:, 0].cpu().numpy(), predictions[:, 1].cpu().numpy()
        
        grid_np = grid_points.cpu().numpy()
        ref_u = np.array([self.interp_u([pt[1], pt[2], pt[0]]).squeeze() for pt in grid_np])
        ref_v = np.array([self.interp_v([pt[1], pt[2], pt[0]]).squeeze() for pt in grid_np])
        
        mse_u, mse_v = np.mean((pred_u - ref_u)**2), np.mean((pred_v - ref_v)**2)
        eps = 1e-10
        l2_u, l2_v = np.sqrt(mse_u) / (np.sqrt(np.mean(ref_u**2)) + eps), np.sqrt(mse_v) / (np.sqrt(np.mean(ref_v**2)) + eps)
        lmax_u, lmax_v = np.max(np.abs(pred_u - ref_u)) / (np.max(np.abs(ref_u)) + eps), np.max(np.abs(pred_v - ref_v)) / (np.max(np.abs(ref_v)) + eps)
        
        return {'predictions_u': pred_u, 'predictions_v': pred_v, 'reference_u': ref_u, 'reference_v': ref_v,
                'mse_u': mse_u, 'mse_v': mse_v, 'l2_u': l2_u, 'l2_v': l2_v, 'lmax_u': lmax_u, 'lmax_v': lmax_v}


# ============================================================
# VISUALIZATION
# ============================================================

class InferenceVisualizer:
    def __init__(self, inference_engine, embedding_results):
        self.inference_engine = inference_engine
        self.embedding_results = embedding_results
        self.T_unique, self.X_unique, self.Y_unique = inference_engine.T_unique, inference_engine.X_unique, inference_engine.Y_unique
    
    def plot_performance_analysis(self, save_dir="result", t_plot=0.5):
        print(f"\n=== Plot 2: Performance Analysis at t={t_plot} ===")
        X_grid, Y_grid = np.meshgrid(self.X_unique, self.Y_unique, indexing='ij')
        t_idx = np.argmin(np.abs(self.T_unique - t_plot))
        actual_t = self.T_unique[t_idx]
        
        for et in ["NONE", "FNN_BASIS", "QNN"]:
            if et not in self.embedding_results: continue
            res = self.embedding_results[et]
            method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[et]
            
            u_pred = res['predictions_u'].reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
            v_pred = res['predictions_v'].reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
            u_ref = res['reference_u'].reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
            v_ref = res['reference_v'].reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            im1 = axes[0,0].contourf(X_grid, Y_grid, u_pred[t_idx], 50, cmap='inferno'); fig.colorbar(im1, ax=axes[0,0]); axes[0,0].set_title(f'u - {method_name}')
            im2 = axes[1,0].contourf(X_grid, Y_grid, v_pred[t_idx], 50, cmap='viridis'); fig.colorbar(im2, ax=axes[1,0]); axes[1,0].set_title(f'v - {method_name}')
            
            eps = 1e-10
            u_l2 = (u_pred[t_idx] - u_ref[t_idx])**2 / (u_ref[t_idx]**2 + eps)
            v_l2 = (v_pred[t_idx] - v_ref[t_idx])**2 / (v_ref[t_idx]**2 + eps)
            im3 = axes[0,1].contourf(X_grid, Y_grid, u_l2, 50, cmap='hot'); fig.colorbar(im3, ax=axes[0,1]); axes[0,1].set_title('u L2 Error')
            im4 = axes[1,1].contourf(X_grid, Y_grid, v_l2, 50, cmap='hot'); fig.colorbar(im4, ax=axes[1,1]); axes[1,1].set_title('v L2 Error')
            
            u_linf = np.abs(u_pred[t_idx] - u_ref[t_idx]) / (np.abs(u_ref[t_idx]) + eps)
            v_linf = np.abs(v_pred[t_idx] - v_ref[t_idx]) / (np.abs(v_ref[t_idx]) + eps)
            im5 = axes[0,2].contourf(X_grid, Y_grid, u_linf, 50, cmap='plasma'); fig.colorbar(im5, ax=axes[0,2]); axes[0,2].set_title('u L∞ Error')
            im6 = axes[1,2].contourf(X_grid, Y_grid, v_linf, 50, cmap='plasma'); fig.colorbar(im6, ax=axes[1,2]); axes[1,2].set_title('v L∞ Error')
            
            plt.suptitle(f'Plot 7: {method_name} Performance (t={actual_t:.2f})', fontsize=16, fontweight='bold')
            plt.tight_layout(); plt.savefig(f'{save_dir}/plot7_{et.lower()}_performance.png'); plt.close()

    def plot_time_evolution(self, save_dir="result", t_max=1.0):
        print(f"\n=== Plot 4: Time Evolution Comparison ===")
        X_grid, Y_grid = np.meshgrid(self.X_unique, self.Y_unique, indexing='ij')
        n_times = 5
        time_indices = np.linspace(0, len(self.T_unique)-1, n_times, dtype=int)
        
        for comp in ['u', 'v']:
            fig, axes = plt.subplots(3, n_times, figsize=(5*n_times, 15))
            for row, et in enumerate(["NONE", "FNN_BASIS", "QNN"]):
                if et not in self.embedding_results: continue
                pred = self.embedding_results[et][f'predictions_{comp}'].reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
                for col, t_idx in enumerate(time_indices):
                    im = axes[row, col].contourf(X_grid, Y_grid, pred[t_idx], 50, cmap='viridis' if comp=='u' else 'plasma')
                    if row == 0: axes[row, col].set_title(f't={self.T_unique[t_idx]:.2f}')
                    fig.colorbar(im, ax=axes[row, col])
            plt.suptitle(f'{comp} Time Evolution Comparison', fontsize=14, fontweight='bold')
            plt.tight_layout(); plt.savefig(f'{save_dir}/extraplot_time_evolution_{comp}.png'); plt.close()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    inference_engine = LengyelEpstein2DQPINNInference(config, device)
    ref_path = os.path.join(config.INPUT_DIR, "lengyel_epstein_3d_reference_solution.npy")
    inference_engine.load_reference(ref_path)
    
    embedding_results = {}
    for et, folder, filename in [("NONE", "pinn", "model.pth"), ("FNN_BASIS", "fnn_basis", "model.npy"), ("QNN", "qnn", "model.npy")]:
        model_path = os.path.join(config.INPUT_DIR, folder, filename)
        if os.path.exists(model_path):
            embedding_results[et] = inference_engine.inference(model_path, et)
            if et in ["FNN_BASIS", "QNN"]:
                plot_quantum_circuit(inference_engine.circuit, et, config, config.OUTPUT_DIR)
                if et == "QNN": plot_qnn_embedding_circuit(inference_engine.qnn_embedding, config, config.OUTPUT_DIR)
    
    visualizer = InferenceVisualizer(inference_engine, embedding_results)
    visualizer.plot_performance_analysis(config.OUTPUT_DIR, t_plot=0.5)
    visualizer.plot_time_evolution(config.OUTPUT_DIR, t_max=1.0)
    
    with open(os.path.join(config.OUTPUT_DIR, 'inference_results.pkl'), 'wb') as f:
        pickle.dump(embedding_results, f)
    print(f"\nInference complete. Results in {config.OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
