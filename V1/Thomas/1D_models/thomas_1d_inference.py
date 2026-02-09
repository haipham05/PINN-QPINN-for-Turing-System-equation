"""
Thomas 1D QPINN - Inference Script
Load trained models and perform inference with visualization

This script loads the trained models and generates:
1. Plot 3: Quantum circuit visualization (FNN and QNN)
2. Plot 7: Performance analysis for each model (predictions, L2 error, L∞ error)
3. Plot 8: Time evolution comparison (t ∈ [0, 1])

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
# QUANTUM CIRCUIT PLOTTING FUNCTIONS
# ============================================================

def plot_quantum_circuit(circuit_func, embedding_type, config, save_dir="result"):
    """Plot and save quantum circuit visualization
    
    Args:
        circuit_func: QNode circuit function
        embedding_type: str, "FNN_BASIS" or "QNN"
        config: Config object
        save_dir: str, directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    method_name = {"FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
    
    # Create dummy inputs for visualization
    x_dummy = np.random.rand(2)  # (t, x) for 1D
    theta_dummy = np.random.rand(config.N_LAYERS, config.N_WIRES, 3)  # Variational parameters
    basis_dummy = np.random.rand(config.N_WIRES)
    
    print(f"\n📊 Generating quantum circuit diagram for {method_name}...")
    
    try:
        fig, ax = qml.draw_mpl(circuit_func)(x_dummy, theta_dummy, basis_dummy)
        
        # Add title
        ax.set_title(f'{method_name} Quantum Circuit Architecture\n'
                    f'({config.N_LAYERS} layers, {config.N_WIRES} qubits)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"plot3_quantum_circuit_{embedding_type.lower()}_inference.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Circuit diagram saved: {save_path}")
        
    except Exception as e:
        print(f"   ⚠ Warning: Could not generate circuit diagram: {e}")


def plot_qnn_embedding_circuit(qnn_embedding, config, save_dir="result"):
    """Plot and save QNN embedding circuit visualization
    
    Args:
        qnn_embedding: QNNEmbedding module
        config: Config object
        save_dir: str, directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n📊 Generating QNN embedding circuit diagram...")
    
    try:
        # Create dummy inputs
        x_dummy = np.random.rand(2)  # (t, x) for 1D
        
        # Get the embedding circuit from the QNNEmbedding module
        fig, ax = qml.draw_mpl(qnn_embedding.qnode_embed)(x_dummy, qnn_embedding.weights_embed)
        
        # Add title
        ax.set_title(f'QNN Embedding Circuit Architecture\n'
                    f'({config.N_LAYERS_EMBED} layers, {config.N_WIRES} qubits)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save figure
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
    """Configuration class for Thomas 1D QPINN"""
    
    # Quantum Circuit Parameters
    N_LAYERS = 5
    N_WIRES = 4
    
    # FNN Basis Parameters
    HIDDEN_LAYERS_FNN = 2
    NEURONS_FNN = 20
    
    # QNN Embedding Parameters
    N_LAYERS_EMBED = 2
    
    # Physics Parameters (Thomas)
    D_U = 0.001
    D_V = 0.02
    A = 20.0
    B = 57.0
    RHO = 13.0
    ALPHA = 1.2
    K = 0.1
    
    # Boundary conditions (Dirichlet)
    U_BOUNDARY = 5.0
    V_BOUNDARY = 10.0
    
    # Domain
    T_MIN = 0.0
    T_MAX = 1.0
    X_MIN = 0.0
    X_MAX = 1.0
    
    # Inference Domain Parameters (choose number of evaluation points)
    T_EVAL_POINTS = 200      # Number of time points for evaluation
    X_EVAL_POINTS = 50      # Number of x points for evaluation
    
    # Directory configuration
    INPUT_DIR = "result"
    OUTPUT_DIR = "result"


# ============================================================
# NEURAL NETWORK MODELS (Same as training)
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
    """Quantum Neural Network for embedding generation"""
    
    def __init__(self, n_wires, n_layers_embed, output_dim, input_dim=2):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers_embed = n_layers_embed
        self.output_dim = output_dim
        
        self.weights_embed = nn.Parameter(
            torch.randn(n_layers_embed, n_wires, 3, requires_grad=True)
        )
        
        self.dev = qml.device("default.qubit", wires=n_wires)
        self.qnode_embed = qml.QNode(
            self._circuit_embed,
            self.dev,
            interface="torch",
            diff_method="best"
        )
    
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

class Thomas1DQPINNInference:
    """Inference engine for Thomas 1D QPINN"""
    
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
            print("✓ 1D Reference solution loaded successfully")
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
        """Scale quantum outputs to physical range"""
        # u in [0.0547, 5.0100], v in [8.9548, 29.6516] (based on reference)
        u_scaled = raw_output[:, 0] * 2.47765 + 2.53235
        v_scaled = raw_output[:, 1] * 10.3484 + 19.3032
        return torch.stack([u_scaled, v_scaled], dim=1)
    
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
# VISUALIZATION
# ============================================================

class InferenceVisualizer:
    """Visualization class for 1D inference results"""
    
    def __init__(self, inference_engine, embedding_results):
        self.inference_engine = inference_engine
        self.embedding_results = embedding_results
        self.T_unique = inference_engine.T_unique
        self.X_unique = inference_engine.X_unique
    
    def plot_performance_analysis(self, save_dir="result"):
        """Plot 7: Performance analysis for each model (1x3 subplots per model)
        
        For each model (PINN, FNN-TE-QPINN, QNN-TE-QPINN):
        - Column 1: Model predictions (heatmap u and v)
        - Column 2: L2 Relative Error (MSE) vs RK45
        - Column 3: L∞ Relative Error (Lmax) vs RK45
        """
        print("\n=== Plot 7: Performance Analysis for Each Model ===")
        
        os.makedirs(save_dir, exist_ok=True)
        
        n_t = len(self.T_unique)
        n_x = len(self.X_unique)
        
        for embedding_type in ["NONE", "FNN_BASIS", "QNN"]:
            method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
            folder_name = {"NONE": "pinn", "FNN_BASIS": "fnn_basis", "QNN": "qnn"}[embedding_type]
            
            if embedding_type not in self.embedding_results:
                print(f"⚠ {method_name} results not found!")
                continue
            
            results = self.embedding_results[embedding_type]
            
            # Create figure with 2 rows (u, v) × 3 columns (prediction, L2 error, Linf error)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Reshape predictions
            u_pred = results['predictions_u'].reshape(n_t, n_x)
            v_pred = results['predictions_v'].reshape(n_t, n_x)
            u_ref = results['reference_u'].reshape(n_t, n_x)
            v_ref = results['reference_v'].reshape(n_t, n_x)
            
            # Compute L2 Relative Error (MSE)
            eps = 1e-10
            u_l2_error = (u_pred - u_ref) ** 2 / (u_ref ** 2 + eps)
            v_l2_error = (v_pred - v_ref) ** 2 / (v_ref ** 2 + eps)
            
            # Compute L∞ Relative Error (pointwise)
            u_linf_error = np.abs(u_pred - u_ref) / (np.abs(u_ref) + eps)
            v_linf_error = np.abs(v_pred - v_ref) / (np.abs(v_ref) + eps)
            
            # === Column 1: Predictions (heatmaps) ===
            # u Prediction
            im1 = axes[0, 0].contourf(self.T_unique, self.X_unique, u_pred.T, 100, cmap='inferno')
            axes[0, 0].set_xlabel('Time t', fontsize=11)
            axes[0, 0].set_ylabel('Space x', fontsize=11)
            axes[0, 0].set_title(f'u(t,x) - {method_name}', fontsize=12, fontweight='bold')
            fig.colorbar(im1, ax=axes[0, 0], label='u')
            
            # v Prediction
            im2 = axes[1, 0].contourf(self.T_unique, self.X_unique, v_pred.T, 100, cmap='viridis')
            axes[1, 0].set_xlabel('Time t', fontsize=11)
            axes[1, 0].set_ylabel('Space x', fontsize=11)
            axes[1, 0].set_title(f'v(t,x) - {method_name}', fontsize=12, fontweight='bold')
            fig.colorbar(im2, ax=axes[1, 0], label='v')
            
            # === Column 2: L2 Relative Error (MSE) ===
            # u L2 Error
            im3 = axes[0, 1].contourf(self.T_unique, self.X_unique, u_l2_error.T, 100, cmap='hot')
            axes[0, 1].set_xlabel('Time t', fontsize=11)
            axes[0, 1].set_ylabel('Space x', fontsize=11)
            axes[0, 1].set_title(f'u - L2 Relative Error (MSE)', fontsize=12, fontweight='bold')
            fig.colorbar(im3, ax=axes[0, 1], label='L2 Error')
            
            # v L2 Error
            im4 = axes[1, 1].contourf(self.T_unique, self.X_unique, v_l2_error.T, 100, cmap='hot')
            axes[1, 1].set_xlabel('Time t', fontsize=11)
            axes[1, 1].set_ylabel('Space x', fontsize=11)
            axes[1, 1].set_title(f'v - L2 Relative Error (MSE)', fontsize=12, fontweight='bold')
            fig.colorbar(im4, ax=axes[1, 1], label='L2 Error')
            
            # === Column 3: L∞ Relative Error (Lmax) ===
            # u Linf Error
            im5 = axes[0, 2].contourf(self.T_unique, self.X_unique, u_linf_error.T, 100, cmap='plasma')
            axes[0, 2].set_xlabel('Time t', fontsize=11)
            axes[0, 2].set_ylabel('Space x', fontsize=11)
            axes[0, 2].set_title(f'u - L∞ Relative Error (Lmax)', fontsize=12, fontweight='bold')
            fig.colorbar(im5, ax=axes[0, 2], label='L∞ Error')
            
            # v Linf Error
            im6 = axes[1, 2].contourf(self.T_unique, self.X_unique, v_linf_error.T, 100, cmap='plasma')
            axes[1, 2].set_xlabel('Time t', fontsize=11)
            axes[1, 2].set_ylabel('Space x', fontsize=11)
            axes[1, 2].set_title(f'v - L∞ Relative Error (Lmax)', fontsize=12, fontweight='bold')
            fig.colorbar(im6, ax=axes[1, 2], label='L∞ Error')
            
            # Add global error statistics as text annotation
            global_mse_u = results['mse_u']
            global_mse_v = results['mse_v']
            global_lmax_u = results['lmax_u']
            global_lmax_v = results['lmax_v']
            
            stats_text = (f"Global Errors:\n"
                         f"u: MSE={global_mse_u:.2E}, Lmax={global_lmax_u:.2E}\n"
                         f"v: MSE={global_mse_v:.2E}, Lmax={global_lmax_v:.2E}")
            fig.text(0.5, 0.01, stats_text, ha='center', fontsize=11, 
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.suptitle(f'Plot 7: {method_name} Performance vs RK45 Reference (Thomas 1D)', 
                         fontsize=15, fontweight='bold')
            plt.tight_layout(rect=[0, 0.05, 1, 0.96])
            
            plt.savefig(f'{save_dir}/plot7_performance_{folder_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Plot 7 for {method_name} saved: {save_dir}/plot7_performance_{folder_name}.png")
    
    def plot_time_evolution(self, save_dir="result", t_max=1.0):
        """Plot 8: Time evolution heatmaps for all methods"""
        print(f"\n=== Plot 8: Time Evolution Comparison (t ∈ [0, {t_max}]) ===")
        
        os.makedirs(save_dir, exist_ok=True)
        
        n_t = len(self.T_unique)
        n_x = len(self.X_unique)
        
        # Filter times
        valid_mask = self.T_unique <= t_max
        t_valid = self.T_unique[valid_mask]
        n_t_valid = len(t_valid)
        
        T_grid, X_grid = np.meshgrid(t_valid, self.X_unique, indexing='ij')
        
        for component in ['u', 'v']:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            cmap = 'viridis' if component == 'u' else 'plasma'
            
            # Reference
            results = self.embedding_results["NONE"]
            ref = results[f'reference_{component}'].reshape(n_t, n_x)[:n_t_valid, :]
            
            im0 = axes[0].pcolormesh(T_grid, X_grid, ref, shading='auto', cmap=cmap)
            axes[0].set_xlabel('Time t', fontsize=11)
            axes[0].set_ylabel('Space x', fontsize=11)
            axes[0].set_title(f'Reference {component}(t,x)', fontsize=12, fontweight='bold')
            plt.colorbar(im0, ax=axes[0])
            
            # Models
            for idx, (embedding_type, ax) in enumerate(zip(["NONE", "FNN_BASIS", "QNN"], axes[1:])):
                method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
                
                if embedding_type in self.embedding_results:
                    pred = self.embedding_results[embedding_type][f'predictions_{component}'].reshape(n_t, n_x)[:n_t_valid, :]
                    
                    im = ax.pcolormesh(T_grid, X_grid, pred, shading='auto', cmap=cmap)
                    ax.set_xlabel('Time t', fontsize=11)
                    ax.set_ylabel('Space x', fontsize=11)
                    ax.set_title(f'{method_name} {component}(t,x)', fontsize=12, fontweight='bold')
                    plt.colorbar(im, ax=ax)
            
            plt.suptitle(f'Plot 8: {component}(t,x) Time Evolution Comparison (Thomas 1D, t∈[0,{t_max}])',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/plot8_time_evolution_{component}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Plot 8 - Time evolution plot for {component} saved")
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*90)
        print("INFERENCE SUMMARY STATISTICS (Thomas 1D)")
        print("="*90)
        print(f"\n{'Method':<18} {'MSE_u':<12} {'MSE_v':<12} {'L2_u':<12} {'L2_v':<12} {'Lmax_u':<12} {'Lmax_v':<12}")
        print("-"*90)
        
        for method in ["NONE", "FNN_BASIS", "QNN"]:
            if method in self.embedding_results:
                method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[method]
                results = self.embedding_results[method]
                print(f"{method_name:<18} {results['mse_u']:<12.6E} {results['mse_v']:<12.6E} "
                      f"{results['l2_u']:<12.6E} {results['l2_v']:<12.6E} "
                      f"{results['lmax_u']:<12.6E} {results['lmax_v']:<12.6E}")
        
        print("="*90)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    # Configuration
    config = Config()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("THOMAS 1D QPINN INFERENCE")
    print("="*80)
    
    # Initialize inference engine
    inference_engine = Thomas1DQPINNInference(config, device)
    
    # Load reference
    ref_path = os.path.join(config.INPUT_DIR, "thomas_reference_solution.npy")
    inference_engine.load_reference(ref_path)
    
    # Run inference for all models
    embedding_results = {}
    
    model_configs = [
        ("NONE", "pinn", "model.pth"),
        ("FNN_BASIS", "fnn_basis", "model.npy"),
        ("QNN", "qnn", "model.npy"),
    ]
    
    for embedding_type, folder, filename in model_configs:
        model_path = os.path.join(config.INPUT_DIR, folder, filename)
        
        # Check for backward compatibility
        if not os.path.exists(model_path):
            alt_folder = "none" if folder == "pinn" else folder
            model_path = os.path.join(config.INPUT_DIR, alt_folder, filename)
        
        if os.path.exists(model_path):
            results = inference_engine.inference(model_path, embedding_type)
            embedding_results[embedding_type] = results
            
            # Plot quantum circuits for FNN and QNN
            if embedding_type in ["FNN_BASIS", "QNN"]:
                # Plot main circuit
                if embedding_type == "FNN_BASIS":
                    dev = qml.device("default.qubit", wires=config.N_WIRES)
                    @qml.qnode(dev, interface="torch")
                    def circuit(x, theta, basis):
                        for i in range(config.N_WIRES):
                            if i % 2 == 0:
                                qml.RY(basis[i] * x[0], wires=i)
                            else:
                                qml.RY(basis[i] * x[1], wires=i)
                        for layer in range(config.N_LAYERS):
                            for qubit in range(config.N_WIRES):
                                qml.RX(theta[layer, qubit, 0], wires=qubit)
                                qml.RY(theta[layer, qubit, 1], wires=qubit)
                                qml.RZ(theta[layer, qubit, 2], wires=qubit)
                            for qubit in range(config.N_WIRES - 1):
                                qml.CNOT(wires=[qubit, qubit + 1])
                        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
                    try:
                        plot_quantum_circuit(circuit, "FNN_BASIS", config, config.OUTPUT_DIR)
                    except Exception as e:
                        print(f"⚠ Could not generate FNN circuit plot: {e}")
                
                elif embedding_type == "QNN":
                    dev = qml.device("default.qubit", wires=config.N_WIRES)
                    @qml.qnode(dev, interface="torch")
                    def circuit(x, theta, basis):
                        for i in range(config.N_WIRES):
                            if i % 2 == 0:
                                qml.RX(x[0], wires=i)
                            else:
                                qml.RY(x[1], wires=i)
                        for layer in range(config.N_LAYERS):
                            for qubit in range(config.N_WIRES):
                                qml.RX(theta[layer, qubit, 0], wires=qubit)
                                qml.RY(theta[layer, qubit, 1], wires=qubit)
                                qml.RZ(theta[layer, qubit, 2], wires=qubit)
                            for qubit in range(config.N_WIRES - 1):
                                qml.CNOT(wires=[qubit, qubit + 1])
                        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
                    try:
                        plot_quantum_circuit(circuit, "QNN", config, config.OUTPUT_DIR)
                    except Exception as e:
                        print(f"⚠ Could not generate QNN circuit plot: {e}")
                    
                    # Plot QNN embedding circuit
                    if hasattr(inference_engine, 'qnn_embedding'):
                        try:
                            plot_qnn_embedding_circuit(inference_engine.qnn_embedding, config, config.OUTPUT_DIR)
                        except Exception as e:
                            print(f"⚠ Could not generate QNN embedding circuit plot: {e}")
        else:
            print(f"⚠ Model not found: {model_path}")
    
    # Visualizations
    visualizer = InferenceVisualizer(inference_engine, embedding_results)
    
    # === Plot 7 & 8: Performance and Time Evolution ===
    visualizer.plot_performance_analysis(config.OUTPUT_DIR)
    visualizer.plot_time_evolution(config.OUTPUT_DIR, t_max=1.0)
    visualizer.print_summary()
    
    # Save results
    with open(os.path.join(config.OUTPUT_DIR, 'inference_results.pkl'), 'wb') as f:
        pickle.dump(embedding_results, f)
    
    print(f"\n✓ Inference results saved: {config.OUTPUT_DIR}/inference_results.pkl")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    print("Generated Plots:")
    print("  - plot3_quantum_circuit_*.png (FNN and QNN circuits)")
    print("  - plot7_performance_*.png (for each model)")
    print("  - plot8_time_evolution_*.png (for u and v)")
    print("="*80)


if __name__ == "__main__":
    main()
