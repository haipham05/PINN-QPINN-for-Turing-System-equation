"""
RD 2D QPINN - Inference Script
Reaction-Diffusion 2D Quantum Physics-Informed Neural Network

This script performs inference on trained models:
1. PINN (Pure Physics-Informed Neural Network)
2. FNN-TE-QPINN (FNN Basis Temporal Embedding QPINN)
3. QNN-TE-QPINN (Quantum Neural Network Temporal Embedding QPINN)

Includes performance analysis adapted for 2D spatial domain

Domain: t ∈ [0, 10], x ∈ [-1, 1], y ∈ [-1, 1]

PLOTS GENERATED:
================
Plot 3: In ra quantum circuits của hai model FNN-TE-QPINN và QNN-TE-QPINN
        - FNN-TE-QPINN main circuit
        - QNN-TE-QPINN main circuit
        - QNN-TE-QPINN embedding circuit (separate)

Plot 7: 3 biểu đồ cho 3 model (PINN, FNN-TE-QPINN, QNN-TE-QPINN)
        - Left: Inference solution (heatmap)
        - Middle: L2 relative error (MSE) vs RK45 reference
        - Right: L∞ relative error (Lmax) vs RK45 reference

Author: QPINN Research
Date: 2024-2025
"""

import os
import sys
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
import matplotlib as mpl

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def count_parameters(model_components):
    """Count total trainable parameters in model components
    
    Args:
        model_components: dict with keys like 'theta', 'basis_net', 'qnn_embedding', 'pinn'
    
    Returns:
        total_params: int, total number of parameters
        param_dict: dict, breakdown of parameters by component
    """
    param_dict = {}
    total_params = 0
    
    for name, component in model_components.items():
        if isinstance(component, torch.Tensor):
            # For theta tensor
            n_params = component.numel()
            param_dict[name] = n_params
            total_params += n_params
        elif isinstance(component, nn.Module):
            # For neural network modules
            n_params = sum(p.numel() for p in component.parameters() if p.requires_grad)
            param_dict[name] = n_params
            total_params += n_params
    
    return total_params, param_dict


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
    x_dummy = np.random.rand(3)  # (t, x, y)
    basis_dummy = np.random.rand(config.N_WIRES)
    
    print(f"\n📊 Generating quantum circuit diagram for {method_name}...")
    
    try:
        fig, ax = qml.draw_mpl(circuit_func)(x_dummy, basis_dummy)
        
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
        
        print(f"   ✓ Main circuit diagram saved: {save_path}")
        
    except Exception as e:
        print(f"   ⚠ Warning: Could not generate main circuit diagram: {e}")


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
        x_dummy = np.random.rand(3)  # (t, x, y)
        
        # Get the embedding circuit from the QNNEmbedding module
        fig, ax = qml.draw_mpl(qnn_embedding.qnode_embed)(x_dummy, qnn_embedding.weights_embed)
        
        # Add title
        ax.set_title(f'QNN Embedding Circuit Architecture\n'
                    f'({config.N_LAYERS_EMBED} layers, {config.N_WIRES_EMBED} qubits)', 
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
    """Configuration class for RD 2D QPINN"""
    
    # Random seed
    SEED = 42
    
    # Quantum Circuit Parameters
    N_LAYERS = 5
    N_WIRES = 6

    # PINN Parameters
    PINN_HIDDEN_LAYERS = 4
    PINN_NEURONS = 50
    
    # FNN Basis Parameters
    HIDDEN_LAYERS_FNN = 2
    NEURONS_FNN = 10
    
    # QNN Embedding Parameters
    N_WIRES_EMBED = 4
    N_LAYERS_EMBED = 2
    
    # Domain Parameters (2D spatial)
    T_COLLOC_POINTS = 50
    X_COLLOC_POINTS = 25
    Y_COLLOC_POINTS = 25
    
    # Physics Parameters (Reaction-Diffusion)
    DA = 1e-5       # Activator diffusion
    DS = 2e-3       # Substrate diffusion
    k1 = 1.0        # Autocatalytic rate
    k2 = 1.0        # Activator decay
    k3 = 1e-3       # Feed into S
    
    # Time domain
    T_MIN = 0.0
    T_MAX = 1.0
    
    # Directory configuration
    BASE_DIR = "result"
    INPUT_DIR = "result"
    OUTPUT_DIR = "result"


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class FNNBasisNet(nn.Module):
    """FNN Basis Network for embedding generation - 3D input (t, x, y)"""
    
    def __init__(self, n_hidden_layers, branch_width, output_size, input_dim=3):
        super().__init__()
        self.n_hidden_layers = n_hidden_layers
        self.branch_width = branch_width
        self.output_size = output_size
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, branch_width))
        for _ in range(n_hidden_layers - 1):
            self.layers.append(nn.Linear(branch_width, branch_width))
        self.layers.append(nn.Linear(branch_width, output_size))
    
    def forward(self, x):
        for i in range(self.n_hidden_layers):
            x = torch.tanh(self.layers[i](x))
        x = self.layers[self.n_hidden_layers](x)
        return x


class QNNEmbedding(nn.Module):
    """QNN Embedding for quantum basis generation - 3D input (t, x, y)"""
    
    def __init__(self, n_wires_embed, n_layers_embed, n_wires_main, input_dim=3):
        super().__init__()
        self.n_wires_embed = n_wires_embed
        self.n_layers_embed = n_layers_embed
        self.n_wires_main = n_wires_main
        self.input_dim = input_dim
        
        self.weights_embed = nn.Parameter(
            torch.randn(n_layers_embed, n_wires_embed, 3, requires_grad=True)
        )
        self.qnode_embed = qml.QNode(
            self.circuit_embed,
            qml.device("default.qubit", wires=n_wires_embed),
            interface="torch",
            diff_method="best",
            max_diff=2,
        )
    
    def circuit_embed(self, x, weights):
        """Embedding circuit for 3D input (t, x, y)"""
        for l in range(self.n_layers_embed):
            for i in range(self.n_wires_embed):
                if i % 3 == 0:
                    qml.RX(x[0], wires=i)  # t
                elif i % 3 == 1:
                    qml.RY(x[1], wires=i)  # x
                else:
                    qml.RZ(x[2], wires=i)  # y
            for i in range(self.n_wires_embed):
                qml.RX(weights[l, i, 0], wires=i)
                qml.RY(weights[l, i, 1], wires=i)
                qml.RZ(weights[l, i, 2], wires=i)
            if self.n_wires_embed > 1:
                for i in range(self.n_wires_embed - 1):
                    qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i % self.n_wires_embed)) for i in range(self.n_wires_main)]
    
    def forward(self, x):
        basis_t = self.qnode_embed(x.T, self.weights_embed)
        if isinstance(basis_t, list):
            basis_t = torch.stack(basis_t) * torch.pi
        else:
            basis_t = basis_t * torch.pi
        return basis_t.T


# ============================================================
# QPINN INFERENCE CLASS
# ============================================================

class RD2DQPINNInference:
    """Inference class for RD 2D QPINN"""
    
    def __init__(self, config, device=None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set seeds
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        
        # Initialize domain
        self._setup_domain()
        
        # Load reference solution
        self._load_reference()
        
        # Current embedding type
        self.embedding_type = None
        self.theta = None
        self.basis_net = None
        self.qnn_embedding = None
        self.pinn = None
        
    def _setup_domain(self):
        """Setup 2D collocation points and domain"""
        t = torch.linspace(self.config.T_MIN, self.config.T_MAX, self.config.T_COLLOC_POINTS)
        x = torch.linspace(-1.0, 1.0, self.config.X_COLLOC_POINTS)
        y = torch.linspace(-1.0, 1.0, self.config.Y_COLLOC_POINTS)
        
        self.input_domain = torch.tensor(list(product(t, x, y)), dtype=torch.float32)
        
        # Move to device
        self.input_domain = self.input_domain.clone().detach().to(self.device)
        self.domain_bounds = torch.tensor(
            [[self.config.T_MIN, -1.0, -1.0], [self.config.T_MAX, 1.0, 1.0]], 
            device=self.device
        )
        
        # Unique values for plotting
        self.T_unique = np.unique(t.numpy())
        self.X_unique = np.unique(x.numpy())
        self.Y_unique = np.unique(y.numpy())
        
    def _load_reference(self):
        """Load 2D reference solution"""
        ref_file = "rd_equations_2d_reference.npy"
        
        if os.path.exists(ref_file):
            loaded = np.load(ref_file, allow_pickle=True)[()]
            A_interp = loaded['A']
            S_interp = loaded['S']
            
            def reference_solution_A(data):
                output = np.zeros(data.shape[0])
                for i in range(data.shape[0]):
                    output[i] = A_interp([data[i, 0], data[i, 1], data[i, 2]]).squeeze()
                return output
            
            def reference_solution_S(data):
                output = np.zeros(data.shape[0])
                for i in range(data.shape[0]):
                    output[i] = S_interp([data[i, 0], data[i, 1], data[i, 2]]).squeeze()
                return output
            
            self.reference_values_A = torch.tensor(
                reference_solution_A(self.input_domain.detach().cpu().numpy()),
                device=self.device, dtype=torch.float32
            )
            self.reference_values_S = torch.tensor(
                reference_solution_S(self.input_domain.detach().cpu().numpy()),
                device=self.device, dtype=torch.float32
            )
            print("✓ 2D Reference solution loaded successfully")
        else:
            print("⚠ Warning: 2D Reference solution not found")
            self.reference_values_A = None
            self.reference_values_S = None
    
    def _create_circuit(self):
        """Create quantum circuit for 3D input (t, x, y)"""
        @qml.qnode(qml.device("default.qubit", wires=self.config.N_WIRES), interface="torch")
        def circuit(x, basis=None):
            if self.embedding_type == "NONE":
                for i in range(self.config.N_WIRES):
                    if i % 3 == 0:
                        qml.RY(x[0], wires=i)
                    elif i % 3 == 1:
                        qml.RY(x[1], wires=i)
                    else:
                        qml.RY(x[2], wires=i)
            elif self.embedding_type == "FNN_BASIS":
                for i in range(self.config.N_WIRES):
                    if i % 3 == 0:
                        qml.RY(basis[i] * x[0], wires=i)
                    elif i % 3 == 1:
                        qml.RY(basis[i] * x[1], wires=i)
                    else:
                        qml.RY(basis[i] * x[2], wires=i)
            elif self.embedding_type == "QNN":
                for i in range(self.config.N_WIRES):
                    if i % 3 == 0:
                        qml.RY(basis[i] + x[0], wires=i)
                    elif i % 3 == 1:
                        qml.RY(basis[i] + x[1], wires=i)
                    else:
                        qml.RY(basis[i] + x[2], wires=i)
            
            for i in range(self.config.N_LAYERS):
                for j in range(self.config.N_WIRES):
                    qml.RX(self.theta[i, j, 0], wires=j)
                    qml.RY(self.theta[i, j, 1], wires=j)
                    qml.RZ(self.theta[i, j, 2], wires=j)
                for j in range(self.config.N_WIRES - 1):
                    qml.CNOT(wires=[j, j + 1])
            
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
        
        return circuit
    
    def _postprocess_output(self, raw_expectations):
        """Convert raw expectations to physical ranges for 2D"""
        angle = raw_expectations * (torch.pi / 2.0)
        sin_transformed = torch.sin(angle)
        zero_to_one = (sin_transformed + 1.0) / 2.0
        
        A_physical = 0.0 + 0.5 * zero_to_one[:, 0]
        S_physical = 0.85 + 0.20 * zero_to_one[:, 1]
        
        return torch.stack([A_physical, S_physical], dim=1)
    
    def model(self, x):
        """Forward pass through model"""
        x_rescaled = 1.9 * (x - self.domain_bounds[0]) / (self.domain_bounds[1] - self.domain_bounds[0]) - 0.95
        
        if self.embedding_type == "FNN_BASIS":
            basis = self.basis_net(x_rescaled)
            raw = self.circuit(x_rescaled.T, basis=basis.T)
            raw_stacked = torch.stack(raw).T
            return self._postprocess_output(raw_stacked)
        elif self.embedding_type == "QNN":
            basis = self.qnn_embedding(x_rescaled)
            raw = self.circuit(x_rescaled.T, basis=basis.T)
            raw_stacked = torch.stack(raw).T
            return self._postprocess_output(raw_stacked)
        else:  # NONE
            return self.pinn(x_rescaled)
    
    def inference(self, model_path, embedding_type):
        """Load model and perform inference"""
        self.embedding_type = embedding_type
        method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
        
        print(f"\n{'='*60}")
        print(f"Running Inference: {method_name}")
        print(f"{'='*60}")
        
        # Load model state (weights_only=False for numpy arrays compatibility)
        model_state = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Initialize based on embedding type
        if embedding_type == "FNN_BASIS":
            self.theta = torch.tensor(model_state['theta'], device=self.device)
            self.basis_net = FNNBasisNet(
                self.config.HIDDEN_LAYERS_FNN,
                self.config.NEURONS_FNN,
                self.config.N_WIRES,
                input_dim=3
            ).to(self.device)
            self.basis_net.load_state_dict(model_state['basis_net'])
            self.basis_net.eval()
            self.circuit = self._create_circuit()
            
            # Count and print parameters
            model_components = {'theta': self.theta, 'basis_net': self.basis_net}
            total_params, param_dict = count_parameters(model_components)
            print(f"\n📊 Model Parameters:")
            print(f"   Theta (quantum weights): {param_dict['theta']:,}")
            print(f"   FNN Basis Network: {param_dict['basis_net']:,}")
            print(f"   Total Trainable Parameters: {total_params:,}")
            
            # Plot quantum circuit
            plot_quantum_circuit(self.circuit, embedding_type, self.config, self.config.OUTPUT_DIR)
            
        elif embedding_type == "QNN":
            self.theta = torch.tensor(model_state['theta'], device=self.device)
            self.qnn_embedding = QNNEmbedding(
                self.config.N_WIRES_EMBED,
                self.config.N_LAYERS_EMBED,
                self.config.N_WIRES,
                input_dim=3
            ).to(self.device)
            self.qnn_embedding.load_state_dict(model_state['qnn_embedding'])
            self.qnn_embedding.eval()
            self.circuit = self._create_circuit()
            
            # Count and print parameters
            model_components = {'theta': self.theta, 'qnn_embedding': self.qnn_embedding}
            total_params, param_dict = count_parameters(model_components)
            print(f"\n📊 Model Parameters:")
            print(f"   Theta (quantum weights): {param_dict['theta']:,}")
            print(f"   QNN Embedding Network: {param_dict['qnn_embedding']:,}")
            print(f"   Total Trainable Parameters: {total_params:,}")
            
            # Plot quantum circuits
            plot_quantum_circuit(self.circuit, embedding_type, self.config, self.config.OUTPUT_DIR)
            plot_qnn_embedding_circuit(self.qnn_embedding, self.config, self.config.OUTPUT_DIR)
            
        else:  # NONE (PINN)
            self.pinn = FNNBasisNet(
                self.config.PINN_HIDDEN_LAYERS,
                self.config.PINN_NEURONS,
                2,
                input_dim=3
            ).to(self.device)
            self.pinn.load_state_dict(model_state['pinn'])
            self.pinn.eval()
            
            # Count and print parameters
            model_components = {'pinn': self.pinn}
            total_params, param_dict = count_parameters(model_components)
            print(f"\n📊 Model Parameters:")
            print(f"   PINN Network: {param_dict['pinn']:,}")
            print(f"   Total Trainable Parameters: {total_params:,}")
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(self.input_domain)
        
        A_pred = predictions[:, 0].cpu().numpy()
        S_pred = predictions[:, 1].cpu().numpy()
        
        # Compute errors if reference available
        if self.reference_values_A is not None:
            A_ref = self.reference_values_A.cpu().numpy()
            S_ref = self.reference_values_S.cpu().numpy()
            
            A_error = np.abs(A_pred - A_ref)
            S_error = np.abs(S_pred - S_ref)
            
            mse_A = np.mean((A_pred - A_ref)**2)
            mse_S = np.mean((S_pred - S_ref)**2)
            l2_A = np.sqrt(np.sum((A_pred - A_ref)**2) / np.sum(A_ref**2 + 1e-10))
            l2_S = np.sqrt(np.sum((S_pred - S_ref)**2) / np.sum(S_ref**2 + 1e-10))
            lmax_A = np.max(A_error)
            lmax_S = np.max(S_error)
        else:
            A_ref = S_ref = A_error = S_error = None
            mse_A = mse_S = l2_A = l2_S = lmax_A = lmax_S = 0.0
        
        results = {
            'predictions_A': A_pred,
            'predictions_S': S_pred,
            'reference_A': A_ref,
            'reference_S': S_ref,
            'error_A': A_error,
            'error_S': S_error,
            'mse_A': mse_A,
            'mse_S': mse_S,
            'l2_A': l2_A,
            'l2_S': l2_S,
            'lmax_A': lmax_A,
            'lmax_S': lmax_S
        }
        
        print(f"✅ Inference Complete: {method_name}")
        print(f"   MSE_A: {mse_A:.6E}, MSE_S: {mse_S:.6E}")
        print(f"   L2_A: {l2_A:.6E}, L2_S: {l2_S:.6E}")
        print(f"   Lmax_A: {lmax_A:.6E}, Lmax_S: {lmax_S:.6E}")
        
        return results


# ============================================================
# VISUALIZATION - Plot 2 Only (Adapted for 2D)
# ============================================================

class InferenceVisualizer:
    """Visualization class for 2D inference results (Plot 2 only)"""
    
    def __init__(self, inference_engine, embedding_results):
        self.inference_engine = inference_engine
        self.embedding_results = embedding_results
        self.T_unique = inference_engine.T_unique
        self.X_unique = inference_engine.X_unique
        self.Y_unique = inference_engine.Y_unique
    
    def plot_performance_analysis(self, save_dir="result", t_plot=0.5):
        """Plot 7: Performance Analysis for each model
        3 biểu đồ cho 3 model. Mỗi biểu đồ gồm 3 cột:
        - Cột bên trái: kết quả load model đã training và inference với solution là heatmap
        - Cột ở giữa: L2 relative error (MSE) so với RK45 reference
        - Cột ngoài cùng bên phải: L infinity relative error (Lmax) so với RK45
        """
        print(f"\n" + "="*70)
        print(f"Generating Plot 7: Performance Analysis for Each Model at t={t_plot}")
        print(f"="*70)
        
        os.makedirs(save_dir, exist_ok=True)
        
        X_grid, Y_grid = np.meshgrid(self.X_unique, self.Y_unique, indexing='ij')
        
        # Find the time index closest to t_plot
        t_idx = np.argmin(np.abs(self.T_unique - t_plot))
        actual_t = self.T_unique[t_idx]
        print(f"   Using t={actual_t:.2f} (closest to t={t_plot})")
        
        # Generate plot for each model
        for embedding_type in ["NONE", "FNN_BASIS", "QNN"]:
            method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
            folder_name = {"NONE": "pinn", "FNN_BASIS": "fnn_basis", "QNN": "qnn"}[embedding_type]
            
            if embedding_type not in self.embedding_results:
                print(f"⚠ {method_name} results not found!")
                continue
            
            results = self.embedding_results[embedding_type]
            
            # Reshape predictions: (T, X, Y)
            A_pred = results['predictions_A'].reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
            S_pred = results['predictions_S'].reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
            
            # Create figure with 2 rows (A, S) x 3 columns
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # === Column 1: Model Predictions ===
            # A prediction
            im1 = axes[0, 0].contourf(X_grid, Y_grid, A_pred[t_idx, :, :], 50, cmap='inferno')
            axes[0, 0].set_xlabel('x', fontsize=12)
            axes[0, 0].set_ylabel('y', fontsize=12)
            axes[0, 0].set_title(f'A - {method_name} Solution', fontsize=13, fontweight='bold')
            axes[0, 0].set_aspect('equal')
            fig.colorbar(im1, ax=axes[0, 0], label='A')
            
            # S prediction
            im2 = axes[1, 0].contourf(X_grid, Y_grid, S_pred[t_idx, :, :], 50, cmap='viridis')
            axes[1, 0].set_xlabel('x', fontsize=12)
            axes[1, 0].set_ylabel('y', fontsize=12)
            axes[1, 0].set_title(f'S - {method_name} Solution', fontsize=13, fontweight='bold')
            axes[1, 0].set_aspect('equal')
            fig.colorbar(im2, ax=axes[1, 0], label='S')
            
            # === Column 2 & 3: Errors vs RK45 Reference ===
            if results['reference_A'] is not None:
                A_ref = results['reference_A'].reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
                S_ref = results['reference_S'].reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
                
                # Compute L2 Relative Error (MSE) at each spatial point
                eps = 1e-10
                A_l2_error = (A_pred[t_idx, :, :] - A_ref[t_idx, :, :]) ** 2 / (A_ref[t_idx, :, :] ** 2 + eps)
                S_l2_error = (S_pred[t_idx, :, :] - S_ref[t_idx, :, :]) ** 2 / (S_ref[t_idx, :, :] ** 2 + eps)
                
                # Compute L∞ Relative Error (pointwise absolute error / reference)
                A_linf_error = np.abs(A_pred[t_idx, :, :] - A_ref[t_idx, :, :]) / (np.abs(A_ref[t_idx, :, :]) + eps)
                S_linf_error = np.abs(S_pred[t_idx, :, :] - S_ref[t_idx, :, :]) / (np.abs(S_ref[t_idx, :, :]) + eps)
                
                # === Column 2: L2 Relative Error ===
                im3 = axes[0, 1].contourf(X_grid, Y_grid, A_l2_error, 50, cmap='hot')
                axes[0, 1].set_xlabel('x', fontsize=12)
                axes[0, 1].set_ylabel('y', fontsize=12)
                axes[0, 1].set_title(f'A - L2 Relative Error (MSE)', fontsize=13, fontweight='bold')
                axes[0, 1].set_aspect('equal')
                fig.colorbar(im3, ax=axes[0, 1], label='L2 Error')
                
                im4 = axes[1, 1].contourf(X_grid, Y_grid, S_l2_error, 50, cmap='hot')
                axes[1, 1].set_xlabel('x', fontsize=12)
                axes[1, 1].set_ylabel('y', fontsize=12)
                axes[1, 1].set_title(f'S - L2 Relative Error (MSE)', fontsize=13, fontweight='bold')
                axes[1, 1].set_aspect('equal')
                fig.colorbar(im4, ax=axes[1, 1], label='L2 Error')
                
                # === Column 3: L∞ Relative Error ===
                im5 = axes[0, 2].contourf(X_grid, Y_grid, A_linf_error, 50, cmap='plasma')
                axes[0, 2].set_xlabel('x', fontsize=12)
                axes[0, 2].set_ylabel('y', fontsize=12)
                axes[0, 2].set_title(f'A - L∞ Relative Error (Lmax)', fontsize=13, fontweight='bold')
                axes[0, 2].set_aspect('equal')
                fig.colorbar(im5, ax=axes[0, 2], label='L∞ Error')
                
                im6 = axes[1, 2].contourf(X_grid, Y_grid, S_linf_error, 50, cmap='plasma')
                axes[1, 2].set_xlabel('x', fontsize=12)
                axes[1, 2].set_ylabel('y', fontsize=12)
                axes[1, 2].set_title(f'S - L∞ Relative Error (Lmax)', fontsize=13, fontweight='bold')
                axes[1, 2].set_aspect('equal')
                fig.colorbar(im6, ax=axes[1, 2], label='L∞ Error')
                
                # Add global error statistics as text annotation
                global_mse_A = results['mse_A']
                global_mse_S = results['mse_S']
                global_lmax_A = results['lmax_A']
                global_lmax_S = results['lmax_S']
                
                stats_text = (f"Global Errors:\n"
                             f"A: MSE={global_mse_A:.2E}, Lmax={global_lmax_A:.2E}\n"
                             f"S: MSE={global_mse_S:.2E}, Lmax={global_lmax_S:.2E}")
                fig.text(0.5, 0.01, stats_text, ha='center', fontsize=11, 
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.suptitle(f'Plot 7: {method_name} Performance vs RK45 Reference (t={actual_t:.2f})', 
                         fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0.05, 1, 0.96])
            
            # Format time for filename (e.g., t=0.5 -> t05, t=1.0 -> t10)
            t_str = f"t{t_plot:.1f}".replace(".", "")
            plt.savefig(f'{save_dir}/plot7_performance_analysis_{folder_name}_{t_str}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Plot 7 for {method_name} saved: {save_dir}/plot7_performance_analysis_{folder_name}_{t_str}.png")
    
    def plot_time_evolution(self, save_dir="result", t_max=1.0):
        """Additional Plot: Time evolution comparison for all methods
        
        Args:
            save_dir: Directory to save plots
            t_max: Maximum time for evolution (default 1.0)
        """
        print("\n=== Plot 4: Time Evolution Comparison (2D) ===")
        
        os.makedirs(save_dir, exist_ok=True)
        
        X_grid, Y_grid = np.meshgrid(self.X_unique, self.Y_unique, indexing='ij')
        
        # Select time slices within [0, t_max]
        n_times = 5
        valid_mask = self.T_unique <= t_max
        valid_times = self.T_unique[valid_mask]
        time_indices = np.linspace(0, len(valid_times)-1, n_times, dtype=int)
        
        for component in ['A', 'S']:
            fig, axes = plt.subplots(3, n_times, figsize=(5*n_times, 15))
            cmap = 'inferno' if component == 'A' else 'viridis'
            
            for row, embedding_type in enumerate(["NONE", "FNN_BASIS", "QNN"]):
                method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
                results = self.embedding_results[embedding_type]
                
                pred = results[f'predictions_{component}'].reshape(
                    len(self.T_unique), len(self.X_unique), len(self.Y_unique)
                )
                
                for col, t_idx in enumerate(time_indices):
                    im = axes[row, col].contourf(X_grid, Y_grid, pred[t_idx, :, :], 50, cmap=cmap)
                    axes[row, col].set_aspect('equal')
                    
                    if row == 0:
                        axes[row, col].set_title(f't={valid_times[t_idx]:.2f}', fontsize=12)
                    if col == 0:
                        axes[row, col].set_ylabel(f'{method_name}\ny', fontsize=11)
                    if row == 2:
                        axes[row, col].set_xlabel('x', fontsize=11)
                    
                    fig.colorbar(im, ax=axes[row, col])
            
            plt.suptitle(f'{component} Time Evolution Comparison (RD 2D, t∈[0,{t_max}])', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/extraplot_time_evolution_{component.lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Time evolution plot for {component} saved")
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*105)
        print("INFERENCE SUMMARY STATISTICS (RD 2D)")
        print("="*105)
        print(f"\n{'Method':<20} {'MSE_A':<15} {'MSE_S':<15} {'L2_A':<15} {'L2_S':<15} {'Lmax_A':<15} {'Lmax_S':<15}")
        print("-"*105)
        
        for method in ["NONE", "FNN_BASIS", "QNN"]:
            method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[method]
            results = self.embedding_results[method]
            print(f"{method_name:<20} {results['mse_A']:<15.6E} {results['mse_S']:<15.6E} "
                  f"{results['l2_A']:<15.6E} {results['l2_S']:<15.6E} "
                  f"{results['lmax_A']:<15.6E} {results['lmax_S']:<15.6E}")
        
        print("="*105)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main function for 2D inference"""
    # Configuration
    config = Config()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("RD 2D QPINN INFERENCE")
    print("="*80)
    
    # Initialize inference engine
    inference_engine = RD2DQPINNInference(config, device)
    
    # Run inference for all methods
    embedding_results = {}
    for embedding_type in ["NONE", "FNN_BASIS", "QNN"]:
        # Use descriptive folder names
        folder_name = {"NONE": "pinn", "FNN_BASIS": "fnn_basis", "QNN": "qnn"}[embedding_type]
        model_path = os.path.join(config.INPUT_DIR, folder_name, 'model.pt')
        
        # Also check for legacy folder names
        if not os.path.exists(model_path):
            model_path = os.path.join(config.INPUT_DIR, embedding_type.lower(), 'model.pt')
        
        if os.path.exists(model_path):
            results = inference_engine.inference(model_path, embedding_type)
            embedding_results[embedding_type] = results
        else:
            print(f"⚠ Model not found: {model_path}")
    
    # Check if we have results
    if not embedding_results:
        print("❌ No models found for inference. Please train first using rd_2d_training.py")
        return
    
    # Visualization (Plot 2 and Plot 4)
    visualizer = InferenceVisualizer(inference_engine, embedding_results)
    visualizer.plot_performance_analysis(config.OUTPUT_DIR)
    visualizer.plot_time_evolution(config.OUTPUT_DIR)
    visualizer.print_summary()
    
    # Save inference results
    with open(os.path.join(config.OUTPUT_DIR, 'inference_results.pkl'), 'wb') as f:
        pickle.dump(embedding_results, f)
    
    print(f"\n✓ Inference results saved: {config.OUTPUT_DIR}/inference_results.pkl")
    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)

# Example usage:
# python rd_2d_inference.py
if __name__ == "__main__":
    main()
