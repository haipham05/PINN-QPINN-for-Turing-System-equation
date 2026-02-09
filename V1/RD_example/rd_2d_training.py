"""
RD 2D QPINN - Training Script
Reaction-Diffusion 2D Quantum Physics-Informed Neural Network

This script trains three models:
1. PINN (Pure Physics-Informed Neural Network)
2. FNN-TE-QPINN (FNN Basis Temporal Embedding QPINN)
3. QNN-TE-QPINN (Quantum Neural Network Temporal Embedding QPINN)

Based on the 1D implementation architecture, extended for 2D spatial domain.

Domain: t ∈ [0, 10], x ∈ [-1, 1], y ∈ [-1, 1]
PDE System:
    ∂A/∂t = D_A (∂²A/∂x² + ∂²A/∂y²) + k1 A² S - k2 A
    ∂S/∂t = D_S (∂²S/∂x² + ∂²S/∂y²) - k1 A² S + k3

PLOTS GENERATED:
================
Plot 1: Biểu đồ thể hiện collocation points 2D (IC, BC, Interior)
        - Initial Condition (IC) points: spatial (x, y) at t=0
        - Boundary Condition (BC) points: temporal (t) vs spatial at boundaries
        - Interior points: spatial (x, y) domain for all interior time steps

Plot 2: Biểu đồ biểu diễn hai reference solution của A và S giải bằng RK45
        - Reference solution heatmaps at 5 time snapshots

Plot 3: In ra quantum circuits của hai model FNN-TE-QPINN và QNN-TE-QPINN
        - FNN-TE-QPINN main circuit
        - QNN-TE-QPINN main circuit
        - QNN-TE-QPINN embedding circuit (separate)

Plot 4: Biểu đồ embedding results (future implementation)

Plot 5: 3 biểu đồ cho 3 model (PINN, FNN-TE-QPINN, QNN-TE-QPINN)
        - Left: Training loss evolution
        - Right: Absolute error vs RK45 reference

Plot 6: 3 biểu đồ line chart show total loss của 3 models
        - PINN total loss
        - FNN-TE-QPINN total loss
        - QNN-TE-QPINN total loss

Author: QPINN Research
Date: 2024-2025
"""

import os
import sys
import json
import time
import pickle
import argparse
from itertools import product

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib as mpl

torch.random.manual_seed(42)

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
        filename = f"plot3_quantum_circuit_{embedding_type.lower()}.png"
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
        filename = "plot3_quantum_circuit_qnn_embedding.png"
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
    T_COLLOC_POINTS = 100   # Reduced for 2D to manage memory
    X_COLLOC_POINTS = 50
    Y_COLLOC_POINTS = 50
    
    # Physics Parameters (Reaction-Diffusion)
    DA = 1e-5       # Activator diffusion
    DS = 2e-3       # Substrate diffusion
    k1 = 1.0        # Autocatalytic rate
    k2 = 1.0        # Activator decay
    k3 = 1e-3       # Feed into S
    
    # Time domain
    T_MIN = 0.0
    T_MAX = 1.0
    
    # Training Parameters
    TRAINING_ITERATIONS = 2
    BOUNDARY_SCALE = 5
    WEIGHT_S = 20
    
    # Output directory
    BASE_DIR = "result"
    OUTPUT_DIR = "result"


# ============================================================
# REFERENCE SOLUTION GENERATOR
# ============================================================

def generate_reference_solution(config, save_path="rd_equations_2d_reference.npy"):
    """Generate 2D reference solution using RK45 solver with ADI method"""
    
    print("=== Generating 2D Reference Solution ===")
    
    # Spatial domain
    Nx = 50
    Ny = 50
    x = np.linspace(-1.0, 1.0, Nx)
    y = np.linspace(-1.0, 1.0, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Time domain
    t_start, t_end = config.T_MIN, config.T_MAX
    Nt = 201
    t_eval = np.linspace(t_start, t_end, Nt)
    
    # Create 2D meshgrid
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Initial conditions: Gaussian bump at center
    def make_ic_2d(X, Y):
        sig = 0.2
        A0 = 0.35 * np.exp(-0.5 * ((X**2 + Y**2) / sig**2))
        S0 = np.ones_like(X)
        return np.clip(A0, 0.0, None), S0
    
    A0, S0 = make_ic_2d(X, Y)
    
    # Flatten for ODE solver
    N = Nx * Ny
    y0 = np.concatenate([A0.flatten(), S0.flatten()])
    
    # 2D Laplacian with periodic boundary conditions
    def laplacian_2d(u_flat, Nx, Ny, dx, dy):
        u = u_flat.reshape(Nx, Ny)
        
        # Periodic BC using roll
        lap = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dx**2
        lap += (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dy**2
        
        return lap.flatten()
    
    # RHS
    def rd_rhs_2d(t, y):
        A = y[:N]
        S = y[N:]
        
        lapA = laplacian_2d(A, Nx, Ny, dx, dy)
        lapS = laplacian_2d(S, Nx, Ny, dx, dy)
        
        A2S = (A * A) * S
        
        dA = config.DA * lapA + config.k1 * A2S - config.k2 * A
        dS = config.DS * lapS - config.k1 * A2S + config.k3
        
        return np.concatenate([dA, dS])
    
    # Solve
    print("Solving 2D RD equations with RK45...")
    sol = solve_ivp(
        rd_rhs_2d,
        [t_eval[0], t_eval[-1]],
        y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-5,
        atol=1e-8
    )
    
    t = sol.t
    A_sol = sol.y[:N, :].reshape(Nx, Ny, -1)  # Shape: (Nx, Ny, Nt)
    S_sol = sol.y[N:, :].reshape(Nx, Ny, -1)
    
    print(f"Status: {sol.message}")
    print(f"Solution shape: A={A_sol.shape}, S={S_sol.shape}")
    
    # Build 3D interpolators (t, x, y)
    interpA = RegularGridInterpolator((t, x, y), np.transpose(A_sol, (2, 0, 1)), 
                                       bounds_error=False, fill_value=None)
    interpS = RegularGridInterpolator((t, x, y), np.transpose(S_sol, (2, 0, 1)), 
                                       bounds_error=False, fill_value=None)
    
    # Save
    np.save(save_path, {'A': interpA, 'S': interpS}, allow_pickle=True)
    print(f"✓ 2D Reference solution saved to {save_path}")
    
    return interpA, interpS, t, x, y, A_sol, S_sol


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
            # Encode t, x, y cyclically across wires
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
        # x shape: [batch, 3] for (t, x, y)
        basis_t = self.qnode_embed(x.T, self.weights_embed)
        if isinstance(basis_t, list):
            basis_t = torch.stack(basis_t) * torch.pi
        else:
            basis_t = basis_t * torch.pi
        return basis_t.T


# ============================================================
# QPINN TRAINER CLASS
# ============================================================

class RD2DQPINNTrainer:
    """Main trainer class for RD 2D QPINN"""
    
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
        
        # Create full 3D grid
        self.input_domain = torch.tensor(list(product(t, x, y)), dtype=torch.float32)
        
        # Masks for 2D spatial domain
        init_val_mask = self.input_domain[:, 0] == self.config.T_MIN
        self.init_val_colloc = self.input_domain[init_val_mask]
        
        # Boundary: x = ±1 or y = ±1
        dir_boundary_mask = (
            (self.input_domain[:, 1] == -1.0) | 
            (self.input_domain[:, 1] == 1.0) |
            (self.input_domain[:, 2] == -1.0) | 
            (self.input_domain[:, 2] == 1.0)
        )
        self.dir_boundary_colloc = self.input_domain[dir_boundary_mask & ~init_val_mask]
        
        boundary_mask = init_val_mask | dir_boundary_mask
        self.interior_colloc = self.input_domain[~boundary_mask]
        
        # Move to device with gradients
        self.input_domain = self.input_domain.clone().detach().requires_grad_(True).to(self.device)
        self.init_val_colloc = self.init_val_colloc.clone().detach().requires_grad_(True).to(self.device)
        self.dir_boundary_colloc = self.dir_boundary_colloc.clone().detach().requires_grad_(True).to(self.device)
        self.interior_colloc = self.interior_colloc.clone().detach().requires_grad_(True).to(self.device)
        
        self.domain_bounds = torch.tensor(
            [[self.config.T_MIN, -1.0, -1.0], [self.config.T_MAX, 1.0, 1.0]], 
            device=self.device
        )
        
        # Unique values for plotting
        self.T_unique = np.unique(t.numpy())
        self.X_unique = np.unique(x.numpy())
        self.Y_unique = np.unique(y.numpy())
        
        print(f"✓ Domain setup complete:")
        print(f"  Total points: {self.input_domain.shape[0]}")
        print(f"  Interior points: {self.interior_colloc.shape[0]}")
        print(f"  IC points: {self.init_val_colloc.shape[0]}")
        print(f"  Boundary points: {self.dir_boundary_colloc.shape[0]}")
        
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
            print("⚠ Warning: 2D Reference solution not found. Generating...")
            # Only need interpolators here, ignore extra return values
            _, _, _, _, _, _, _ = generate_reference_solution(self.config, ref_file)
            self._load_reference()
    
    def _create_circuit(self):
        """Create quantum circuit for 3D input (t, x, y)"""
        @qml.qnode(qml.device("default.qubit", wires=self.config.N_WIRES), interface="torch")
        def circuit(x, basis=None):
            if self.embedding_type == "NONE":
                # Direct encoding of t, x, y
                for i in range(self.config.N_WIRES):
                    if i % 3 == 0:
                        qml.RY(x[0], wires=i)  # t
                    elif i % 3 == 1:
                        qml.RY(x[1], wires=i)  # x
                    else:
                        qml.RY(x[2], wires=i)  # y
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
        
        # Ranges based on 2D reference solution
        A_physical = 0.0 + 0.5 * zero_to_one[:, 0]   # A: [0.0, 0.5]
        S_physical = 0.85 + 0.20 * zero_to_one[:, 1]  # S: [0.85, 1.05]
        
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
        else:  # NONE (PINN)
            return self.pinn(x_rescaled)
    
    def _init_val_loss(self):
        """Initial condition loss for 2D: A(0,x,y) = Gaussian, S(0,x,y) = 1"""
        predictions = self.model(self.init_val_colloc)
        A_pred = predictions[:, 0]
        S_pred = predictions[:, 1]
        
        # 2D Gaussian initial condition for A
        x_ic = self.init_val_colloc[:, 1]
        y_ic = self.init_val_colloc[:, 2]
        sig = 0.2
        A_ic = 0.35 * torch.exp(-0.5 * ((x_ic**2 + y_ic**2) / sig**2))
        S_ic = torch.ones_like(S_pred)
        
        loss_A = torch.mean((A_pred - A_ic)**2)
        loss_S = torch.mean((S_pred - S_ic)**2)
        
        return loss_A + self.config.WEIGHT_S * loss_S
    
    def _boundary_loss(self):
        """Periodic boundary condition loss for 2D domain
        
        Enforces BOTH value continuity (Dirichlet) AND gradient continuity (Neumann)
        to match the true periodic BC used in the reference solution (via np.roll).
        """
        # Enable gradient computation for boundary points
        self.dir_boundary_colloc.requires_grad_(True)
        
        predictions = self.model(self.dir_boundary_colloc)
        
        x_boundary = self.dir_boundary_colloc[:, 1]
        y_boundary = self.dir_boundary_colloc[:, 2]
        
        # Identify boundary faces
        left_x_mask = (x_boundary == -1.0)
        right_x_mask = (x_boundary == 1.0)
        bottom_y_mask = (y_boundary == -1.0)
        top_y_mask = (y_boundary == 1.0)
        
        loss_A_bc = torch.tensor(0.0, device=self.device)
        loss_S_bc = torch.tensor(0.0, device=self.device)
        
        # ====== X BOUNDARIES (x = ±1) ======
        if torch.sum(left_x_mask) > 0 and torch.sum(right_x_mask) > 0:
            min_count = min(torch.sum(left_x_mask), torch.sum(right_x_mask))
            
            # Values at boundaries
            A_left = predictions[left_x_mask, 0][:min_count]
            A_right = predictions[right_x_mask, 0][:min_count]
            S_left = predictions[left_x_mask, 1][:min_count]
            S_right = predictions[right_x_mask, 1][:min_count]
            
            # Value continuity loss (Dirichlet BC)
            loss_A_dirichlet = torch.mean((A_left - A_right)**2)
            loss_S_dirichlet = torch.mean((S_left - S_right)**2)
            
            # Gradient continuity loss (Neumann BC)
            # Compute ∂A/∂x and ∂S/∂x at boundaries
            grad_outputs_A = torch.ones_like(A_left)
            grad_outputs_S = torch.ones_like(S_left)
            
            # Gradients at left boundary
            dA_left = torch.autograd.grad(
                A_left, self.dir_boundary_colloc, grad_outputs=grad_outputs_A, 
                create_graph=True, allow_unused=True
            )[0][left_x_mask, 1][:min_count]
            
            dS_left = torch.autograd.grad(
                S_left, self.dir_boundary_colloc, grad_outputs=grad_outputs_S,
                create_graph=True, allow_unused=True
            )[0][left_x_mask, 1][:min_count]
            
            # Gradients at right boundary  
            dA_right = torch.autograd.grad(
                A_right, self.dir_boundary_colloc, grad_outputs=grad_outputs_A,
                create_graph=True, allow_unused=True
            )[0][right_x_mask, 1][:min_count]
            
            dS_right = torch.autograd.grad(
                S_right, self.dir_boundary_colloc, grad_outputs=grad_outputs_S,
                create_graph=True, allow_unused=True
            )[0][right_x_mask, 1][:min_count]
            
            # Gradient continuity loss (Neumann BC)
            loss_A_neumann = torch.mean((dA_left - dA_right)**2)
            loss_S_neumann = torch.mean((dS_left - dS_right)**2)
            
            # Combine Dirichlet and Neumann losses
            loss_A_bc = loss_A_bc + loss_A_dirichlet + loss_A_neumann
            loss_S_bc = loss_S_bc + loss_S_dirichlet + loss_S_neumann
        
        # ====== Y BOUNDARIES (y = ±1) ======
        if torch.sum(bottom_y_mask) > 0 and torch.sum(top_y_mask) > 0:
            min_count = min(torch.sum(bottom_y_mask), torch.sum(top_y_mask))
            
            # Values at boundaries
            A_bottom = predictions[bottom_y_mask, 0][:min_count]
            A_top = predictions[top_y_mask, 0][:min_count]
            S_bottom = predictions[bottom_y_mask, 1][:min_count]
            S_top = predictions[top_y_mask, 1][:min_count]
            
            # Value continuity loss (Dirichlet BC)
            loss_A_dirichlet = torch.mean((A_bottom - A_top)**2)
            loss_S_dirichlet = torch.mean((S_bottom - S_top)**2)
            
            # Gradient continuity loss (Neumann BC)
            # Compute ∂A/∂y and ∂S/∂y at boundaries
            grad_outputs_A = torch.ones_like(A_bottom)
            grad_outputs_S = torch.ones_like(S_bottom)
            
            # Gradients at bottom boundary
            dA_bottom = torch.autograd.grad(
                A_bottom, self.dir_boundary_colloc, grad_outputs=grad_outputs_A,
                create_graph=True, allow_unused=True
            )[0][bottom_y_mask, 2][:min_count]
            
            dS_bottom = torch.autograd.grad(
                S_bottom, self.dir_boundary_colloc, grad_outputs=grad_outputs_S,
                create_graph=True, allow_unused=True
            )[0][bottom_y_mask, 2][:min_count]
            
            # Gradients at top boundary
            dA_top = torch.autograd.grad(
                A_top, self.dir_boundary_colloc, grad_outputs=grad_outputs_A,
                create_graph=True, allow_unused=True
            )[0][top_y_mask, 2][:min_count]
            
            dS_top = torch.autograd.grad(
                S_top, self.dir_boundary_colloc, grad_outputs=grad_outputs_S,
                create_graph=True, allow_unused=True
            )[0][top_y_mask, 2][:min_count]
            
            # Gradient continuity loss (Neumann BC)
            loss_A_neumann = torch.mean((dA_bottom - dA_top)**2)
            loss_S_neumann = torch.mean((dS_bottom - dS_top)**2)
            
            # Combine Dirichlet and Neumann losses
            loss_A_bc = loss_A_bc + loss_A_dirichlet + loss_A_neumann
            loss_S_bc = loss_S_bc + loss_S_dirichlet + loss_S_neumann
        
        return loss_A_bc + self.config.WEIGHT_S * loss_S_bc
    
    def _pde_residual(self):
        """PDE residual loss for 2D reaction-diffusion"""
        predictions = self.model(self.interior_colloc)
        A_pred = predictions[:, 0]
        S_pred = predictions[:, 1]
        
        grad_outputs = torch.ones_like(A_pred)
        
        # Gradients for A
        dA = torch.autograd.grad(A_pred, self.interior_colloc, grad_outputs=grad_outputs, create_graph=True)[0]
        dA_dt = dA[:, 0]
        dA_dx = dA[:, 1]
        dA_dy = dA[:, 2]
        
        # Second derivatives for A
        dA_dA_dx = torch.autograd.grad(dA_dx, self.interior_colloc, grad_outputs=grad_outputs, create_graph=True)[0]
        dA_dx_dx = dA_dA_dx[:, 1]
        dA_dA_dy = torch.autograd.grad(dA_dy, self.interior_colloc, grad_outputs=grad_outputs, create_graph=True)[0]
        dA_dy_dy = dA_dA_dy[:, 2]
        laplacian_A = dA_dx_dx + dA_dy_dy
        
        # Gradients for S
        dS = torch.autograd.grad(S_pred, self.interior_colloc, grad_outputs=grad_outputs, create_graph=True)[0]
        dS_dt = dS[:, 0]
        dS_dx = dS[:, 1]
        dS_dy = dS[:, 2]
        
        # Second derivatives for S
        dS_dS_dx = torch.autograd.grad(dS_dx, self.interior_colloc, grad_outputs=grad_outputs, create_graph=True)[0]
        dS_dx_dx = dS_dS_dx[:, 1]
        dS_dS_dy = torch.autograd.grad(dS_dy, self.interior_colloc, grad_outputs=grad_outputs, create_graph=True)[0]
        dS_dy_dy = dS_dS_dy[:, 2]
        laplacian_S = dS_dx_dx + dS_dy_dy
        
        # Reaction term
        A2S = (A_pred ** 2) * S_pred
        
        # PDE residuals
        res_A = dA_dt - self.config.DA * laplacian_A - self.config.k1 * A2S + self.config.k2 * A_pred
        res_S = dS_dt - self.config.DS * laplacian_S + self.config.k1 * A2S - self.config.k3
        
        return torch.mean(res_A**2) + self.config.WEIGHT_S * torch.mean(res_S**2)
    
    def loss_fnc(self):
        """Total loss function"""
        return self.config.BOUNDARY_SCALE * (self._init_val_loss() + self._boundary_loss()) + self._pde_residual()
    
    def compute_mse(self):
        """Compute MSE against reference"""
        if self.reference_values_A is None:
            return 0.0, 0.0
        
        predictions = self.model(self.input_domain)
        mse_A = torch.mean((predictions[:, 0] - self.reference_values_A) ** 2).item()
        mse_S = torch.mean((predictions[:, 1] - self.reference_values_S) ** 2).item()
        
        return mse_A, mse_S
    
    def compute_component_losses(self):
        """Compute component-wise losses"""
        predictions = self.model(self.input_domain)
        
        if self.reference_values_A is not None:
            loss_A = torch.mean((predictions[:, 0] - self.reference_values_A) ** 2).item()
            loss_S = torch.mean((predictions[:, 1] - self.reference_values_S) ** 2).item()
        else:
            loss_A = torch.mean(predictions[:, 0] ** 2).item()
            loss_S = torch.mean(predictions[:, 1] ** 2).item()
        
        return loss_A, loss_S
    
    def train(self, embedding_type):
        """Train the model with specified embedding type"""
        self.embedding_type = embedding_type
        method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
        
        print(f"\n{'='*60}")
        print(f"Training {method_name}")
        print(f"{'='*60}")
        
        # Initialize parameters
        self.theta = torch.rand(
            self.config.N_LAYERS, self.config.N_WIRES, 3,
            device=self.device, requires_grad=True
        )
        
        # Initialize embedding networks (input_dim=3 for 2D spatial)
        if embedding_type == "FNN_BASIS":
            self.basis_net = FNNBasisNet(
                self.config.HIDDEN_LAYERS_FNN,
                self.config.NEURONS_FNN,
                self.config.N_WIRES,
                input_dim=3  # t, x, y
            ).to(self.device)
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
            
            opt = torch.optim.LBFGS(
                [self.theta, *self.basis_net.parameters()],
                line_search_fn="strong_wolfe"
            )
        elif embedding_type == "QNN":
            self.qnn_embedding = QNNEmbedding(
                self.config.N_WIRES_EMBED,
                self.config.N_LAYERS_EMBED,
                self.config.N_WIRES,
                input_dim=3  # t, x, y
            ).to(self.device)
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
            
            opt = torch.optim.LBFGS(
                [self.theta, *self.qnn_embedding.parameters()],
                line_search_fn="strong_wolfe"
            )
        else:  # NONE (PINN)
            self.pinn = FNNBasisNet(
                self.config.PINN_HIDDEN_LAYERS,
                self.config.PINN_NEURONS,
                2,  # Output A and S
                input_dim=3  # t, x, y
            ).to(self.device)
            
            # Count and print parameters
            model_components = {'pinn': self.pinn}
            total_params, param_dict = count_parameters(model_components)
            print(f"\n📊 Model Parameters:")
            print(f"   PINN Network: {param_dict['pinn']:,}")
            print(f"   Total Trainable Parameters: {total_params:,}")
            
            opt = torch.optim.LBFGS(
                self.pinn.parameters(),
                line_search_fn="strong_wolfe"
            )
        
        # Training history
        history = {
            'loss_history': [],
            'loss_A_history': [],
            'loss_S_history': [],
            'mse_A_history': [],
            'mse_S_history': []
        }
        
        def closure():
            opt.zero_grad()
            l = self.loss_fnc()
            l.backward()
            return l
        
        previous_loss = float('inf')
        
        # Training loop
        for epoch in range(self.config.TRAINING_ITERATIONS):
            start_time = time.time()
            opt.step(closure)
            
            current_loss = self.loss_fnc().item()
            mse_A, mse_S = self.compute_mse()
            loss_A, loss_S = self.compute_component_losses()
            
            history['loss_history'].append(current_loss)
            history['loss_A_history'].append(loss_A)
            history['loss_S_history'].append(loss_S)
            history['mse_A_history'].append(mse_A)
            history['mse_S_history'].append(mse_S)
            
            print(f"Epoch {epoch:03d} | Total: {current_loss:.2E} | Loss_A: {loss_A:.2E} | "
                  f"Loss_S: {loss_S:.2E} | MSE_A: {mse_A:.2E} | Time: {time.time()-start_time:.2f}s", end='\r')
        
            if abs(previous_loss - current_loss) < 1e-10:
                print("Early stopping: Loss change < 1e-10")
                break
            
            previous_loss = current_loss

        print(f"\n✅ {method_name} Training Complete")
        print(f"   Final Total Loss: {current_loss:.2E}")
        
        # Get predictions
        predictions = self.model(self.input_domain)
        A_pred = predictions[:, 0].detach().cpu().numpy()
        S_pred = predictions[:, 1].detach().cpu().numpy()
        
        # Store results
        results = {
            **history,
            'final_loss': current_loss,
            'final_loss_A': loss_A,
            'final_loss_S': loss_S,
            'final_mse_A': mse_A,
            'final_mse_S': mse_S,
            'predictions_A': A_pred,
            'predictions_S': S_pred,
            'theta': self.theta.detach().cpu().numpy()
        }
        
        # Store model state
        if embedding_type == "FNN_BASIS":
            results['basis_net_state'] = self.basis_net.state_dict()
        elif embedding_type == "QNN":
            results['qnn_embedding_state'] = self.qnn_embedding.state_dict()
        elif embedding_type == "NONE":
            results['pinn_state'] = self.pinn.state_dict()
        
        return results


# ============================================================
# PLOT 2: REFERENCE SOLUTION VISUALIZATION
# ============================================================

def plot_collocation_points_2d(config, save_dir="result"):
    """
    Plot 1: Collocation Points Visualization (matching actual training points)
    Uses uniform grid and masks to match trainer._setup_domain().
    """
    print("\n=== Generating Plot 1: Collocation Points (Grid-Based) ===")
    
    # Match trainer: Uniform linspace for t, x, y
    t_points = np.linspace(config.T_MIN, config.T_MAX, config.T_COLLOC_POINTS)
    x_points = np.linspace(-1.0, 1.0, config.X_COLLOC_POINTS)
    y_points = np.linspace(-1.0, 1.0, config.Y_COLLOC_POINTS)
    
    # Full domain grid (matches input_domain)
    full_domain = np.array(list(product(t_points, x_points, y_points)))
    
    # 1. IC Points: t == T_MIN
    ic_mask = full_domain[:, 0] == config.T_MIN
    ic_points = full_domain[ic_mask]
    
    # 2. BC Points: x=±1 or y=±1, excluding IC (t > T_MIN)
    bc_mask = (
        ((full_domain[:, 1] == -1.0) | (full_domain[:, 1] == 1.0) |
         (full_domain[:, 2] == -1.0) | (full_domain[:, 2] == 1.0)) &
        (full_domain[:, 0] != config.T_MIN)
    )
    bc_points = full_domain[bc_mask]
    
    # 3. Interior Points: Neither IC nor BC
    interior_mask = ~(ic_mask | bc_mask)
    interior_points = full_domain[interior_mask]
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Subplot 1: Initial Condition Points (IC) - Red
    ax1 = fig.add_subplot(131)
    if len(ic_points) > 0:
        ax1.scatter(ic_points[:, 1], ic_points[:, 2], c="r", s=10, alpha=0.8, label="Initial Condition")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Initial Condition Points (t=0)")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # Subplot 2: Boundary Condition Points (BC) - Blue
    ax2 = fig.add_subplot(132)
    if len(bc_points) > 0:
        ax2.scatter(bc_points[:, 1], bc_points[:, 2], c="blue", s=10, alpha=0.6, label="Boundary")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Boundary Points (t>0)")
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    
    # Subplot 3: Interior Points - Black (no sampling needed for small grids)
    ax3 = fig.add_subplot(133)
    if len(interior_points) > 0:
        ax3.scatter(interior_points[:, 1], interior_points[:, 2], c="black", s=10, alpha=0.5, label="Interior")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_title("Interior Points")
    ax3.legend()
    ax3.grid(True, alpha=0.2)
    
    plt.suptitle('Plot 1: Collocation Points (IC, BC, Interior) - Uniform Grid', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'plot1_collocation_points.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"✓ Plot 1 saved: {save_path}")
    print(f"  IC points: {len(ic_points)}")
    print(f"  BC points: {len(bc_points)}")
    print(f"  Interior points: {len(interior_points)}")
    print(f"  Total: {len(ic_points) + len(bc_points) + len(interior_points)}")


def plot_reference_solution(t, x, y, A_sol, S_sol, save_dir="result"):
    """
    Plot 2: Reference Solution from RK45
    Biểu đồ biểu diễn hai reference solution của A và S giải bằng RK45
    Creates 2D heatmaps of A and S at 5 time snapshots
    
    Args:
        t: Time array from RK45 solver
        x, y: Spatial grids
        A_sol, S_sol: Solution arrays of shape (Nx, Ny, Nt)
        save_dir: Directory to save the plot
    """
    print("\n" + "="*60)
    print("Generating Plot 2: Reference Solution (RK45)")
    print("="*60)
    
    # Select 5 time snapshots
    n_times = 5
    time_indices = np.linspace(0, len(t)-1, n_times, dtype=int)
    
    # Create 2x5 grid: 2 rows (A, S) × 5 columns (time snapshots)
    fig, axes = plt.subplots(2, n_times, figsize=(20, 8))
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    for col, t_idx in enumerate(time_indices):
        current_time = t[t_idx]
        
        # Row 1: Activator A
        A_slice = A_sol[:, :, t_idx]
        im1 = axes[0, col].contourf(X, Y, A_slice, levels=20, cmap='inferno')
        axes[0, col].set_title(f't = {current_time:.3f}', fontsize=10)
        axes[0, col].set_xlabel('x')
        if col == 0:
            axes[0, col].set_ylabel('A (Activator)', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=axes[0, col])
        
        # Row 2: Substrate S
        S_slice = S_sol[:, :, t_idx]
        im2 = axes[1, col].contourf(X, Y, S_slice, levels=20, cmap='viridis')
        axes[1, col].set_title(f't = {current_time:.3f}', fontsize=10)
        axes[1, col].set_xlabel('x')
        if col == 0:
            axes[1, col].set_ylabel('S (Substrate)', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=axes[1, col])
    
    plt.suptitle('Plot 2: Reference Solution (RK45) - 2D Reaction-Diffusion', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'plot2_reference_solution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot 2 saved: {save_path}")
    print("="*60)


# ============================================================
# VISUALIZATION - Plot 1, Plot 2, and Plot 3
# Note: Plot 2 is generated in main() before training
# ============================================================

class TrainingVisualizer:
    """Visualization class for 2D training results (Plot 1 and Plot 3)"""
    
    def __init__(self, trainer, embedding_results):
        self.trainer = trainer
        self.embedding_results = embedding_results
        self.T_unique = trainer.T_unique
        self.X_unique = trainer.X_unique
        self.Y_unique = trainer.Y_unique
        self.reference_values_A = trainer.reference_values_A
        self.reference_values_S = trainer.reference_values_S
    
    def plot_embedding_results(self, save_dir="result"):
        """Plot 4: Embedding Results Visualization
        1 biểu đồ biểu diễn kết quả sau khi embedding của equation với các giá trị là tích của φ(x)·x
        Shows embedding outputs for FNN-TE-QPINN and QNN-TE-QPINN
        """
        print("\n=== Plot 4: Embedding Results ===")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Create grid for visualization
        t_plot = np.linspace(self.T_unique[0], self.T_unique[-1], 50)
        x_plot = np.linspace(self.X_unique[0], self.X_unique[-1], 50)
        y_plot = np.linspace(self.Y_unique[0], self.Y_unique[-1], 50)
        
        T_grid, X_grid = np.meshgrid(t_plot, x_plot, indexing='ij')
        Y_fixed = 0.0  # Fix y at 0 for 2D visualization
        
        # Create grid points (t, x, y_fixed)
        grid_points = np.column_stack([
            T_grid.flatten(),
            X_grid.flatten(),
            np.full(T_grid.size, Y_fixed)
        ])
        
        # Convert to torch and rescale
        grid_torch = torch.tensor(grid_points, dtype=torch.float32, device=self.trainer.device)
        
        # Rescale to [-0.95, 0.95] domain for models
        domain_bounds = torch.tensor([[0.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=self.trainer.device)
        grid_rescaled = 1.9 * (grid_torch - domain_bounds[0]) / (domain_bounds[1] - domain_bounds[0]) - 0.95
        
        # Process FNN-TE-QPINN embedding
        fnn_basis_outputs = None
        qnn_basis_outputs = None
        
        # Get basis networks from trainer if they were stored
        if hasattr(self.trainer, 'basis_net') and self.trainer.basis_net is not None:
            with torch.no_grad():
                fnn_basis_outputs = self.trainer.basis_net(grid_rescaled).detach().cpu().numpy()
        
        if hasattr(self.trainer, 'qnn_embedding') and self.trainer.qnn_embedding is not None:
            with torch.no_grad():
                qnn_basis_outputs = self.trainer.qnn_embedding(grid_rescaled).detach().cpu().numpy()
        
        # If embeddings not available from trainer, skip this plot
        if fnn_basis_outputs is None and qnn_basis_outputs is None:
            print("⚠ Embedding networks not available for visualization")
            return
        
        # Determine number of wires from embedding output
        n_wires = fnn_basis_outputs.shape[1] if fnn_basis_outputs is not None else qnn_basis_outputs.shape[1]
        
        # Plot embeddings
        fig, axes = plt.subplots(2, n_wires, figsize=(4 * n_wires, 10))
        
        if n_wires == 1:
            axes = axes.reshape(2, 1)
        
        t_vals = grid_rescaled[:, 0].detach().cpu().numpy()
        x_vals = grid_rescaled[:, 1].detach().cpu().numpy()
        
        # Row 1: FNN-TE-QPINN embeddings
        if fnn_basis_outputs is not None:
            for i in range(n_wires):
                if i % 2 == 0:
                    z = fnn_basis_outputs[:, i] * t_vals
                    label = r"$\phi_{{{}}}^{{FNN}} \cdot \tilde{{t}}$".format(i + 1)
                else:
                    z = fnn_basis_outputs[:, i] * x_vals
                    label = r"$\phi_{{{}}}^{{FNN}} \cdot \tilde{{x}}$".format(i + 1)
                
                Z_grid = z.reshape(len(t_plot), len(x_plot))
                im = axes[0, i].contourf(t_plot, x_plot, Z_grid.T, 50, cmap='viridis')
                axes[0, i].set_title(label, fontsize=11)
                axes[0, i].set_xlabel('t', fontsize=10)
                if i == 0:
                    axes[0, i].set_ylabel('FNN-TE-QPINN\nx', fontsize=11)
                plt.colorbar(im, ax=axes[0, i])
        
        # Row 2: QNN-TE-QPINN embeddings
        if qnn_basis_outputs is not None:
            for i in range(n_wires):
                if i % 2 == 0:
                    z = qnn_basis_outputs[:, i] * t_vals
                    label = r"$\phi_{{{}}}^{{QNN}} \cdot \tilde{{t}}$".format(i + 1)
                else:
                    z = qnn_basis_outputs[:, i] * x_vals
                    label = r"$\phi_{{{}}}^{{QNN}} \cdot \tilde{{x}}$".format(i + 1)
                
                Z_grid = z.reshape(len(t_plot), len(x_plot))
                im = axes[1, i].contourf(t_plot, x_plot, Z_grid.T, 50, cmap='viridis')
                axes[1, i].set_title(label, fontsize=11)
                axes[1, i].set_xlabel('t', fontsize=10)
                if i == 0:
                    axes[1, i].set_ylabel('QNN-TE-QPINN\nx', fontsize=11)
                plt.colorbar(im, ax=axes[1, i])
        
        plt.suptitle(r"Plot 4: Trainable Embedding Functions: $\phi(x) \cdot x$ (RD 2D at y=0)", 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/plot4_embedding_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot 4 saved: {save_dir}/plot4_embedding_results.png")
    
    def plot_training_analysis(self, save_dir="result"):
        """Plot 1: Training analysis for each model - 2D adapted (2x3 subplots)
        Shows spatial slices at t=0, t=T/2, t=T for A and S
        """
        print("\n=== Plot 1: Training Analysis for Each Model ===")
        
        os.makedirs(save_dir, exist_ok=True)
        
        for embedding_type in ["NONE", "FNN_BASIS", "QNN"]:
            method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
            results = self.embedding_results[embedding_type]
            
            fig, axes = plt.subplots(2, 3, figsize=(24, 16))
            
            # Column 1: Training Loss Evolution
            axes[0,0].semilogy(results['loss_history'], 'b-', linewidth=2, label='Total Loss')
            axes[0,0].semilogy(results['loss_A_history'], 'r--', linewidth=2, label='Loss A')
            axes[0,0].semilogy(results['loss_S_history'], 'g--', linewidth=2, label='Loss S')
            axes[0,0].set_xlabel('Epoch', fontsize=12)
            axes[0,0].set_ylabel('Loss (log scale)', fontsize=12)
            axes[0,0].set_title(f'{method_name} Training Loss Evolution', fontsize=13)
            axes[0,0].legend(loc='best')
            axes[0,0].grid(True, alpha=0.3, which='both')
            
            axes[1,0].semilogy(results['mse_A_history'], 'r-', linewidth=2, label='MSE A')
            axes[1,0].semilogy(results['mse_S_history'], 'g-', linewidth=2, label='MSE S')
            axes[1,0].set_xlabel('Epoch', fontsize=12)
            axes[1,0].set_ylabel('MSE (log scale)', fontsize=12)
            axes[1,0].set_title(f'{method_name} MSE Evolution', fontsize=13)
            axes[1,0].legend(loc='best')
            axes[1,0].grid(True, alpha=0.3, which='both')
            
            # Reshape predictions for 2D spatial visualization
            # Shape: (T, X, Y)
            A_pred = results['predictions_A'].reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
            S_pred = results['predictions_S'].reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
            
            # Column 2: A at t=T/2 (middle time)
            t_mid_idx = len(self.T_unique) // 2
            X_grid, Y_grid = np.meshgrid(self.X_unique, self.Y_unique, indexing='ij')
            
            im1 = axes[0,1].contourf(X_grid, Y_grid, A_pred[t_mid_idx, :, :], 50, cmap='inferno')
            axes[0,1].set_xlabel('x', fontsize=12)
            axes[0,1].set_ylabel('y', fontsize=12)
            axes[0,1].set_title(f'A(t={self.T_unique[t_mid_idx]:.1f}, x, y) - {method_name}', fontsize=13)
            axes[0,1].set_aspect('equal')
            fig.colorbar(im1, ax=axes[0,1], label='A')
            
            im2 = axes[1,1].contourf(X_grid, Y_grid, S_pred[t_mid_idx, :, :], 50, cmap='viridis')
            axes[1,1].set_xlabel('x', fontsize=12)
            axes[1,1].set_ylabel('y', fontsize=12)
            axes[1,1].set_title(f'S(t={self.T_unique[t_mid_idx]:.1f}, x, y) - {method_name}', fontsize=13)
            axes[1,1].set_aspect('equal')
            fig.colorbar(im2, ax=axes[1,1], label='S')
            
            # Column 3: Errors at t=T/2
            if self.reference_values_A is not None:
                A_ref = self.reference_values_A.cpu().numpy().reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
                S_ref = self.reference_values_S.cpu().numpy().reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
                
                A_error = np.abs(A_pred[t_mid_idx, :, :] - A_ref[t_mid_idx, :, :])
                S_error = np.abs(S_pred[t_mid_idx, :, :] - S_ref[t_mid_idx, :, :])
                
                im3 = axes[0,2].contourf(X_grid, Y_grid, A_error, 50, cmap='inferno')
                axes[0,2].set_xlabel('x', fontsize=12)
                axes[0,2].set_ylabel('y', fontsize=12)
                axes[0,2].set_title(f'|A_pred - A_ref| at t={self.T_unique[t_mid_idx]:.1f}', fontsize=13)
                axes[0,2].set_aspect('equal')
                fig.colorbar(im3, ax=axes[0,2], label='|Error|')
                
                im4 = axes[1,2].contourf(X_grid, Y_grid, S_error, 50, cmap='viridis')
                axes[1,2].set_xlabel('x', fontsize=12)
                axes[1,2].set_ylabel('y', fontsize=12)
                axes[1,2].set_title(f'|S_pred - S_ref| at t={self.T_unique[t_mid_idx]:.1f}', fontsize=13)
                axes[1,2].set_aspect('equal')
                fig.colorbar(im4, ax=axes[1,2], label='|Error|')
            
            plt.suptitle(f'Plot 5: {method_name} Training Analysis (RD 2D)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/plot5_training_{embedding_type.lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Plot 5 for {method_name} saved: {save_dir}/plot5_training_{embedding_type.lower()}.png")
    
    def plot_methods_comparison(self, save_dir="result"):
        """Plot 6: Methods comparison - Total loss during training
        3 biểu đồ line chart show kết quả total loss của 3 models PINN, FNN-TE-QPINN 
        và QNN-TE-QPINN (3 lines) trong quá trình training
        """
        print("\n=== Plot 6: Methods Comparison - Total Loss ===")
        
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Total Loss for all 3 models
        axes[0].semilogy(self.embedding_results["NONE"]['loss_history'], 'b-', linewidth=2, label='PINN')
        axes[0].semilogy(self.embedding_results["FNN_BASIS"]['loss_history'], 'r--', linewidth=2, label='FNN-TE-QPINN')
        axes[0].semilogy(self.embedding_results["QNN"]['loss_history'], 'g-', linewidth=2, label='QNN-TE-QPINN')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Total Loss (log scale)', fontsize=12)
        axes[0].set_title('Total Loss Comparison', fontsize=13)
        axes[0].legend(fontsize=10, loc='best')
        axes[0].grid(True, alpha=0.3, which='both')
        
        # Plot 2: Loss A for all 3 models
        axes[1].semilogy(self.embedding_results["NONE"]['loss_A_history'], 'b-', linewidth=2, label='PINN')
        axes[1].semilogy(self.embedding_results["FNN_BASIS"]['loss_A_history'], 'r--', linewidth=2, label='FNN-TE-QPINN')
        axes[1].semilogy(self.embedding_results["QNN"]['loss_A_history'], 'g-', linewidth=2, label='QNN-TE-QPINN')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss A (log scale)', fontsize=12)
        axes[1].set_title('Activator A Loss Comparison', fontsize=13)
        axes[1].legend(fontsize=10, loc='best')
        axes[1].grid(True, alpha=0.3, which='both')
        
        # Plot 3: Loss S for all 3 models
        axes[2].semilogy(self.embedding_results["NONE"]['loss_S_history'], 'b-', linewidth=2, label='PINN')
        axes[2].semilogy(self.embedding_results["FNN_BASIS"]['loss_S_history'], 'r--', linewidth=2, label='FNN-TE-QPINN')
        axes[2].semilogy(self.embedding_results["QNN"]['loss_S_history'], 'g-', linewidth=2, label='QNN-TE-QPINN')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Loss S (log scale)', fontsize=12)
        axes[2].set_title('Substrate S Loss Comparison', fontsize=13)
        axes[2].legend(fontsize=10, loc='best')
        axes[2].grid(True, alpha=0.3, which='both')
        
        plt.suptitle('Plot 6: Three Methods Comparison (PINN vs FNN-TE-QPINN vs QNN-TE-QPINN)',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/plot6_methods_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot 6 saved: {save_dir}/plot6_methods_comparison.png")
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*70)
        print("TRAINING SUMMARY STATISTICS (RD 2D)")
        print("="*70)
        print(f"\n{'Method':<20} {'Final Total Loss':<18} {'Final Loss A':<18} {'Final Loss S':<18}")
        print("-"*70)
        
        for method in ["NONE", "FNN_BASIS", "QNN"]:
            method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[method]
            results = self.embedding_results[method]
            print(f"{method_name:<20} {results['final_loss']:<18.2E} "
                  f"{results['final_loss_A']:<18.2E} {results['final_loss_S']:<18.2E}")
        
        print("="*70)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main function for 2D training"""
    # Configuration
    config = Config()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("RD 2D QPINN TRAINING")
    print("="*80)
    print(f"Domain: t ∈ [{config.T_MIN}, {config.T_MAX}], x ∈ [-1, 1], y ∈ [-1, 1]")
    print(f"Grid: {config.T_COLLOC_POINTS} × {config.X_COLLOC_POINTS} × {config.Y_COLLOC_POINTS}")
    print(f"Iterations: {config.TRAINING_ITERATIONS}")
    
    # Plot 1: Collocation Points
    plot_collocation_points_2d(config, config.OUTPUT_DIR)
    
    # Generate reference solution and Plot 2
    ref_file = os.path.join(config.OUTPUT_DIR, "rd_equations_2d_reference.npy")
    if not os.path.exists(ref_file):
        print("\n" + "="*80)
        print("Generating Reference Solution")
        print("="*80)
        interpA, interpS, t_ref, x_ref, y_ref, A_sol, S_sol = generate_reference_solution(config, ref_file)
        
        # Create Plot 2: Reference Solution
        plot_reference_solution(t_ref, x_ref, y_ref, A_sol, S_sol, config.OUTPUT_DIR)
    else:
        print(f"\n✓ Reference solution already exists: {ref_file}")
        print("  Skipping Plot 2 generation. Delete the file to regenerate.")
    
    # Initialize trainer
    trainer = RD2DQPINNTrainer(config, device)
    
    # Train all methods
    embedding_results = {}
    for embedding_type in ["NONE", "FNN_BASIS", "QNN"]:
        results = trainer.train(embedding_type)
        embedding_results[embedding_type] = results
        
        # Save model - use descriptive folder names
        folder_name = {"NONE": "pinn", "FNN_BASIS": "fnn_basis", "QNN": "qnn"}[embedding_type]
        model_dir = os.path.join(config.OUTPUT_DIR, folder_name)
        os.makedirs(model_dir, exist_ok=True)
        
        model_state = {'theta': results['theta']}
        if embedding_type == "FNN_BASIS":
            model_state['basis_net'] = results['basis_net_state']
        elif embedding_type == "QNN":
            model_state['qnn_embedding'] = results['qnn_embedding_state']
        elif embedding_type == "NONE":
            model_state['pinn'] = results['pinn_state']
        
        torch.save(model_state, os.path.join(model_dir, 'model.pt'))
        print(f"✓ Model saved: {model_dir}/model.pt")
    
    # Visualization (Plot 4, Plot 5 and Plot 6)
    visualizer = TrainingVisualizer(trainer, embedding_results)
    visualizer.plot_embedding_results(config.OUTPUT_DIR)
    visualizer.plot_training_analysis(config.OUTPUT_DIR)
    visualizer.plot_methods_comparison(config.OUTPUT_DIR)
    visualizer.print_summary()
    
    # Save results for inference
    with open(os.path.join(config.OUTPUT_DIR, 'training_results.pkl'), 'wb') as f:
        pickle.dump(embedding_results, f)
    
    print(f"\n✓ Training results saved: {config.OUTPUT_DIR}/training_results.pkl")
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTo run inference, use:")
    print(f"  python rd_2d_inference.py")

# Example usage: python rd_2d_training.py
if __name__ == "__main__":
    main()
