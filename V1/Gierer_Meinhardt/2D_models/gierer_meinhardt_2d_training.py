"""
Gierer-Meinhardt 2D QPINN - Training Script
Reaction-Diffusion 2D Spatial + Time Quantum Physics-Informed Neural Network

This script trains three models:
1. PINN (Pure Physics-Informed Neural Network)
2. FNN-TE-QPINN (FNN Basis Temporal Embedding QPINN)
3. QNN-TE-QPINN (Quantum Neural Network Temporal Embedding QPINN)

Based on paper 2024112448454, extended to 2D spatial domain

Domain: t ∈ [0, 1], x ∈ [0, 1], y ∈ [0, 1]
PDE System (Gierer-Meinhardt 2D):
    ∂u/∂t = D_u (∂²u/∂x² + ∂²u/∂y²) + a*u²/v - u + ρ
    ∂v/∂t = D_v (∂²v/∂x² + ∂²v/∂y²) + a*u² - v

Initial Conditions:
    u(x, y, 0) = 0.5 + 0.1*sin(2πx)sin(2πy)
    v(x, y, 0) = 0.5 + 0.1*sin(2πx)sin(2πy)

Boundary Conditions (Dirichlet):
    u = 0.5, v = 0.5 on all boundaries

Author: QPINN Research
Date: 2024-2025
"""

import os
import sys
import json
import time
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
        filename = f"plot3_quantum_circuit_{embedding_type.lower()}_training.png"
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
                    f'({config.N_LAYERS_EMBED} layers, {config.N_WIRES} qubits)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save figure
        filename = "plot3_quantum_circuit_qnn_embedding_training.png"
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
    """Configuration class for Gierer-Meinhardt 2D QPINN"""
    
    # Random seed
    SEED = 42
    
    # Quantum Circuit Parameters
    N_LAYERS = 4
    N_WIRES = 4
    
    # FNN Basis Parameters
    HIDDEN_LAYERS_FNN = 2
    NEURONS_FNN = 16
    
    # QNN Embedding Parameters
    N_LAYERS_EMBED = 2
    
    # PINN-specific Parameters
    PINN_HIDDEN_LAYERS = 4
    PINN_NEURONS = 50
    
    # Domain Parameters (reduced for 2D spatial)
    T_COLLOC_POINTS = 10
    X_COLLOC_POINTS = 5
    Y_COLLOC_POINTS = 5
    
    # Physics Parameters (Gierer-Meinhardt)
    D_U = 0.1
    D_V = 0.1
    A = 0.1
    RHO = 0.9
    
    # Boundary conditions (Dirichlet)
    U_BOUNDARY = 0.5
    V_BOUNDARY = 0.5
    
    # Time domain
    T_MIN = 0.0
    T_MAX = 1.0
    
    # Spatial domain
    X_MIN = 0.0
    X_MAX = 1.0
    Y_MIN = 0.0
    Y_MAX = 1.0
    
    # Training Parameters
    TRAINING_ITERATIONS = 2
    LAMBDA_SCALE = 1e3   # Weight for IC + BC loss
    
    # Output directory
    BASE_DIR = "result"
    OUTPUT_DIR = "result"


# ============================================================
# VISUALIZATION - Plot 5, Plot 6, Plot 7
# ============================================================

def plot_collocation_points_2d(config, save_dir="result"):
    """
    Plot 5: Collocation Points Distribution (IC, BC, Interior)
    Biểu đồ thể hiện collocation points được tách thành IC, BC, và Interior points
    Adapted for Brusselator domain [0,1] × [0,1]
    
    IC points: All spatial (x,y) points at t=T_MIN (full grid)
    BC points: Random points at spatial boundaries for t > T_MIN
    Interior points: Random interior points (not at boundaries, not at t=T_MIN)
    """
    print("\n=== Plot 5: Collocation Points Distribution ===")
    
    # Create collocation points
    t_points = np.linspace(config.T_MIN, config.T_MAX, config.T_COLLOC_POINTS)
    x_points = np.linspace(config.X_MIN, config.X_MAX, config.X_COLLOC_POINTS)
    y_points = np.linspace(config.Y_MIN, config.Y_MAX, config.Y_COLLOC_POINTS)
    
    # 1. Initial Condition Points: All (x,y) at t=T_MIN
    t_ic = np.array([config.T_MIN])
    ic_points_list = list(product(t_ic, x_points, y_points))
    ic_points = np.array(ic_points_list)
    
    # 2. Boundary Condition Points: Random sampling at spatial boundaries for t > T_MIN
    N_bc_per_boundary = config.T_COLLOC_POINTS * config.X_COLLOC_POINTS
    
    bc_t_x_min = np.random.uniform(config.T_MIN, config.T_MAX, N_bc_per_boundary)
    bc_y_x_min = np.random.uniform(config.Y_MIN, config.Y_MAX, N_bc_per_boundary)
    bc_x_min = np.column_stack([bc_t_x_min, np.full(N_bc_per_boundary, config.X_MIN), bc_y_x_min])
    
    bc_t_x_max = np.random.uniform(config.T_MIN, config.T_MAX, N_bc_per_boundary)
    bc_y_x_max = np.random.uniform(config.Y_MIN, config.Y_MAX, N_bc_per_boundary)
    bc_x_max = np.column_stack([bc_t_x_max, np.full(N_bc_per_boundary, config.X_MAX), bc_y_x_max])
    
    bc_t_y_min = np.random.uniform(config.T_MIN, config.T_MAX, N_bc_per_boundary)
    bc_x_y_min = np.random.uniform(config.X_MIN, config.X_MAX, N_bc_per_boundary)
    bc_y_min = np.column_stack([bc_t_y_min, bc_x_y_min, np.full(N_bc_per_boundary, config.Y_MIN)])
    
    bc_t_y_max = np.random.uniform(config.T_MIN, config.T_MAX, N_bc_per_boundary)
    bc_x_y_max = np.random.uniform(config.X_MIN, config.X_MAX, N_bc_per_boundary)
    bc_y_max = np.column_stack([bc_t_y_max, bc_x_y_max, np.full(N_bc_per_boundary, config.Y_MAX)])
    
    bc_points = np.vstack([bc_x_min, bc_x_max, bc_y_min, bc_y_max])
    
    # 3. Interior Points: All other points in the domain
    full_domain = np.array(list(product(t_points, x_points, y_points)))
    
    ic_mask = full_domain[:, 0] == config.T_MIN
    bc_mask = (
        ((full_domain[:, 1] == config.X_MIN) | (full_domain[:, 1] == config.X_MAX) |
         (full_domain[:, 2] == config.Y_MIN) | (full_domain[:, 2] == config.Y_MAX)) &
        (full_domain[:, 0] != config.T_MIN)
    )
    interior_mask = ~(ic_mask | bc_mask)
    interior_points = full_domain[interior_mask]
    
    # Create figure with 3 subplots (matching rd_2d_qpinn.py style)
    fig = plt.figure(figsize=(15, 5))
    
    # Subplot 1: Initial Condition Points (IC) - Red
    ax1 = fig.add_subplot(131)
    if len(ic_points) > 0:
        ax1.scatter(ic_points[:, 1], ic_points[:, 2], c="r", s=1, alpha=0.6, label="Initial Condition")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Initial Condition Points (t=0)")
    ax1.legend()
    
    # Subplot 2: Boundary Condition Points (BC) - Blue
    ax2 = fig.add_subplot(132)
    if len(bc_points) > 0:
        ax2.scatter(bc_points[:, 1], bc_points[:, 2], c="blue", s=1, alpha=0.3, label="Boundary")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Boundary Points (t>0)")
    ax2.legend()
    
    # Subplot 3: Interior Points - Black (sample if too many)
    ax3 = fig.add_subplot(133)
    if len(interior_points) > 0:
        sample_idx = np.random.choice(interior_points.shape[0], min(5000, interior_points.shape[0]), replace=False)
        ax3.scatter(interior_points[sample_idx, 1], interior_points[sample_idx, 2], c="black", s=1, alpha=0.1, label="Interior")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_title("Interior Points (sample)")
    ax3.legend()
    
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


def plot_reference_solution(t, x, y, u_sol, v_sol, save_dir="result"):
    """
    Plot 6: Reference Solution from RK45
    Biểu đồ biểu diễn hai reference solution của u và v giải bằng RK45
    Creates 2D heatmaps of u and v at 5 time snapshots
    Adapted for Brusselator (u, v) variables
    
    Args:
        t: Time array from RK45 solver
        x, y: Spatial grids
        u_sol, v_sol: Solution arrays of shape (Nx, Ny, Nt)
        save_dir: Directory to save the plot
    """
    print("\n" + "="*60)
    print("Generating Plot 6: Reference Solution (RK45)")
    print("="*60)
    
    # Select 5 time snapshots
    n_times = 5
    time_indices = np.linspace(0, len(t)-1, n_times, dtype=int)
    
    # Create 2x5 grid: 2 rows (u, v) × 5 columns (time snapshots)
    fig, axes = plt.subplots(2, n_times, figsize=(20, 8))
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    for col, t_idx in enumerate(time_indices):
        current_time = t[t_idx]
        
        # Row 1: u (Activator-like)
        u_slice = u_sol[:, :, t_idx]
        im1 = axes[0, col].contourf(X, Y, u_slice, levels=20, cmap='inferno')
        axes[0, col].set_title(f't = {current_time:.3f}', fontsize=10)
        axes[0, col].set_xlabel('x')
        if col == 0:
            axes[0, col].set_ylabel('u (Activator)', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=axes[0, col])
        
        # Row 2: v (Substrate-like)
        v_slice = v_sol[:, :, t_idx]
        im2 = axes[1, col].contourf(X, Y, v_slice, levels=20, cmap='viridis')
        axes[1, col].set_title(f't = {current_time:.3f}', fontsize=10)
        axes[1, col].set_xlabel('x')
        if col == 0:
            axes[1, col].set_ylabel('v (Substrate)', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=axes[1, col])
    
    plt.suptitle('Plot 2: Reference Solution (RK45) - 2D Gierer-Meinhardt', 
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
# REFERENCE SOLUTION GENERATOR
# ============================================================

def generate_reference_solution(config, save_path="gierer_meinhardt_3d_reference_solution.npy"):
    """Generate 2D spatial reference solution using RK45 solver"""
    
    print("=== Generating 3D (2D spatial + time) Reference Solution ===")
    
    # Spatial domain
    Nx = 64
    Ny = 64
    x = np.linspace(config.X_MIN, config.X_MAX, Nx)
    y = np.linspace(config.Y_MIN, config.Y_MAX, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Time domain
    Nt = 101
    t_eval = np.linspace(config.T_MIN, config.T_MAX, Nt)
    
    # Initial conditions (2D)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u0 = 0.5 + 0.1 * np.sin(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)
    v0 = 0.5 + 0.1 * np.sin(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)
    u0 = np.clip(u0, 0.0, None)
    v0 = np.clip(v0, 0.0, None)
    y0 = np.concatenate([u0.ravel(), v0.ravel()])
    
    def laplacian_2d(u, dx, dy):
        """2D Laplacian with Dirichlet BC"""
        d2 = np.zeros_like(u)
        d2[1:-1, 1:-1] = (
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
            (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
        )
        return d2
    
    def rd_rhs_3d(t, y_flat):
        """RHS for 2D Gierer-Meinhardt system with Dirichlet BC enforcement"""
        u = y_flat[:Nx*Ny].reshape(Nx, Ny)
        v = y_flat[Nx*Ny:].reshape(Nx, Ny)
        
        # Enforce Dirichlet BC
        u[0, :] = config.U_BOUNDARY
        u[-1, :] = config.U_BOUNDARY
        u[:, 0] = config.U_BOUNDARY
        u[:, -1] = config.U_BOUNDARY
        
        v[0, :] = config.V_BOUNDARY
        v[-1, :] = config.V_BOUNDARY
        v[:, 0] = config.V_BOUNDARY
        v[:, -1] = config.V_BOUNDARY
        
        lapU = laplacian_2d(u, dx, dy)
        lapV = laplacian_2d(v, dx, dy)
        
        du = config.D_U * lapU + (config.A * (u**2) / (v + 1e-6)) - u + config.RHO
        dv = config.D_V * lapV + config.A * (u**2) - v
        
        # Set du/dt = 0 and dv/dt = 0 at boundaries (Dirichlet BC)
        du[0, :] = 0.0
        du[-1, :] = 0.0
        du[:, 0] = 0.0
        du[:, -1] = 0.0
        
        dv[0, :] = 0.0
        dv[-1, :] = 0.0
        dv[:, 0] = 0.0
        dv[:, -1] = 0.0
        
        return np.concatenate([du.ravel(), dv.ravel()])
    
    print(f"Solving 2D Gierer-Meinhardt PDE with RK45...")
    print(f"Parameters: D_U={config.D_U}, D_V={config.D_V}, A={config.A}, RHO={config.RHO}")
    print(f"Grid: Nx={Nx}, Ny={Ny}, Nt={Nt}")
    
    sol = solve_ivp(rd_rhs_3d, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-9)
    
    t = sol.t
    u_sol = sol.y[:Nx*Ny, :].reshape(Nx, Ny, -1)
    v_sol = sol.y[Nx*Ny:, :].reshape(Nx, Ny, -1)
    print("Status:", sol.message)
    
    # Build interpolators: (x, y, t) ordering
    print("Building interpolators...")
    interpU = RegularGridInterpolator((x, y, t), u_sol, bounds_error=False, fill_value=None)
    interpV = RegularGridInterpolator((x, y, t), v_sol, bounds_error=False, fill_value=None)
    
    print(f"Saving reference solution to '{save_path}'...")
    np.save(save_path, {
        'u': interpU, 'v': interpV,
        't': t, 'x': x, 'y': y,
        'u_sol': u_sol, 'v_sol': v_sol
    }, allow_pickle=True)
    
    print(f"✓ Reference solution generated: u∈[{u_sol.min():.4f}, {u_sol.max():.4f}], "
          f"v∈[{v_sol.min():.4f}, {v_sol.max():.4f}]")
    
    return interpU, interpV


# ============================================================
# NEURAL NETWORK MODELS
# ============================================================

class FNNBasisNet(nn.Module):
    """FNN network to generate basis for quantum circuit encoding"""
    
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
    """Quantum Neural Network for embedding generation (dual-circuit approach)"""
    
    def __init__(self, n_wires, n_layers_embed, output_dim, input_dim=3):
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
        """Embedding circuit for 3D input (t, x, y)"""
        for layer in range(self.n_layers_embed):
            for i in range(self.n_wires):
                qubit_idx = i % 3
                if qubit_idx == 0:
                    qml.RX(x[0], wires=i)  # t
                elif qubit_idx == 1:
                    qml.RY(x[1], wires=i)  # x
                else:
                    qml.RZ(x[2], wires=i)  # y
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
# TRAINER CLASS
# ============================================================

class GiererMeinhardt2DQPINNTrainer:
    """Trainer class for Brusselator 3D (2D spatial + time) QPINN models"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.domain_min = torch.tensor([config.T_MIN, config.X_MIN, config.Y_MIN], device=device)
        self.domain_max = torch.tensor([config.T_MAX, config.X_MAX, config.Y_MAX], device=device)
        
        # Generate collocation points
        self._setup_collocation_points()
        
        # Load reference solution
        self._load_reference_solution()
        
        # Store training results
        self.training_results = {}
    
    def _setup_collocation_points(self):
        """Setup collocation points for training"""
        t_torch = torch.linspace(self.config.T_MIN, self.config.T_MAX, self.config.T_COLLOC_POINTS)
        x_torch = torch.linspace(self.config.X_MIN, self.config.X_MAX, self.config.X_COLLOC_POINTS)
        y_torch = torch.linspace(self.config.Y_MIN, self.config.Y_MAX, self.config.Y_COLLOC_POINTS)
        
        # Store unique values for visualization
        self.T_unique = t_torch.cpu().numpy()
        self.X_unique = x_torch.cpu().numpy()
        self.Y_unique = y_torch.cpu().numpy()
        
        # Domain: (t, x, y)
        domain = torch.tensor(list(product(t_torch, x_torch, y_torch)), dtype=torch.float32)
        
        # Initial condition mask (t = 0)
        init_val_mask = domain[:, 0] == self.config.T_MIN
        self.init_val_colloc = domain[init_val_mask].clone().detach().requires_grad_(True).to(self.device)
        
        # Boundary mask (x or y at boundary)
        boundary_mask = (
            (domain[:, 1] == self.config.X_MIN) | (domain[:, 1] == self.config.X_MAX) |
            (domain[:, 2] == self.config.Y_MIN) | (domain[:, 2] == self.config.Y_MAX)
        )
        self.boundary_colloc = domain[boundary_mask & ~init_val_mask].clone().detach().requires_grad_(True).to(self.device)
        
        # Interior points
        interior_mask = ~(init_val_mask | boundary_mask)
        self.interior_colloc = domain[interior_mask].clone().detach().requires_grad_(True).to(self.device)
        
        # Full domain
        self.input_domain = domain.clone().detach().requires_grad_(True).to(self.device)
        
        # High-res IC points
        Nx = 64
        Ny = 64
        x_ic = np.linspace(self.config.X_MIN, self.config.X_MAX, Nx)
        y_ic = np.linspace(self.config.Y_MIN, self.config.Y_MAX, Ny)
        X_ic, Y_ic = np.meshgrid(x_ic, y_ic, indexing='ij')
        
        self.u0_ic = 0.5 + 0.1 * np.sin(2.0 * np.pi * X_ic) * np.sin(2.0 * np.pi * Y_ic)
        self.v0_ic = 0.5 + 0.1 * np.sin(2.0 * np.pi * X_ic) * np.sin(2.0 * np.pi * Y_ic)
        
        t_ic = torch.full((Nx*Ny, 1), self.config.T_MIN, device=self.device)
        x_ic_t = torch.tensor(X_ic.ravel(), device=self.device).float().view(-1, 1)
        y_ic_t = torch.tensor(Y_ic.ravel(), device=self.device).float().view(-1, 1)
        self.ic_points_highres = torch.cat([t_ic, x_ic_t, y_ic_t], dim=1)
        
        print(f"✓ Collocation points: Interior={len(self.interior_colloc)}, "
              f"Boundary={len(self.boundary_colloc)}, IC={len(self.init_val_colloc)}")
    
    def _load_reference_solution(self):
        """Load or generate reference solution"""
        ref_path = os.path.join(self.config.BASE_DIR, "gierer_meinhardt_3d_reference_solution.npy")
        
        if os.path.exists(ref_path):
            print(f"Loading reference solution from {ref_path}")
            loaded = np.load(ref_path, allow_pickle=True)[()]
            self.interp_u = loaded['u']
            self.interp_v = loaded['v']
        else:
            os.makedirs(self.config.BASE_DIR, exist_ok=True)
            self.interp_u, self.interp_v = generate_reference_solution(self.config, ref_path)
        
        # Compute reference on domain
        domain_np = self.input_domain.detach().cpu().numpy()
        # Reference expects (x, y, t), domain is (t, x, y)
        ref_u = np.array([self.interp_u([pt[1], pt[2], pt[0]]).squeeze() for pt in domain_np])
        ref_v = np.array([self.interp_v([pt[1], pt[2], pt[0]]).squeeze() for pt in domain_np])
        
        self.reference_u = torch.tensor(ref_u, device=self.device, dtype=torch.float32)
        self.reference_v = torch.tensor(ref_v, device=self.device, dtype=torch.float32)
        
        print(f"✓ Reference solution loaded: u∈[{self.reference_u.min():.4f}, {self.reference_u.max():.4f}], "
              f"v∈[{self.reference_v.min():.4f}, {self.reference_v.max():.4f}]")
    
    def _create_circuit(self):
        """Create the main quantum circuit for TE-QPINN"""
        dev = qml.device("default.qubit", wires=self.config.N_WIRES)
        
        @qml.qnode(dev, interface="torch")
        def circuit(x, theta, basis):
            # Tensor encoding with basis for 3D input
            for i in range(self.config.N_WIRES):
                qubit_idx = i % 3
                if qubit_idx == 0:
                    qml.RY(basis[i] * x[0], wires=i)  # t
                elif qubit_idx == 1:
                    qml.RY(basis[i] * x[1], wires=i)  # x
                else:
                    qml.RY(basis[i] * x[2], wires=i)  # y
            
            # Variational layers
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
        u_scaled = raw_output[:, 0] * 2.0 + 2.0
        v_scaled = raw_output[:, 1] * 2.0 + 2.0
        return torch.stack([u_scaled, v_scaled], dim=1)
    
    def model(self, x):
        """Forward pass through the current model"""
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
        else:  # NONE (pure PINN)
            return self.pinn(x_rescaled)
    
    def _create_loss_functions(self):
        """Create loss functions for training"""
        
        def extract_u(multi_output):
            return multi_output[:, 0] if multi_output.dim() > 1 else multi_output[0]
        
        def extract_v(multi_output):
            return multi_output[:, 1] if multi_output.dim() > 1 else multi_output[1]
        
        def pde_loss():
            pred = self.model(self.interior_colloc)
            u = extract_u(pred)
            v = extract_v(pred)
            
            # u gradients
            grad_u = torch.autograd.grad(u.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            du_dt = grad_u[:, 0]
            du_dx = grad_u[:, 1]
            du_dy = grad_u[:, 2]
            
            grad_du_dx = torch.autograd.grad(du_dx.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            d2u_dx2 = grad_du_dx[:, 1]
            
            grad_du_dy = torch.autograd.grad(du_dy.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            d2u_dy2 = grad_du_dy[:, 2]
            
            # v gradients
            grad_v = torch.autograd.grad(v.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            dv_dt = grad_v[:, 0]
            dv_dx = grad_v[:, 1]
            dv_dy = grad_v[:, 2]
            
            grad_dv_dx = torch.autograd.grad(dv_dx.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            d2v_dx2 = grad_dv_dx[:, 1]
            
            grad_dv_dy = torch.autograd.grad(dv_dy.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            d2v_dy2 = grad_dv_dy[:, 2]
            
            coupling = (u ** 2) * v
            laplacian_u = d2u_dx2 + d2u_dy2
            laplacian_v = d2v_dx2 + d2v_dy2
            
            # PDE residuals (Gierer-Meinhardt)
            residual_u = du_dt - self.config.D_U * laplacian_u - (self.config.A * (u**2) / (v + 1e-6)) + u - self.config.RHO
            residual_v = dv_dt - self.config.D_V * laplacian_v - self.config.A * (u**2) + v
            
            return torch.mean(residual_u ** 2) + torch.mean(residual_v ** 2)
        
        def initial_condition_loss():
            pred = self.model(self.ic_points_highres)
            u = extract_u(pred)
            v = extract_v(pred)
            
            u_true = torch.tensor(self.u0_ic.ravel(), device=self.device, dtype=torch.float32)
            v_true = torch.tensor(self.v0_ic.ravel(), device=self.device, dtype=torch.float32)
            
            return torch.mean((u - u_true) ** 2) + torch.mean((v - v_true) ** 2)
        
        def boundary_loss():
            if len(self.boundary_colloc) == 0:
                return torch.tensor(0.0, device=self.device)
            
            pred = self.model(self.boundary_colloc)
            u = extract_u(pred)
            v = extract_v(pred)
            
            # Dirichlet BC: u=1, v=3 at boundaries
            loss_u = torch.mean((u - self.config.U_BOUNDARY) ** 2)
            loss_v = torch.mean((v - self.config.V_BOUNDARY) ** 2)
            
            return loss_u + loss_v
        
        def total_loss():
            loss_pde = pde_loss()
            loss_ic = initial_condition_loss()
            loss_bc = boundary_loss()
            return loss_pde + self.config.LAMBDA_SCALE * (loss_ic + loss_bc)
        
        def compute_metrics():
            pred = self.model(self.input_domain)
            u = extract_u(pred)
            v = extract_v(pred)
            
            mse_u = torch.mean((u - self.reference_u) ** 2).item()
            mse_v = torch.mean((v - self.reference_v) ** 2).item()
            linf_u = torch.max(torch.abs(u - self.reference_u)).item()
            linf_v = torch.max(torch.abs(v - self.reference_v)).item()
            
            return {
                'mse_u': mse_u, 'mse_v': mse_v,
                'mse_total': mse_u + mse_v,
                'linf_u': linf_u, 'linf_v': linf_v,
                'linf_max': max(linf_u, linf_v)
            }
        
        return total_loss, compute_metrics
    
    def train_model(self, embedding_type, iterations):
        """Train a single model type"""
        self.embedding_type = embedding_type
        method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
        
        print(f"\n{'='*70}")
        print(f"TRAINING: {method_name}")
        print(f"{'='*70}")
        
        # Initialize model
        if embedding_type == "FNN_BASIS":
            self.theta = torch.rand(self.config.N_LAYERS, self.config.N_WIRES, 3,
                                    device=self.device, requires_grad=True)
            self.basis_net = FNNBasisNet(
                self.config.HIDDEN_LAYERS_FNN,
                self.config.NEURONS_FNN,
                self.config.N_WIRES,
                input_dim=3
            ).to(self.device)
            self.circuit = self._create_circuit()
            params = [self.theta] + list(self.basis_net.parameters())
            
            # Count parameters
            model_components = {'theta': self.theta, 'basis_net': self.basis_net}
            total_params, param_dict = count_parameters(model_components)
            print(f"\n📊 Model Parameters ({method_name}):")
            print(f"   theta: {param_dict.get('theta', 0):,}")
            print(f"   basis_net: {param_dict.get('basis_net', 0):,}")
            print(f"   Total: {total_params:,}")
            
        elif embedding_type == "QNN":
            self.theta = torch.rand(self.config.N_LAYERS, self.config.N_WIRES, 3,
                                    device=self.device, requires_grad=True)
            self.qnn_embedding = QNNEmbedding(
                self.config.N_WIRES,
                self.config.N_LAYERS_EMBED,
                self.config.N_WIRES,
                input_dim=3
            ).to(self.device)
            self.circuit = self._create_circuit()
            params = [self.theta] + list(self.qnn_embedding.parameters())
            
            # Count parameters
            model_components = {'theta': self.theta, 'qnn_embedding': self.qnn_embedding}
            total_params, param_dict = count_parameters(model_components)
            print(f"\n📊 Model Parameters ({method_name}):")
            print(f"   theta: {param_dict.get('theta', 0):,}")
            print(f"   qnn_embedding: {param_dict.get('qnn_embedding', 0):,}")
            print(f"   Total: {total_params:,}")
            
        else:  # NONE (PINN)
            self.pinn = FNNBasisNet(
                self.config.PINN_HIDDEN_LAYERS,
                self.config.PINN_NEURONS,
                2,  # Output: u, v
                input_dim=3
            ).to(self.device)
            params = list(self.pinn.parameters())
            
            # Count parameters
            model_components = {'pinn': self.pinn}
            total_params, param_dict = count_parameters(model_components)
            print(f"\n📊 Model Parameters ({method_name}):")
            print(f"   pinn: {param_dict.get('pinn', 0):,}")
            print(f"   Total: {total_params:,}")
        
        # Optimizer
        optimizer = torch.optim.LBFGS(params, line_search_fn="strong_wolfe")

        
        # Loss functions
        total_loss_fn, compute_metrics_fn = self._create_loss_functions()
        
        def closure():
            optimizer.zero_grad()
            loss = total_loss_fn()
            loss.backward()
            return loss
        
        # Training loop with separate loss tracking
        loss_history = []
        loss_u_history = []
        loss_v_history = []
        mse_u_history = []
        mse_v_history = []
        start_time = time.time()
        
        def extract_u(multi_output):
            return multi_output[:, 0] if multi_output.dim() > 1 else multi_output[0]
        
        def extract_v(multi_output):
            return multi_output[:, 1] if multi_output.dim() > 1 else multi_output[1]
        
        previous_loss = float('inf')
        for epoch in range(iterations):
            optimizer.step(closure)
            current_loss = total_loss_fn().item()
            metrics = compute_metrics_fn()
            
            # Early stopping
            if abs(previous_loss - current_loss) < 1e-10:
                print("Early stopping: Loss change < 1e-10")
                break
            previous_loss = current_loss
            
            # Compute separate losses for u and v
            pred = self.model(self.interior_colloc)
            u = extract_u(pred)
            v = extract_v(pred)
            
            # u gradients
            grad_u = torch.autograd.grad(u.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            du_dt = grad_u[:, 0]
            du_dx = grad_u[:, 1]
            du_dy = grad_u[:, 2]
            grad_du_dx = torch.autograd.grad(du_dx.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            grad_du_dy = torch.autograd.grad(du_dy.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            d2u_dx2 = grad_du_dx[:, 1]
            d2u_dy2 = grad_du_dy[:, 2]
            
            # v gradients
            grad_v = torch.autograd.grad(v.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            dv_dt = grad_v[:, 0]
            dv_dx = grad_v[:, 1]
            dv_dy = grad_v[:, 2]
            grad_dv_dx = torch.autograd.grad(dv_dx.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            grad_dv_dy = torch.autograd.grad(dv_dy.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            d2v_dx2 = grad_dv_dx[:, 1]
            d2v_dy2 = grad_dv_dy[:, 2]
            
            # Gierer-Meinhardt residuals
            residual_u = du_dt - self.config.D_U * (d2u_dx2 + d2u_dy2) - (self.config.A * (u**2) / (v + 1e-6)) + u - self.config.RHO
            residual_v = dv_dt - self.config.D_V * (d2v_dx2 + d2v_dy2) - self.config.A * (u**2) + v
            
            loss_u = torch.mean(residual_u ** 2).item()
            loss_v = torch.mean(residual_v ** 2).item()
            
            loss_history.append(current_loss)
            loss_u_history.append(loss_u)
            loss_v_history.append(loss_v)
            mse_u_history.append(metrics['mse_u'])
            mse_v_history.append(metrics['mse_v'])
            
            if epoch % 10 == 0 or epoch == iterations - 1:
                print(f"Epoch {epoch:04d} | Loss: {current_loss:.2E} | "
                      f"MSE_u: {metrics['mse_u']:.2E} | MSE_v: {metrics['mse_v']:.2E} | "
                      f"L∞: {metrics['linf_max']:.2E}")
        
        training_time = time.time() - start_time
        final_metrics = compute_metrics_fn()
        
        # Get final predictions
        with torch.no_grad():
            final_pred = self.model(self.input_domain)
            predictions_u = extract_u(final_pred).cpu().numpy()
            predictions_v = extract_v(final_pred).cpu().numpy()
        
        print(f"\n✅ {method_name} Training completed in {training_time:.2f}s")
        print(f"   Final Loss: {loss_history[-1]:.2E}")
        print(f"   MSE (u): {final_metrics['mse_u']:.2E}")
        print(f"   MSE (v): {final_metrics['mse_v']:.2E}")
        print(f"   L∞ max: {final_metrics['linf_max']:.2E}")
        
        # Save results
        self.training_results[embedding_type] = {
            'loss_history': loss_history,
            'loss_u_history': loss_u_history,
            'loss_v_history': loss_v_history,
            'mse_u_history': mse_u_history,
            'mse_v_history': mse_v_history,
            'predictions_u': predictions_u,
            'predictions_v': predictions_v,
            'final_loss': current_loss,
            'final_loss_u': loss_u,
            'final_loss_v': loss_v,
            'final_metrics': final_metrics,
            'training_time': training_time
        }
        
        return loss_history, final_metrics
    
    def save_model(self, embedding_type, save_dir):
        """Save trained model"""
        folder_name = {"NONE": "pinn", "FNN_BASIS": "fnn_basis", "QNN": "qnn"}[embedding_type]
        model_dir = os.path.join(save_dir, folder_name)
        os.makedirs(model_dir, exist_ok=True)
        
        if embedding_type == "FNN_BASIS":
            save_dict = {
                'theta': self.theta.detach().cpu().numpy(),
                'basis_net': self.basis_net.state_dict(),
                'config': {
                    'N_LAYERS': self.config.N_LAYERS,
                    'N_WIRES': self.config.N_WIRES,
                    'HIDDEN_LAYERS_FNN': self.config.HIDDEN_LAYERS_FNN,
                    'NEURONS_FNN': self.config.NEURONS_FNN,
                }
            }
            np.save(os.path.join(model_dir, 'model.npy'), save_dict, allow_pickle=True)
            
        elif embedding_type == "QNN":
            save_dict = {
                'theta': self.theta.detach().cpu().numpy(),
                'qnn_embedding': self.qnn_embedding.state_dict(),
                'config': {
                    'N_LAYERS': self.config.N_LAYERS,
                    'N_WIRES': self.config.N_WIRES,
                    'N_LAYERS_EMBED': self.config.N_LAYERS_EMBED,
                }
            }
            np.save(os.path.join(model_dir, 'model.npy'), save_dict, allow_pickle=True)
            
        else:  # PINN
            torch.save(self.pinn.state_dict(), os.path.join(model_dir, 'model.pth'))
        
        # Save training results (exclude large arrays for JSON)
        with open(os.path.join(model_dir, 'training_results.json'), 'w') as f:
            results = {
                'loss_history': [float(x) for x in self.training_results[embedding_type]['loss_history']],
                'loss_u_history': [float(x) for x in self.training_results[embedding_type]['loss_u_history']],
                'loss_v_history': [float(x) for x in self.training_results[embedding_type]['loss_v_history']],
                'mse_u_history': [float(x) for x in self.training_results[embedding_type]['mse_u_history']],
                'mse_v_history': [float(x) for x in self.training_results[embedding_type]['mse_v_history']],
                'final_loss': float(self.training_results[embedding_type]['final_loss']),
                'final_loss_u': float(self.training_results[embedding_type]['final_loss_u']),
                'final_loss_v': float(self.training_results[embedding_type]['final_loss_v']),
                'final_metrics': {k: float(v) for k, v in self.training_results[embedding_type]['final_metrics'].items()},
                'training_time': float(self.training_results[embedding_type]['training_time'])
            }
            json.dump(results, f, indent=2)
        
        print(f"✓ {folder_name} model saved to {model_dir}")


# ============================================================
# VISUALIZATION - Plot 8 (Plot 1), Plot 9 (Plot 3) and others
# ============================================================

class TrainingVisualizer:
    """Visualization class for training results"""
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.results = trainer.training_results
        self.T_unique = trainer.T_unique
        self.X_unique = trainer.X_unique
        self.Y_unique = trainer.Y_unique
        self.reference_u = trainer.reference_u.cpu().numpy()
        self.reference_v = trainer.reference_v.cpu().numpy()
    
    def plot_embedding_results(self, save_dir="result"):
        """Plot 8: Embedding Results Visualization (adapted for Brusselator)
        Shows embedding outputs for FNN-TE-QPINN and QNN-TE-QPINN
        """
        print("\n=== Plot 8: Embedding Results ===")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Create grid for visualization
        t_plot = np.linspace(self.T_unique[0], self.T_unique[-1], 50)
        x_plot = np.linspace(self.X_unique[0], self.X_unique[-1], 50)
        y_plot = np.linspace(self.Y_unique[0], self.Y_unique[-1], 50)
        
        T_grid, X_grid = np.meshgrid(t_plot, x_plot, indexing='ij')
        Y_fixed = (self.Y_unique[0] + self.Y_unique[-1]) / 2.0  # Fix y at midpoint
        
        # Create grid points (t, x, y_fixed)
        grid_points = np.column_stack([
            T_grid.flatten(),
            X_grid.flatten(),
            np.full(T_grid.size, Y_fixed)
        ])
        
        # Convert to torch and rescale
        grid_torch = torch.tensor(grid_points, dtype=torch.float32, device=self.trainer.device)
        
        # Rescale to [-0.95, 0.95] domain for models
        domain_bounds = torch.tensor(
            [[self.trainer.config.T_MIN, self.trainer.config.X_MIN, self.trainer.config.Y_MIN],
             [self.trainer.config.T_MAX, self.trainer.config.X_MAX, self.trainer.config.Y_MAX]],
            device=self.trainer.device
        )
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
        
        plt.suptitle(r"Plot 4: Trainable Embedding Functions: $\phi(x) \cdot x$ (Brusselator 2D)", 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/plot4_embedding_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot 4 saved: {save_dir}/plot4_embedding_results.png")
    
    def plot_training_analysis(self, save_dir="result"):
        """Plot 1: Training analysis for each model - 2D adapted
        Renamed to Training Analysis for Brusselator
        """
        print("\n=== Plot 1: Training Analysis for Each Model ===")
        
        os.makedirs(save_dir, exist_ok=True)
        
        for embedding_type in ["NONE", "FNN_BASIS", "QNN"]:
            method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
            results = self.results[embedding_type]
            
            fig, axes = plt.subplots(2, 3, figsize=(24, 16))
            
            # Column 1: Training Loss Evolution
            axes[0,0].semilogy(results['loss_history'], 'b-', linewidth=2, label='Total Loss')
            axes[0,0].semilogy(results['loss_u_history'], 'r--', linewidth=2, label='Loss u')
            axes[0,0].semilogy(results['loss_v_history'], 'g--', linewidth=2, label='Loss v')
            axes[0,0].set_xlabel('Epoch', fontsize=12)
            axes[0,0].set_ylabel('Loss (log scale)', fontsize=12)
            axes[0,0].set_title(f'{method_name} Training Loss Evolution', fontsize=13)
            axes[0,0].legend(loc='best')
            axes[0,0].grid(True, alpha=0.3, which='both')
            
            axes[1,0].semilogy(results['mse_u_history'], 'r-', linewidth=2, label='MSE u')
            axes[1,0].semilogy(results['mse_v_history'], 'g-', linewidth=2, label='MSE v')
            axes[1,0].set_xlabel('Epoch', fontsize=12)
            axes[1,0].set_ylabel('MSE (log scale)', fontsize=12)
            axes[1,0].set_title(f'{method_name} MSE Evolution', fontsize=13)
            axes[1,0].legend(loc='best')
            axes[1,0].grid(True, alpha=0.3, which='both')
            
            # Reshape predictions for 2D spatial visualization
            # Shape: (T, X, Y)
            u_pred = results['predictions_u'].reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
            v_pred = results['predictions_v'].reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
            
            # Column 2: u/v at t=T/2 (middle time)
            t_mid_idx = len(self.T_unique) // 2
            X_grid, Y_grid = np.meshgrid(self.X_unique, self.Y_unique, indexing='ij')
            
            im1 = axes[0,1].contourf(X_grid, Y_grid, u_pred[t_mid_idx, :, :], 50, cmap='inferno')
            axes[0,1].set_xlabel('x', fontsize=12)
            axes[0,1].set_ylabel('y', fontsize=12)
            axes[0,1].set_title(f'u(t={self.T_unique[t_mid_idx]:.2f}, x, y) - {method_name}', fontsize=13)
            axes[0,1].set_aspect('equal')
            fig.colorbar(im1, ax=axes[0,1], label='u')
            
            im2 = axes[1,1].contourf(X_grid, Y_grid, v_pred[t_mid_idx, :, :], 50, cmap='viridis')
            axes[1,1].set_xlabel('x', fontsize=12)
            axes[1,1].set_ylabel('y', fontsize=12)
            axes[1,1].set_title(f'v(t={self.T_unique[t_mid_idx]:.2f}, x, y) - {method_name}', fontsize=13)
            axes[1,1].set_aspect('equal')
            fig.colorbar(im2, ax=axes[1,1], label='v')
            
            # Column 3: Errors at t=T/2
            u_ref = self.reference_u.reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
            v_ref = self.reference_v.reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
            
            u_error = np.abs(u_pred[t_mid_idx, :, :] - u_ref[t_mid_idx, :, :])
            v_error = np.abs(v_pred[t_mid_idx, :, :] - v_ref[t_mid_idx, :, :])
            
            im3 = axes[0,2].contourf(X_grid, Y_grid, u_error, 50, cmap='inferno')
            axes[0,2].set_xlabel('x', fontsize=12)
            axes[0,2].set_ylabel('y', fontsize=12)
            axes[0,2].set_title(f'|u_pred - u_ref| at t={self.T_unique[t_mid_idx]:.2f}', fontsize=13)
            axes[0,2].set_aspect('equal')
            fig.colorbar(im3, ax=axes[0,2], label='|Error|')
            
            im4 = axes[1,2].contourf(X_grid, Y_grid, v_error, 50, cmap='viridis')
            axes[1,2].set_xlabel('x', fontsize=12)
            axes[1,2].set_ylabel('y', fontsize=12)
            axes[1,2].set_title(f'|v_pred - v_ref| at t={self.T_unique[t_mid_idx]:.2f}', fontsize=13)
            axes[1,2].set_aspect('equal')
            fig.colorbar(im4, ax=axes[1,2], label='|Error|')
            
            plt.suptitle(f'Plot 5: {method_name} Training Analysis (Brusselator 3D)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/plot5_training_{embedding_type.lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Plot 5 for {method_name} saved: {save_dir}/plot5_training_{embedding_type.lower()}.png")
    
    def plot_methods_comparison(self, save_dir="result"):
        """Plot 3: Methods comparison - Total loss during training"""
        print("\n=== Plot 3: Methods Comparison - Total Loss ===")
        
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Total Loss for all 3 models
        axes[0].semilogy(self.results["NONE"]['loss_history'], 'b-', linewidth=2, label='PINN')
        axes[0].semilogy(self.results["FNN_BASIS"]['loss_history'], 'r--', linewidth=2, label='FNN-TE-QPINN')
        axes[0].semilogy(self.results["QNN"]['loss_history'], 'g-', linewidth=2, label='QNN-TE-QPINN')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Total Loss (log scale)', fontsize=12)
        axes[0].set_title('Total Loss Comparison', fontsize=13)
        axes[0].legend(fontsize=10, loc='best')
        axes[0].grid(True, alpha=0.3, which='both')
        
        # Plot 2: Loss u for all 3 models
        axes[1].semilogy(self.results["NONE"]['loss_u_history'], 'b-', linewidth=2, label='PINN')
        axes[1].semilogy(self.results["FNN_BASIS"]['loss_u_history'], 'r--', linewidth=2, label='FNN-TE-QPINN')
        axes[1].semilogy(self.results["QNN"]['loss_u_history'], 'g-', linewidth=2, label='QNN-TE-QPINN')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss u (log scale)', fontsize=12)
        axes[1].set_title('Activator u Loss Comparison', fontsize=13)
        axes[1].legend(fontsize=10, loc='best')
        axes[1].grid(True, alpha=0.3, which='both')
        
        # Plot 3: Loss v for all 3 models
        axes[2].semilogy(self.results["NONE"]['loss_v_history'], 'b-', linewidth=2, label='PINN')
        axes[2].semilogy(self.results["FNN_BASIS"]['loss_v_history'], 'r--', linewidth=2, label='FNN-TE-QPINN')
        axes[2].semilogy(self.results["QNN"]['loss_v_history'], 'g-', linewidth=2, label='QNN-TE-QPINN')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Loss v (log scale)', fontsize=12)
        axes[2].set_title('Substrate v Loss Comparison', fontsize=13)
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
        print("TRAINING SUMMARY STATISTICS (Gierer-Meinhardt 2D)")
        print("="*70)
        print(f"\n{'Method':<20} {'Final Total Loss':<18} {'Final Loss u':<18} {'Final Loss v':<18}")
        print("-"*70)
        
        for method in ["NONE", "FNN_BASIS", "QNN"]:
            method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[method]
            results = self.results[method]
            print(f"{method_name:<20} {results['final_loss']:<18.2E} "
                  f"{results['final_loss_u']:<18.2E} {results['final_loss_v']:<18.2E}")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    # Configuration
    config = Config()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("GIERER-MEINHARDT 2D QPINN TRAINING (2D Spatial + Time)")
    print("="*80)
    print(f"Domain: t ∈ [{config.T_MIN}, {config.T_MAX}], "
          f"x ∈ [{config.X_MIN}, {config.X_MAX}], y ∈ [{config.Y_MIN}, {config.Y_MAX}]")
    print(f"Parameters: D_U={config.D_U}, D_V={config.D_V}, A={config.A}, RHO={config.RHO}")
    print(f"Training iterations: {config.TRAINING_ITERATIONS}")
    print("="*80)
    
    # Generate reference solution and reference plots (Plot 5, 6)
    ref_path = os.path.join(config.BASE_DIR, "gierer_meinhardt_3d_reference_solution.npy")
    if not os.path.exists(ref_path):
        generate_reference_solution(config, ref_path)
    
    # Load reference solution to access interpolators
    ref_data = np.load(ref_path, allow_pickle=True).item()
    interp_u, interp_v = ref_data['u'], ref_data['v']
    t_ref = ref_data['t']
    x_ref = ref_data['x']
    y_ref = ref_data['y']
    u_sol = ref_data['u_sol']
    v_sol = ref_data['v_sol']
    
    # Plot collocation points (Plot 5)
    plot_collocation_points_2d(config, config.OUTPUT_DIR)
    
    # Plot reference solution (Plot 6)
    plot_reference_solution(t_ref, x_ref, y_ref, u_sol, v_sol, config.OUTPUT_DIR)
    
    # Initialize trainer
    trainer = GiererMeinhardt2DQPINNTrainer(config, device)
    
    # Train all models
    for embedding_type in ["NONE", "FNN_BASIS", "QNN"]:
        trainer.train_model(embedding_type, config.TRAINING_ITERATIONS)
        trainer.save_model(embedding_type, config.OUTPUT_DIR)
        
        # Plot quantum circuit after training each model (Plot 7)
        if embedding_type == "FNN_BASIS":
            if hasattr(trainer, 'circuit'):
                plot_quantum_circuit(trainer.circuit, embedding_type, config, config.OUTPUT_DIR)
        elif embedding_type == "QNN":
            if hasattr(trainer, 'circuit'):
                plot_quantum_circuit(trainer.circuit, embedding_type, config, config.OUTPUT_DIR)
            if hasattr(trainer, 'qnn_embedding'):
                plot_qnn_embedding_circuit(trainer.qnn_embedding, config, config.OUTPUT_DIR)
    
    # Generate visualizations
    visualizer = TrainingVisualizer(trainer)
    visualizer.plot_training_analysis(config.OUTPUT_DIR)  # Plot 5
    visualizer.plot_embedding_results(config.OUTPUT_DIR)   # Plot 8
    visualizer.plot_methods_comparison(config.OUTPUT_DIR)  # Plot 3
    visualizer.print_summary()
    
    # Save combined results
    combined_results = {}
    for method, data in trainer.training_results.items():
        combined_results[method] = {
            'loss_history': [float(x) for x in data['loss_history']],
            'loss_u_history': [float(x) for x in data['loss_u_history']],
            'loss_v_history': [float(x) for x in data['loss_v_history']],
            'mse_u_history': [float(x) for x in data['mse_u_history']],
            'mse_v_history': [float(x) for x in data['mse_v_history']],
            'final_loss': float(data['final_loss']),
            'final_loss_u': float(data['final_loss_u']),
            'final_loss_v': float(data['final_loss_v']),
            'final_metrics': {k: float(v) for k, v in data['final_metrics'].items()},
            'training_time': float(data['training_time'])
        }
    
    with open(os.path.join(config.OUTPUT_DIR, 'training_summary.json'), 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Results saved to: {config.OUTPUT_DIR}/")
    print("\nGenerated Plots:")
    print("  Plot 1 - plot1_training_*.png (for each model)")
    print("  Plot 3 - plot3_methods_comparison.png")
    print("  Plot 5 - plot5_collocation_points.png")
    print("  Plot 6 - plot6_reference_solution.png")
    print("  Plot 7 - plot7_quantum_circuit_*.png (for FNN and QNN)")
    print("  Plot 8 - plot8_embedding_results.png")
    print("\nOther outputs:")
    print("  - pinn/, fnn_basis/, qnn/ (model checkpoints)")
    print("  - training_summary.json")
    print("="*80)


if __name__ == "__main__":
    main()
