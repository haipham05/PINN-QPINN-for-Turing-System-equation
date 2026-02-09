"""
Gray-Scott 1D QPINN - Training Script
Reaction-Diffusion 1D Quantum Physics-Informed Neural Network

This script trains three models:
1. PINN (Pure Physics-Informed Neural Network)
2. FNN-TE-QPINN (FNN Basis Temporal Embedding QPINN)
3. QNN-TE-QPINN (Quantum Neural Network Temporal Embedding QPINN)

Based on paper 2024112448454

Domain: t ∈ [0, 1], x ∈ [0, 1]
PDE System (Brusselator):
    ∂u/∂t = D_u ∂²u/∂x² - u*v² + f*(1-u)
    ∂v/∂t = D_v ∂²v/∂x² + u*v² - (k+f)*v

Initial Conditions:
    u(x, 0) = 1
    v(x, 0) = 0 (with localized perturbation)

Boundary Conditions (Dirichlet):
    u(0, t) = u(1, t) = 1
    v(0, t) = v(1, t) = 0

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
# CONFIGURATION
# ============================================================

class Config:
    """Configuration class for Gray-Scott 1D QPINN"""
    
    # Random seed
    SEED = 42
    
    # Quantum Circuit Parameters
    N_LAYERS = 5
    N_WIRES = 4
    
    # FNN Basis Parameters
    HIDDEN_LAYERS_FNN = 2
    NEURONS_FNN = 20
    
    # QNN Embedding Parameters
    N_LAYERS_EMBED = 2
    
    # PINN-specific Parameters
    PINN_HIDDEN_LAYERS = 4
    PINN_NEURONS = 50
    
    # Domain Parameters
    T_COLLOC_POINTS = 5
    X_COLLOC_POINTS = 10
    
    # Physics Parameters (Gray-Scott)
    D_U = 2.0e-5    # Diffusion coefficient for u
    D_V = 1.0e-5    # Diffusion coefficient for v
    F = 0.04        # Feed rate
    K = 0.06        # Kill rate
    
    # Boundary conditions (Dirichlet)
    U_BOUNDARY = 1.0
    V_BOUNDARY = 0.0
    
    # Time domain
    T_MIN = 0.0
    T_MAX = 1.0
    
    # Spatial domain
    X_MIN = 0.0
    X_MAX = 1.0
    
    # Training Parameters
    TRAINING_ITERATIONS = 2
    LAMBDA_SCALE = 10.0   # Weight for IC + BC loss
    
    # Output directory
    BASE_DIR = "result"
    OUTPUT_DIR = "result"


# ============================================================
# REFERENCE SOLUTION GENERATOR
# ============================================================

def generate_reference_solution(config, save_path="grayscott_reference_solution.npy"):
    """Generate 1D reference solution using RK45 solver"""
    
    print("=== Generating 1D Gray-Scott Reference Solution ===")
    
    # Spatial domain
    Nx = 400
    x = np.linspace(config.X_MIN, config.X_MAX, Nx)
    dx = x[1] - x[0]
    
    # Time domain
    t_start, t_end = config.T_MIN, config.T_MAX
    Nt = 401
    t_eval = np.linspace(t_start, t_end, Nt)
    
    # Initial conditions from paper
    u0 = np.ones(Nx)
    v0 = np.zeros(Nx)
    mid = Nx // 2
    v0[mid-5:mid+5] = 0.5  # Localized perturbation
    u0 = np.clip(u0, 0.0, None)
    v0 = np.clip(v0, 0.0, None)
    y0 = np.concatenate([u0, v0])
    
    def laplacian_1d(u, dx, bc='dirichlet'):
        d2 = np.zeros_like(u)
        d2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        d2[0] = 0.0
        d2[-1] = 0.0
        return d2
    
    def rd_rhs(t, y):
        """RHS for Gray-Scott system with Dirichlet BC enforcement"""
        u = y[:Nx]
        v = y[Nx:]
        
        # Enforce Dirichlet BCs
        u[0] = config.U_BOUNDARY; u[-1] = config.U_BOUNDARY
        v[0] = config.V_BOUNDARY; v[-1] = config.V_BOUNDARY
        
        lapU = laplacian_1d(u, dx)
        lapV = laplacian_1d(v, dx)
        
        reaction = u * (v ** 2)
        du = config.D_U * lapU - reaction + config.F * (1.0 - u)
        dv = config.D_V * lapV + reaction - (config.K + config.F) * v
        
        # Set du/dt = 0 and dv/dt = 0 at boundaries (Dirichlet BC)
        du[0] = 0.0
        du[-1] = 0.0
        dv[0] = 0.0
        dv[-1] = 0.0
        
        return np.concatenate([du, dv])
    
    print(f"Solving Gray-Scott PDE with RK45...")
    print(f"Parameters: D_U={config.D_U}, D_V={config.D_V}, F={config.F}, K={config.K}")
    
    sol = solve_ivp(rd_rhs, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, 
                    method='RK45', rtol=1e-6, atol=1e-9)
    
    t = sol.t
    u_sol = sol.y[:Nx, :]
    v_sol = sol.y[Nx:, :]
    print("Status:", sol.message)
    
    # Build interpolators
    print("Building interpolators...")
    interpU = RegularGridInterpolator((t, x), u_sol.T, bounds_error=False, fill_value=None)
    interpV = RegularGridInterpolator((t, x), v_sol.T, bounds_error=False, fill_value=None)
    
    print(f"Saving reference solution to '{save_path}'...")
    np.save(save_path, {'u': interpU, 'v': interpV, 't': t, 'x': x, 
                        'u_sol': u_sol, 'v_sol': v_sol}, allow_pickle=True)
    
    print(f"✓ Reference solution generated: u∈[{u_sol.min():.4f}, {u_sol.max():.4f}], "
          f"v∈[{v_sol.min():.4f}, {v_sol.max():.4f}]")
    
    return interpU, interpV


# ============================================================
# PARAMETER COUNTING
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


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_collocation_points_1d(config, save_dir="result"):
    """
    Plot 1: Collocation Points Distribution (1D version)
    Single subplot showing IC, BC, and Interior points with different colors
    Adapted to match rd_2d_qpinn.py visual style
    """
    print("\n=== Plot 1: Collocation Points Distribution ===")
    
    # Create collocation points
    t_points = np.linspace(config.T_MIN, config.T_MAX, config.T_COLLOC_POINTS)
    x_points = np.linspace(config.X_MIN, config.X_MAX, config.X_COLLOC_POINTS)
    
    # 1. Initial Condition Points: All x at t=T_MIN
    t_ic = np.array([config.T_MIN])
    ic_points = np.column_stack([np.full(len(x_points), config.T_MIN), x_points])
    
    # 2. Boundary Condition Points: Random points at x=0 or x=1 for t > T_MIN
    N_bc = config.T_COLLOC_POINTS * config.X_COLLOC_POINTS
    
    bc_t = np.random.uniform(config.T_MIN, config.T_MAX, N_bc)
    bc_x_min = np.column_stack([bc_t[:N_bc//2], np.full(N_bc//2, config.X_MIN)])
    bc_x_max = np.column_stack([bc_t[N_bc//2:], np.full(N_bc - N_bc//2, config.X_MAX)])
    bc_points = np.vstack([bc_x_min, bc_x_max])
    
    # 3. Interior Points: Random sampling in the interior
    full_domain = np.array(list(product(t_points, x_points)))
    ic_mask = full_domain[:, 0] == config.T_MIN
    bc_mask = (
        ((full_domain[:, 1] == config.X_MIN) | (full_domain[:, 1] == config.X_MAX)) &
        (full_domain[:, 0] != config.T_MIN)
    )
    interior_mask = ~(ic_mask | bc_mask)
    interior_points = full_domain[interior_mask]
    
    # Create single subplot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot IC points (red) - matching rd_2d_qpinn.py style
    if len(ic_points) > 0:
        ax.scatter(ic_points[:, 0], ic_points[:, 1], c="r", s=1, alpha=0.6, label="Initial Condition", zorder=3)
    
    # Plot BC points (blue)
    if len(bc_points) > 0:
        ax.scatter(bc_points[:, 0], bc_points[:, 1], c="blue", s=1, alpha=0.3, label="Boundary", zorder=2)
    
    # Plot Interior points (black)
    if len(interior_points) > 0:
        sample_idx = np.random.choice(interior_points.shape[0], min(5000, interior_points.shape[0]), replace=False)
        ax.scatter(interior_points[sample_idx, 0], interior_points[sample_idx, 1], c="black", s=1, alpha=0.1, label="Interior", zorder=1)
    
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title("Collocation Points Distribution (IC, BC, Interior)")
    ax.legend()
    
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


def plot_reference_solution(t, x, u_sol, v_sol, save_dir="result"):
    """
    Plot 2: Reference Solution from RK45
    2D heatmaps (t-x plane) for u and v
    
    Args:
        t: Time array (Nt,)
        x: Space array (Nx,)
        u_sol: Solution array of shape (Nx, Nt)
        v_sol: Solution array of shape (Nx, Nt)
    """
    print("\n" + "="*60)
    print("Generating Plot 2: Reference Solution (RK45)")
    print("="*60)
    
    # Create figure with 1 row × 2 columns (u and v heatmaps)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create meshgrid for heatmaps (T, X both need to match data dimensions)
    # u_sol and v_sol have shape (Nx, Nt)
    T_grid, X_grid = np.meshgrid(t, x)  # This creates (Nx, Nt) grids
    
    # Heatmap 1: u(t, x)
    im1 = axes[0].pcolormesh(T_grid, X_grid, u_sol, shading='auto', cmap='inferno')
    axes[0].set_xlabel('Time (t)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Space (x)', fontsize=12, fontweight='bold')
    axes[0].set_title('u(t,x) - Activator (Reference)', fontsize=13, fontweight='bold')
    cbar1 = fig.colorbar(im1, ax=axes[0], label='u')
    
    # Heatmap 2: v(t, x)
    im2 = axes[1].pcolormesh(T_grid, X_grid, v_sol, shading='auto', cmap='viridis')
    axes[1].set_xlabel('Time (t)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Space (x)', fontsize=12, fontweight='bold')
    axes[1].set_title('v(t,x) - Substrate (Reference)', fontsize=13, fontweight='bold')
    cbar2 = fig.colorbar(im2, ax=axes[1], label='v')
    
    plt.suptitle('Plot 2: Reference Solution (RK45) - 1D Gray-Scott', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'plot2_reference_solution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot 2 saved: {save_path}")
    print(f"  u range: [{u_sol.min():.4f}, {u_sol.max():.4f}]")
    print(f"  v range: [{v_sol.min():.4f}, {v_sol.max():.4f}]")
    print("="*60)


def plot_quantum_circuit(circuit_func, embedding_type, config, save_dir="result"):
    """Plot and save quantum circuit visualization"""
    os.makedirs(save_dir, exist_ok=True)
    
    method_name = {"FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
    
    # Create dummy inputs for visualization
    x_dummy = np.random.rand(2)  # (t, x)
    theta_dummy = np.random.rand(config.N_LAYERS, config.N_WIRES, 3)
    basis_dummy = np.random.rand(config.N_WIRES)
    
    print(f"\n📊 Generating quantum circuit diagram for {method_name}...")
    
    try:
        fig, ax = qml.draw_mpl(circuit_func)(x_dummy, theta_dummy, basis_dummy)
        
        ax.set_title(f'{method_name} Quantum Circuit Architecture\n'
                    f'({config.N_LAYERS} layers, {config.N_WIRES} qubits)', 
                    fontsize=12, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        filename = f"plot3_quantum_circuit_{embedding_type.lower()}.png"
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
                    fontsize=12, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        # Save figure
        filename = "plot3_quantum_circuit_qnn_embedding.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ QNN embedding circuit diagram saved: {save_path}")
        
    except Exception as e:
        print(f"   ⚠ Warning: Could not generate QNN embedding circuit diagram: {e}")


def plot_embedding_results(trainer, save_dir="result"):
    """Plot 4: Embedding Results Visualization (1D adapted)"""
    print("\n=== Plot 4: Embedding Results ===")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create grid for visualization
    t_plot = np.linspace(trainer.T_unique[0], trainer.T_unique[-1], 100)
    x_plot = np.linspace(trainer.X_unique[0], trainer.X_unique[-1], 100)
    
    T_grid, X_grid = np.meshgrid(t_plot, x_plot, indexing='ij')
    
    # Create grid points (t, x)
    grid_points = np.column_stack([T_grid.flatten(), X_grid.flatten()])
    
    # Convert to torch and rescale
    grid_torch = torch.tensor(grid_points, dtype=torch.float32, device=trainer.device)
    
    # Rescale to [-0.95, 0.95] domain for models
    domain_bounds = torch.tensor(
        [[trainer.config.T_MIN, trainer.config.X_MIN],
         [trainer.config.T_MAX, trainer.config.X_MAX]],
        device=trainer.device
    )
    grid_rescaled = 1.9 * (grid_torch - domain_bounds[0]) / (domain_bounds[1] - domain_bounds[0]) - 0.95
    
    # Process embeddings
    fnn_basis_outputs = None
    qnn_basis_outputs = None
    
    if hasattr(trainer, 'basis_net') and trainer.basis_net is not None:
        with torch.no_grad():
            fnn_basis_outputs = trainer.basis_net(grid_rescaled).detach().cpu().numpy()
    
    if hasattr(trainer, 'qnn_embedding') and trainer.qnn_embedding is not None:
        with torch.no_grad():
            qnn_basis_outputs = trainer.qnn_embedding(grid_rescaled).detach().cpu().numpy()
    
    if fnn_basis_outputs is None and qnn_basis_outputs is None:
        print("⚠ Embedding networks not available for visualization")
        return
    
    n_wires = fnn_basis_outputs.shape[1] if fnn_basis_outputs is not None else qnn_basis_outputs.shape[1]
    
    # Plot embeddings (2 rows × n_wires columns)
    fig, axes = plt.subplots(2, n_wires, figsize=(5 * n_wires, 10))
    
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
            axes[0, i].set_title(label, fontsize=10)
            axes[0, i].set_xlabel('t', fontsize=9)
            if i == 0:
                axes[0, i].set_ylabel('FNN-TE-QPINN\nx', fontsize=10)
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
            axes[1, i].set_title(label, fontsize=10)
            axes[1, i].set_xlabel('t', fontsize=9)
            if i == 0:
                axes[1, i].set_ylabel('QNN-TE-QPINN\nx', fontsize=10)
            plt.colorbar(im, ax=axes[1, i])
    
    plt.suptitle(r"Plot 4: Trainable Embedding Functions: $\phi(x) \cdot x$ (Gray-Scott 1D)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/plot4_embedding_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot 4 saved: {save_dir}/plot4_embedding_results.png")


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
    """Quantum Neural Network for embedding generation (dual-circuit approach)"""
    
    def __init__(self, n_wires, n_layers, output_dim, input_dim=2):
        super().__init__()
        self.n_wires = n_wires  # Updated to use n_wires
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        self.weights_embed = nn.Parameter(
            torch.randn(n_layers, n_wires, 3, requires_grad=True)
        )
        
        self.dev = qml.device("default.qubit", wires=n_wires)
        self.qnode_embed = qml.QNode(
            self._circuit_embed,
            self.dev,
            interface="torch",
            diff_method="best"
        )
    
    def _circuit_embed(self, x, weights):
        """Embedding circuit for 1D input (t, x)"""
        for layer in range(self.n_layers):
            for i in range(self.n_wires):
                if i % 2 == 0:
                    qml.RX(x[0], wires=i)  # t
                else:
                    qml.RY(x[1], wires=i)  # x
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

class GrayScott1DQPINNTrainer:
    """Trainer class for Gray-Scott 1D QPINN models"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.domain_bounds = torch.tensor(
            [[config.T_MIN, config.T_MAX], [config.X_MIN, config.X_MAX]],
            device=device, dtype=torch.float32
        )
        self.domain_min = torch.tensor([config.T_MIN, config.X_MIN], device=device)
        self.domain_max = torch.tensor([config.T_MAX, config.X_MAX], device=device)
        
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
        
        # Store unique values for visualization
        self.T_unique = t_torch.cpu().numpy()
        self.X_unique = x_torch.cpu().numpy()
        
        domain = torch.tensor(list(product(t_torch, x_torch)), dtype=torch.float32)
        
        # Initial condition mask (t = 0)
        init_val_mask = domain[:, 0] == self.config.T_MIN
        self.init_val_colloc = domain[init_val_mask].clone().detach().requires_grad_(True).to(self.device)
        
        # Boundary mask (x = 0 or x = 1)
        boundary_mask = (domain[:, 1] == self.config.X_MIN) | (domain[:, 1] == self.config.X_MAX)
        self.boundary_colloc = domain[boundary_mask & ~init_val_mask].clone().detach().requires_grad_(True).to(self.device)
        
        # Interior points
        interior_mask = ~(init_val_mask | boundary_mask)
        self.interior_colloc = domain[interior_mask].clone().detach().requires_grad_(True).to(self.device)
        
        # Full domain
        self.input_domain = domain.clone().detach().requires_grad_(True).to(self.device)
        
        # High-res IC points
        Nx = 400
        x_ic_np = np.linspace(self.config.X_MIN, self.config.X_MAX, Nx)
        self.u0_ic = 1.0 + np.sin(2.0 * np.pi * x_ic_np)
        self.v0_ic = np.ones(Nx) * 3.0
        
        x_ic_torch = torch.tensor(x_ic_np, device=self.device).float().view(-1, 1)
        t_ic_torch = torch.full_like(x_ic_torch, self.config.T_MIN)
        self.ic_points_highres = torch.cat([t_ic_torch, x_ic_torch], dim=1)
        
        print(f"✓ Collocation points: Interior={len(self.interior_colloc)}, "
              f"Boundary={len(self.boundary_colloc)}, IC={len(self.init_val_colloc)}")
    
    def _load_reference_solution(self):
        """Load or generate reference solution"""
        ref_path = os.path.join(self.config.BASE_DIR, "grayscott_reference_solution.npy")
        
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
        ref_u = np.array([self.interp_u([pt[0], pt[1]]).squeeze() for pt in domain_np])
        ref_v = np.array([self.interp_v([pt[0], pt[1]]).squeeze() for pt in domain_np])
        
        self.reference_u = torch.tensor(ref_u, device=self.device, dtype=torch.float32)
        self.reference_v = torch.tensor(ref_v, device=self.device, dtype=torch.float32)
        
        print(f"✓ Reference solution loaded: u∈[{self.reference_u.min():.4f}, {self.reference_u.max():.4f}], "
              f"v∈[{self.reference_v.min():.4f}, {self.reference_v.max():.4f}]")
    
    def _create_circuit(self):
        """Create the main quantum circuit for TE-QPINN"""
        dev = qml.device("default.qubit", wires=self.config.N_WIRES)
        
        @qml.qnode(dev, interface="torch")
        def circuit(x, theta, basis):
            # Tensor encoding with basis
            for i in range(self.config.N_WIRES):
                if i % 2 == 0:
                    qml.RY(basis[i] * x[0], wires=i)  # t
                else:
                    qml.RY(basis[i] * x[1], wires=i)  # x
            
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
        # u in [0, 1], v in [0, 1] (based on reference)
        u_scaled = raw_output[:, 0] * 0.5 + 0.5
        v_scaled = raw_output[:, 1] * 0.5 + 0.5
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
            grad_du_dx = torch.autograd.grad(du_dx.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            d2u_dx2 = grad_du_dx[:, 1]
            
            # v gradients
            grad_v = torch.autograd.grad(v.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            dv_dt = grad_v[:, 0]
            dv_dx = grad_v[:, 1]
            grad_dv_dx = torch.autograd.grad(dv_dx.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            d2v_dx2 = grad_dv_dx[:, 1]
            
            # PDE residuals (Gray-Scott)
            residual_u = du_dt - self.config.D_U * d2u_dx2 + u * (v**2) - self.config.F * (1.0 - u)
            residual_v = dv_dt - self.config.D_V * d2v_dx2 - u * (v**2) + (self.config.F + self.config.K) * v
            
            return torch.mean(residual_u ** 2) + torch.mean(residual_v ** 2)
        
        def initial_condition_loss():
            pred = self.model(self.ic_points_highres)
            u = extract_u(pred)
            v = extract_v(pred)
            
            u_true = torch.tensor(self.u0_ic, device=self.device, dtype=torch.float32)
            v_true = torch.tensor(self.v0_ic, device=self.device, dtype=torch.float32)
            
            return torch.mean((u - u_true) ** 2) + torch.mean((v - v_true) ** 2)
        
        def boundary_loss():
            if len(self.boundary_colloc) == 0:
                return torch.tensor(0.0, device=self.device)
            
            left_mask = self.boundary_colloc[:, 1] == self.config.X_MIN
            right_mask = self.boundary_colloc[:, 1] == self.config.X_MAX
            
            loss = torch.tensor(0.0, device=self.device)
            
            if left_mask.sum() > 0:
                pred_left = self.model(self.boundary_colloc[left_mask])
                u_left = extract_u(pred_left)
                v_left = extract_v(pred_left)
                loss = loss + torch.mean((u_left - self.config.U_BOUNDARY) ** 2)
                loss = loss + torch.mean((v_left - self.config.V_BOUNDARY) ** 2)
            
            if right_mask.sum() > 0:
                pred_right = self.model(self.boundary_colloc[right_mask])
                u_right = extract_u(pred_right)
                v_right = extract_v(pred_right)
                loss = loss + torch.mean((u_right - self.config.U_BOUNDARY) ** 2)
                loss = loss + torch.mean((v_right - self.config.V_BOUNDARY) ** 2)
            
            return loss
        
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
                input_dim=2
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
                self.config.N_WIRES,  # Updated to use N_WIRES
                self.config.N_LAYERS_EMBED,
                self.config.N_WIRES,
                input_dim=2
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
                input_dim=2
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
            grad_du_dx = torch.autograd.grad(du_dx.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            d2u_dx2 = grad_du_dx[:, 1]
            
            # v gradients
            grad_v = torch.autograd.grad(v.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            dv_dt = grad_v[:, 0]
            dv_dx = grad_v[:, 1]
            grad_dv_dx = torch.autograd.grad(dv_dx.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            d2v_dx2 = grad_dv_dx[:, 1]
            
            # Gray-Scott residuals
            residual_u = du_dt - self.config.D_U * d2u_dx2 + u * (v**2) - self.config.F * (1.0 - u)
            residual_v = dv_dt - self.config.D_V * d2v_dx2 - u * (v**2) + (self.config.F + self.config.K) * v
            
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
# VISUALIZATION - Plot 1 and Plot 3 Only
# ============================================================

class TrainingVisualizer:
    """Visualization class for training results (Plot 1 and Plot 3)"""
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.results = trainer.training_results
        self.T_unique = trainer.T_unique
        self.X_unique = trainer.X_unique
        self.reference_u = trainer.reference_u.cpu().numpy()
        self.reference_v = trainer.reference_v.cpu().numpy()
    
    def plot_training_analysis(self, save_dir="result"):
        """Plot 5: Training analysis for each model (2x3 subplots)"""
        print("\n=== Plot 5: Training Analysis for Each Model ===")
        
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
            
            # Column 2: Predictions
            u_pred = results['predictions_u'].reshape(len(self.T_unique), len(self.X_unique))
            v_pred = results['predictions_v'].reshape(len(self.T_unique), len(self.X_unique))
            
            im1 = axes[0,1].contourf(self.T_unique, self.X_unique, u_pred.T, 100, cmap='inferno')
            axes[0,1].set_xlabel('Time t', fontsize=12)
            axes[0,1].set_ylabel('Space x', fontsize=12)
            axes[0,1].set_title(f'u(t,x) Prediction - {method_name}', fontsize=13)
            fig.colorbar(im1, ax=axes[0,1], label='u')
            
            im2 = axes[1,1].contourf(self.T_unique, self.X_unique, v_pred.T, 100, cmap='viridis')
            axes[1,1].set_xlabel('Time t', fontsize=12)
            axes[1,1].set_ylabel('Space x', fontsize=12)
            axes[1,1].set_title(f'v(t,x) Prediction - {method_name}', fontsize=13)
            fig.colorbar(im2, ax=axes[1,1], label='v')
            
            # Column 3: Absolute Errors
            u_ref = self.reference_u.reshape(len(self.T_unique), len(self.X_unique))
            v_ref = self.reference_v.reshape(len(self.T_unique), len(self.X_unique))
            
            u_error = np.abs(u_pred - u_ref)
            v_error = np.abs(v_pred - v_ref)
            
            im3 = axes[0,2].contourf(self.T_unique, self.X_unique, u_error.T, 100, cmap='inferno')
            axes[0,2].set_xlabel('Time t', fontsize=12)
            axes[0,2].set_ylabel('Space x', fontsize=12)
            axes[0,2].set_title(f'|u_pred - u_ref| - {method_name}', fontsize=13)
            fig.colorbar(im3, ax=axes[0,2], label='|Error|')
            
            im4 = axes[1,2].contourf(self.T_unique, self.X_unique, v_error.T, 100, cmap='viridis')
            axes[1,2].set_xlabel('Time t', fontsize=12)
            axes[1,2].set_ylabel('Space x', fontsize=12)
            axes[1,2].set_title(f'|v_pred - v_ref| - {method_name}', fontsize=13)
            fig.colorbar(im4, ax=axes[1,2], label='|Error|')
            
            plt.suptitle(f'Plot 5: {method_name} Training Analysis (Brusselator 1D)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/plot5_training_{embedding_type.lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Plot 5 for {method_name} saved: {save_dir}/plot5_training_{embedding_type.lower()}.png")
    
    def plot_methods_comparison(self, save_dir="result"):
        """Plot 6: Methods comparison (1x3 subplots)"""
        print("\n=== Plot 6: Methods Comparison ===")
        
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Total Loss
        axes[0].semilogy(self.results["NONE"]['loss_history'], 'b-', linewidth=2, label='PINN')
        axes[0].semilogy(self.results["FNN_BASIS"]['loss_history'], 'r--', linewidth=2, label='FNN-TE-QPINN')
        axes[0].semilogy(self.results["QNN"]['loss_history'], 'g-', linewidth=3, label='QNN-TE-QPINN')
        axes[0].set_xlabel('Epoch', fontsize=14)
        axes[0].set_ylabel('Total Loss (log scale)', fontsize=14)
        axes[0].set_title('Total Loss Comparison', fontsize=15, fontweight='bold')
        axes[0].legend(fontsize=12, loc='best')
        axes[0].grid(True, alpha=0.3, which='both')
        
        # Loss u
        axes[1].semilogy(self.results["NONE"]['loss_u_history'], 'b-', linewidth=2, label='PINN')
        axes[1].semilogy(self.results["FNN_BASIS"]['loss_u_history'], 'r--', linewidth=2, label='FNN-TE-QPINN')
        axes[1].semilogy(self.results["QNN"]['loss_u_history'], 'g-', linewidth=3, label='QNN-TE-QPINN')
        axes[1].set_xlabel('Epoch', fontsize=14)
        axes[1].set_ylabel('Loss u (log scale)', fontsize=14)
        axes[1].set_title('Activator u Loss Comparison', fontsize=15, fontweight='bold')
        axes[1].legend(fontsize=12, loc='best')
        axes[1].grid(True, alpha=0.3, which='both')
        
        # Loss v
        axes[2].semilogy(self.results["NONE"]['loss_v_history'], 'b-', linewidth=2, label='PINN')
        axes[2].semilogy(self.results["FNN_BASIS"]['loss_v_history'], 'r--', linewidth=2, label='FNN-TE-QPINN')
        axes[2].semilogy(self.results["QNN"]['loss_v_history'], 'g-', linewidth=3, label='QNN-TE-QPINN')
        axes[2].set_xlabel('Epoch', fontsize=14)
        axes[2].set_ylabel('Loss v (log scale)', fontsize=14)
        axes[2].set_title('Substrate v Loss Comparison', fontsize=15, fontweight='bold')
        axes[2].legend(fontsize=12, loc='best')
        axes[2].grid(True, alpha=0.3, which='both')
        
        plt.suptitle('Plot 6: Three Methods Comparison (PINN vs FNN-TE-QPINN vs QNN-TE-QPINN)',
                     fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/plot6_methods_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot 6 saved: {save_dir}/plot6_methods_comparison.png")
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*70)
        print("TRAINING SUMMARY STATISTICS")
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
    print("GRAY-SCOTT 1D QPINN TRAINING")
    print("="*80)
    print(f"Domain: t ∈ [{config.T_MIN}, {config.T_MAX}], x ∈ [{config.X_MIN}, {config.X_MAX}]")
    print(f"Parameters: D_U={config.D_U}, D_V={config.D_V}, F={config.F}, K={config.K}")
    print(f"Training iterations: {config.TRAINING_ITERATIONS}")
    print("="*80)
    
    # Initialize trainer
    trainer = GrayScott1DQPINNTrainer(config, device)
    
    # === Plot 1: Collocation Points ===
    plot_collocation_points_1d(config, config.OUTPUT_DIR)
    
    # === Plot 2: Reference Solution ===
    ref_path = os.path.join(config.BASE_DIR, "grayscott_reference_solution.npy")
    if os.path.exists(ref_path):
        loaded = np.load(ref_path, allow_pickle=True)[()]
        plot_reference_solution(loaded['t'], loaded['x'], loaded['u_sol'], loaded['v_sol'], config.OUTPUT_DIR)
    
    # Train all models
    for embedding_type in ["NONE", "FNN_BASIS", "QNN"]:
        trainer.train_model(embedding_type, config.TRAINING_ITERATIONS)
        trainer.save_model(embedding_type, config.OUTPUT_DIR)
    
    # === Plot 3 & 4: Quantum Circuits (FNN and QNN) ===
    # Create dummy circuits for visualization
    from scipy.integrate import solve_ivp
    
    # FNN Circuit
    dev_fnn = qml.device("default.qubit", wires=config.N_WIRES)
    
    @qml.qnode(dev_fnn, interface="torch")
    def circuit_fnn(x, theta, basis):
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
        plot_quantum_circuit(circuit_fnn, "FNN_BASIS", config, config.OUTPUT_DIR)
    except Exception as e:
        print(f"⚠ Could not generate FNN circuit plot: {e}")
    
    # QNN Circuit
    dev_qnn = qml.device("default.qubit", wires=config.N_WIRES)
    
    @qml.qnode(dev_qnn, interface="torch")
    def circuit_qnn(x, theta, basis):
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
        plot_quantum_circuit(circuit_qnn, "QNN", config, config.OUTPUT_DIR)
    except Exception as e:
        print(f"⚠ Could not generate QNN circuit plot: {e}")
    
    # Plot QNN embedding circuit
    if hasattr(trainer, 'qnn_embedding') and trainer.qnn_embedding is not None:
        try:
            plot_qnn_embedding_circuit(trainer.qnn_embedding, config, config.OUTPUT_DIR)
        except Exception as e:
            print(f"⚠ Could not generate QNN embedding circuit plot: {e}")
    
    # === Plot 4: Embedding Results ===
    try:
        plot_embedding_results(trainer, config.OUTPUT_DIR)
    except Exception as e:
        print(f"⚠ Could not generate embedding plot: {e}")
    
    # === Generate visualizations (Plot 5 and 6) ===
    visualizer = TrainingVisualizer(trainer)
    visualizer.plot_training_analysis(config.OUTPUT_DIR)
    visualizer.plot_methods_comparison(config.OUTPUT_DIR)
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
    print("Generated Plots:")
    print("  - plot1_collocation_points.png")
    print("  - plot2_reference_solution.png")
    print("  - plot3_quantum_circuit_*.png (FNN and QNN circuits)")
    print("  - plot4_embedding_results.png")
    print("  - plot5_training_*.png (for each model)")
    print("  - plot6_methods_comparison.png")
    print("  - pinn/, fnn_basis/, qnn/ (model checkpoints)")
    print("  - training_summary.json")
    print("="*80)


if __name__ == "__main__":
    main()
