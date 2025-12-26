"""
Lengyel-Epstein 2D QPINN - Training Script
Reaction-Diffusion 2D Spatial + Time Quantum Physics-Informed Neural Network

This script trains three models:
1. PINN (Pure Physics-Informed Neural Network)
2. FNN-TE-QPINN (FNN Basis Temporal Embedding QPINN)
3. QNN-TE-QPINN (Quantum Neural Network Temporal Embedding QPINN)

Domain: t ∈ [0, 1], x ∈ [0, 1], y ∈ [0, 1]
PDE System (Lengyel-Epstein 2D):
    ∂u/∂t = D_u (∂²u/∂x² + ∂²u/∂y²) + a - u - 4uv/(1 + u²)
    ∂v/∂t = D_v (∂²v/∂x² + ∂²v/∂y²) + b(u - uv/(1 + u²))

Initial Conditions:
    u(x, y, 0) = 2.0 + 0.01*sin(2πx)sin(2πy)
    v(x, y, 0) = 5.0

Boundary Conditions (Dirichlet):
    u = 2.0, v = 5.0 on all boundaries

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
        filename = f"plot3_quantum_circuit_{embedding_type.lower()}_training.png"
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
    """Configuration class for Lengyel-Epstein 2D QPINN"""
    
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
    T_COLLOC_POINTS = 10
    X_COLLOC_POINTS = 5
    Y_COLLOC_POINTS = 5
    
    # Physics Parameters (Lengyel-Epstein)
    D_U = 1.0
    D_V = 10.0
    A = 10.0
    B = 1.5
    
    # Boundary conditions (Dirichlet)
    U_BOUNDARY = 2.0
    V_BOUNDARY = 5.0
    
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
    LAMBDA_SCALE = 1e3
    
    # Output directory
    BASE_DIR = "result"
    OUTPUT_DIR = "result"


# ============================================================
# VISUALIZATION
# ============================================================

def plot_collocation_points_2d(config, save_dir="result"):
    """Plot 1: Collocation Points Distribution"""
    print("\n=== Plot 1: Collocation Points Distribution ===")
    
    t_points = np.linspace(config.T_MIN, config.T_MAX, config.T_COLLOC_POINTS)
    x_points = np.linspace(config.X_MIN, config.X_MAX, config.X_COLLOC_POINTS)
    y_points = np.linspace(config.Y_MIN, config.Y_MAX, config.Y_COLLOC_POINTS)
    
    t_ic = np.array([config.T_MIN])
    ic_points = np.array(list(product(t_ic, x_points, y_points)))
    
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
    
    full_domain = np.array(list(product(t_points, x_points, y_points)))
    ic_mask = full_domain[:, 0] == config.T_MIN
    bc_mask = (
        ((full_domain[:, 1] == config.X_MIN) | (full_domain[:, 1] == config.X_MAX) |
         (full_domain[:, 2] == config.Y_MIN) | (full_domain[:, 2] == config.Y_MAX)) &
        (full_domain[:, 0] != config.T_MIN)
    )
    interior_mask = ~(ic_mask | bc_mask)
    interior_points = full_domain[interior_mask]
    
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131)
    if len(ic_points) > 0:
        ax1.scatter(ic_points[:, 1], ic_points[:, 2], c="r", s=1, alpha=0.6, label="Initial Condition")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Initial Condition Points (t=0)")
    ax1.legend()
    
    ax2 = fig.add_subplot(132)
    if len(bc_points) > 0:
        ax2.scatter(bc_points[:, 1], bc_points[:, 2], c="blue", s=1, alpha=0.3, label="Boundary")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Boundary Points (t>0)")
    ax2.legend()
    
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


def plot_reference_solution(t, x, y, u_sol, v_sol, save_dir="result"):
    """Plot 2: Reference Solution from RK45"""
    print("\n=== Plot 2: Reference Solution (RK45) ===")
    n_times = 5
    time_indices = np.linspace(0, len(t)-1, n_times, dtype=int)
    fig, axes = plt.subplots(2, n_times, figsize=(20, 8))
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    for col, t_idx in enumerate(time_indices):
        current_time = t[t_idx]
        u_slice = u_sol[:, :, t_idx]
        im1 = axes[0, col].contourf(X, Y, u_slice, levels=20, cmap='inferno')
        axes[0, col].set_title(f't = {current_time:.3f}', fontsize=10)
        axes[0, col].set_xlabel('x')
        if col == 0:
            axes[0, col].set_ylabel('u (Activator)', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=axes[0, col])
        
        v_slice = v_sol[:, :, t_idx]
        im2 = axes[1, col].contourf(X, Y, v_slice, levels=20, cmap='viridis')
        axes[1, col].set_title(f't = {current_time:.3f}', fontsize=10)
        axes[1, col].set_xlabel('x')
        if col == 0:
            axes[1, col].set_ylabel('v (Inhibitor)', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=axes[1, col])
    
    plt.suptitle('Plot 2: Reference Solution (RK45) - 2D Lengyel-Epstein', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'plot2_reference_solution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot 2 saved: {save_path}")


# ============================================================
# REFERENCE SOLUTION GENERATOR
# ============================================================

def generate_reference_solution(config, save_path="lengyel_epstein_3d_reference_solution.npy"):
    """Generate 2D spatial reference solution using RK45 solver"""
    print("=== Generating 3D (2D spatial + time) Reference Solution ===")
    Nx, Ny = 64, 64
    x = np.linspace(config.X_MIN, config.X_MAX, Nx)
    y = np.linspace(config.Y_MIN, config.Y_MAX, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    Nt = 101
    t_eval = np.linspace(config.T_MIN, config.T_MAX, Nt)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    u0 = 2.0 + 0.01 * np.sin(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)
    v0 = np.ones_like(X) * 5.0
    y0 = np.concatenate([u0.ravel(), v0.ravel()])
    
    def laplacian_2d(u, dx, dy):
        d2 = np.zeros_like(u)
        d2[1:-1, 1:-1] = (
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
            (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
        )
        return d2
    
    def rd_rhs_3d(t, y_flat):
        u = y_flat[:Nx*Ny].reshape(Nx, Ny)
        v = y_flat[Nx*Ny:].reshape(Nx, Ny)
        
        u[0, :], u[-1, :], u[:, 0], u[:, -1] = config.U_BOUNDARY, config.U_BOUNDARY, config.U_BOUNDARY, config.U_BOUNDARY
        v[0, :], v[-1, :], v[:, 0], v[:, -1] = config.V_BOUNDARY, config.V_BOUNDARY, config.V_BOUNDARY, config.V_BOUNDARY
        
        lapU = laplacian_2d(u, dx, dy)
        lapV = laplacian_2d(v, dx, dy)
        
        denom = 1.0 + u**2
        du = config.D_U * lapU + config.A - u - (4.0 * u * v) / denom
        dv = config.D_V * lapV + config.B * (u - (u * v) / denom)
        
        du[0, :], du[-1, :], du[:, 0], du[:, -1] = 0, 0, 0, 0
        dv[0, :], dv[-1, :], dv[:, 0], dv[:, -1] = 0, 0, 0, 0
        return np.concatenate([du.ravel(), dv.ravel()])
    
    print(f"Solving 2D Lengyel-Epstein PDE with RK45...")
    sol = solve_ivp(rd_rhs_3d, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)
    u_sol = sol.y[:Nx*Ny, :].reshape(Nx, Ny, -1)
    v_sol = sol.y[Nx*Ny:, :].reshape(Nx, Ny, -1)
    
    interpU = RegularGridInterpolator((x, y, sol.t), u_sol, bounds_error=False, fill_value=None)
    interpV = RegularGridInterpolator((x, y, sol.t), v_sol, bounds_error=False, fill_value=None)
    
    np.save(save_path, {'u': interpU, 'v': interpV, 't': sol.t, 'x': x, 'y': y, 'u_sol': u_sol, 'v_sol': v_sol}, allow_pickle=True)
    print(f"✓ Reference solution generated: u∈[{u_sol.min():.4f}, {u_sol.max():.4f}], v∈[{v_sol.min():.4f}, {v_sol.max():.4f}]")
    return interpU, interpV


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
# TRAINER CLASS
# ============================================================

class LengyelEpstein2DQPINNTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.domain_min = torch.tensor([config.T_MIN, config.X_MIN, config.Y_MIN], device=device)
        self.domain_max = torch.tensor([config.T_MAX, config.X_MAX, config.Y_MAX], device=device)
        self._setup_collocation_points()
        self._load_reference_solution()
        self.training_results = {}
    
    def _setup_collocation_points(self):
        t_torch = torch.linspace(self.config.T_MIN, self.config.T_MAX, self.config.T_COLLOC_POINTS)
        x_torch = torch.linspace(self.config.X_MIN, self.config.X_MAX, self.config.X_COLLOC_POINTS)
        y_torch = torch.linspace(self.config.Y_MIN, self.config.Y_MAX, self.config.Y_COLLOC_POINTS)
        self.T_unique, self.X_unique, self.Y_unique = t_torch.numpy(), x_torch.numpy(), y_torch.numpy()
        
        domain = torch.tensor(list(product(t_torch, x_torch, y_torch)), dtype=torch.float32)
        init_val_mask = domain[:, 0] == self.config.T_MIN
        self.init_val_colloc = domain[init_val_mask].clone().detach().requires_grad_(True).to(self.device)
        
        boundary_mask = ((domain[:, 1] == self.config.X_MIN) | (domain[:, 1] == self.config.X_MAX) |
                         (domain[:, 2] == self.config.Y_MIN) | (domain[:, 2] == self.config.Y_MAX))
        self.boundary_colloc = domain[boundary_mask & ~init_val_mask].clone().detach().requires_grad_(True).to(self.device)
        
        interior_mask = ~(init_val_mask | boundary_mask)
        self.interior_colloc = domain[interior_mask].clone().detach().requires_grad_(True).to(self.device)
        self.input_domain = domain.clone().detach().requires_grad_(True).to(self.device)
        
        Nx, Ny = 64, 64
        x_ic = np.linspace(self.config.X_MIN, self.config.X_MAX, Nx)
        y_ic = np.linspace(self.config.Y_MIN, self.config.Y_MAX, Ny)
        X_ic, Y_ic = np.meshgrid(x_ic, y_ic, indexing='ij')
        self.u0_ic = 2.0 + 0.01 * np.sin(2.0 * np.pi * X_ic) * np.sin(2.0 * np.pi * Y_ic)
        self.v0_ic = np.ones_like(X_ic) * 5.0
        
        t_ic = torch.full((Nx*Ny, 1), self.config.T_MIN, device=self.device)
        x_ic_t = torch.tensor(X_ic.ravel(), device=self.device).float().view(-1, 1)
        y_ic_t = torch.tensor(Y_ic.ravel(), device=self.device).float().view(-1, 1)
        self.ic_points_highres = torch.cat([t_ic, x_ic_t, y_ic_t], dim=1)
    
    def _load_reference_solution(self):
        ref_path = os.path.join(self.config.BASE_DIR, "lengyel_epstein_3d_reference_solution.npy")
        if os.path.exists(ref_path):
            loaded = np.load(ref_path, allow_pickle=True)[()]
            self.interp_u, self.interp_v = loaded['u'], loaded['v']
        else:
            os.makedirs(self.config.BASE_DIR, exist_ok=True)
            self.interp_u, self.interp_v = generate_reference_solution(self.config, ref_path)
        
        domain_np = self.input_domain.detach().cpu().numpy()
        ref_u = np.array([self.interp_u([pt[1], pt[2], pt[0]]).squeeze() for pt in domain_np])
        ref_v = np.array([self.interp_v([pt[1], pt[2], pt[0]]).squeeze() for pt in domain_np])
        self.reference_u = torch.tensor(ref_u, device=self.device, dtype=torch.float32)
        self.reference_v = torch.tensor(ref_v, device=self.device, dtype=torch.float32)
    
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
    
    def _create_loss_functions(self):
        def pde_loss():
            pred = self.model(self.interior_colloc)
            u, v = pred[:, 0], pred[:, 1]
            
            grad_u = torch.autograd.grad(u.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            du_dt, du_dx, du_dy = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2]
            d2u_dx2 = torch.autograd.grad(du_dx.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0][:, 1]
            d2u_dy2 = torch.autograd.grad(du_dy.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0][:, 2]
            
            grad_v = torch.autograd.grad(v.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            dv_dt, dv_dx, dv_dy = grad_v[:, 0], grad_v[:, 1], grad_v[:, 2]
            d2v_dx2 = torch.autograd.grad(dv_dx.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0][:, 1]
            d2v_dy2 = torch.autograd.grad(dv_dy.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0][:, 2]
            
            denom = 1.0 + u**2
            res_u = du_dt - self.config.D_U * (d2u_dx2 + d2u_dy2) - (self.config.A - u - (4.0 * u * v) / denom)
            res_v = dv_dt - self.config.D_V * (d2v_dx2 + d2v_dy2) - self.config.B * (u - (u * v) / denom)
            return torch.mean(res_u**2) + torch.mean(res_v**2)
        
        def ic_loss():
            pred = self.model(self.ic_points_highres)
            u_true = torch.tensor(self.u0_ic.ravel(), device=self.device, dtype=torch.float32)
            v_true = torch.tensor(self.v0_ic.ravel(), device=self.device, dtype=torch.float32)
            return torch.mean((pred[:, 0] - u_true)**2) + torch.mean((pred[:, 1] - v_true)**2)
        
        def bc_loss():
            pred = self.model(self.boundary_colloc)
            return torch.mean((pred[:, 0] - self.config.U_BOUNDARY)**2) + torch.mean((pred[:, 1] - self.config.V_BOUNDARY)**2)
        
        def total_loss():
            return pde_loss() + self.config.LAMBDA_SCALE * (ic_loss() + bc_loss())
        
        def compute_metrics():
            pred = self.model(self.input_domain)
            u, v = pred[:, 0], pred[:, 1]
            mse_u, mse_v = torch.mean((u - self.reference_u)**2).item(), torch.mean((v - self.reference_v)**2).item()
            linf_u, linf_v = torch.max(torch.abs(u - self.reference_u)).item(), torch.max(torch.abs(v - self.reference_v)).item()
            return {'mse_u': mse_u, 'mse_v': mse_v, 'mse_total': mse_u + mse_v, 'linf_u': linf_u, 'linf_v': linf_v, 'linf_max': max(linf_u, linf_v)}
        
        return total_loss, compute_metrics
    
    def train_model(self, embedding_type, iterations):
        self.embedding_type = embedding_type
        method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
        print(f"\nTRAINING: {method_name}")
        
        if embedding_type == "FNN_BASIS":
            self.theta = torch.rand(self.config.N_LAYERS, self.config.N_WIRES, 3, device=self.device, requires_grad=True)
            self.basis_net = FNNBasisNet(self.config.HIDDEN_LAYERS_FNN, self.config.NEURONS_FNN, self.config.N_WIRES).to(self.device)
            self.circuit = self._create_circuit()
            params = [self.theta] + list(self.basis_net.parameters())
        elif embedding_type == "QNN":
            self.theta = torch.rand(self.config.N_LAYERS, self.config.N_WIRES, 3, device=self.device, requires_grad=True)
            self.qnn_embedding = QNNEmbedding(self.config.N_WIRES, self.config.N_LAYERS_EMBED, self.config.N_WIRES).to(self.device)
            self.circuit = self._create_circuit()
            params = [self.theta] + list(self.qnn_embedding.parameters())
        else:
            self.pinn = FNNBasisNet(self.config.HIDDEN_LAYERS_FNN, self.config.NEURONS_FNN, 2).to(self.device)
            params = list(self.pinn.parameters())
        
        optimizer = torch.optim.LBFGS(params, line_search_fn="strong_wolfe")
        total_loss_fn, compute_metrics_fn = self._create_loss_functions()
        
        def closure():
            optimizer.zero_grad(); loss = total_loss_fn(); loss.backward(); return loss
        
        loss_history, mse_u_history, mse_v_history = [], [], []
        start_time = time.time()
        for epoch in range(iterations):
            optimizer.step(closure)
            metrics = compute_metrics_fn()
            loss_history.append(total_loss_fn().item())
            mse_u_history.append(metrics['mse_u']); mse_v_history.append(metrics['mse_v'])
            if epoch % 10 == 0 or epoch == iterations - 1:
                print(f"Epoch {epoch:04d} | Loss: {loss_history[-1]:.2E} | MSE_u: {metrics['mse_u']:.2E} | MSE_v: {metrics['mse_v']:.2E}")
        
        training_time = time.time() - start_time
        with torch.no_grad():
            final_pred = self.model(self.input_domain)
            predictions_u, predictions_v = final_pred[:, 0].cpu().numpy(), final_pred[:, 1].cpu().numpy()
        
        self.training_results[embedding_type] = {
            'loss_history': loss_history, 'mse_u_history': mse_u_history, 'mse_v_history': mse_v_history,
            'predictions_u': predictions_u, 'predictions_v': predictions_v,
            'final_loss': loss_history[-1], 'final_metrics': compute_metrics_fn(), 'training_time': training_time
        }
    
    def save_model(self, embedding_type, save_dir):
        folder_name = {"NONE": "pinn", "FNN_BASIS": "fnn_basis", "QNN": "qnn"}[embedding_type]
        model_dir = os.path.join(save_dir, folder_name)
        os.makedirs(model_dir, exist_ok=True)
        if embedding_type == "FNN_BASIS":
            np.save(os.path.join(model_dir, 'model.npy'), {'theta': self.theta.detach().cpu().numpy(), 'basis_net': self.basis_net.state_dict()}, allow_pickle=True)
        elif embedding_type == "QNN":
            np.save(os.path.join(model_dir, 'model.npy'), {'theta': self.theta.detach().cpu().numpy(), 'qnn_embedding': self.qnn_embedding.state_dict()}, allow_pickle=True)
        else:
            torch.save(self.pinn.state_dict(), os.path.join(model_dir, 'model.pth'))
        
        with open(os.path.join(model_dir, 'training_results.json'), 'w') as f:
            json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in self.training_results[embedding_type].items() if k not in ['predictions_u', 'predictions_v']}, f, indent=2)


# ============================================================
# VISUALIZATION CLASS
# ============================================================

class TrainingVisualizer:
    def __init__(self, trainer):
        self.trainer = trainer
        self.results = trainer.training_results
        self.T_unique, self.X_unique, self.Y_unique = trainer.T_unique, trainer.X_unique, trainer.Y_unique
        self.reference_u = trainer.reference_u.cpu().numpy()
        self.reference_v = trainer.reference_v.cpu().numpy()
    
    def plot_training_analysis(self, save_dir="result"):
        for embedding_type in ["NONE", "FNN_BASIS", "QNN"]:
            method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
            res = self.results[embedding_type]
            fig, axes = plt.subplots(2, 3, figsize=(24, 16))
            axes[0,0].semilogy(res['loss_history'], label='Total Loss'); axes[0,0].set_title(f'{method_name} Loss'); axes[0,0].legend()
            axes[1,0].semilogy(res['mse_u_history'], label='MSE u'); axes[1,0].semilogy(res['mse_v_history'], label='MSE v'); axes[1,0].set_title(f'{method_name} MSE'); axes[1,0].legend()
            
            u_pred = res['predictions_u'].reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
            v_pred = res['predictions_v'].reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
            t_mid = len(self.T_unique) // 2
            X_grid, Y_grid = np.meshgrid(self.X_unique, self.Y_unique, indexing='ij')
            
            im1 = axes[0,1].contourf(X_grid, Y_grid, u_pred[t_mid], 50, cmap='inferno'); fig.colorbar(im1, ax=axes[0,1]); axes[0,1].set_title(f'u pred t={self.T_unique[t_mid]:.2f}')
            im2 = axes[1,1].contourf(X_grid, Y_grid, v_pred[t_mid], 50, cmap='viridis'); fig.colorbar(im2, ax=axes[1,1]); axes[1,1].set_title(f'v pred t={self.T_unique[t_mid]:.2f}')
            
            u_ref = self.reference_u.reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
            v_ref = self.reference_v.reshape(len(self.T_unique), len(self.X_unique), len(self.Y_unique))
            im3 = axes[0,2].contourf(X_grid, Y_grid, np.abs(u_pred[t_mid] - u_ref[t_mid]), 50, cmap='inferno'); fig.colorbar(im3, ax=axes[0,2]); axes[0,2].set_title('u Error')
            im4 = axes[1,2].contourf(X_grid, Y_grid, np.abs(v_pred[t_mid] - v_ref[t_mid]), 50, cmap='viridis'); fig.colorbar(im4, ax=axes[1,2]); axes[1,2].set_title('v Error')
            
            plt.tight_layout(); plt.savefig(f'{save_dir}/plot5_training_{embedding_type.lower()}.png'); plt.close()

    def plot_methods_comparison(self, save_dir="result"):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for m, c in zip(["NONE", "FNN_BASIS", "QNN"], ['b', 'r', 'g']):
            ax.semilogy(self.results[m]['loss_history'], color=c, label={"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[m])
        ax.set_title('Methods Comparison - Total Loss'); ax.legend(); plt.savefig(f'{save_dir}/plot6_methods_comparison.png'); plt.close()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    ref_path = os.path.join(config.BASE_DIR, "lengyel_epstein_3d_reference_solution.npy")
    if not os.path.exists(ref_path): generate_reference_solution(config, ref_path)
    ref_data = np.load(ref_path, allow_pickle=True).item()
    
    plot_collocation_points_2d(config, config.OUTPUT_DIR)
    plot_reference_solution(ref_data['t'], ref_data['x'], ref_data['y'], ref_data['u_sol'], ref_data['v_sol'], config.OUTPUT_DIR)
    
    trainer = LengyelEpstein2DQPINNTrainer(config, device)
    for et in ["NONE", "FNN_BASIS", "QNN"]:
        trainer.train_model(et, config.TRAINING_ITERATIONS)
        trainer.save_model(et, config.OUTPUT_DIR)
    
    visualizer = TrainingVisualizer(trainer)
    visualizer.plot_training_analysis(config.OUTPUT_DIR)
    visualizer.plot_methods_comparison(config.OUTPUT_DIR)
    
    summary = {m: {k: v for k, v in trainer.training_results[m].items() if k not in ['predictions_u', 'predictions_v']} for m in trainer.training_results}
    with open(os.path.join(config.OUTPUT_DIR, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else float(x))
    print(f"\nTraining complete. Results in {config.OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
