"""
Lengyel-Epstein 1D QPINN - Training Script
Reaction-Diffusion 1D Quantum Physics-Informed Neural Network

This script trains three models:
1. PINN (Pure Physics-Informed Neural Network)
2. FNN-TE-QPINN (FNN Basis Temporal Embedding QPINN)
3. QNN-TE-QPINN (Quantum Neural Network Temporal Embedding QPINN)

Domain: t ∈ [0, 1], x ∈ [0, 1]
PDE System (Lengyel-Epstein):
    ∂u/∂t = D_u ∂²u/∂x² + a - u - 4uv/(1 + u²)
    ∂v/∂t = D_v ∂²v/∂x² + b(u - uv/(1 + u²))

Initial Conditions:
    u(x, 0) = 2.0 + 0.1*sin(2πx)
    v(x, 0) = 5.0 + 0.1*cos(2πx)

Boundary Conditions (Dirichlet):
    u(0, t) = u(1, t) = 2.0
    v(0, t) = v(1, t) = 5.0

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
    """Configuration class for Lengyel-Epstein 1D QPINN"""
    
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
    
    # Domain Parameters
    T_COLLOC_POINTS = 5
    X_COLLOC_POINTS = 10
    
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
    
    # Training Parameters
    TRAINING_ITERATIONS = 2
    LAMBDA_SCALE = 10.0   # Weight for IC + BC loss
    
    # Output directory
    BASE_DIR = "result"
    OUTPUT_DIR = "result"


# ============================================================
# REFERENCE SOLUTION GENERATOR
# ============================================================

def generate_reference_solution(config, save_path="lengyel_epstein_reference_solution.npy"):
    """Generate 1D reference solution using RK45 solver"""
    
    print("=== Generating 1D Lengyel-Epstein Reference Solution ===")
    
    # Spatial domain
    Nx = 400
    x = np.linspace(config.X_MIN, config.X_MAX, Nx)
    dx = x[1] - x[0]
    
    # Time domain
    t_start, t_end = config.T_MIN, config.T_MAX
    Nt = 401
    t_eval = np.linspace(t_start, t_end, Nt)
    
    # Initial conditions
    u0 = config.U_BOUNDARY + 0.1 * np.sin(2 * np.pi * x)
    v0 = config.V_BOUNDARY + 0.1 * np.cos(2 * np.pi * x)
    y0 = np.concatenate([u0, v0])
    
    def laplacian_1d(u, dx):
        d2 = np.zeros_like(u)
        d2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        # Dirichlet BC: derivatives at boundaries are handled by the solver
        return d2
    
    def rd_rhs(t, y):
        u = y[:Nx]
        v = y[Nx:]
        
        # Enforce Dirichlet BC
        u[0] = config.U_BOUNDARY
        u[-1] = config.U_BOUNDARY
        v[0] = config.V_BOUNDARY
        v[-1] = config.V_BOUNDARY
        
        lapU = laplacian_1d(u, dx)
        lapV = laplacian_1d(v, dx)
        
        denom = 1.0 + u**2
        reaction_u = config.A - u - (4.0 * u * v) / denom
        reaction_v = config.B * (u - (u * v) / denom)
        
        du = config.D_U * lapU + reaction_u
        dv = config.D_V * lapV + reaction_v
        
        # Dirichlet BC: du/dt = 0 at boundaries
        du[0] = 0.0
        du[-1] = 0.0
        dv[0] = 0.0
        dv[-1] = 0.0
        
        return np.concatenate([du, dv])
    
    print(f"Solving Lengyel-Epstein PDE with RK45...")
    sol = solve_ivp(rd_rhs, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, 
                    method='RK45', rtol=1e-6, atol=1e-9)
    
    t = sol.t
    u_sol = sol.y[:Nx, :]
    v_sol = sol.y[Nx:, :]
    
    # Build interpolators
    interpU = RegularGridInterpolator((t, x), u_sol.T, bounds_error=False, fill_value=None)
    interpV = RegularGridInterpolator((t, x), v_sol.T, bounds_error=False, fill_value=None)
    
    np.save(save_path, {'u': interpU, 'v': interpV, 't': t, 'x': x, 
                        'u_sol': u_sol, 'v_sol': v_sol}, allow_pickle=True)
    
    print(f"✓ Reference solution generated: u∈[{u_sol.min():.4f}, {u_sol.max():.4f}], "
          f"v∈[{v_sol.min():.4f}, {v_sol.max():.4f}]")
    
    return interpU, interpV


# ============================================================
# PARAMETER COUNTING
# ============================================================

def count_parameters(model_components):
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


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_collocation_points_1d(config, save_dir="result"):
    print("\n=== Plot 1: Collocation Points Distribution ===")
    t_points = np.linspace(config.T_MIN, config.T_MAX, config.T_COLLOC_POINTS)
    x_points = np.linspace(config.X_MIN, config.X_MAX, config.X_COLLOC_POINTS)
    
    ic_points = np.column_stack([np.full(len(x_points), config.T_MIN), x_points])
    N_bc = config.T_COLLOC_POINTS * config.X_COLLOC_POINTS
    bc_t = np.random.uniform(config.T_MIN, config.T_MAX, N_bc)
    bc_x_min = np.column_stack([bc_t[:N_bc//2], np.full(N_bc//2, config.X_MIN)])
    bc_x_max = np.column_stack([bc_t[N_bc//2:], np.full(N_bc - N_bc//2, config.X_MAX)])
    bc_points = np.vstack([bc_x_min, bc_x_max])
    
    full_domain = np.array(list(product(t_points, x_points)))
    ic_mask = full_domain[:, 0] == config.T_MIN
    bc_mask = ((full_domain[:, 1] == config.X_MIN) | (full_domain[:, 1] == config.X_MAX)) & (full_domain[:, 0] != config.T_MIN)
    interior_points = full_domain[~(ic_mask | bc_mask)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(ic_points[:, 0], ic_points[:, 1], c="r", s=1, alpha=0.6, label="Initial Condition")
    ax.scatter(bc_points[:, 0], bc_points[:, 1], c="blue", s=1, alpha=0.3, label="Boundary")
    ax.scatter(interior_points[:, 0], interior_points[:, 1], c="black", s=1, alpha=0.1, label="Interior")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title("Collocation Points Distribution")
    ax.legend()
    plt.savefig(os.path.join(save_dir, 'plot1_collocation_points.png'), dpi=150)
    plt.close()


def plot_reference_solution(t, x, u_sol, v_sol, save_dir="result"):
    print("\n=== Plot 2: Reference Solution (RK45) ===")
    T, X = np.meshgrid(t, x)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    im1 = axes[0].pcolormesh(T, X, u_sol, cmap='inferno', shading='auto')
    axes[0].set_title("u(x, t) Reference")
    fig.colorbar(im1, ax=axes[0])
    im2 = axes[1].pcolormesh(T, X, v_sol, cmap='viridis', shading='auto')
    axes[1].set_title("v(x, t) Reference")
    fig.colorbar(im2, ax=axes[1])
    plt.savefig(os.path.join(save_dir, 'plot2_reference_solution.png'), dpi=150)
    plt.close()


# ============================================================
# NEURAL NETWORK MODELS
# ============================================================

class FNNBasisNet(nn.Module):
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
    def __init__(self, n_wires, n_layers, output_dim, input_dim=2):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.weights_embed = nn.Parameter(torch.randn(n_layers, n_wires, 3, requires_grad=True))
        self.dev = qml.device("default.qubit", wires=n_wires)
        self.qnode_embed = qml.QNode(self._circuit_embed, self.dev, interface="torch")
    
    def _circuit_embed(self, x, weights):
        for layer in range(self.n_layers):
            for i in range(self.n_wires):
                if i % 2 == 0: qml.RX(x[0], wires=i)
                else: qml.RY(x[1], wires=i)
            for i in range(self.n_wires):
                qml.RX(weights[layer, i, 0], wires=i)
                qml.RY(weights[layer, i, 1], wires=i)
                qml.RZ(weights[layer, i, 2], wires=i)
            if self.n_wires > 1:
                for i in range(self.n_wires - 1): qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.output_dim)]
    
    def forward(self, x):
        basis_t = self.qnode_embed(x.T, self.weights_embed)
        basis_t = (torch.stack(basis_t) if isinstance(basis_t, list) else basis_t) * torch.pi
        return basis_t.T


# ============================================================
# TRAINER CLASS
# ============================================================

class LengyelEpstein1DQPINNTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.domain_min = torch.tensor([config.T_MIN, config.X_MIN], device=device)
        self.domain_max = torch.tensor([config.T_MAX, config.X_MAX], device=device)
        self._setup_collocation_points()
        self._load_reference()
        self.training_results = {}

    def _setup_collocation_points(self):
        t_points = np.linspace(self.config.T_MIN, self.config.T_MAX, self.config.T_COLLOC_POINTS)
        x_points = np.linspace(self.config.X_MIN, self.config.X_MAX, self.config.X_COLLOC_POINTS)
        
        # Interior
        full_grid = np.array(list(product(t_points, x_points)))
        ic_mask = full_grid[:, 0] == self.config.T_MIN
        bc_mask = (full_grid[:, 1] == self.config.X_MIN) | (full_grid[:, 1] == self.config.X_MAX)
        self.interior_colloc = torch.tensor(full_grid[~(ic_mask | bc_mask)], dtype=torch.float32, device=self.device, requires_grad=True)
        
        # IC
        self.ic_points = torch.tensor(full_grid[ic_mask], dtype=torch.float32, device=self.device)
        self.u0_ic = self.config.U_BOUNDARY + 0.1 * np.sin(2 * np.pi * self.ic_points[:, 1].cpu().numpy())
        self.v0_ic = self.config.V_BOUNDARY + 0.1 * np.cos(2 * np.pi * self.ic_points[:, 1].cpu().numpy())
        
        # BC
        self.boundary_colloc = torch.tensor(full_grid[bc_mask & ~ic_mask], dtype=torch.float32, device=self.device)
        
        # Full domain for metrics
        self.input_domain = torch.tensor(full_grid, dtype=torch.float32, device=self.device)

    def _load_reference(self):
        ref_path = os.path.join(self.config.BASE_DIR, "lengyel_epstein_reference_solution.npy")
        if os.path.exists(ref_path):
            loaded = np.load(ref_path, allow_pickle=True)[()]
            self.interp_u, self.interp_v = loaded['u'], loaded['v']
        else:
            os.makedirs(self.config.BASE_DIR, exist_ok=True)
            self.interp_u, self.interp_v = generate_reference_solution(self.config, ref_path)
        
        domain_np = self.input_domain.cpu().numpy()
        self.reference_u = torch.tensor([self.interp_u([pt[0], pt[1]]) for pt in domain_np], device=self.device, dtype=torch.float32).squeeze()
        self.reference_v = torch.tensor([self.interp_v([pt[0], pt[1]]) for pt in domain_np], device=self.device, dtype=torch.float32).squeeze()

    def _create_circuit(self):
        dev = qml.device("default.qubit", wires=self.config.N_WIRES)
        @qml.qnode(dev, interface="torch")
        def circuit(x, theta, basis):
            for i in range(self.config.N_WIRES):
                qml.RY(basis[i] * (x[0] if i % 2 == 0 else x[1]), wires=i)
            for layer in range(self.config.N_LAYERS):
                for q in range(self.config.N_WIRES):
                    qml.RX(theta[layer, q, 0], wires=q)
                    qml.RY(theta[layer, q, 1], wires=q)
                    qml.RZ(theta[layer, q, 2], wires=q)
                for q in range(self.config.N_WIRES - 1): qml.CNOT(wires=[q, q + 1])
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
        return circuit

    def _postprocess_output(self, raw_output):
        return raw_output * 5.0 + 5.0

    def model(self, x):
        x_rescaled = 2.0 * (x - self.domain_min) / (self.domain_max - self.domain_min) - 1.0
        if self.embedding_type == "FNN_BASIS":
            basis = self.basis_net(x_rescaled)
            raw = self.circuit(x_rescaled.T, self.theta, basis.T)
            return self._postprocess_output(torch.stack(raw).T)
        elif self.embedding_type == "QNN":
            basis = self.qnn_embedding(x_rescaled)
            raw = self.circuit(x_rescaled.T, self.theta, basis.T)
            return self._postprocess_output(torch.stack(raw).T)
        else:
            return self.pinn(x_rescaled)

    def _create_loss_functions(self):
        def pde_loss():
            pred = self.model(self.interior_colloc)
            u, v = pred[:, 0], pred[:, 1]
            grad_u = torch.autograd.grad(u.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            du_dt, du_dx = grad_u[:, 0], grad_u[:, 1]
            d2u_dx2 = torch.autograd.grad(du_dx.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0][:, 1]
            
            grad_v = torch.autograd.grad(v.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            dv_dt, dv_dx = grad_v[:, 0], grad_v[:, 1]
            d2v_dx2 = torch.autograd.grad(dv_dx.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0][:, 1]
            
            denom = 1.0 + u**2
            res_u = du_dt - self.config.D_U * d2u_dx2 - (self.config.A - u - (4.0 * u * v) / denom)
            res_v = dv_dt - self.config.D_V * d2v_dx2 - self.config.B * (u - (u * v) / denom)
            return torch.mean(res_u**2) + torch.mean(res_v**2)

        def ic_loss():
            pred = self.model(self.ic_points)
            return torch.mean((pred[:, 0] - torch.tensor(self.u0_ic, device=self.device))**2) + \
                   torch.mean((pred[:, 1] - torch.tensor(self.v0_ic, device=self.device))**2)

        def bc_loss():
            pred = self.model(self.boundary_colloc)
            return torch.mean((pred[:, 0] - self.config.U_BOUNDARY)**2) + \
                   torch.mean((pred[:, 1] - self.config.V_BOUNDARY)**2)

        def total_loss():
            return pde_loss() + self.config.LAMBDA_SCALE * (ic_loss() + bc_loss())

        def compute_metrics():
            pred = self.model(self.input_domain)
            mse_u = torch.mean((pred[:, 0] - self.reference_u)**2).item()
            mse_v = torch.mean((pred[:, 1] - self.reference_v)**2).item()
            return {'mse_u': mse_u, 'mse_v': mse_v, 'linf_max': torch.max(torch.abs(pred - torch.stack([self.reference_u, self.reference_v], dim=1))).item()}

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
        
        loss_history = []
        start_time = time.time()
        for epoch in range(iterations):
            optimizer.step(closure)
            loss_history.append(total_loss_fn().item())
            if epoch % 1 == 0:
                m = compute_metrics_fn()
                print(f"Epoch {epoch} | Loss: {loss_history[-1]:.2E} | MSE_u: {m['mse_u']:.2E} | MSE_v: {m['mse_v']:.2E}")
        
        self.training_results[embedding_type] = {'loss_history': loss_history, 'training_time': time.time() - start_time}
        return loss_history

    def save_model(self, embedding_type, save_dir):
        folder = {"NONE": "pinn", "FNN_BASIS": "fnn_basis", "QNN": "qnn"}[embedding_type]
        path = os.path.join(save_dir, folder)
        os.makedirs(path, exist_ok=True)
        if embedding_type == "FNN_BASIS":
            np.save(os.path.join(path, 'model.npy'), {'theta': self.theta.detach().cpu().numpy(), 'basis_net': self.basis_net.state_dict()}, allow_pickle=True)
        elif embedding_type == "QNN":
            np.save(os.path.join(path, 'model.npy'), {'theta': self.theta.detach().cpu().numpy(), 'qnn_embedding': self.qnn_embedding.state_dict()}, allow_pickle=True)
        else:
            torch.save(self.pinn.state_dict(), os.path.join(path, 'model.pth'))


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    plot_collocation_points_1d(config, config.OUTPUT_DIR)
    ref_path = os.path.join(config.BASE_DIR, "lengyel_epstein_reference_solution.npy")
    if os.path.exists(ref_path):
        ref = np.load(ref_path, allow_pickle=True)[()]
        plot_reference_solution(ref['t'], ref['x'], ref['u_sol'], ref['v_sol'], config.OUTPUT_DIR)
    
    trainer = LengyelEpstein1DQPINNTrainer(config, device)
    for mode in ["NONE", "FNN_BASIS", "QNN"]:
        trainer.train_model(mode, config.TRAINING_ITERATIONS)
        trainer.save_model(mode, config.OUTPUT_DIR)
    
    with open(os.path.join(config.OUTPUT_DIR, 'training_results.json'), 'w') as f:
        json.dump({k: {'final_loss': v['loss_history'][-1], 'time': v['training_time']} for k, v in trainer.training_results.items()}, f, indent=4)
    print("\n=== TRAINING COMPLETE ===")

if __name__ == "__main__":
    main()
