"""
Brusselator 3D QPINN - Training Script
Reaction-Diffusion 2D Spatial + Time Quantum Physics-Informed Neural Network

This script trains three models:
1. PINN (Pure Physics-Informed Neural Network)
2. FNN-TE-QPINN (FNN Basis Temporal Embedding QPINN)
3. QNN-TE-QPINN (Quantum Neural Network Temporal Embedding QPINN)

Based on paper 2024112448454, extended to 2D spatial domain

Domain: t ∈ [0, 1], x ∈ [0, 1], y ∈ [0, 1]
PDE System (Brusselator 2D):
    ∂u/∂t = μ (∂²u/∂x² + ∂²u/∂y²) + u²v - (ε+1)u + β
    ∂v/∂t = μ (∂²v/∂x² + ∂²v/∂y²) - u²v + εu

Initial Conditions:
    u(x, y, 0) = 1 + sin(2πx)sin(2πy)
    v(x, y, 0) = 3

Boundary Conditions (Dirichlet):
    u = 1, v = 3 on all boundaries

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
    """Configuration class for Brusselator 3D QPINN"""
    
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
    
    # Domain Parameters (reduced for 2D spatial)
    T_COLLOC_POINTS = 10
    X_COLLOC_POINTS = 5
    Y_COLLOC_POINTS = 5
    
    # Physics Parameters (Brusselator from paper 2024112448454)
    MU = 0.01       # Diffusion coefficient
    EPSILON = 0.5   # Reaction parameter
    BETA = 0.1      # Source constant
    
    # Boundary conditions (Dirichlet)
    U_BOUNDARY = 1.0
    V_BOUNDARY = 3.0
    
    # Time domain
    T_MIN = 0.0
    T_MAX = 1.0
    
    # Spatial domain
    X_MIN = 0.0
    X_MAX = 1.0
    Y_MIN = 0.0
    Y_MAX = 1.0
    
    # Training Parameters
    TRAINING_ITERATIONS = 200
    LAMBDA_SCALE = 1e3   # Weight for IC + BC loss
    
    # Output directory
    BASE_DIR = "result"
    OUTPUT_DIR = "result"


# ============================================================
# REFERENCE SOLUTION GENERATOR
# ============================================================

def generate_reference_solution(config, save_path="brusselator_3d_reference_solution.npy"):
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
    u0 = 1.0 + np.sin(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)
    v0 = np.ones_like(X) * 3.0
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
        """RHS for 2D Brusselator system"""
        u = y_flat[:Nx*Ny].reshape(Nx, Ny)
        v = y_flat[Nx*Ny:].reshape(Nx, Ny)
        
        lapU = laplacian_2d(u, dx, dy)
        lapV = laplacian_2d(v, dx, dy)
        
        coupling = (u * u) * v
        du = config.MU * lapU + coupling - (config.EPSILON + 1.0) * u + config.BETA
        dv = config.MU * lapV - coupling + config.EPSILON * u
        
        return np.concatenate([du.ravel(), dv.ravel()])
    
    print(f"Solving 2D Brusselator PDE with RK45...")
    print(f"Parameters: μ={config.MU}, ε={config.EPSILON}, β={config.BETA}")
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

class Brusselator3DQPINNTrainer:
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
        
        self.u0_ic = 1.0 + np.sin(2.0 * np.pi * X_ic) * np.sin(2.0 * np.pi * Y_ic)
        self.v0_ic = np.ones_like(X_ic) * 3.0
        
        t_ic = torch.full((Nx*Ny, 1), self.config.T_MIN, device=self.device)
        x_ic_t = torch.tensor(X_ic.ravel(), device=self.device).float().view(-1, 1)
        y_ic_t = torch.tensor(Y_ic.ravel(), device=self.device).float().view(-1, 1)
        self.ic_points_highres = torch.cat([t_ic, x_ic_t, y_ic_t], dim=1)
        
        print(f"✓ Collocation points: Interior={len(self.interior_colloc)}, "
              f"Boundary={len(self.boundary_colloc)}, IC={len(self.init_val_colloc)}")
    
    def _load_reference_solution(self):
        """Load or generate reference solution"""
        ref_path = os.path.join(self.config.BASE_DIR, "brusselator_3d_reference_solution.npy")
        
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
            
            # PDE residuals
            residual_u = du_dt - self.config.MU * laplacian_u - coupling + (self.config.EPSILON + 1.0) * u - self.config.BETA
            residual_v = dv_dt - self.config.MU * laplacian_v + coupling - self.config.EPSILON * u
            
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
            
        else:  # NONE (PINN)
            self.pinn = FNNBasisNet(
                self.config.HIDDEN_LAYERS_FNN,
                self.config.NEURONS_FNN,
                2,  # Output: u, v
                input_dim=3
            ).to(self.device)
            params = list(self.pinn.parameters())
        
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
        
        for epoch in range(iterations):
            optimizer.step(closure)
            current_loss = total_loss_fn().item()
            metrics = compute_metrics_fn()
            
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
            
            coupling = (u ** 2) * v
            residual_u = du_dt - self.config.MU * (d2u_dx2 + d2u_dy2) - coupling + (self.config.EPSILON + 1.0) * u - self.config.BETA
            residual_v = dv_dt - self.config.MU * (d2v_dx2 + d2v_dy2) + coupling - self.config.EPSILON * u
            
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
        self.Y_unique = trainer.Y_unique
        self.reference_u = trainer.reference_u.cpu().numpy()
        self.reference_v = trainer.reference_v.cpu().numpy()
    
    def plot_training_analysis(self, save_dir="result"):
        """Plot 1: Training analysis for each model - 2D adapted (2x3 subplots)
        Shows spatial slices at t=T/2 for u and v
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
            
            plt.suptitle(f'Plot 1: {method_name} Training Analysis (Brusselator 3D)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/plot1_training_{embedding_type.lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Plot 1 for {method_name} saved: {save_dir}/plot1_training_{embedding_type.lower()}.png")
    
    def plot_methods_comparison(self, save_dir="result"):
        """Plot 3: Methods comparison (1x3 subplots)"""
        print("\n=== Plot 3: Methods Comparison ===")
        
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
        
        plt.suptitle('Plot 3: Three Methods Comparison (PINN vs FNN-TE-QPINN vs QNN-TE-QPINN) - Brusselator 3D',
                     fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/plot3_methods_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot 3 saved: {save_dir}/plot3_methods_comparison.png")
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*70)
        print("TRAINING SUMMARY STATISTICS (Brusselator 3D)")
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
    print("BRUSSELATOR 3D QPINN TRAINING (2D Spatial + Time)")
    print("="*80)
    print(f"Domain: t ∈ [{config.T_MIN}, {config.T_MAX}], "
          f"x ∈ [{config.X_MIN}, {config.X_MAX}], y ∈ [{config.Y_MIN}, {config.Y_MAX}]")
    print(f"Parameters: μ={config.MU}, ε={config.EPSILON}, β={config.BETA}")
    print(f"Training iterations: {config.TRAINING_ITERATIONS}")
    print("="*80)
    
    # Initialize trainer
    trainer = Brusselator3DQPINNTrainer(config, device)
    
    # Train all models
    for embedding_type in ["NONE", "FNN_BASIS", "QNN"]:
        trainer.train_model(embedding_type, config.TRAINING_ITERATIONS)
        trainer.save_model(embedding_type, config.OUTPUT_DIR)
    
    # Generate visualizations
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
    print("  - plot1_training_*.png (for each model)")
    print("  - plot3_methods_comparison.png")
    print("  - pinn/, fnn_basis/, qnn/ (model checkpoints)")
    print("  - training_summary.json")
    print("="*80)


if __name__ == "__main__":
    main()
