"""
Gray-Scott 2D QPINN - Training Script
Reaction-Diffusion 2D Spatial + Time Quantum Physics-Informed Neural Network

This script trains three models:
1. PINN (Pure Physics-Informed Neural Network)
2. FNN-TE-QPINN (FNN Basis Temporal Embedding QPINN)
3. QNN-TE-QPINN (Quantum Neural Network Temporal Embedding QPINN)

Domain: t ∈ [0, 1], x ∈ [0, 1], y ∈ [0, 1]
PDE System (Gray-Scott 2D):
    ∂u/∂t = D_u (∂²u/∂x² + ∂²u/∂y²) - u*v² + f*(1-u)
    ∂v/∂t = D_v (∂²v/∂x² + ∂²v/∂y²) + u*v² - (k+f)*v

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
    """Configuration class for Gray-Scott 2D QPINN"""
    
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
    T_COLLOC_POINTS = 10
    X_COLLOC_POINTS = 5
    Y_COLLOC_POINTS = 5
    
    # Physics Parameters
    D_U = 2.0e-5
    D_V = 1.0e-5
    F = 0.04
    K = 0.06
    
    # Boundary conditions
    U_BOUNDARY = 1.0
    V_BOUNDARY = 0.0
    
    # Time domain
    T_MIN = 0.0
    T_MAX = 1.0
    
    # Spatial domain
    X_MIN = 0.0
    X_MAX = 1.0
    Y_MIN = 0.0
    Y_MAX = 1.0
    
    # Training Parameters
    TRAINING_ITERATIONS = 100
    LAMBDA_SCALE = 1e3
    
    # Output directory
    BASE_DIR = "result"
    OUTPUT_DIR = "result"


# ============================================================
# REFERENCE SOLUTION GENERATOR
# ============================================================

def generate_reference_solution(config, save_path="grayscott_3d_reference_solution.npy"):
    """Generate 2D spatial reference solution using RK45 solver"""
    
    print("=== Generating 3D (2D spatial + time) Gray-Scott Reference Solution ===")
    
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
    u0 = np.ones_like(X)
    v0 = np.zeros_like(X)
    mid_x, mid_y = X.shape[0] // 2, X.shape[1] // 2
    v0[mid_x-3:mid_x+3, mid_y-3:mid_y+3] = 0.5
    
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
        """RHS for 2D Gray-Scott system"""
        u = y_flat[:Nx*Ny].reshape(Nx, Ny)
        v = y_flat[Nx*Ny:].reshape(Nx, Ny)
        
        lapU = laplacian_2d(u, dx, dy)
        lapV = laplacian_2d(v, dx, dy)
        
        reaction = u * (v ** 2)
        du = config.D_U * lapU - reaction + config.F * (1.0 - u)
        dv = config.D_V * lapV + reaction - (config.K + config.F) * v
        
        return np.concatenate([du.ravel(), dv.ravel()])
    
    print(f"Solving 2D Gray-Scott PDE with RK45...")
    print(f"Parameters: D_u={config.D_U}, D_v={config.D_V}, f={config.F}, k={config.K}")
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
    """Quantum Neural Network for embedding generation"""
    
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

class GrayScott2DQPINNTrainer:
    """Trainer class for Gray-Scott 2D QPINN models"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.domain_bounds = torch.tensor(
            [[config.T_MIN, config.T_MAX], [config.X_MIN, config.X_MAX], [config.Y_MIN, config.Y_MAX]],
            device=device, dtype=torch.float32
        )
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
        
        # Store unique values
        self.T_unique = t_torch.cpu().numpy()
        self.X_unique = x_torch.cpu().numpy()
        self.Y_unique = y_torch.cpu().numpy()
        
        domain = torch.tensor(list(product(t_torch, x_torch, y_torch)), dtype=torch.float32)
        
        # Initial condition mask (t = 0)
        init_val_mask = domain[:, 0] == self.config.T_MIN
        self.init_val_colloc = domain[init_val_mask].clone().detach().requires_grad_(True).to(self.device)
        
        # Boundary mask
        boundary_mask = ((domain[:, 1] == self.config.X_MIN) | (domain[:, 1] == self.config.X_MAX) |
                        (domain[:, 2] == self.config.Y_MIN) | (domain[:, 2] == self.config.Y_MAX))
        self.boundary_colloc = domain[boundary_mask & ~init_val_mask].clone().detach().requires_grad_(True).to(self.device)
        
        # Interior points
        interior_mask = ~(init_val_mask | boundary_mask)
        self.interior_colloc = domain[interior_mask].clone().detach().requires_grad_(True).to(self.device)
        
        # Full domain
        self.input_domain = domain.clone().detach().requires_grad_(True).to(self.device)
        
        # High-res IC points
        Nx = 32
        Ny = 32
        x_ic_np = np.linspace(self.config.X_MIN, self.config.X_MAX, Nx)
        y_ic_np = np.linspace(self.config.Y_MIN, self.config.Y_MAX, Ny)
        X_ic, Y_ic = np.meshgrid(x_ic_np, y_ic_np, indexing='ij')
        
        self.u0_ic = np.ones_like(X_ic)
        self.v0_ic = np.zeros_like(X_ic)
        mid_x, mid_y = X_ic.shape[0] // 2, X_ic.shape[1] // 2
        self.v0_ic[mid_x-2:mid_x+2, mid_y-2:mid_y+2] = 0.5
        
        x_ic_torch = torch.tensor(X_ic.ravel(), device=self.device).float().view(-1, 1)
        y_ic_torch = torch.tensor(Y_ic.ravel(), device=self.device).float().view(-1, 1)
        t_ic_torch = torch.full_like(x_ic_torch, self.config.T_MIN)
        self.ic_points_highres = torch.cat([t_ic_torch, x_ic_torch, y_ic_torch], dim=1)
        
        print(f"✓ Collocation points: Interior={len(self.interior_colloc)}, "
              f"Boundary={len(self.boundary_colloc)}, IC={len(self.init_val_colloc)}")
    
    def _load_reference_solution(self):
        """Load or generate reference solution"""
        ref_path = os.path.join(self.config.BASE_DIR, "grayscott_3d_reference_solution.npy")
        
        if os.path.exists(ref_path):
            print(f"Loading reference solution from {ref_path}")
            loaded = np.load(ref_path, allow_pickle=True)[()]
            self.interp_u = loaded['u']
            self.interp_v = loaded['v']
            u_sol = loaded['u_sol']
            v_sol = loaded['v_sol']
        else:
            os.makedirs(self.config.BASE_DIR, exist_ok=True)
            self.interp_u, self.interp_v = generate_reference_solution(self.config, ref_path)
            loaded = np.load(ref_path, allow_pickle=True)[()]
            u_sol = loaded['u_sol']
            v_sol = loaded['v_sol']
        
        # Compute reference on domain
        domain_np = self.input_domain.detach().cpu().numpy()
        ref_u = np.array([self.interp_u([pt[1], pt[2], pt[0]]) for pt in domain_np])
        ref_v = np.array([self.interp_v([pt[1], pt[2], pt[0]]) for pt in domain_np])
        
        self.reference_u = torch.tensor(ref_u, device=self.device, dtype=torch.float32)
        self.reference_v = torch.tensor(ref_v, device=self.device, dtype=torch.float32)
        
        print(f"✓ Reference solution loaded: u∈[{u_sol.min():.4f}, {u_sol.max():.4f}], "
              f"v∈[{v_sol.min():.4f}, {v_sol.max():.4f}]")
    
    def _create_circuit(self):
        """Create the main quantum circuit for TE-QPINN"""
        dev = qml.device("default.qubit", wires=self.config.N_WIRES)
        
        @qml.qnode(dev, interface="torch")
        def circuit(x, theta, basis):
            # Tensor encoding with basis
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
    
    def model(self, x):
        """Forward pass through the current model (no postprocessing)"""
        x_rescaled = 2.0 * (x - self.domain_min) / (self.domain_max - self.domain_min) - 1.0
        
        if self.embedding_type == "FNN_BASIS":
            basis = self.basis_net(x_rescaled)
            raw = self.circuit(x_rescaled.T, self.theta, basis.T)
            raw_stacked = torch.stack(raw).T
            return raw_stacked
        elif self.embedding_type == "QNN":
            basis = self.qnn_embedding(x_rescaled)
            raw = self.circuit(x_rescaled.T, self.theta, basis.T)
            raw_stacked = torch.stack(raw).T
            return raw_stacked
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
            
            reaction = u * (v ** 2)
            
            # Gray-Scott PDE residuals
            residual_u = du_dt - self.config.D_U * (d2u_dx2 + d2u_dy2) + reaction - self.config.F * (1.0 - u)
            residual_v = dv_dt - self.config.D_V * (d2v_dx2 + d2v_dy2) - reaction + (self.config.K + self.config.F) * v
            
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
            
            loss = torch.mean((u - self.config.U_BOUNDARY) ** 2)
            loss = loss + torch.mean((v - self.config.V_BOUNDARY) ** 2)
            
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
            
        else:  # NONE (pure PINN)
            self.pinn = FNNBasisNet(
                self.config.HIDDEN_LAYERS_FNN,
                self.config.NEURONS_FNN,
                2,
                input_dim=3
            ).to(self.device)
            params = list(self.pinn.parameters())
        
        # Setup optimizer
        optimizer = torch.optim.Adam(params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
        # Get loss functions
        total_loss_fn, compute_metrics_fn = self._create_loss_functions()
        
        # Training loop
        print(f"Training for {iterations} iterations...")
        start_time = time.time()
        
        for iteration in range(iterations):
            optimizer.zero_grad()
            loss = total_loss_fn()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (iteration + 1) % max(1, iterations // 10) == 0:
                metrics = compute_metrics_fn()
                elapsed = time.time() - start_time
                print(f"Iter {iteration+1:4d}/{iterations} | Loss: {loss.item():.4e} | "
                      f"MSE_u: {metrics['mse_u']:.4e} | MSE_v: {metrics['mse_v']:.4e} | "
                      f"L∞_u: {metrics['linf_u']:.4e} | Time: {elapsed:.1f}s")
        
        # Save model
        self.save_model(embedding_type, method_name)
        self.training_results[embedding_type] = compute_metrics_fn()
        
        print(f"✓ Training complete!")
    
    def save_model(self, embedding_type, method_name):
        """Save trained model"""
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        if embedding_type == "FNN_BASIS":
            model_state = {
                'theta': self.theta.detach().cpu().numpy(),
                'basis_net': self.basis_net.state_dict()
            }
            path = os.path.join(self.config.OUTPUT_DIR, f"grayscott_2d_fnn_basis.npy")
        elif embedding_type == "QNN":
            model_state = {
                'theta': self.theta.detach().cpu().numpy(),
                'qnn_embedding': self.qnn_embedding.state_dict()
            }
            path = os.path.join(self.config.OUTPUT_DIR, f"grayscott_2d_qnn.npy")
        else:  # PINN
            torch.save(self.pinn.state_dict(), os.path.join(self.config.OUTPUT_DIR, f"grayscott_2d_pinn.pt"))
            return
        
        np.save(path, model_state, allow_pickle=True)
        print(f"✓ {method_name} model saved to {path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = Config()
    trainer = GrayScott2DQPINNTrainer(config, device)
    
    # Train models
    trainer.train_model("NONE", config.TRAINING_ITERATIONS)  # PINN
    trainer.train_model("FNN_BASIS", config.TRAINING_ITERATIONS)  # FNN-TE-QPINN
    trainer.train_model("QNN", config.TRAINING_ITERATIONS)  # QNN-TE-QPINN
    
    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE")
    print("="*70)
