"""
Thomas 2D QPINN - Training Script
3D domain (t, x, y) for 2D spatial reaction-diffusion

PDE System (Thomas):
    ∂u/∂t = D_u (∂²u/∂x² + ∂²u/∂y²) - u + a*v²
    ∂v/∂t = D_v (∂²v/∂x² + ∂²v/∂y²) - v + b*u

Author: QPINN Research
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

torch.random.manual_seed(42)


class Config:
    """Configuration for Thomas 2D QPINN"""
    
    SEED = 42
    N_LAYERS = 5
    N_WIRES = 4
    HIDDEN_LAYERS_FNN = 2
    NEURONS_FNN = 20
    N_LAYERS_EMBED = 2
    
    T_COLLOC_POINTS = 10
    X_COLLOC_POINTS = 5
    Y_COLLOC_POINTS = 5
    
    # Thomas parameters
    D_U = 0.1
    D_V = 0.1
    A = 0.1
    B = 0.9
    
    U_BOUNDARY = 0.5
    V_BOUNDARY = 0.5
    
    T_MIN = 0.0
    T_MAX = 1.0
    X_MIN = 0.0
    X_MAX = 1.0
    Y_MIN = 0.0
    Y_MAX = 1.0
    
    TRAINING_ITERATIONS = 2
    LAMBDA_SCALE = 10.0
    
    BASE_DIR = "result"
    OUTPUT_DIR = "result"


def generate_2d_reference_solution(config, save_path="thomas_reference_solution_2d.npy"):
    """Generate 2D reference solution"""
    
    print("=== Generating 2D Thomas Reference Solution ===")
    
    Nx, Ny = 32, 32
    x = np.linspace(config.X_MIN, config.X_MAX, Nx)
    y = np.linspace(config.Y_MIN, config.Y_MAX, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    Nt = 51
    t_eval = np.linspace(config.T_MIN, config.T_MAX, Nt)
    
    # Initial conditions
    X, Y = np.meshgrid(x, y, indexing='ij')
    u0 = 0.5 + 0.1 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    v0 = 0.5 + 0.1 * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    
    u0 = np.clip(u0, 0.0, None).flatten()
    v0 = np.clip(v0, 0.0, None).flatten()
    y0 = np.concatenate([u0, v0])
    
    def laplacian_2d(u, dx, dy, nx, ny):
        u_grid = u.reshape(nx, ny)
        lap = np.zeros_like(u_grid)
        lap[1:-1, 1:-1] = (
            (u_grid[2:, 1:-1] - 2*u_grid[1:-1, 1:-1] + u_grid[:-2, 1:-1]) / dx**2 +
            (u_grid[1:-1, 2:] - 2*u_grid[1:-1, 1:-1] + u_grid[1:-1, :-2]) / dy**2
        )
        return lap.flatten()
    
    def rd_rhs(t, y):
        """RHS for 2D Thomas system"""
        u = y[:Nx*Ny]
        v = y[Nx*Ny:]
        lapU = laplacian_2d(u, dx, dy, Nx, Ny)
        lapV = laplacian_2d(v, dx, dy, Nx, Ny)
        
        du = config.D_U * lapU - u + config.A * (v ** 2)
        dv = config.D_V * lapV - v + config.B * u
        return np.concatenate([du, dv])
    
    print(f"Solving 2D Thomas PDE with RK45...")
    print(f"Grid: {Nx}×{Ny}, Time steps: {Nt}")
    
    sol = solve_ivp(rd_rhs, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, 
                    method='RK45', rtol=1e-6, atol=1e-9)
    
    t = sol.t
    u_sol = sol.y[:Nx*Ny, :].reshape(Nx, Ny, -1)
    v_sol = sol.y[Nx*Ny:, :].reshape(Nx, Ny, -1)
    print("Status:", sol.message)
    
    interpU = RegularGridInterpolator((t, x, y), u_sol, bounds_error=False, fill_value=None)
    interpV = RegularGridInterpolator((t, x, y), v_sol, bounds_error=False, fill_value=None)
    
    np.save(save_path, {'u': interpU, 'v': interpV, 't': t, 'x': x, 'y': y,
                        'u_sol': u_sol, 'v_sol': v_sol}, allow_pickle=True)
    
    print(f"✓ 2D reference solution generated: u∈[{u_sol.min():.4f}, {u_sol.max():.4f}], "
          f"v∈[{v_sol.min():.4f}, {v_sol.max():.4f}]")
    
    return interpU, interpV


class FNNBasisNet(nn.Module):
    """FNN for basis generation (3D input for 2D space)"""
    
    def __init__(self, n_hidden_layers, width, output_dim, input_dim=3):
        super().__init__()
        layers = [nn.Linear(input_dim, width)]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, output_dim))
        self.layers = nn.ModuleList(layers)
        self.n_hidden_layers = n_hidden_layers
    
    def forward(self, x):
        for i in range(self.n_hidden_layers):
            x = torch.tanh(self.layers[i](x))
        return self.layers[-1](x)


class QNNEmbedding(nn.Module):
    """Quantum embedding network"""
    
    def __init__(self, n_wires, n_layers_embed, output_dim, input_dim=3):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers_embed = n_layers_embed
        self.output_dim = output_dim
        
        self.weights_embed = nn.Parameter(
            torch.randn(n_layers_embed, n_wires, 3, requires_grad=True)
        )
        
        self.dev = qml.device("default.qubit", wires=n_wires)
        self.qnode_embed = qml.QNode(self._circuit_embed, self.dev, interface="torch", diff_method="best")
    
    def _circuit_embed(self, x, weights):
        for layer in range(self.n_layers_embed):
            for i in range(self.n_wires):
                qml.RX(x[i % len(x)], wires=i)
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


class Thomas2DQPINNTrainer:
    """Trainer for 2D Thomas QPINN"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.domain_min = torch.tensor([config.T_MIN, config.X_MIN, config.Y_MIN], device=device)
        self.domain_max = torch.tensor([config.T_MAX, config.X_MAX, config.Y_MAX], device=device)
        
        self._setup_collocation_points()
        self._load_reference_solution()
        self.training_results = {}
    
    def _setup_collocation_points(self):
        """Setup 3D collocation points"""
        t_torch = torch.linspace(self.config.T_MIN, self.config.T_MAX, self.config.T_COLLOC_POINTS)
        x_torch = torch.linspace(self.config.X_MIN, self.config.X_MAX, self.config.X_COLLOC_POINTS)
        y_torch = torch.linspace(self.config.Y_MIN, self.config.Y_MAX, self.config.Y_COLLOC_POINTS)
        
        domain = torch.tensor(list(product(t_torch, x_torch, y_torch)), dtype=torch.float32)
        
        init_val_mask = domain[:, 0] == self.config.T_MIN
        self.init_val_colloc = domain[init_val_mask].clone().detach().requires_grad_(True).to(self.device)
        
        boundary_mask = ((domain[:, 1] == self.config.X_MIN) | (domain[:, 1] == self.config.X_MAX) |
                         (domain[:, 2] == self.config.Y_MIN) | (domain[:, 2] == self.config.Y_MAX))
        self.boundary_colloc = domain[boundary_mask & ~init_val_mask].clone().detach().requires_grad_(True).to(self.device)
        
        interior_mask = ~(init_val_mask | boundary_mask)
        self.interior_colloc = domain[interior_mask].clone().detach().requires_grad_(True).to(self.device)
        
        self.input_domain = domain.clone().detach().requires_grad_(True).to(self.device)
        
        # Initial conditions on fine grid
        Nx, Ny = 32, 32
        x_ic_np = np.linspace(self.config.X_MIN, self.config.X_MAX, Nx)
        y_ic_np = np.linspace(self.config.Y_MIN, self.config.Y_MAX, Ny)
        X, Y = np.meshgrid(x_ic_np, y_ic_np, indexing='ij')
        self.u0_ic = 0.5 + 0.1 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        self.v0_ic = 0.5 + 0.1 * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        
        x_ic_torch = torch.tensor(x_ic_np, device=self.device).float()
        y_ic_torch = torch.tensor(y_ic_np, device=self.device).float()
        X_torch, Y_torch = torch.meshgrid(x_ic_torch, y_ic_torch, indexing='ij')
        T_torch = torch.full_like(X_torch, self.config.T_MIN)
        
        self.ic_points_highres = torch.stack([T_torch, X_torch, Y_torch], dim=-1).reshape(-1, 3)
        
        print(f"✓ Collocation points (3D): Interior={len(self.interior_colloc)}, "
              f"Boundary={len(self.boundary_colloc)}, IC={len(self.init_val_colloc)}")
    
    def _load_reference_solution(self):
        """Load reference solution"""
        ref_path = os.path.join(self.config.BASE_DIR, "thomas_reference_solution_2d.npy")
        
        if os.path.exists(ref_path):
            print(f"Loading reference solution from {ref_path}")
            loaded = np.load(ref_path, allow_pickle=True)[()]
            self.interp_u = loaded['u']
            self.interp_v = loaded['v']
            u_sol = loaded['u_sol']
            v_sol = loaded['v_sol']
        else:
            os.makedirs(self.config.BASE_DIR, exist_ok=True)
            self.interp_u, self.interp_v = generate_2d_reference_solution(self.config, ref_path)
            loaded = np.load(ref_path, allow_pickle=True)[()]
            u_sol = loaded['u_sol']
            v_sol = loaded['v_sol']
        
        domain_np = self.input_domain.detach().cpu().numpy()
        ref_u = np.array([self.interp_u(pt).squeeze() for pt in domain_np])
        ref_v = np.array([self.interp_v(pt).squeeze() for pt in domain_np])
        
        self.reference_u = torch.tensor(ref_u, device=self.device, dtype=torch.float32)
        self.reference_v = torch.tensor(ref_v, device=self.device, dtype=torch.float32)
        
        print(f"✓ Reference solution loaded: u∈[{u_sol.min():.4f}, {u_sol.max():.4f}], "
              f"v∈[{v_sol.min():.4f}, {v_sol.max():.4f}]")
    
    def _create_circuit(self):
        """Create quantum circuit for 3D input"""
        dev = qml.device("default.qubit", wires=self.config.N_WIRES)
        
        @qml.qnode(dev, interface="torch")
        def circuit(x, theta, basis):
            for i in range(self.config.N_WIRES):
                qml.RY(basis[i] * x[i % 3], wires=i)
            
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
        """Forward pass"""
        x_rescaled = 2.0 * (x - self.domain_min) / (self.domain_max - self.domain_min) - 1.0
        
        if self.embedding_type == "FNN_BASIS":
            basis = self.basis_net(x_rescaled)
            raw = self.circuit(x_rescaled.T, self.theta, basis.T)
            return torch.stack(raw).T
        elif self.embedding_type == "QNN":
            basis = self.qnn_embedding(x_rescaled)
            raw = self.circuit(x_rescaled.T, self.theta, basis.T)
            return torch.stack(raw).T
        else:
            return self.pinn(x_rescaled)
    
    def _create_loss_functions(self):
        """Create loss functions"""
        
        def extract_u(multi_output):
            return multi_output[:, 0] if multi_output.dim() > 1 else multi_output[0]
        
        def extract_v(multi_output):
            return multi_output[:, 1] if multi_output.dim() > 1 else multi_output[1]
        
        def pde_loss():
            pred = self.model(self.interior_colloc)
            u = extract_u(pred)
            v = extract_v(pred)
            
            grad_u = torch.autograd.grad(u.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            du_dt = grad_u[:, 0]
            du_dx = grad_u[:, 1]
            du_dy = grad_u[:, 2]
            
            grad_du_dx = torch.autograd.grad(du_dx.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            d2u_dx2 = grad_du_dx[:, 1]
            
            grad_du_dy = torch.autograd.grad(du_dy.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            d2u_dy2 = grad_du_dy[:, 2]
            
            grad_v = torch.autograd.grad(v.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            dv_dt = grad_v[:, 0]
            dv_dx = grad_v[:, 1]
            dv_dy = grad_v[:, 2]
            
            grad_dv_dx = torch.autograd.grad(dv_dx.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            d2v_dx2 = grad_dv_dx[:, 1]
            
            grad_dv_dy = torch.autograd.grad(dv_dy.sum(), self.interior_colloc, create_graph=True, retain_graph=True)[0]
            d2v_dy2 = grad_dv_dy[:, 2]
            
            # Thomas PDE residuals
            residual_u = du_dt - self.config.D_U * (d2u_dx2 + d2u_dy2) + u - self.config.A * (v ** 2)
            residual_v = dv_dt - self.config.D_V * (d2v_dx2 + d2v_dy2) + v - self.config.B * u
            
            return torch.mean(residual_u ** 2) + torch.mean(residual_v ** 2)
        
        def initial_condition_loss():
            pred = self.model(self.ic_points_highres)
            u = extract_u(pred)
            v = extract_v(pred)
            
            u_true = torch.tensor(self.u0_ic.flatten(), device=self.device, dtype=torch.float32)
            v_true = torch.tensor(self.v0_ic.flatten(), device=self.device, dtype=torch.float32)
            
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
        """Train model"""
        self.embedding_type = embedding_type
        method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
        
        print(f"\n{'='*70}")
        print(f"TRAINING: {method_name} (2D)")
        print(f"{'='*70}")
        
        if embedding_type == "FNN_BASIS":
            self.theta = torch.rand(self.config.N_LAYERS, self.config.N_WIRES, 3,
                                    device=self.device, requires_grad=True)
            self.basis_net = FNNBasisNet(self.config.HIDDEN_LAYERS_FNN, self.config.NEURONS_FNN,
                                        self.config.N_WIRES, input_dim=3).to(self.device)
            self.circuit = self._create_circuit()
            params = [self.theta] + list(self.basis_net.parameters())
        elif embedding_type == "QNN":
            self.theta = torch.rand(self.config.N_LAYERS, self.config.N_WIRES, 3,
                                    device=self.device, requires_grad=True)
            self.qnn_embedding = QNNEmbedding(self.config.N_WIRES, self.config.N_LAYERS_EMBED,
                                             self.config.N_WIRES, input_dim=3).to(self.device)
            self.circuit = self._create_circuit()
            params = [self.theta] + list(self.qnn_embedding.parameters())
        else:
            self.pinn = FNNBasisNet(self.config.HIDDEN_LAYERS_FNN, self.config.NEURONS_FNN,
                                   2, input_dim=3).to(self.device)
            params = list(self.pinn.parameters())
        
        optimizer = torch.optim.Adam(params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
        total_loss_fn, compute_metrics_fn = self._create_loss_functions()
        
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
                      f"Time: {elapsed:.1f}s")
        
        self.save_model(embedding_type, method_name)
        self.training_results[embedding_type] = compute_metrics_fn()
        print(f"✓ Training complete!")
    
    def save_model(self, embedding_type, method_name):
        """Save model"""
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        if embedding_type == "FNN_BASIS":
            model_state = {
                'theta': self.theta.detach().cpu().numpy(),
                'basis_net': self.basis_net.state_dict()
            }
            path = os.path.join(self.config.OUTPUT_DIR, f"thomas_2d_fnn_basis.npy")
        elif embedding_type == "QNN":
            model_state = {
                'theta': self.theta.detach().cpu().numpy(),
                'qnn_embedding': self.qnn_embedding.state_dict()
            }
            path = os.path.join(self.config.OUTPUT_DIR, f"thomas_2d_qnn.npy")
        else:
            torch.save(self.pinn.state_dict(), os.path.join(self.config.OUTPUT_DIR, f"thomas_2d_pinn.pt"))
            return
        
        np.save(path, model_state, allow_pickle=True)
        print(f"✓ {method_name} model saved to {path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = Config()
    trainer = Thomas2DQPINNTrainer(config, device)
    
    trainer.train_model("NONE", config.TRAINING_ITERATIONS)
    trainer.train_model("FNN_BASIS", config.TRAINING_ITERATIONS)
    trainer.train_model("QNN", config.TRAINING_ITERATIONS)
    
    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE (2D)")
    print("="*70)
