"""
Schnakenberg 1D QPINN - Inference Script (Version 2)
Load trained models and perform inference with visualization

This script loads trained models and generates:
- Plot 6: Inference results with L2 and L∞ relative errors

Version 2 Features:
- For each of 3 models: 3 columns layout
  - Column 1: QNN-TE-QPINN inference (u and v heatmaps)
  - Column 2: L2 relative error heatmap
  - Column 3: L∞ relative error heatmap

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
# CONFIGURATION
# ============================================================

class Config:
    """Configuration class for Schnakenberg 1D QPINN"""
    
    # Quantum Circuit Parameters
    N_LAYERS = 5
    N_WIRES = 4
    
    # FNN Basis Parameters
    HIDDEN_LAYERS_FNN = 2
    NEURONS_FNN = 20
    
    # QNN Embedding Parameters
    N_LAYERS_EMBED = 2
    
    # Physics Parameters
    MU = 0.01
    EPSILON = 0.5
    BETA = 0.1
    
    # Boundary conditions
    U_BOUNDARY = 1.0
    V_BOUNDARY = 3.0
    
    # Domain
    T_MIN = 0.0
    T_MAX = 1.0
    X_MIN = 0.0
    X_MAX = 1.0
    
    # Inference Domain Parameters
    T_EVAL_POINTS = 51
    X_EVAL_POINTS = 51
    
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

class Schnakenberg1DQPINNInference:
    """Inference engine for Schnakenberg 1D QPINN"""
    
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
        
        # Create evaluation grid from config parameters
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
        u_scaled = raw_output[:, 0] * 2.0 + 2.0
        v_scaled = raw_output[:, 1] * 2.0 + 2.0
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
        
        # Compute L2 relative error
        eps = 1e-10
        l2_u = np.sqrt(np.mean((pred_u - ref_u) ** 2)) / (np.sqrt(np.mean(ref_u ** 2)) + eps)
        l2_v = np.sqrt(np.mean((pred_v - ref_v) ** 2)) / (np.sqrt(np.mean(ref_v ** 2)) + eps)
        
        # Compute L∞ relative error
        lmax_u = np.max(np.abs(pred_u - ref_u)) / (np.max(np.abs(ref_u)) + eps)
        lmax_v = np.max(np.abs(pred_v - ref_v)) / (np.max(np.abs(ref_v)) + eps)
        
        print(f"✅ Inference Complete: {method_name}")
        print(f"   L2_u: {l2_u:.6E}, L2_v: {l2_v:.6E}")
        print(f"   Lmax_u: {lmax_u:.6E}, Lmax_v: {lmax_v:.6E}")
        
        return {
            'predictions_u': pred_u,
            'predictions_v': pred_v,
            'reference_u': ref_u,
            'reference_v': ref_v,
            'l2_u': l2_u, 'l2_v': l2_v,
            'lmax_u': lmax_u, 'lmax_v': lmax_v
        }


# ============================================================
# VISUALIZATION - Plot 6
# ============================================================

class InferenceVisualizerV2:
    """Visualization class for inference results (Version 2)"""
    
    def __init__(self, inference_engine, embedding_results):
        self.inference_engine = inference_engine
        self.embedding_results = embedding_results
        self.T_unique = inference_engine.T_unique
        self.X_unique = inference_engine.X_unique
    
    def plot_inference_results(self, save_dir="result"):
        """Plot 6: Inference results with L2 and L∞ relative errors"""
        print("\n=== Plot 6: Inference Results with Error Analysis ===")
        
        os.makedirs(save_dir, exist_ok=True)
        
        n_t = len(self.T_unique)
        n_x = len(self.X_unique)
        
        for embedding_type in ["NONE", "FNN_BASIS", "QNN"]:
            method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
            
            if embedding_type not in self.embedding_results:
                print(f"⚠ {method_name} results not found!")
                continue
            
            results = self.embedding_results[embedding_type]
            
            fig, axes = plt.subplots(1, 3, figsize=(22, 6))
            
            # Reshape predictions
            u_pred = results['predictions_u'].reshape(n_t, n_x)
            v_pred = results['predictions_v'].reshape(n_t, n_x)
            u_ref = results['reference_u'].reshape(n_t, n_x)
            v_ref = results['reference_v'].reshape(n_t, n_x)
            
            # Column 1: Predictions (u and v combined)
            pred_combined = np.abs(u_pred) + np.abs(v_pred)
            im1 = axes[0].contourf(self.T_unique, self.X_unique, pred_combined.T, 100, cmap='viridis')
            axes[0].set_xlabel('Time t', fontsize=12)
            axes[0].set_ylabel('Space x', fontsize=12)
            axes[0].set_title(f'Predictions (|u|+|v|) - {method_name}', fontsize=13, fontweight='bold')
            cbar1 = fig.colorbar(im1, ax=axes[0], label='Magnitude')
            
            # Column 2: L2 relative error
            l2_error_u = np.abs(u_pred - u_ref) / (np.abs(u_ref) + 1e-10)
            l2_error_v = np.abs(v_pred - v_ref) / (np.abs(v_ref) + 1e-10)
            l2_error_combined = (l2_error_u + l2_error_v) / 2.0
            
            im2 = axes[1].contourf(self.T_unique, self.X_unique, l2_error_combined.T, 100, cmap='Reds')
            axes[1].set_xlabel('Time t', fontsize=12)
            axes[1].set_ylabel('Space x', fontsize=12)
            axes[1].set_title(f'L2 Relative Error - {method_name}\nL2_u={results["l2_u"]:.3E}, L2_v={results["l2_v"]:.3E}', 
                             fontsize=12, fontweight='bold')
            cbar2 = fig.colorbar(im2, ax=axes[1], label='L2 Error')
            
            # Column 3: L∞ relative error
            lmax_error_u = np.abs(u_pred - u_ref) / (np.max(np.abs(u_ref)) + 1e-10)
            lmax_error_v = np.abs(v_pred - v_ref) / (np.max(np.abs(v_ref)) + 1e-10)
            lmax_error_combined = np.maximum(lmax_error_u, lmax_error_v)
            
            im3 = axes[2].contourf(self.T_unique, self.X_unique, lmax_error_combined.T, 100, cmap='Oranges')
            axes[2].set_xlabel('Time t', fontsize=12)
            axes[2].set_ylabel('Space x', fontsize=12)
            axes[2].set_title(f'L∞ Relative Error - {method_name}\nLmax_u={results["lmax_u"]:.3E}, Lmax_v={results["lmax_v"]:.3E}', 
                             fontsize=12, fontweight='bold')
            cbar3 = fig.colorbar(im3, ax=axes[2], label='L∞ Error')
            
            plt.suptitle(f'Plot 6: Inference Results - {method_name} (Brusselator 1D v2)', 
                        fontsize=15, fontweight='bold', y=1.00)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/plot6_inference_{embedding_type.lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Plot 6 for {method_name} saved: {save_dir}/plot6_inference_{embedding_type.lower()}.png")
    
    def print_summary(self):
        """Print inference summary"""
        print("\n" + "="*70)
        print("INFERENCE SUMMARY STATISTICS")
        print("="*70)
        print(f"\n{'Method':<20} {'L2_u':<15} {'L2_v':<15} {'Lmax_u':<15} {'Lmax_v':<15}")
        print("-"*70)
        
        for method in ["NONE", "FNN_BASIS", "QNN"]:
            method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[method]
            if method in self.embedding_results:
                results = self.embedding_results[method]
                print(f"{method_name:<20} {results['l2_u']:<15.3E} {results['l2_v']:<15.3E} "
                      f"{results['lmax_u']:<15.3E} {results['lmax_v']:<15.3E}")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    # Configuration
    config = Config()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("BRUSSELATOR 1D QPINN INFERENCE (VERSION 2)")
    print("="*80)
    
    # Initialize inference engine
    inference_engine = Brusselator1DQPINNInference(config, device)
    
    # Load reference solution
    ref_path = os.path.join(config.INPUT_DIR, "brusselator_reference_solution.npy")
    inference_engine.load_reference(ref_path)
    
    # Run inference for all models
    embedding_results = {}
    
    for embedding_type in ["NONE", "FNN_BASIS", "QNN"]:
        folder_name = {"NONE": "pinn", "FNN_BASIS": "fnn_basis", "QNN": "qnn"}[embedding_type]
        model_path = os.path.join(config.INPUT_DIR, folder_name, "model.npy" if embedding_type != "NONE" else "model.pth")
        
        if not os.path.exists(model_path):
            print(f"⚠ Model not found: {model_path}")
            continue
        
        results = inference_engine.inference(model_path, embedding_type)
        embedding_results[embedding_type] = results
    
    # Generate visualizations
    visualizer = InferenceVisualizerV2(inference_engine, embedding_results)
    visualizer.plot_inference_results(config.OUTPUT_DIR)
    visualizer.print_summary()
    
    # Save inference results
    with open(os.path.join(config.OUTPUT_DIR, 'inference_summary.json'), 'w') as f:
        summary = {}
        for method, results in embedding_results.items():
            summary[method] = {
                'l2_u': float(results['l2_u']),
                'l2_v': float(results['l2_v']),
                'lmax_u': float(results['lmax_u']),
                'lmax_v': float(results['lmax_v'])
            }
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    print(f"Results saved to: {config.OUTPUT_DIR}/")
    print("  - plot6_inference_*.png (for each model)")
    print("  - inference_summary.json")
    print("="*80)


if __name__ == "__main__":
    main()
