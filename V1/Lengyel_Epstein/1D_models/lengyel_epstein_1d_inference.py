"""
Lengyel-Epstein 1D QPINN - Inference Script
Load trained models and perform inference with visualization

This script loads the trained models and generates:
1. Plot 3: Quantum circuit visualization (FNN and QNN)
2. Plot 7: Performance analysis for each model (predictions, L2 error, L∞ error)
3. Plot 8: Time evolution comparison (t ∈ [0, 1])

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
# QUANTUM CIRCUIT PLOTTING FUNCTIONS
# ============================================================

def plot_quantum_circuit(circuit_func, embedding_type, config, save_dir="result"):
    os.makedirs(save_dir, exist_ok=True)
    method_name = {"FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
    x_dummy = np.random.rand(2)
    theta_dummy = np.random.rand(config.N_LAYERS, config.N_WIRES, 3)
    basis_dummy = np.random.rand(config.N_WIRES)
    try:
        fig, ax = qml.draw_mpl(circuit_func)(x_dummy, theta_dummy, basis_dummy)
        ax.set_title(f'{method_name} Quantum Circuit Architecture\n({config.N_LAYERS} layers, {config.N_WIRES} qubits)', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        filename = f"plot3_quantum_circuit_{embedding_type.lower()}_inference.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e: print(f"⚠ Warning: {e}")

def plot_qnn_embedding_circuit(qnn_embedding, config, save_dir="result"):
    os.makedirs(save_dir, exist_ok=True)
    try:
        x_dummy = np.random.rand(2)
        fig, ax = qml.draw_mpl(qnn_embedding.qnode_embed)(x_dummy, qnn_embedding.weights_embed)
        ax.set_title(f'QNN Embedding Circuit Architecture\n({config.N_LAYERS_EMBED} layers, {config.N_WIRES} qubits)', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "plot3_quantum_circuit_qnn_embedding_inference.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e: print(f"⚠ Warning: {e}")


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Configuration class for Lengyel-Epstein 1D QPINN"""
    N_LAYERS = 5
    N_WIRES = 4
    HIDDEN_LAYERS_FNN = 2
    NEURONS_FNN = 20
    PINN_HIDDEN_LAYERS = 4
    PINN_NEURONS = 50
    N_LAYERS_EMBED = 2
    D_U = 1.0
    D_V = 10.0
    A = 10.0
    B = 1.5
    U_BOUNDARY = 2.0
    V_BOUNDARY = 5.0
    T_MIN = 0.0
    T_MAX = 1.0
    X_MIN = 0.0
    X_MAX = 1.0
    T_EVAL_POINTS = 200
    X_EVAL_POINTS = 50
    INPUT_DIR = "result"
    OUTPUT_DIR = "result"
    BASE_DIR = "result"


# ============================================================
# NEURAL NETWORK MODELS
# ============================================================

class FNNBasisNet(nn.Module):
    def __init__(self, n_hidden_layers, width, output_dim, input_dim=2):
        super().__init__()
        self.n_hidden_layers = n_hidden_layers
        layers = [nn.Linear(input_dim, width)]
        for _ in range(n_hidden_layers - 1): layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, output_dim))
        self.layers = nn.ModuleList(layers)
    def forward(self, x):
        for i in range(self.n_hidden_layers): x = torch.tanh(self.layers[i](x))
        return self.layers[-1](x)

class QNNEmbedding(nn.Module):
    def __init__(self, n_wires, n_layers_embed, output_dim, input_dim=2):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers_embed = n_layers_embed
        self.output_dim = output_dim
        self.weights_embed = nn.Parameter(torch.randn(n_layers_embed, n_wires, 3, requires_grad=True))
        self.dev = qml.device("default.qubit", wires=n_wires)
        self.qnode_embed = qml.QNode(self._circuit_embed, self.dev, interface="torch")
    def _circuit_embed(self, x, weights):
        for layer in range(self.n_layers_embed):
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
# INFERENCE ENGINE
# ============================================================

class LengyelEpstein1DQPINNInference:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.domain_min = torch.tensor([config.T_MIN, config.X_MIN], device=device)
        self.domain_max = torch.tensor([config.T_MAX, config.X_MAX], device=device)
        self.interp_u = None
        self.interp_v = None
        self.T_unique = None
        self.X_unique = None

    def load_reference(self, ref_path):
        if os.path.exists(ref_path):
            loaded = np.load(ref_path, allow_pickle=True)[()]
            # Load pre-built interpolators
            self.interp_u = loaded['u']
            self.interp_v = loaded['v']
            print("✓ 1D Reference solution loaded successfully")
        else:
            raise FileNotFoundError(f"Reference solution not found: {ref_path}")
        
        # Create evaluation grid from config parameters (independent of reference)
        self.T_unique = np.linspace(self.config.T_MIN, self.config.T_MAX, self.config.T_EVAL_POINTS)
        self.X_unique = np.linspace(self.config.X_MIN, self.config.X_MAX, self.config.X_EVAL_POINTS)
        
        print(f"   Evaluation grid: t={self.config.T_EVAL_POINTS} points, x={self.config.X_EVAL_POINTS} points")

    def _create_circuit(self):
        dev = qml.device("default.qubit", wires=self.config.N_WIRES)
        @qml.qnode(dev, interface="torch")
        def circuit(x, theta, basis):
            for i in range(self.config.N_WIRES): qml.RY(basis[i] * (x[0] if i % 2 == 0 else x[1]), wires=i)
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
        else: return self.pinn(x_rescaled)

    def inference(self, model_path, embedding_type):
        self.embedding_type = embedding_type
        method_name = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[embedding_type]
        print(f"\nRunning Inference: {method_name}")
        if embedding_type == "FNN_BASIS":
            state = np.load(model_path, allow_pickle=True)[()]
            self.theta = torch.tensor(state['theta'], device=self.device)
            self.basis_net = FNNBasisNet(self.config.HIDDEN_LAYERS_FNN, self.config.NEURONS_FNN, self.config.N_WIRES).to(self.device)
            self.basis_net.load_state_dict(state['basis_net']); self.basis_net.eval()
            self.circuit = self._create_circuit()
        elif embedding_type == "QNN":
            state = np.load(model_path, allow_pickle=True)[()]
            self.theta = torch.tensor(state['theta'], device=self.device)
            self.qnn_embedding = QNNEmbedding(self.config.N_WIRES, self.config.N_LAYERS_EMBED, self.config.N_WIRES).to(self.device)
            self.qnn_embedding.load_state_dict(state['qnn_embedding']); self.qnn_embedding.eval()
            self.circuit = self._create_circuit()
        else:
            self.pinn = FNNBasisNet(self.config.PINN_HIDDEN_LAYERS, self.config.PINN_NEURONS, 2).to(self.device)
            self.pinn.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False)); self.pinn.eval()
        
        grid = torch.tensor(list(product(self.T_unique, self.X_unique)), dtype=torch.float32, device=self.device)
        with torch.no_grad(): pred = self.model(grid)
        pred_u, pred_v = pred[:, 0].cpu().numpy(), pred[:, 1].cpu().numpy()
        ref_u = np.array([self.interp_u([pt[0], pt[1]]) for pt in grid.cpu().numpy()]).squeeze()
        ref_v = np.array([self.interp_v([pt[0], pt[1]]) for pt in grid.cpu().numpy()]).squeeze()
        
        mse_u, mse_v = np.mean((pred_u - ref_u)**2), np.mean((pred_v - ref_v)**2)
        print(f"✅ Inference Complete: MSE_u: {mse_u:.6E}, MSE_v: {mse_v:.6E}")
        return {'predictions_u': pred_u, 'predictions_v': pred_v, 'reference_u': ref_u, 'reference_v': ref_v, 'mse_u': mse_u, 'mse_v': mse_v}


# ============================================================
# VISUALIZATION
# ============================================================

class InferenceVisualizer:
    def __init__(self, engine, results):
        self.engine, self.results = engine, results
    def plot_performance(self, save_dir="result"):
        os.makedirs(save_dir, exist_ok=True)
        T, X = np.meshgrid(self.engine.T_unique, self.engine.X_unique)
        for mode in ["NONE", "FNN_BASIS", "QNN"]:
            if mode not in self.results: continue
            res = self.results[mode]
            method = {"NONE": "PINN", "FNN_BASIS": "FNN-TE-QPINN", "QNN": "QNN-TE-QPINN"}[mode]
            u_pred = res['predictions_u'].reshape(len(self.engine.T_unique), len(self.engine.X_unique)).T
            u_ref = res['reference_u'].reshape(len(self.engine.T_unique), len(self.engine.X_unique)).T
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            im1 = axes[0].pcolormesh(T, X, u_pred, cmap='inferno', shading='auto'); axes[0].set_title(f"u {method}"); fig.colorbar(im1, ax=axes[0])
            im2 = axes[1].pcolormesh(T, X, u_ref, cmap='inferno', shading='auto'); axes[1].set_title("u Reference"); fig.colorbar(im2, ax=axes[1])
            im3 = axes[2].pcolormesh(T, X, np.abs(u_pred - u_ref), cmap='hot', shading='auto'); axes[2].set_title("Absolute Error"); fig.colorbar(im3, ax=axes[2])
            plt.savefig(os.path.join(save_dir, f"plot7_{mode.lower()}_performance.png"), dpi=150); plt.close()

def main():
    # Configuration
    config = Config()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("LENGYEL-EPSTEIN 1D QPINN INFERENCE")
    print("="*80)
    
    # Initialize inference engine
    inference_engine = LengyelEpstein1DQPINNInference(config, device)
    
    # Load reference
    ref_path = os.path.join(config.INPUT_DIR, "lengyel_epstein_reference_solution.npy")
    inference_engine.load_reference(ref_path)
    
    # Run inference for all models
    embedding_results = {}
    
    model_configs = [
        ("NONE", "pinn", "model.pth"),
        ("FNN_BASIS", "fnn_basis", "model.npy"),
        ("QNN", "qnn", "model.npy"),
    ]
    
    for embedding_type, folder, filename in model_configs:
        model_path = os.path.join(config.INPUT_DIR, folder, filename)
        
        # Check for backward compatibility
        if not os.path.exists(model_path):
            alt_folder = "none" if folder == "pinn" else folder
            model_path = os.path.join(config.INPUT_DIR, alt_folder, filename)
        
        if os.path.exists(model_path):
            results = inference_engine.inference(model_path, embedding_type)
            embedding_results[embedding_type] = results
            
            # Plot quantum circuits for FNN and QNN
            if embedding_type in ["FNN_BASIS", "QNN"]:
                # Plot main circuit
                if embedding_type == "FNN_BASIS":
                    dev = qml.device("default.qubit", wires=config.N_WIRES)
                    @qml.qnode(dev, interface="torch")
                    def circuit(x, theta, basis):
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
                        plot_quantum_circuit(circuit, "FNN_BASIS", config, config.OUTPUT_DIR)
                    except Exception as e:
                        print(f"⚠ Could not generate FNN circuit plot: {e}")
                
                elif embedding_type == "QNN":
                    dev = qml.device("default.qubit", wires=config.N_WIRES)
                    @qml.qnode(dev, interface="torch")
                    def circuit(x, theta, basis):
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
                        plot_quantum_circuit(circuit, "QNN", config, config.OUTPUT_DIR)
                    except Exception as e:
                        print(f"⚠ Could not generate QNN circuit plot: {e}")
                    
                    # Plot QNN embedding circuit
                    if hasattr(inference_engine, 'qnn_embedding'):
                        try:
                            plot_qnn_embedding_circuit(inference_engine.qnn_embedding, config, config.OUTPUT_DIR)
                        except Exception as e:
                            print(f"⚠ Could not generate QNN embedding circuit plot: {e}")
        else:
            print(f"⚠ Model not found: {model_path}")
    
    # Visualizations
    visualizer = InferenceVisualizer(inference_engine, embedding_results)
    
    # === Plot 7 & 8: Performance and Time Evolution ===
    visualizer.plot_performance_analysis(config.OUTPUT_DIR)
    visualizer.plot_time_evolution(config.OUTPUT_DIR, t_max=1.0)
    visualizer.print_summary()
    
    # Save results
    with open(os.path.join(config.OUTPUT_DIR, 'inference_results.pkl'), 'wb') as f:
        pickle.dump(embedding_results, f)
    
    print(f"\n✓ Inference results saved: {config.OUTPUT_DIR}/inference_results.pkl")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    print("Generated Plots:")
    print("  - plot3_quantum_circuit_*.png (FNN and QNN circuits)")
    print("  - plot7_performance_*.png (for each model)")
    print("  - plot8_time_evolution_*.png (for u and v)")
    print("="*80)

if __name__ == "__main__": main()
