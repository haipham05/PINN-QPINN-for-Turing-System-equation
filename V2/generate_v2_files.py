#!/usr/bin/env python3
"""
Generator script to create v2 training and inference files for all equations.
This script reads the Brussel_v2 template and adapts it for each equation.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# BASE PATHS
WORKSPACE_ROOT = Path("/home/haipham2407/Documents/QPINN/rd/Turing Patern")
TEMPLATE_1D_TRAINING = WORKSPACE_ROOT / "Brussel_v2" / "1D_models" / "brussel_1d_training_v2.py"
TEMPLATE_1D_INFERENCE = WORKSPACE_ROOT / "Brussel_v2" / "1D_models" / "brussel_1d_inference_v2.py"
TEMPLATE_2D_TRAINING = WORKSPACE_ROOT / "Brussel_v2" / "2D_models" / "brussel_2d_training_v2.py" if (WORKSPACE_ROOT / "Brussel_v2" / "2D_models").exists() else None
TEMPLATE_2D_INFERENCE = WORKSPACE_ROOT / "Brussel_v2" / "2D_models" / "brussel_2d_inference_v2.py" if (WORKSPACE_ROOT / "Brussel_v2" / "2D_models").exists() else None

V2_OUTPUT_ROOT = WORKSPACE_ROOT / "V2"

# EQUATION CONFIGURATIONS
# Each equation specifies: parameters, RHS function, ICs, BCs
EQUATIONS_CONFIG = {
    "GrayScott": {
        "full_name": "Gray-Scott",
        "class_prefix": "GrayScott",
        "ref_filename": "grayscott_reference_solution.npy",
        "params": {
            "D_U": "2.0e-5",
            "D_V": "1.0e-5",
            "F": "0.04",
            "K": "0.06",
        },
        "u_boundary": "1.0",
        "v_boundary": "0.0",
        "pde_docstring": """    ∂u/∂t = D_u ∂²u/∂x² - u*v² + f*(1-u)
    ∂v/∂t = D_v ∂²v/∂x² + u*v² - (k+f)*v""",
        "ic_1d": ("np.ones(Nx)", "np.zeros(Nx); mid = Nx // 2; v0[mid-5:mid+5] = 0.5"),
        "ic_2d": ("np.ones((Nx, Ny))", "np.zeros((Nx, Ny)); mid_x = Nx // 2; mid_y = Ny // 2; v0[mid_x-5:mid_x+5, mid_y-5:mid_y+5] = 0.5"),
        "rhs_1d": """        reaction = u * (v ** 2)
        du = config.D_U * lapU - reaction + config.F * (1.0 - u)
        dv = config.D_V * lapV + reaction - (config.K + config.F) * v""",
        "rhs_2d": """            reaction = u * (v ** 2)
            du = config.D_U * lapU - reaction + config.F * (1.0 - u)
            dv = config.D_V * lapV + reaction - (config.K + config.F) * v""",
        "param_string": "d_u={config.D_U}, d_v={config.D_V}, f={config.F}, k={config.K}",
    },
    "Thomas": {
        "full_name": "Thomas",
        "class_prefix": "Thomas",
        "ref_filename": "thomas_reference_solution.npy",
        "params": {
            "D_U": "0.1",
            "D_V": "0.1",
            "A": "0.1",
            "B": "0.9",
        },
        "u_boundary": "1.0",
        "v_boundary": "3.0",
        "pde_docstring": """    ∂u/∂t = D_u ∂²u/∂x² - u + a*v²
    ∂v/∂t = D_v ∂²v/∂x² - v + b*u""",
        "ic_1d": ("np.ones(Nx)", "np.ones(Nx) * 3.0"),
        "ic_2d": ("np.ones((Nx, Ny))", "np.ones((Nx, Ny)) * 3.0"),
        "rhs_1d": """        du = config.D_U * lapU - u + config.A * (v ** 2)
        dv = config.D_V * lapV - v + config.B * u""",
        "rhs_2d": """            du = config.D_U * lapU - u + config.A * (v ** 2)
            dv = config.D_V * lapV - v + config.B * u""",
        "param_string": "d_u={config.D_U}, d_v={config.D_V}, a={config.A}, b={config.B}",
    },
    "Schnakenberg": {
        "full_name": "Schnakenberg",
        "class_prefix": "Schnakenberg",
        "ref_filename": "schnakenberg_reference_solution.npy",
        "params": {
            "D_U": "0.1",
            "D_V": "0.1",
            "ALPHA": "0.1",
            "BETA": "0.9",
            "GAMMA": "1.0",
        },
        "u_boundary": "0.5",
        "v_boundary": "1.0",
        "pde_docstring": """    ∂u/∂t = D_u ∂²u/∂x² + α - u + γ*u²*v
    ∂v/∂t = D_v ∂²v/∂x² + β - γ*u²*v""",
        "ic_1d": ("np.ones(Nx) * 0.5", "np.ones(Nx) * 1.0"),
        "ic_2d": ("np.ones((Nx, Ny)) * 0.5", "np.ones((Nx, Ny)) * 1.0"),
        "rhs_1d": """        du = config.D_U * lapU + config.ALPHA - u + config.GAMMA * (u**2) * v
        dv = config.D_V * lapV + config.BETA - config.GAMMA * (u**2) * v""",
        "rhs_2d": """            du = config.D_U * lapU + config.ALPHA - u + config.GAMMA * (u**2) * v
            dv = config.D_V * lapV + config.BETA - config.GAMMA * (u**2) * v""",
        "param_string": "d_u={config.D_U}, d_v={config.D_V}, alpha={config.ALPHA}, beta={config.BETA}, gamma={config.GAMMA}",
    },
    "Gierer-Meinhardt": {
        "full_name": "Gierer-Meinhardt",
        "class_prefix": "GiererMeinhardt",
        "ref_filename": "gierer_meinhardt_reference_solution.npy",
        "params": {
            "D_U": "0.001",
            "D_V": "0.1",
            "A": "0.1",
            "RHO": "0.01",
        },
        "u_boundary": "0.5",
        "v_boundary": "1.0",
        "pde_docstring": """    ∂u/∂t = D_u ∂²u/∂x² + (a*u²/v) - u + ρ
    ∂v/∂t = D_v ∂²v/∂x² + a*u² - v""",
        "ic_1d": ("0.5 + 0.1 * np.sin(2 * np.pi * x)", "np.ones(Nx) * 1.0"),
        "ic_2d": ("0.5 + 0.1 * np.sin(2 * np.pi * x[:, np.newaxis]) * np.sin(2 * np.pi * y[np.newaxis, :])", "np.ones((Nx, Ny)) * 1.0"),
        "rhs_1d": """        du = config.D_U * lapU + (config.A * u**2 / (v + 1e-6)) - u + config.RHO
        dv = config.D_V * lapV + config.A * u**2 - v""",
        "rhs_2d": """            du = config.D_U * lapU + (config.A * u**2 / (v + 1e-6)) - u + config.RHO
            dv = config.D_V * lapV + config.A * u**2 - v""",
        "param_string": "d_u={config.D_U}, d_v={config.D_V}, a={config.A}, rho={config.RHO}",
    },
}

def generate_config_block(eq_config: Dict) -> str:
    """Generate Config class parameters"""
    lines = []
    for param_name, param_value in eq_config["params"].items():
        lines.append(f"    {param_name} = {param_value}")
    return "\n".join(lines)

def adapt_template_training_1d(template_content: str, eq_key: str, eq_config: Dict) -> str:
    """Adapt 1D training template for specific equation"""
    content = template_content
    
    # Replace class name
    class_name = f"{eq_config['class_prefix']}1DQPINNTrainer"
    content = content.replace("class Brusselator1DQPINNTrainer", f"class {class_name}")
    content = content.replace("Brusselator 1D QPINN", f"{eq_config['full_name']} 1D QPINN")
    content = content.replace("class Brusselator1DQPINNTrainer", f"class {class_name}")
    
    # Replace config
    config_block = generate_config_block(eq_config)
    content = re.sub(
        r"class Config:.*?(# Boundary conditions)",
        f"class Config:\n    \"\"\"Configuration class for {eq_config['full_name']} 1D QPINN\"\"\"\n    \n    # Random seed\n    SEED = 42\n    \n    # Quantum Circuit Parameters\n    N_LAYERS = 5\n    N_WIRES = 4\n    \n    # FNN Basis Parameters\n    HIDDEN_LAYERS_FNN = 2\n    NEURONS_FNN = 20\n    \n    # QNN Embedding Parameters\n    N_LAYERS_EMBED = 2\n    \n    # Domain Parameters\n    T_COLLOC_POINTS = 5\n    X_COLLOC_POINTS = 10\n    \n    # Physics Parameters ({eq_config['full_name']})\n{config_block}\n    \n    # Boundary conditions",
        content,
        flags=re.DOTALL
    )
    
    # Replace PDE documentation in docstring
    content = content.replace(
        """Domain: t ∈ [0, 1], x ∈ [0, 1]
PDE System (Brusselator):
    ∂u/∂t = μ ∂²u/∂x² + u²v - (ε+1)u + β
    ∂v/∂t = μ ∂²v/∂x² - u²v + εu

Initial Conditions:
    u(x, 0) = 1 + sin(2πx)
    v(x, 0) = 3

Boundary Conditions (Dirichlet):
    u(0, t) = u(1, t) = 1
    v(0, t) = v(1, t) = 3""",
        f"""Domain: t ∈ [0, 1], x ∈ [0, 1]
PDE System ({eq_config['full_name']}):
{eq_config['pde_docstring']}

Boundary Conditions (Dirichlet):
    u(0, t) = u(1, t) = {eq_config['u_boundary']}
    v(0, t) = v(1, t) = {eq_config['v_boundary']}"""
    )
    
    # Replace reference solution generation
    content = content.replace("generate_reference_solution.npy", eq_config["ref_filename"])
    content = content.replace('def generate_reference_solution(config, save_path="brusselator_reference_solution.npy")',
                             f'def generate_reference_solution(config, save_path="{eq_config["ref_filename"]}")')
    content = content.replace('print("=== Generating 1D Reference Solution ===")',
                             f'print("=== Generating 1D {eq_config["full_name"]} Reference Solution ===")')
    
    # Replace initial conditions
    ic_u, ic_v = eq_config["ic_1d"]
    content = re.sub(
        r"u0 = 1\.0 \+ np\.sin\(2\.0 \* np\.pi \* x\)\n\s+v0 = np\.ones\(Nx\) \* 3\.0",
        f"u0 = {ic_u}\nv0 = {ic_v}",
        content
    )
    
    # Replace parameter print statement
    param_str = eq_config["param_string"]
    content = content.replace('print(f"Parameters: μ={config.MU}, ε={config.EPSILON}, β={config.BETA}")',
                             f'print(f"Parameters: {param_str}")')
    
    # Replace RHS function
    content = re.sub(
        r"def rd_rhs\(t, y\):.*?return np\.concatenate\(\[du, dv\]\)",
        f"""def rd_rhs(t, y):
        \"\"\"RHS for {eq_config['full_name']} system\"\"\"
        u = y[:Nx]
        v = y[Nx:]
        lapU = laplacian_1d(u, dx)
        lapV = laplacian_1d(v, dx)
        
{eq_config['rhs_1d']}
        return np.concatenate([du, dv])""",
        content,
        flags=re.DOTALL
    )
    
    # Replace in trainer class
    content = content.replace("Solving Brusselator PDE", f"Solving {eq_config['full_name']} PDE")
    content = content.replace('print(f"Solving Brusselator PDE with RK45...")',
                             f'print(f"Solving {eq_config["full_name"]} PDE with RK45...")')
    
    # Update boundary conditions in setup
    content = content.replace("U_BOUNDARY = 1.0", f"U_BOUNDARY = {eq_config['u_boundary']}")
    content = content.replace("V_BOUNDARY = 3.0", f"V_BOUNDARY = {eq_config['v_boundary']}")
    
    # Replace residual calculations
    content = content.replace(
        "coupling_term = (u * u) * v",
        "coupling_term = (u * u) * v"  # May need equation-specific residuals
    )
    
    return content

def main():
    """Generate all v2 files"""
    
    print("\n" + "="*80)
    print("QPINN V2 FILE GENERATOR")
    print("="*80)
    
    if not TEMPLATE_1D_TRAINING.exists():
        print(f"❌ ERROR: Template not found at {TEMPLATE_1D_TRAINING}")
        return
    
    print(f"✓ Loading template from: {TEMPLATE_1D_TRAINING}")
    
    with open(TEMPLATE_1D_TRAINING, 'r') as f:
        template_1d_training = f.read()
    
    # Generate files for each equation
    for eq_key, eq_config in EQUATIONS_CONFIG.items():
        print(f"\n{'='*80}")
        print(f"Generating files for {eq_config['full_name']}")
        print(f"{'='*80}")
        
        eq_dir_name = eq_key if eq_key != "Gierer-Meinhardt" else "Gierer-Meinhardt"
        eq_dir = V2_OUTPUT_ROOT / eq_dir_name / "1D_models"
        eq_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate 1D training file
        adapted_training = adapt_template_training_1d(template_1d_training, eq_key, eq_config)
        
        training_file = eq_dir / f"{eq_key.lower()}_1d_training_v2.py"
        with open(training_file, 'w') as f:
            f.write(adapted_training)
        
        print(f"✓ Created: {training_file}")

if __name__ == "__main__":
    main()
