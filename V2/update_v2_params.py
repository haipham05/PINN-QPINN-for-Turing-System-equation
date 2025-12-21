#!/usr/bin/env python3
"""
Intelligent parameter replacer for v2 equation files.
Modifies copied template files to match each equation's parameters.
"""

import re
import os
from pathlib import Path

# Configuration for each equation
EQUATION_CONFIGS = {
    "GrayScott": {
        "full_name": "Gray-Scott",
        "class_base": "GrayScott",
        "ref_file": "grayscott_reference_solution.npy",
        "params": {
            "D_U": "2.0e-5",
            "D_V": "1.0e-5", 
            "F": "0.04",
            "K": "0.06",
        },
        "u_boundary": "1.0",
        "v_boundary": "0.0",
        "u_ic_1d": "np.ones(Nx)",
        "v_ic_1d": "np.zeros(Nx); mid = Nx // 2; v0[mid-5:mid+5] = 0.5",
        "u_ic_2d": "np.ones((Nx, Ny))",
        "v_ic_2d": "np.zeros((Nx, Ny)); mid_x = Nx // 2; mid_y = Ny // 2; v0[mid_x-5:mid_x+5, mid_y-5:mid_y+5] = 0.5",
        "rhs_comment": "Gray-Scott system",
        "param_comment": "Feed rate f and kill rate k",
    },
    "Thomas": {
        "full_name": "Thomas",
        "class_base": "Thomas",
        "ref_file": "thomas_reference_solution.npy",
        "params": {
            "D_U": "0.1",
            "D_V": "0.1",
            "A": "0.1",
            "B": "0.9",
        },
        "u_boundary": "1.0",
        "v_boundary": "3.0",
        "u_ic_1d": "np.ones(Nx)",
        "v_ic_1d": "np.ones(Nx) * 3.0",
        "u_ic_2d": "np.ones((Nx, Ny))",
        "v_ic_2d": "np.ones((Nx, Ny)) * 3.0",
        "rhs_comment": "Thomas system",
        "param_comment": "Reaction parameters a and b",
    },
    "Schnakenberg": {
        "full_name": "Schnakenberg",
        "class_base": "Schnakenberg",
        "ref_file": "schnakenberg_reference_solution.npy",
        "params": {
            "D_U": "0.1",
            "D_V": "0.1",
            "ALPHA": "0.1",
            "BETA": "0.9",
            "GAMMA": "1.0",
        },
        "u_boundary": "0.5",
        "v_boundary": "1.0",
        "u_ic_1d": "np.ones(Nx) * 0.5",
        "v_ic_1d": "np.ones(Nx) * 1.0",
        "u_ic_2d": "np.ones((Nx, Ny)) * 0.5",
        "v_ic_2d": "np.ones((Nx, Ny)) * 1.0",
        "rhs_comment": "Schnakenberg system",
        "param_comment": "Reaction parameters alpha, beta, gamma",
    },
    "Gierer-Meinhardt": {
        "full_name": "Gierer-Meinhardt",
        "class_base": "GiererMeinhardt",
        "ref_file": "gierer_meinhardt_reference_solution.npy",
        "params": {
            "D_U": "0.001",
            "D_V": "0.1",
            "A": "0.1",
            "RHO": "0.01",
        },
        "u_boundary": "0.5",
        "v_boundary": "1.0",
        "u_ic_1d": "0.5 + 0.1 * np.sin(2 * np.pi * x)",
        "v_ic_1d": "np.ones(Nx) * 1.0",
        "u_ic_2d": "0.5 + 0.1 * np.sin(2 * np.pi * x[:, np.newaxis]) * np.sin(2 * np.pi * y[np.newaxis, :])",
        "v_ic_2d": "np.ones((Nx, Ny)) * 1.0",
        "rhs_comment": "Gierer-Meinhardt system",
        "param_comment": "Activator-inhibitor with ratio-dependent kinetics",
    },
}

def replace_in_file(filepath: str, replacements: list):
    """Apply multiple replacements to a file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    with open(filepath, 'w') as f:
        f.write(content)

def update_equation_files(v2_root: str):
    """Update all copied v2 files with correct parameters"""
    
    v2_path = Path(v2_root)
    
    for eq_key, config in EQUATION_CONFIGS.items():
        print(f"\nProcessing {config['full_name']}...")
        
        eq_dir = v2_path / eq_key / "1D_models"
        training_file = eq_dir / f"{eq_key.lower()}_1d_training_v2.py"
        inference_file = eq_dir / f"{eq_key.lower()}_1d_inference_v2.py"
        
        if not training_file.exists():
            print(f"  ⚠ Training file not found: {training_file}")
            continue
        
        # Generate parameter block
        param_lines = []
        for param_name, param_value in config['params'].items():
            param_lines.append(f"    {param_name} = {param_value}")
        params_block = "\n".join(param_lines)
        
        # Update training file
        replacements = [
            ("Brusselator 1D QPINN", f"{config['full_name']} 1D QPINN"),
            ("class Brusselator1DQPINNTrainer", f"class {config['class_base']}1DQPINNTrainer"),
            (
                "# Physics Parameters (Brusselator from paper 2024112448454)\n    MU = 0.01       # Diffusion coefficient\n    EPSILON = 0.5   # Reaction parameter\n    BETA = 0.1      # Source constant",
                f"# Physics Parameters ({config['full_name']})\n{params_block}"
            ),
            (
                "    U_BOUNDARY = 1.0\n    V_BOUNDARY = 3.0",
                f"    U_BOUNDARY = {config['u_boundary']}\n    V_BOUNDARY = {config['v_boundary']}"
            ),
            ("Default save path: brusselator_reference_solution.npy", f"Default save path: {config['ref_file']}"),
            ('save_path="brusselator_reference_solution.npy"', f'save_path="{config["ref_file"]}"'),
            ("Generate 1D reference solution using RK45 solver", f"Generate 1D {config['full_name']} reference solution using RK45 solver"),
            (
                "u0 = 1.0 + np.sin(2.0 * np.pi * x)\n    v0 = np.ones(Nx) * 3.0",
                f"u0 = {config['u_ic_1d']}\nv0 = {config['v_ic_1d']}"
            ),
            ("\"\"\"RHS for Brusselator system\"\"\"", f"\"\"\"RHS for {config['rhs_comment']}\"\"\""),
            ("Solving Brusselator PDE", f"Solving {config['full_name']} PDE"),
        ]
        
        replace_in_file(str(training_file), replacements)
        print(f"  ✓ Updated: {training_file.name}")
        
        if inference_file.exists():
            replacements_inf = [
                ("Brusselator 1D QPINN", f"{config['full_name']} 1D QPINN"),
                ("class Brusselator1DQPINNInference", f"class {config['class_base']}1DQPINNInference"),
                ('save_path="brusselator_reference_solution.npy"', f'save_path="{config["ref_file"]}"'),
            ]
            replace_in_file(str(inference_file), replacements_inf)
            print(f"  ✓ Updated: {inference_file.name}")

if __name__ == "__main__":
    V2_ROOT = "/home/haipham2407/Documents/QPINN/rd/Turing Patern/V2"
    
    print("\n" + "="*70)
    print("UPDATING V2 FILES WITH EQUATION-SPECIFIC PARAMETERS")
    print("="*70)
    
    update_equation_files(V2_ROOT)
    
    print("\n" + "="*70)
    print("UPDATE COMPLETE")
    print("="*70)
    print("\nNote: Files have been updated with basic parameter replacements.")
    print("For complex RHS functions, manual review of each file is recommended.")
