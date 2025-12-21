#!/usr/bin/env python3
"""
Automated v2 file generator for all reaction-diffusion equations.
Reads Brussel_v2 template and adapts it for all other equations.
"""

import os
import re
from pathlib import Path

# Define equation-specific parameters
EQUATIONS = {
    "GrayScott": {
        "name": "Gray-Scott",
        "1d": {
            "class_name": "GrayScott1DQPINNTrainer",
            "ref_filename": "grayscott_reference_solution.npy",
            "params": {
                "D_U": "2.0e-5",
                "D_V": "1.0e-5",
                "F": "0.04",
                "K": "0.06",
            },
            "u_boundary": "1.0",
            "v_boundary": "0.0",
            "ic_u": "np.ones(Nx)",
            "ic_v": "np.zeros(Nx); mid = Nx // 2; v0[mid-5:mid+5] = 0.5",
            "rhs": """reaction = u * (v ** 2)
        du = config.D_U * lapU - reaction + config.F * (1.0 - u)
        dv = config.D_V * lapV + reaction - (config.K + config.F) * v""",
            "pde_string": """
    ∂u/∂t = D_u ∂²u/∂x² - u*v² + f*(1-u)
    ∂v/∂t = D_v ∂²v/∂x² + u*v² - (k+f)*v""",
        },
        "2d": {
            "class_name": "GrayScott2DQPINNTrainer",
            "ref_filename": "grayscott_reference_solution.npy",
            "params": {
                "D_U": "2.0e-5",
                "D_V": "1.0e-5",
                "F": "0.04",
                "K": "0.06",
            },
            "u_boundary": "1.0",
            "v_boundary": "0.0",
            "ic_u": "np.ones((Nx, Ny))",
            "ic_v": "np.zeros((Nx, Ny)); mid_x = Nx // 2; mid_y = Ny // 2; v0[mid_x-5:mid_x+5, mid_y-5:mid_y+5] = 0.5",
            "rhs": """reaction = u * (v ** 2)
            du = config.D_U * lapU - reaction + config.F * (1.0 - u)
            dv = config.D_V * lapV + reaction - (config.K + config.F) * v""",
            "pde_string": """
    ∂u/∂t = D_u (∂²u/∂x² + ∂²u/∂y²) - u*v² + f*(1-u)
    ∂v/∂t = D_v (∂²v/∂x² + ∂²v/∂y²) + u*v² - (k+f)*v""",
        },
    },
    "Thomas": {
        "name": "Thomas",
        "1d": {
            "class_name": "Thomas1DQPINNTrainer",
            "ref_filename": "thomas_reference_solution.npy",
            "params": {
                "D_U": "0.1",
                "D_V": "0.1",
                "A": "0.1",
                "B": "0.9",
            },
            "u_boundary": "1.0",
            "v_boundary": "3.0",
            "ic_u": "np.ones(Nx)",
            "ic_v": "np.ones(Nx) * 3.0",
            "rhs": """du = config.D_U * lapU - u + config.A * (v ** 2)
        dv = config.D_V * lapV - v + config.B * u""",
            "pde_string": """
    ∂u/∂t = D_u ∂²u/∂x² - u + a*v²
    ∂v/∂t = D_v ∂²v/∂x² - v + b*u""",
        },
        "2d": {
            "class_name": "Thomas2DQPINNTrainer",
            "ref_filename": "thomas_reference_solution.npy",
            "params": {
                "D_U": "0.1",
                "D_V": "0.1",
                "A": "0.1",
                "B": "0.9",
            },
            "u_boundary": "1.0",
            "v_boundary": "3.0",
            "ic_u": "np.ones((Nx, Ny))",
            "ic_v": "np.ones((Nx, Ny)) * 3.0",
            "rhs": """du = config.D_U * lapU - u + config.A * (v ** 2)
            dv = config.D_V * lapV - v + config.B * u""",
            "pde_string": """
    ∂u/∂t = D_u (∂²u/∂x² + ∂²u/∂y²) - u + a*v²
    ∂v/∂t = D_v (∂²v/∂x² + ∂²v/∂y²) - v + b*u""",
        },
    },
    "Schnakenberg": {
        "name": "Schnakenberg",
        "1d": {
            "class_name": "Schnakenberg1DQPINNTrainer",
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
            "ic_u": "np.ones(Nx) * 0.5",
            "ic_v": "np.ones(Nx) * 1.0",
            "rhs": """du = config.D_U * lapU + config.ALPHA - u + config.GAMMA * (u**2) * v
        dv = config.D_V * lapV + config.BETA - config.GAMMA * (u**2) * v""",
            "pde_string": """
    ∂u/∂t = D_u ∂²u/∂x² + α - u + γ*u²*v
    ∂v/∂t = D_v ∂²v/∂x² + β - γ*u²*v""",
        },
        "2d": {
            "class_name": "Schnakenberg2DQPINNTrainer",
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
            "ic_u": "np.ones((Nx, Ny)) * 0.5",
            "ic_v": "np.ones((Nx, Ny)) * 1.0",
            "rhs": """du = config.D_U * lapU + config.ALPHA - u + config.GAMMA * (u**2) * v
            dv = config.D_V * lapV + config.BETA - config.GAMMA * (u**2) * v""",
            "pde_string": """
    ∂u/∂t = D_u (∂²u/∂x² + ∂²u/∂y²) + α - u + γ*u²*v
    ∂v/∂t = D_v (∂²v/∂x² + ∂²v/∂y²) + β - γ*u²*v""",
        },
    },
    "Gierer-Meinhardt": {
        "name": "Gierer-Meinhardt",
        "1d": {
            "class_name": "GiererMeinhardt1DQPINNTrainer",
            "ref_filename": "gierer_meinhardt_reference_solution.npy",
            "params": {
                "D_U": "0.001",
                "D_V": "0.1",
                "A": "0.1",
                "RHO": "0.01",
            },
            "u_boundary": "0.5",
            "v_boundary": "1.0",
            "ic_u": "0.5 + 0.1 * np.sin(2 * np.pi * x)",
            "ic_v": "np.ones(Nx) * 1.0",
            "rhs": """du = config.D_U * lapU + (config.A * u**2 / v) - u + config.RHO
        dv = config.D_V * lapV + config.A * u**2 - v""",
            "pde_string": """
    ∂u/∂t = D_u ∂²u/∂x² + (a*u²/v) - u + ρ
    ∂v/∂t = D_v ∂²v/∂x² + a*u² - v""",
        },
        "2d": {
            "class_name": "GiererMeinhardt2DQPINNTrainer",
            "ref_filename": "gierer_meinhardt_reference_solution.npy",
            "params": {
                "D_U": "0.001",
                "D_V": "0.1",
                "A": "0.1",
                "RHO": "0.01",
            },
            "u_boundary": "0.5",
            "v_boundary": "1.0",
            "ic_u": "0.5 + 0.1 * np.sin(2 * np.pi * x[:, np.newaxis]) * np.sin(2 * np.pi * y[np.newaxis, :])",
            "ic_v": "np.ones((Nx, Ny)) * 1.0",
            "rhs": """du = config.D_U * lapU + (config.A * u**2 / v) - u + config.RHO
            dv = config.D_V * lapV + config.A * u**2 - v""",
            "pde_string": """
    ∂u/∂t = D_u (∂²u/∂x² + ∂²u/∂y²) + (a*u²/v) - u + ρ
    ∂v/∂t = D_v (∂²v/∂x² + ∂²v/∂y²) + a*u² - v""",
        },
    },
}

def get_config_block(eq_key, dim):
    """Generate the Config class for each equation"""
    eq_data = EQUATIONS[eq_key][dim]
    params = eq_data["params"]
    
    config_lines = []
    for param_name, param_value in params.items():
        config_lines.append(f"    {param_name} = {param_value}")
    
    return "\n".join(config_lines)

def get_param_string(eq_key, dim):
    """Generate parameter string for printing"""
    eq_data = EQUATIONS[eq_key][dim]
    params = eq_data["params"]
    
    param_names = list(params.keys())
    param_prints = ", ".join([f"{name}={{config.{name}}}" for name in param_names])
    param_lower = ", ".join([f"{name.lower()}={{config.{name}}}" for name in param_names])
    
    return param_lower

# Create output files
base_path = Path("/home/haipham2407/Documents/QPINN/rd/Turing Patern/V2")

print(f"✓ Generated parameter specifications for all equations")
print(f"✓ Ready to generate v2 files")

