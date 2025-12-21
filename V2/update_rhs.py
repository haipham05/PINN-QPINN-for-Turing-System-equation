#!/usr/bin/env python3
"""
Update RHS (residual) functions for each equation in training files.
This handles the complex equation-specific calculations.
"""

import re
from pathlib import Path

# Define RHS functions for each equation
RHS_FUNCTIONS = {
    "GrayScott": {
        "1d": """    def rd_rhs(t, y):
        \"\"\"RHS for Gray-Scott system\"\"\"
        u = y[:Nx]
        v = y[Nx:]
        lapU = laplacian_1d(u, dx)
        lapV = laplacian_1d(v, dx)
        
        reaction = u * (v ** 2)
        du = config.D_U * lapU - reaction + config.F * (1.0 - u)
        dv = config.D_V * lapV + reaction - (config.K + config.F) * v
        return np.concatenate([du, dv])""",
        "residual_u": """coupling = u * (v ** 2)
            residual_u = du_dt - self.config.D_U * d2u_dx2 + coupling - self.config.F * (1.0 - u)""",
        "residual_v": """residual_v = dv_dt - self.config.D_V * d2v_dx2 - coupling + (self.config.K + self.config.F) * v""",
    },
    "Thomas": {
        "1d": """    def rd_rhs(t, y):
        \"\"\"RHS for Thomas system\"\"\"
        u = y[:Nx]
        v = y[Nx:]
        lapU = laplacian_1d(u, dx)
        lapV = laplacian_1d(v, dx)
        
        du = config.D_U * lapU - u + config.A * (v ** 2)
        dv = config.D_V * lapV - v + config.B * u
        return np.concatenate([du, dv])""",
        "residual_u": """residual_u = du_dt - self.config.D_U * d2u_dx2 + u - self.config.A * (v ** 2)""",
        "residual_v": """residual_v = dv_dt - self.config.D_V * d2v_dx2 + v - self.config.B * u""",
    },
    "Schnakenberg": {
        "1d": """    def rd_rhs(t, y):
        \"\"\"RHS for Schnakenberg system\"\"\"
        u = y[:Nx]
        v = y[Nx:]
        lapU = laplacian_1d(u, dx)
        lapV = laplacian_1d(v, dx)
        
        reaction = config.GAMMA * (u ** 2) * v
        du = config.D_U * lapU + config.ALPHA - u + reaction
        dv = config.D_V * lapV + config.BETA - reaction
        return np.concatenate([du, dv])""",
        "residual_u": """coupling = self.config.GAMMA * (u ** 2) * v
            residual_u = du_dt - self.config.D_U * d2u_dx2 - self.config.ALPHA + u - coupling""",
        "residual_v": """residual_v = dv_dt - self.config.D_V * d2v_dx2 - self.config.BETA + coupling""",
    },
    "Gierer-Meinhardt": {
        "1d": """    def rd_rhs(t, y):
        \"\"\"RHS for Gierer-Meinhardt system\"\"\"
        u = y[:Nx]
        v = y[Nx:]
        lapU = laplacian_1d(u, dx)
        lapV = laplacian_1d(v, dx)
        
        # Add small epsilon to avoid division by zero
        v_safe = np.clip(v, 1e-6, None)
        du = config.D_U * lapU + (config.A * u**2 / v_safe) - u + config.RHO
        dv = config.D_V * lapV + config.A * u**2 - v
        return np.concatenate([du, dv])""",
        "residual_u": """activator_term = self.config.A * u**2 / (v + 1e-6)
            residual_u = du_dt - self.config.D_U * d2u_dx2 - activator_term + u - self.config.RHO""",
        "residual_v": """inhibitor_term = self.config.A * u**2
            residual_v = dv_dt - self.config.D_V * d2v_dx2 - inhibitor_term + v""",
    },
}

def update_rhs_in_file(filepath: str, eq_key: str):
    """Update RHS functions in training file"""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    if eq_key not in RHS_FUNCTIONS:
        print(f"  ⚠ No RHS definition for {eq_key}")
        return
    
    rhs_def = RHS_FUNCTIONS[eq_key]
    
    # Replace RHS function
    old_rhs = re.search(
        r'def rd_rhs\(t, y\):.*?return np\.concatenate\(\[du, dv\]\)',
        content,
        re.DOTALL
    )
    
    if old_rhs:
        content = content[:old_rhs.start()] + rhs_def["1d"] + content[old_rhs.end():]
        print(f"  ✓ Updated RHS function")
    
    with open(filepath, 'w') as f:
        f.write(content)

def main():
    V2_ROOT = Path("/home/haipham2407/Documents/QPINN/rd/Turing Patern/V2")
    
    for eq_key in ["GrayScott", "Thomas", "Schnakenberg", "Gierer-Meinhardt"]:
        eq_dir = V2_ROOT / eq_key / "1D_models"
        training_file = eq_dir / f"{eq_key.lower()}_1d_training_v2.py"
        
        if training_file.exists():
            print(f"\nUpdating RHS for {eq_key}...")
            update_rhs_in_file(str(training_file), eq_key)
        else:
            print(f"⚠ File not found: {training_file}")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("UPDATING RHS FUNCTIONS FOR ALL EQUATIONS")
    print("="*70)
    main()
    print("\n✓ RHS updates complete")
