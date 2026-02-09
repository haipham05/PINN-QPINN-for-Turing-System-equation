# Comparison: RD_EXAMPLE vs Existing Models

## 📊 Overview

**RD_EXAMPLE** is an **enhanced template/reference implementation** of the QPINN training and inference framework. It serves as a more feature-rich baseline that can be adapted to any 2D reaction-diffusion system, including all 6 Turing patterns (Brusselator, Thomas, Schnakenberg, etc.).

---

## 1. File Structure Comparison

| Aspect | RD_EXAMPLE | Existing Models |
|--------|-----------|-----------------|
| **Training file size** | 1,457 lines (58 KB) | 1,409-1,413 lines (58 KB) |
| **Inference file** | ✓ Included | ✓ Included |
| **Classes** | 4 main classes | 4 main classes |
| **Trainer methods** | 13 methods | 9 methods |
| **Domain** | t ∈ [0, 10], x,y ∈ [-1, 1] | t ∈ [0, 1], x,y ∈ [0, 1] |

---

## 2. Configuration Parameters

### RD_EXAMPLE Config
```python
# Quantum Parameters
N_LAYERS = 5              # Quantum circuit layers
N_WIRES = 6               # Quantum wires (qubits)

# PINN-specific Parameters (NEW!)
PINN_HIDDEN_LAYERS = 4    # Hidden layers for PINN
PINN_NEURONS = 50         # Neurons in PINN

# FNN Basis Parameters
HIDDEN_LAYERS_FNN = 2     # FNN hidden layers
NEURONS_FNN = 10          # FNN neurons per layer

# QNN Embedding Parameters  
N_WIRES_EMBED = 4         # Separate embedding wires (NEW!)
N_LAYERS_EMBED = 2        # Embedding layers
```

### Existing Models Config
```python
# Quantum Parameters
N_LAYERS = 5
N_WIRES = 4               # Only 4 wires

# FNN Basis Parameters
HIDDEN_LAYERS_FNN = 2
NEURONS_FNN = 20          # More neurons (20 vs 10)

# QNN Embedding Parameters
N_LAYERS_EMBED = 2
# No separate N_WIRES_EMBED parameter
```

### Key Configuration Differences

| Parameter | RD_EXAMPLE | Existing | Note |
|-----------|-----------|----------|------|
| N_WIRES | 6 | 4 | RD has more quantum wires |
| NEURONS_FNN | 10 | 20 | RD uses fewer FNN neurons |
| N_WIRES_EMBED | 4 | N/A | RD separates embedding wires |
| PINN_HIDDEN_LAYERS | 4 | N/A | RD has dedicated PINN config |
| PINN_NEURONS | 50 | N/A | RD has dedicated PINN config |

---

## 3. Class Structure Comparison

### Common Classes (Identical/Similar)

| Class | RD_EXAMPLE | Existing | Status |
|-------|-----------|----------|--------|
| **FNNBasisNet** | 2 methods | 2 methods | ✅ Identical |
| **QNNEmbedding** | 3 methods | 3 methods | ✅ Identical |
| **TrainingVisualizer** | 5 methods | 5 methods | ✅ Identical |

### Trainer Class Comparison

**Existing Models (9 methods):**
```
✓ __init__
✓ _setup_collocation_points
✓ _load_reference_solution
✓ _create_circuit
✓ _postprocess_output
✓ model
✓ _create_loss_functions
✓ train_model
✓ save_model
```

**RD_EXAMPLE (13 methods):**
```
✓ __init__
✓ _setup_domain                    (similar to _setup_collocation_points)
✓ _load_reference                  (similar to _load_reference_solution)
✓ _create_circuit
✓ _postprocess_output
✓ model
✓ _create_loss_functions
✓ train_model
✓ save_model
+ run_comparison                   (NEW - compare all 3 models)
+ plot_training_results            (NEW - advanced visualization)
+ get_predictions                  (NEW - inference wrapper)
+ analyze_performance              (NEW - metrics analysis)
```

---

## 4. Method Naming Conventions

| Function | RD_EXAMPLE | Existing |
|----------|-----------|----------|
| Setup collocation | `_setup_domain()` | `_setup_collocation_points()` |
| Load reference | `_load_reference()` | `_load_reference_solution()` |
| Main training loop | `train_model()` | `train_model()` |
| Model forward pass | `model()` | `model()` |

**Observation:** Slightly different names but equivalent functionality.

---

## 5. Physics Parameters

### RD_EXAMPLE PDE
```
Domain: t ∈ [0, 10], x ∈ [-1, 1], y ∈ [-1, 1]

∂A/∂t = D_A (∂²A/∂x² + ∂²A/∂y²) + k1 A² S - k2 A
∂S/∂t = D_S (∂²S/∂x² + ∂²S/∂y²) - k1 A² S + k3

Variables: A (Activator), S (Substrate)
```

### Existing Models (Example: Brusselator 2D)
```
Domain: t ∈ [0, 1], x ∈ [0, 1], y ∈ [0, 1]

∂u/∂t = μ (∂²u/∂x² + ∂²u/∂y²) + u²v - (ε+1)u + β
∂v/∂t = μ (∂²v/∂x² + ∂²v/∂y²) - u²v + εu

Variables: u (species 1), v (species 2)
```

---

## 6. Extra Features in RD_EXAMPLE

### ✨ Additional Methods

1. **`run_comparison()`**
   - Train all 3 models (PINN, FNN-TE-QPINN, QNN-TE-QPINN)
   - Compare performance side-by-side
   - Generate comparison plots

2. **`plot_training_results()`**
   - More comprehensive visualization
   - Loss evolution for each model
   - Error comparison

3. **`get_predictions()`**
   - Wrapper for inference
   - Easier model evaluation

4. **`analyze_performance()`**
   - Compute detailed metrics
   - Performance statistics

### 📊 Visualization Enhancements
- More plot types (7 plots vs 4-5 in existing)
- Better organized visualization code
- More detailed training analysis

---

## 7. Similarities (Core Architecture)

### 🎯 Identical Components

✅ **Neural Network Classes:**
- `FNNBasisNet` - exactly same
- `QNNEmbedding` - exactly same

✅ **Training Methodology:**
- LBFGS optimizer
- Loss computation (PDE + IC + BC)
- Gradient-based optimization

✅ **Quantum Circuit Logic:**
- Same gate sequences
- Same tensor encoding
- Same parameterization

✅ **Collocation Point Strategy:**
- Interior points for PDE loss
- Boundary points for BC loss
- Initial condition points for IC loss

✅ **Reference Solution:**
- RK45 solver
- RegularGridInterpolator
- Same numerical methodology

---

## 8. Differences Summary

| Aspect | RD_EXAMPLE | Existing | Significance |
|--------|-----------|----------|--------------|
| **PINN Architecture** | Separate config | Uses FNN config | More flexible |
| **Quantum Wires** | 6 | 4 | Larger quantum system |
| **FNN Neurons** | 10 | 20 | Different capacity |
| **Embedding Wires** | Separate (4) | Same as main (4) | Explicit separation |
| **Analysis Methods** | 4 extra | None | Better metrics |
| **Domain Size** | t ∈ [0,10] | t ∈ [0,1] | Longer time horizon |
| **Comparison Tools** | Built-in | N/A | Easier benchmarking |

---

## 9. Architecture Diagram

```
Both implementations share identical architecture:

INPUT (t, x, y)
    ↓
[Domain Rescaling to [-0.95, 0.95]]
    ↓
┌───────────────────────────────────┐
│  Three Model Pathways              │
└───────────────────────────────────┘
    ↓                ↓                ↓
[PINN]      [FNN-TE-QPINN]    [QNN-TE-QPINN]
  │              │                  │
  │          FNNBasisNet        QNNEmbedding
  │              │                  │
  └──────┬───────┴──────────────────┘
         ↓
    [Quantum Circuit]
         ↓
    [Post-processing]
         ↓
    OUTPUT (u, v) or (A, S)
```

---

## 10. Use Cases

### RD_EXAMPLE Is Best For:
- **Reference Implementation** - Shows best practices
- **Template Development** - Base for new PDEs
- **Benchmarking** - Built-in comparison tools
- **Analysis** - Advanced visualization
- **Teaching** - Well-commented code

### Existing Models Are Best For:
- **Specific PDEs** - Optimized for particular systems
- **Production Use** - Tuned hyperparameters
- **Publication** - Research-ready code
- **Reproducibility** - Fixed parameters

---

## 11. Conclusion

**RD_EXAMPLE** and **Existing Models** share the same **core QPINN framework** with differences in:

1. ✅ **Configuration flexibility** - RD_EXAMPLE has more options
2. ✅ **Analysis capabilities** - RD_EXAMPLE has more tools
3. ✅ **Hyperparameters** - Different tuning choices
4. ✅ **Domain specifications** - Different problem sizes

**Both are essentially the same architecture**, just with RD_EXAMPLE being a more **feature-complete template** version.

### Recommendation:
- Use **RD_EXAMPLE** as starting point for new reaction-diffusion systems
- Use **Existing Models** for validated/published research
- Both can be cross-adapted with minimal modifications
