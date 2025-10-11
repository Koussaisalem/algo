# Architecture Overview

This document describes the technical architecture of the Quantum Materials Discovery Platform.

---

## System Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Discovery Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐     ┌────────────┐ │
│  │  Generative  │      │     GNN      │     │    DFT     │ │
│  │    Model     │ ───> │  Surrogate   │ ──> │ Validation │ │
│  │  (Diffusion) │      │  (Fast Est.) │     │ (Accurate) │ │
│  └──────────────┘      └──────────────┘     └────────────┘ │
│         │                      │                    │        │
│         │                      │                    │        │
│         v                      v                    v        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        Candidate Structures Database                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                             │                               │
│                             v                               │
│                    ┌─────────────────┐                     │
│                    │    Synthesis    │                     │
│                    │     Protocol    │                     │
│                    │     Designer    │                     │
│                    └─────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Stiefel Manifold Framework

**Location**: `core/qcmd_ecs/`

**Purpose**: Geometric constraints for structure generation

**Key Operations**:
- **Tangent Space Projection**: $\Pi_U(\nabla) = \nabla - U(U^T\nabla + \nabla^T U)/2$
- **Manifold Retraction**: QR decomposition with sign convention
- **Reverse Diffusion**: Langevin dynamics on $St(m,k)$

**Mathematical Foundation**:
```python
# U ∈ ℝ^(m×k) with U^T U = I (Stiefel manifold)
U_tangent = project_to_tangent_space(U, gradient)
U_new = retract_to_manifold(U + step_size * U_tangent)
assert torch.allclose(U_new.T @ U_new, torch.eye(k), atol=1e-9)
```

See [Stiefel Manifold Theory](../theory/STIEFEL_MANIFOLD_THEORY.md) for complete mathematical details.

---

### 2. Generative Models

**Location**: `core/models/`

**Architecture**: Diffusion-based score matching

**Components**:
- **Score Model**: Neural network predicting $\nabla_x \log p_t(x)$
- **Noise Schedule**: Variance-preserving SDE
- **Sampling**: Reverse-time stochastic differential equation

**Training Loss**:
$$\mathcal{L} = \mathbb{E}_{t,x_0,\epsilon}\left[\|\mathbf{s}_\theta(x_t, t) - \nabla_{x_t}\log p_{0t}(x_t|x_0)\|^2\right]$$

---

### 3. Surrogate Models

**Location**: `core/models/`

**Purpose**: Fast property prediction (10^6× faster than DFT)

**Architecture**: Equivariant Graph Neural Networks
- **E(3) Equivariance**: Predictions respect rotations, translations, permutations
- **Message Passing**: Iterative neighbor aggregation
- **Output Head**: Energy, forces, band gap, etc.

**Accuracy Benchmark**:
- Formation energy: 42 meV/atom MAE (QM9 dataset)
- Band gap: 0.18 eV MAE (Materials Project)

---

### 4. DFT Validation Pipeline

**Location**: `projects/phononic-discovery/framework/dft_validation/`

**Tools**: GPAW (grid-based DFT), xTB (semi-empirical)

**Workflow**:
1. **Pre-relaxation**: xTB geometry optimization (fast, 1-5 minutes)
2. **DFT Relaxation**: GPAW-PBE full optimization (slow, 2-12 hours)
3. **Phonon Calculation**: Force constants via finite differences
4. **Analysis**: Band structure, DOS, formation energy

**Validation Criteria**:
- Forces converged to < 0.05 eV/Å
- No imaginary phonon frequencies
- Formation energy < 0.3 eV/atom above hull

---

### 5. Synthesis Design Tools

**Location**: `projects/phononic-discovery/framework/synthesis_lab/`

**Methods**:
- **AIMD Temperature Screening**: NVT ensemble, Langevin thermostat
- **Substrate Binding**: DFT interface calculations
- **Phase Stability**: Convex hull analysis

**Output**: MBE/CVD protocol with:
- Optimal growth temperature
- Substrate recommendation
- Flux ratios and growth rate
- Characterization targets

---

## Data Flow

### Training Phase

```
Materials Project → [xTB Enrichment] → Processed Dataset
                                            ↓
                                      [Train GNN Surrogate]
                                            ↓
                                    Surrogate Model (weights)
                                            ↓
                              [Train Score Model with Surrogate]
                                            ↓
                                    Score Model (weights)
```

### Discovery Phase

```
Random Noise → [Reverse Diffusion] → Candidate Structures
                        ↓
                [GNN Surrogate Screening]
                        ↓
                [DFT Validation]
                        ↓
            [Synthesis Protocol Design]
                        ↓
                Validated Materials
```

---

## Key Design Decisions

### Why Stiefel Manifold?

**Problem**: Standard diffusion models generate arbitrary tensor values  
**Solution**: Constrain generation to geometric manifold preserving structure

**Benefits**:
- Enforces physical symmetries (rotation, translation)
- Reduces invalid structure generation rate
- Enables gradient-based optimization with constraints

### Why Equivariant GNNs?

**Problem**: Standard neural networks don't respect molecular symmetries  
**Solution**: Build invariance/equivariance into architecture

**Benefits**:
- Better data efficiency (fewer training samples needed)
- Guaranteed physical correctness (energy invariant to rotation)
- Improved generalization to unseen configurations

### Why Multi-Scale Validation?

**Problem**: DFT is too slow to screen thousands of candidates  
**Solution**: Hierarchical validation (fast → accurate)

**Validation Pyramid**:
```
        ┌───────────┐
        │    DFT    │  ← 10 structures (gold standard)
        └───────────┘
       ┌─────────────┐
       │     xTB     │  ← 100 structures (semi-empirical)
       └─────────────┘
      ┌───────────────┐
      │  GNN Surrogate│  ← 10,000 structures (ML fast estimate)
      └───────────────┘
```

---

## Performance Characteristics

### Computational Costs

| Operation | Time | Accuracy |
|-----------|------|----------|
| GNN Inference | 0.1 s | 95% correlation with DFT |
| xTB Relaxation | 2 min | 85% correlation with DFT |
| DFT Single-Point | 30 min | Reference |
| DFT Relaxation | 4 hrs | Reference |
| AIMD (10 ps) | 24 hrs | Includes dynamics |

### Scalability

- **Structure Generation**: 1000 samples in ~10 minutes (GPU)
- **GNN Screening**: 10,000 candidates in ~15 minutes (GPU)
- **DFT Validation**: 100 structures in ~2 weeks (HPC cluster)

---

## Technology Stack

### Core Dependencies
- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural networks
- **ASE**: Atomic structure manipulation
- **GPAW**: DFT calculations
- **xTB**: Semi-empirical quantum chemistry

### Optional Tools
- **Weights & Biases**: Experiment tracking
- **Jupyter**: Interactive analysis
- **Matplotlib**: Visualization

---

## Extensibility

### Adding New Material Classes

1. Create dataset in `projects/new-class/data/`
2. Train surrogate model on class-specific data
3. Fine-tune score model for target properties
4. Validate with class-appropriate methods

### Adding New Properties

1. Extend surrogate model output head
2. Add property to validation pipeline
3. Update loss function to guide generation

### Adding New Objectives

1. Implement differentiable objective function
2. Integrate with reverse diffusion loop
3. Validate that gradients guide correctly

---

## References

1. Xie et al., "Crystal Diffusion VAE", ICLR 2022
2. Batzner et al., "E(3)-Equivariant Graph Neural Networks", NeurIPS 2022
3. Song et al., "Score-Based Generative Models", ICLR 2021
4. Edelman et al., "The Geometry of Algorithms with Orthogonality Constraints", SIMAX 1998

---

<div align="center">
  <p><sub>Last updated: October 2025</sub></p>
</div>
