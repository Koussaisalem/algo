# Scripts Reference Guide

Quick reference for all training and utility scripts in the repository.

---

## Training Scripts

### 🧬 QM9 Score Model Training

**Location:** `projects/phononic-discovery/framework/scripts/`

#### Primary Training Scripts:

1. **`04_train_score_model.py`** ⭐
   - **Purpose:** Train denoising score model on QM9 molecular structures
   - **Model:** GNN-based score network for manifold diffusion
   - **Output:** `models/score_model_*.pt`
   - **Usage:**
     ```bash
     cd projects/phononic-discovery/framework/scripts
     python 04_train_score_model.py --epochs 100 --batch_size 32
     ```

2. **`04_train_score_model_v2.py`** ⭐
   - **Purpose:** Updated score model training with improvements
   - **Enhancements:** Better noise schedules, improved architecture
   - **Output:** `models/score_model_v2_*.pt`
   - **Usage:**
     ```bash
     python 04_train_score_model_v2.py --config configs/score_v2.yaml
     ```

### 🔷 TMD Score Model Training

**Location:** `projects/phononic-discovery/framework/scripts/tmd/`

#### TMD-Specific Training:

**`03_train_tmd_score_model.py`** ⭐
- **Purpose:** Train score model specifically for transition metal dichalcogenide (TMD) structures
- **Dataset:** Materials Project TMD subset
- **Special features:** 2D-constrained geometry, layered structure handling
- **Output:** `models/tmd_score_model_*.pt`
- **Usage:**
  ```bash
  cd projects/phononic-discovery/framework/scripts/tmd
  python 03_train_tmd_score_model.py --data_path ../../data/tmd_enriched.pt
  ```

---

## Surrogate Model Training

### Energy Prediction Models

**Location:** `projects/phononic-discovery/framework/scripts/`

**`03_train_surrogate.py`**
- **Purpose:** Train GNN surrogate for fast energy prediction
- **Architecture:** SchNet/PaiNN (E(3)-equivariant)
- **Training objective:** Joint energy + force matching
- **Output:** `models/surrogate/surrogate_state_dict.pt`
- **Usage:**
  ```bash
  python 03_train_surrogate.py --hidden_dim 256 --num_layers 8
  ```

**TMD Surrogate:** `tmd/02_train_tmd_surrogate.py`
- TMD-specific energy model
- Optimized for 2D structures

---

## Utilities

### Core Utilities

**Location:** `projects/phononic-discovery/framework/`

**`manifold_utils.py`** 🛠️
- **Purpose:** Stiefel manifold operations and unit conversions
- **Key functions:**
  - `mass_weighted_coordinates()`: Convert to mass-weighted frame
  - `project_to_tangent_space()`: Tangent space projection
  - `retract_to_manifold()`: QR-based retraction
  - `unit_conversions()`: eV ↔ Hartree, Å ↔ Bohr
- **Import:**
  ```python
  from manifold_utils import project_to_tangent_space, retract_to_manifold
  ```

---

## Complete Pipeline Scripts

### QM9 Molecular Discovery

**Location:** `projects/phononic-discovery/framework/scripts/`

| Script | Purpose | Runtime |
|--------|---------|---------|
| `01_prepare_data.py` | Extract QM9 subset | ~2 min |
| `02_enrich_dataset.py` | Compute xTB energies | ~30 min |
| `03_train_surrogate.py` | Train energy GNN | ~2 hours |
| `04_train_score_model.py` | ⭐ Score training | ~8 hours |
| `04_train_score_model_v2.py` | ⭐ Updated score | ~8 hours |
| `05_advanced_benchmark.py` | Evaluate performance | ~1 hour |
| `06_generate_molecules.py` | Generate structures | ~30 min |
| `analyze_enriched_dataset.py` | Visualizations | ~5 min |

### TMD Materials Discovery

**Location:** `projects/phononic-discovery/framework/scripts/tmd/`

| Script | Purpose | Runtime |
|--------|---------|---------|
| `00_download_materialsproject.py` | Download MP data | ~10 min |
| `01_enrich_tmd_dataset.py` | xTB enrichment | ~1 hour |
| `02_train_tmd_surrogate.py` | TMD energy model | ~3 hours |
| `03_train_tmd_score_model.py` | ⭐ TMD score | ~10 hours |
| `04_generate_tmd_structures.py` | Generate TMDs | ~45 min |
| `05_validate_with_dft.py` | DFT validation | ~variable |

**Additional TMD utilities:**
- Analysis scripts: `analyze_*.py` (15 files)
- Rescue scripts: `rescue_*.py` (for failed jobs)
- Validation scripts: `validate_*.py`

---

## Model Architectures

### Score Models

**QM9 Score Model:**
```
Input: (A, R, U, t)  # Adjacency, coords, orbitals, time
Architecture: 6-layer GNN with attention
Hidden dim: 256
Output: (score_R, score_U)  # Coordinate & orbital scores
```

**TMD Score Model:**
```
Input: (crystal_graph, coords, orbitals, t)
Architecture: E(3)-equivariant GNN
Layers: 8 message-passing blocks
Special: Periodic boundary handling
```

### Surrogate Models

**Energy Surrogate:**
```
Input: (A, R, U)
Architecture: SchNet (continuous convolutions)
Layers: 8
Output: E_pred, F_pred  # Energy and forces
Loss: λ_E ||E - E_DFT||² + λ_F ||F - F_DFT||²
```

---

## Quick Start by Task

### "I want to train a score model on QM9"

```bash
cd projects/phononic-discovery/framework/scripts
python 04_train_score_model.py
```

### "I want to train a score model on TMDs"

```bash
cd projects/phononic-discovery/framework/scripts/tmd
python 03_train_tmd_score_model.py
```

### "I want to train an energy surrogate"

```bash
cd projects/phononic-discovery/framework/scripts
python 03_train_surrogate.py
```

### "I want to use manifold utilities"

```python
import sys
sys.path.append('projects/phononic-discovery/framework')
from manifold_utils import project_to_tangent_space, retract_to_manifold

# Use the utilities
U_tangent = project_to_tangent_space(U, gradient)
U_new = retract_to_manifold(U + step_size * U_tangent)
```

---

## Configuration Files

**Default locations:**
- `configs/score_model.yaml` - Score model hyperparameters
- `configs/surrogate.yaml` - Surrogate training config
- `configs/diffusion.yaml` - Diffusion process settings

**Common hyperparameters:**
```yaml
# Score Model
learning_rate: 5e-4
batch_size: 64
num_epochs: 100
hidden_dim: 256
num_layers: 6

# Surrogate
learning_rate: 1e-3
batch_size: 128
lambda_energy: 1.0
lambda_forces: 10.0

# Diffusion
num_steps: 1000
beta_schedule: 'cosine'
gamma_schedule: 'cos_annealing'
```

---

## Troubleshooting

### Common Issues

**Issue:** "Can't find module `manifold_utils`"
```bash
# Solution: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/projects/phononic-discovery/framework"
```

**Issue:** "CUDA out of memory during score training"
```bash
# Solution: Reduce batch size
python 04_train_score_model.py --batch_size 16
```

**Issue:** "xTB not found for enrichment"
```bash
# Solution: Install via conda
conda install -c conda-forge xtb-python
```

---

## Related Documentation

- 💎 [**Stiefel Manifold Theory**](../theory/STIEFEL_MANIFOLD_THEORY.md) - Mathematical foundation
- 📐 [Architecture Overview](../architecture/OVERVIEW.md) - System design
- 🎯 [Phononic Discovery Project](../../projects/phononic-discovery/README.md) - Project guide

---

## Script Inventory Summary

**Total scripts:** 29+
- **Main pipeline:** 9 scripts (QM9 workflow)
- **TMD pipeline:** 20+ scripts (TMD-specific)
- **Utilities:** 1 core module (`manifold_utils.py`)

**All scripts preserved during restructure** ✅ (verified via `git mv` history)

---

**Last Updated:** 2025-01-11  
**Maintainer:** Koussai Salem
