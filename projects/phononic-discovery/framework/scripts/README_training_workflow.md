# CMD-ECS Training and Generation Workflow

This document explains the complete training and inference pipeline for the CMD-ECS framework.

## Overview

The CMD-ECS system requires two neural models:

1. **Surrogate Model**: Predicts molecular energies (already trained)
2. **Score Model**: Predicts denoising directions on the manifold (to be trained)

## Current Status

### ✅ Completed Components

- **Data Preparation** (`01_prepare_data.py`): QM9 micro-dataset ready
- **Dataset Enrichment** (`02_enrich_dataset.py`): 5k molecules enriched with xTB
- **Surrogate Training** (`03_train_surrogate.py`): Energy predictor trained
- **Benchmarking** (`05_advanced_benchmark.py`): Manifold vs Euclidean comparison

### 🚧 In Development

- **Score Model Training** (`04_train_score_model.py`): Architecture challenge
- **Full Generation** (`06_generate_molecules.py`): Uses oracle scores for now

## The Score Model Challenge

### The Problem

The score model needs to:
- Take a **noisy manifold frame** U_t (shape: `n_atoms × k`)
- Predict the **denoising direction** toward the clean frame
- Work with **arbitrary orbital dimensions** k (not just k=3)

However, NequIP expects:
- **Cartesian coordinates** (shape: `n_atoms × 3`)
- Per-atom **force vectors** (shape: `n_atoms × 3`)

### Current Workaround

The benchmark uses an **oracle score**:
```python
def oracle_score(target_frame):
    return lambda U_t, t: target_frame - U_t
```

This is mathematically sound but doesn't generalize to new molecules.

### Solution Options

#### Option 1: Position-Based Score Model (Recommended for now)
Since manifold frames have rank k=3 for most molecules, we can:
1. Reconstruct Cartesian positions from manifold frames
2. Train score model to denoise positions
3. Project position updates back to tangent space

**Pros**: Works with existing NequIP architecture
**Cons**: Limited to k=3, indirect manifold learning

#### Option 2: Custom Manifold-Native Model
Build a score model that directly operates on `(n_atoms, k)` frames:
1. Use SE(3)-equivariant layers (e3nn, MACE, etc.)
2. Design architecture for variable k
3. Enforce tangent space constraints in outputs

**Pros**: Mathematically elegant, general k
**Cons**: Requires significant architecture work

#### Option 3: Hybrid Approach
Keep using oracle scores for validation, deploy energy steering via gamma:
```bash
python scripts/05_advanced_benchmark.py --gamma 0.1  # Enable MAECS
python scripts/06_generate_molecules.py --gamma 0.2  # Energy-guided generation
```

**Pros**: Validates MAECS immediately, surrogate already trained
**Cons**: Limited to template-based generation

## Recommended Immediate Workflow

### 1. Test MAECS with Oracle + Energy Guidance

```bash
cd /workspaces/algo/qcmd_hybrid_framework

# Run benchmark with energy steering enabled
python scripts/05_advanced_benchmark.py \
  --num-samples 64 \
  --num-steps 40 \
  --gamma 0.15 \
  --noise-scale 0.2 \
  --output-dir results/maecs_gamma_0.15

# Compare against pure manifold (gamma=0)
python scripts/05_advanced_benchmark.py \
  --num-samples 64 \
  --num-steps 40 \
  --gamma 0.0 \
  --noise-scale 0.2 \
  --output-dir results/pure_manifold_gamma_0.0
```

This validates the **Energy-Consistent Score (ECS)** component of QCMD-ECS!

### 2. Generate Molecules with Energy Guidance

```bash
# Generate with MAECS active
python scripts/06_generate_molecules.py \
  --num-samples 10 \
  --num-steps 50 \
  --gamma 0.1 \
  --noise-scale 0.3 \
  --output-dir results/generated_molecules_maecs

# Generate without energy guidance (comparison)
python scripts/06_generate_molecules.py \
  --num-samples 10 \
  --num-steps 50 \
  --gamma 0.0 \
  --noise-scale 0.3 \
  --output-dir results/generated_molecules_pure
```

### 3. Analyze Energy Distributions

The generated molecules will show:
- Whether MAECS produces lower-energy configurations
- How energy steering affects molecular geometry
- The interplay between manifold constraints and physics priors

## Future Development Path

### Phase 1: Validate MAECS (This Week)
- [x] Implement energy gradient computation
- [ ] Benchmark multiple gamma values (0.0, 0.05, 0.1, 0.2, 0.5)
- [ ] Analyze energy distributions in generated molecules
- [ ] Document when energy guidance helps vs hurts

### Phase 2: Scale Up Dataset (Next Week)
- [ ] Enrich 10k-50k molecules (current: 5k)
- [ ] Retrain surrogate on larger dataset
- [ ] Measure surrogate accuracy improvement
- [ ] Re-run benchmarks with better energy predictor

### Phase 3: Score Model (Future)
Choose architecture path:
- **Short-term**: Position-based denoising with NequIP
- **Long-term**: Custom SE(3)-equivariant manifold model

## Performance Notes

### Current Limitations

1. **Energy Accuracy**: Surrogate trained on only ~3600 molecules
   - High absolute errors (~90-100 eV)
   - Relative rankings likely still informative
   
2. **Score Model**: Using oracle (target - current)
   - Perfect for validation, limited for generation
   - Can't generalize beyond template molecules
   
3. **Computational Cost**: CPU-only mode in current setup
   - Each diffusion: ~5-6 ms per step
   - 50 steps: ~0.25-0.3s per molecule

### Scaling Recommendations

- **Immediate**: Test gamma sweep on current 5k dataset
- **Short-term**: Scale to 10k enriched molecules
- **Medium-term**: GPU acceleration for training/inference
- **Long-term**: 50k+ molecules, custom score architecture

## Key Research Questions

1. **Does MAECS improve generation quality?**
   - Compare final energies with gamma=0 vs gamma>0
   - Check if low-gamma helps vs high-gamma

2. **How robust is manifold constraint to energy steering?**
   - Does strong energy guidance break orthogonality?
   - Optimal gamma schedule?

3. **Can we scale the surrogate accuracy?**
   - More data vs deeper networks?
   - Transfer learning from pre-trained models?

## Contact & Collaboration

This framework is under active development. The architecture is sound, but practical
deployment requires resolving the score model training challenge. Contributions welcome!
