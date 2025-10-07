# QCMD-ECS TMD Knockout: Quick Reference Checklist

**Goal:** Demonstrate 3-5x improvement in 2D semiconductor generation  
**Timeline:** 4 weeks to submission  
**Target Journal:** Nature Materials (IF: 47.6)

---

## 📋 Week 1: Data Infrastructure

### Day 1-2: Dataset Acquisition
- [ ] Download C2DB database (~2GB)
- [ ] Filter for TMD structures (target: 300-500)
- [ ] Extract properties: positions, cell, energy, band gap
- [ ] **Output:** `data/tmd/tmd_raw.pt`

### Day 3-5: Electronic Structure Enrichment
- [ ] Run GFN2-xTB on all structures
- [ ] Handle 2D periodic boundary conditions
- [ ] Extract orbital coefficients (or use fallback)
- [ ] Compute mass-weighted manifold frames
- [ ] **Output:** `data/tmd/tmd_xtb_enriched.pt` (300-500 structures)

### Day 6-7: Data Validation
- [ ] Analyze band gap distribution
- [ ] Check orbital orthogonality
- [ ] Identify outliers
- [ ] Generate data quality report
- [ ] **Decision Point:** Dataset quality check

---

## 📋 Week 2: Model Training

### Day 8-10: Surrogate Model
- [ ] Adapt architecture for l_max=2 (d-orbitals)
- [ ] Train on band gap + formation energy
- [ ] Monitor validation MAE (target: <0.2 eV)
- [ ] Save best checkpoint
- [ ] **Output:** `models/tmd_surrogate/best_model.pt`

### Day 11-14: Score Model
- [ ] Implement VectorOutputModel with l_max=2
- [ ] Multi-noise training (0.05, 0.1, 0.2, 0.3)
- [ ] Position-based score prediction
- [ ] Monitor validation loss (target: <0.01)
- [ ] **Output:** `models/tmd_score_model/best_model.pt`
- [ ] **Decision Point:** Training convergence check

---

## 📋 Week 3: Generation & Validation

### Day 15-16: Structure Generation
- [ ] Generate 200 structures with γ=0.0 (pure score)
- [ ] Generate 200 structures with γ=0.1 (optimal)
- [ ] Generate 200 structures with γ=0.3 (strong guidance)
- [ ] **Output:** 600 TMD structures (.xyz files)
- [ ] **Decision Point:** Generation quality check

### Day 17-19: DFT Validation
- [ ] Select top 100 structures (lowest energy)
- [ ] Run GPAW PBE calculations
- [ ] Relax structures (BFGS, fmax=0.05)
- [ ] Compute band structures
- [ ] **Output:** `results/tmd_validation/summary.json`

### Day 20-21: Baseline Comparison
- [ ] Run Euclidean baseline (200 samples)
- [ ] Run Euclidean+Retract baseline (200 samples)
- [ ] Validate baselines with DFT (100 samples)
- [ ] **Output:** `results/baselines/`
- [ ] **Decision Point:** Results quality check

---

## 📋 Week 4: Analysis & Paper

### Day 22-24: Comprehensive Analysis
- [ ] Compute all metrics (validity, band gap MAE, orthogonality)
- [ ] Generate comparison tables
- [ ] Create figures (structures, band gaps, distributions)
- [ ] Ablation study (gamma effects)
- [ ] Novelty analysis (t-SNE, diversity metrics)
- [ ] **Output:** `results/paper_figures/`

### Day 25-28: Manuscript Writing
- [ ] Day 25: Draft Introduction + Methods
- [ ] Day 26: Draft Results + create all figures
- [ ] Day 27: Draft Discussion + Abstract
- [ ] Day 28: Polish, format, prepare supplementary
- [ ] **Output:** Complete manuscript for Nature Materials
- [ ] **Final Decision Point:** Submit or iterate?

---

## 🎯 Success Criteria

### Minimum Viable Success (Go to Publication)
- [ ] ≥85% valid TMD structures (vs 60-70% SOTA)
- [ ] <0.3 eV band gap MAE (vs 0.9 eV SOTA)
- [ ] <10^-10 orthogonality error (vs 10^1 SOTA)
- [ ] ≥3 novel structures validated as stable

### Target Success (Nature Materials)
- [ ] ≥92% valid structures
- [ ] <0.2 eV band gap MAE
- [ ] <10^-12 orthogonality error
- [ ] ≥5 novel structures with interesting properties

### Stretch Goals
- [ ] ≥95% valid structures
- [ ] <0.1 eV band gap MAE
- [ ] Nature/Science consideration
- [ ] Experimental synthesis initiated

---

## 🚧 Go/No-Go Decision Points

### Day 7: Data Quality
**Question:** Is dataset ready?  
**Criteria:** 300+ structures, <10% xTB failures, band gaps reasonable  
**If No-Go:** Extra 3-5 days for GPAW enrichment

### Day 14: Training
**Question:** Are models learning?  
**Criteria:** Surrogate MAE <0.3 eV, score loss decreasing  
**If No-Go:** Debug architecture, try different hyperparameters

### Day 21: Generation
**Question:** Are structures reasonable?  
**Criteria:** 70%+ pass validity checks, sensible energies  
**If No-Go:** Re-train with better hyperparameters

### Day 24: Validation
**Question:** Do results support claims?  
**Criteria:** Band gap MAE < SOTA by 2x, orthogonality < SOTA by 10x  
**If No-Go:** Revise claims, methodology paper instead

---

## 📊 Expected Results

| Metric | SOTA | QCMD-ECS | Improvement |
|--------|------|----------|-------------|
| Valid structures | 70% | **92%** | **31% ↑** |
| Band gap MAE | 0.9 eV | **0.15 eV** | **6x ↓** |
| Orthogonality | 10^1 | **10^-14** | **10^15x ↓** |
| Novel stable | 35% | **75%** | **114% ↑** |

---

## 💰 Budget Estimate

- **Training:** $120-180 (60 GPU-hours)
- **Generation:** $10-20 (100 CPU-hours)
- **DFT Validation:** $600-900 (300 GPU-hours)
- **Total:** ~$730-1100

---

## 🚀 Immediate Action

**Right now:**
```bash
# Wait for QM9 score training to complete
tail -f qcmd_hybrid_framework/score_training.log

# Then execute:
cd /workspaces/algo/qcmd_hybrid_framework
mkdir -p scripts/tmd data/tmd models/tmd_surrogate models/tmd_score_model

# Start Week 1, Day 1
python scripts/tmd/00_download_c2db.py
```

---

## 📞 Status Updates

Track progress daily with this format:

```
Date: YYYY-MM-DD
Week: X, Day: Y
Current task: [Task name]
Status: On track / Behind / Ahead
Blockers: [None / Description]
Next 24h: [Plan]
```

---

## 🎉 Success Declaration

When all checkboxes are ticked:

✅ Dataset: 300+ enriched TMD structures  
✅ Models: Trained surrogate + score model  
✅ Generation: 600+ structures generated  
✅ Validation: 100+ DFT-validated structures  
✅ Results: 3-5x improvement demonstrated  
✅ Manuscript: Complete and polished  

**→ SUBMIT TO NATURE MATERIALS** 🚀

---

**Last Updated:** 2025-10-06  
**Status:** Ready to execute Week 1  
**Confidence:** 85% knockout probability
