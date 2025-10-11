# Fast-Track Validation for Mo₂Te₄

## Overview

This workflow implements a two-phase validation strategy to quickly assess whether the AI-generated Mo₂Te₄ structure is stable, before investing compute time in full DFT optimization.

**Total Time: ~30-35 minutes** (vs. hours for full validation)

## The Strategy

### Phase 1: xTB Pre-Relaxation (5 minutes)
- **Purpose**: Fix geometric issues (atom overlaps, bad angles) using fast semi-empirical method
- **Tool**: xTB (GFN2-xTB for transition metals)
- **Output**: Cleaned structure ready for DFT

### Phase 2: DFT Single-Point (20-30 minutes)
- **Purpose**: Calculate energy and forces at cleaned geometry (no optimization)
- **Tool**: GPAW real-space DFT with conservative settings
- **Output**: Stability verdict (GO/NO-GO signal)

## Quick Start

### Option 1: Automated (Recommended)
Run the all-in-one script:
```bash
cd /workspaces/algo/qcmd_hybrid_framework
bash scripts/tmd/fast_track_mo2te4.sh
```

This runs both phases automatically and gives you a clear verdict at the end.

### Option 2: Manual (Step-by-Step)

**Phase 1: Pre-relaxation**
```bash
cd /workspaces/algo/qcmd_hybrid_framework
python scripts/tmd/prerelax_mo2te4.py
```

**Phase 2: Single-point DFT**
```bash
python scripts/tmd/05_validate_with_dft.py \
    dft_validation/priority/Mo2Te4_prerelaxed.xyz \
    --mode fast \
    --single-point
```

## Interpreting Results

### ✅ GO SIGNAL (Max Force < 1.5 eV/Å)
**Meaning**: Structure is in a stable energy basin  
**Action**: Proceed with full optimization
```bash
python scripts/tmd/05_validate_with_dft.py \
    dft_validation/priority/Mo2Te4_prerelaxed.xyz \
    --mode production
```

### ⚠️ CAUTION (Max Force 1.5-3.0 eV/Å)
**Meaning**: Moderate instability, may converge with care  
**Action**: Try with more conservative settings:
```bash
# Use even lower mixing parameter
# Edit setup_dft_calculator() to use beta=0.015
python scripts/tmd/05_validate_with_dft.py \
    dft_validation/priority/Mo2Te4_prerelaxed.xyz \
    --mode fast
```

### ❌ NO-GO SIGNAL (Max Force > 3.0 eV/Å)
**Meaning**: Structure is highly unstable  
**Action**: Don't waste compute time. Options:
1. Try a different generated structure (you have 16 others!)
2. Analyze failure mode to improve generation
3. Consider this composition may not be viable

## Files Generated

```
dft_validation/priority/
├── Mo2Te4_candidate.xyz           # Original AI-generated structure
├── Mo2Te4_prerelaxed.xyz          # After xTB cleanup (Phase 1)
├── Mo2Te4_prerelax.traj           # xTB trajectory
└── xtb_opt.log                    # xTB optimization log

dft_validation/results/
├── Mo2Te4_prerelaxed_initial.xyz  # Structure with vacuum added
├── Mo2Te4_prerelaxed_results.json # Full results (energy, forces, etc.)
└── Mo2Te4_prerelaxed_opt.log      # DFT calculation log
```

## Troubleshooting

### xTB not installed
```bash
conda install -c conda-forge xtb-python
```

### DFT still fails with "charge sloshing"
The script uses very conservative settings (beta=0.025), but if issues persist:
1. Check the xTB pre-relaxation actually improved geometry
2. Try even coarser grid: edit `setup_dft_calculator()` to use `h=0.20`
3. Consider the structure may genuinely be unstable

### Memory errors
For this 6-atom system, memory should be fine. If issues occur:
- Use `mode='fast'` (h=0.18, ~2 GB RAM)
- Don't use `mode='converged'` for single-point (overkill)

## Why This Works

**Traditional approach**: Run full DFT optimization on raw AI structure
- Problem: AI structures may have geometric issues
- Result: DFT wastes hours trying to converge from bad starting point
- Outcome: Frustration and wasted compute

**Fast-track approach**: Clean first, then test
- xTB fixes geometric issues in 5 minutes (cheap)
- Single-point DFT tests stability in 30 minutes (moderate cost)
- Only invest hours if structure passes litmus test
- Outcome: Efficient use of compute budget

## Next Steps After Success

If you get a GO SIGNAL:

1. **Full optimization** (1-2 hours):
   ```bash
   python scripts/tmd/05_validate_with_dft.py \
       dft_validation/priority/Mo2Te4_prerelaxed.xyz \
       --mode production
   ```

2. **Band structure calculation** (2-3 hours):
   - Determine if semiconductor or metal
   - Calculate band gap

3. **Formation enthalpy** (1-2 hours):
   - Compare to competing phases (Mo₂, Te₄, MoTe₂)
   - Assess thermodynamic stability

4. **Materials Project comparison**:
   - Search for Mo₂Te₄ in database
   - If not found → Novel phase discovery! 🎉

## Time Budget Summary

| Stage | Time | Cumulative |
|-------|------|------------|
| xTB pre-relax | 5 min | 5 min |
| DFT single-point | 30 min | 35 min |
| **Decision point** | - | **35 min** |
| Full optimization | 1-2 hr | 2-3 hr |
| Band structure | 2-3 hr | 4-6 hr |
| Formation enthalpy | 1-2 hr | 5-8 hr |

**Key insight**: You get a reliable stability signal after just 35 minutes!
