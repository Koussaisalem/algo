# DFT Validation Guide

## 🎯 Quick Start

### Option 1: Parallel Execution (RECOMMENDED, 3× faster)

```bash
cd /workspaces/algo/qcmd_hybrid_framework
bash scripts/tmd/parallel_dft_launcher.sh
```

**Expected time:** ~20-30 minutes for all 3 structures  
**CPU usage:** 3 of 4 cores

---

### Option 2: Manual Parallel (More Control)

Open 3 separate terminals and run:

**Terminal 1:**
```bash
cd /workspaces/algo/qcmd_hybrid_framework
python scripts/tmd/05_validate_with_dft.py dft_validation/priority/Mo2Te4_candidate.xyz
```

**Terminal 2:**
```bash
cd /workspaces/algo/qcmd_hybrid_framework
python scripts/tmd/05_validate_with_dft.py dft_validation/priority/CrCuSe2_candidate.xyz
```

**Terminal 3:**
```bash
cd /workspaces/algo/qcmd_hybrid_framework
python scripts/tmd/05_validate_with_dft.py dft_validation/priority/VTe2_candidate.xyz
```

---

### Option 3: Serial Execution (Slower)

```bash
cd /workspaces/algo/qcmd_hybrid_framework
python scripts/tmd/05_validate_with_dft.py
```

**Expected time:** ~60-90 minutes  
**CPU usage:** 1 core

---

## 🔧 Technical Details

### What the Fixes Do

#### 1. Conservative Mixer (`beta=0.025`)
- **Problem:** Default mixer caused charge density oscillations ("charge sloshing")
- **Solution:** Use only 2.5% of new density per iteration
- **Trade-off:** More SCF iterations (~100-200 vs ~30-50), but guaranteed convergence
- **Analogy:** Like cruise control that adjusts speed gently instead of aggressively

#### 2. Parallel Execution
- **Problem:** Serial execution wastes 3 of 4 CPU cores
- **Solution:** Run 3 independent calculations simultaneously
- **Speedup:** 3× faster (30 min vs 90 min)
- **Implementation:** Each structure runs in separate Python process

### Mixer Parameters Explained

```python
Mixer(beta=0.025, nmaxold=5, weight=50.0)
```

- `beta=0.025`: Mix 2.5% new + 97.5% old density (conservative)
- `nmaxold=5`: Use last 5 iterations for extrapolation (improves convergence)
- `weight=50.0`: Kerker preconditioning (handles metallic/semi-metallic systems)

### GPAW Settings

| Parameter | Fast | Production | Converged |
|-----------|------|------------|-----------|
| Grid spacing (`h`) | 0.18 Å | 0.15 Å | 0.12 Å |
| k-points | (4,4,1) | (4,4,1) | (8,8,1) |
| Smearing | 0.1 eV | 0.05 eV | 0.01 eV |
| Max SCF iter | 300 | 300 | 400 |
| Time/structure | ~20 min | ~60 min | ~240 min |

---

## 📊 Monitoring Progress

### Check if jobs are running:
```bash
ps aux | grep validate_with_dft
```

### Monitor log files (in separate terminal):
```bash
# For parallel launcher:
tail -f dft_validation/parallel_logs/Mo2Te4_*.log
tail -f dft_validation/parallel_logs/CrCuSe2_*.log
tail -f dft_validation/parallel_logs/VTe2_*.log

# For manual execution:
tail -f gpaw_output.txt
```

### Check results:
```bash
ls -lh dft_validation/results/
cat dft_validation/results/*_results.json
```

---

## 🎯 Expected Convergence Behavior

### With Conservative Mixer (Fixed)

```
Iter  Energy (eV)   Change (eV)   Status
----  -----------   -----------   ------
  1   -123.456      -            Initial guess
  2   -125.234      -1.778       Large initial change (OK)
  3   -126.012      -0.778       Decreasing change
 ...
 50   -127.891      -0.012       Small oscillations
100   -127.892      -0.001       Converging smoothly
150   -127.892      -0.0001      Nearly converged
180   -127.892       0.00005     ✅ CONVERGED
```

**Key signs of success:**
- Energy changes decrease monotonically
- No large oscillations after iteration 10
- Converges within 300 iterations

### Without Mixer (What We Had)

```
Iter  Energy (eV)   Change (eV)   Status
----  -----------   -----------   ------
  1   -123.456      -            Initial guess
  2   -128.234      -4.778       Large change
  3   -121.891       6.343       ⚠️ Oscillating!
  4   -129.456      -7.565       ⚠️ Getting worse!
 ...
 50   -125.123       3.897       ⚠️ Still oscillating
100   -127.234      -2.111       ⚠️ Not converging
150   -125.891       1.343       ❌ FAILED (charge sloshing)
```

**Signs of failure (fixed by our mixer):**
- Energy oscillates wildly
- No convergence even after 100+ iterations
- Error: "Did not converge" or "charge sloshing detected"

---

## 🚨 Troubleshooting

### Error: "Too few plane waves or grid points"
- **Cause:** Grid spacing too coarse
- **Fix:** Already fixed (switched to `mode='fd'` with appropriate `h`)

### Error: "Did not converge" (charge sloshing)
- **Cause:** Default mixer too aggressive
- **Fix:** Already fixed (`Mixer(beta=0.025, ...)`)

### Error: "Memory allocation failed"
- **Cause:** Grid too fine or system too large
- **Fix:** Increase `h` (e.g., from 0.18 to 0.20) or reduce k-points

### Job killed unexpectedly
- **Cause:** Out of memory or timeout
- **Check:** `dmesg | grep -i kill` for OOM killer
- **Fix:** Run with coarser settings or request more RAM

---

## 🎉 After Validation Completes

### 1. Check Results
```bash
cd dft_validation/results
ls -lh
```

Expected files:
- `Mo2Te4_candidate_results.json`
- `CrCuSe2_candidate_results.json`
- `VTe2_candidate_results.json`
- `*_relaxed.xyz` (relaxed structures)
- `*_relaxed.cif` (for visualization)
- `validation_summary.json` (overall summary)

### 2. Extract Key Metrics
```bash
# Quick summary
for f in *_results.json; do
    echo "=== $f ==="
    jq '.structure_name, .total_energy_eV, .converged' "$f"
done
```

### 3. Compare to Surrogate Predictions
```bash
cat validation_summary.json | jq '.surrogate_comparison'
```

### 4. Next Steps
- If stable (E < 0, min_bond > 1.5 Å): **Run production mode** for accurate energies
- If unstable: **Analyze decomposition pathway**
- If validated: **Calculate band structures** (next script)

---

## 📝 Command Reference

```bash
# Show help
python scripts/tmd/05_validate_with_dft.py --help

# Validate single structure (fast mode)
python scripts/tmd/05_validate_with_dft.py path/to/structure.xyz

# Production mode (more accurate)
python scripts/tmd/05_validate_with_dft.py path/to/structure.xyz --mode production

# Force recomputation
python scripts/tmd/05_validate_with_dft.py path/to/structure.xyz --force

# Validate all structures (serial)
python scripts/tmd/05_validate_with_dft.py

# Parallel launcher (recommended)
bash scripts/tmd/parallel_dft_launcher.sh
```

---

## ⏱️ Time Estimates

| Structure | Atoms | Fast Mode | Production | Converged |
|-----------|-------|-----------|------------|-----------|
| Mo₂Te₄    | 6     | ~20 min   | ~60 min    | ~4 hours  |
| CrCuSe₂   | 4     | ~15 min   | ~45 min    | ~3 hours  |
| VTe₂      | 3     | ~12 min   | ~35 min    | ~2.5 hours|

**Total (serial):** ~47 min (fast), ~140 min (production)  
**Total (parallel):** ~20 min (fast), ~60 min (production)

---

## 🎓 Understanding the Output

### Structure Results JSON

```json
{
  "structure_name": "Mo2Te4_candidate",
  "converged": true,
  "total_energy_eV": -127.892,
  "energy_per_atom_eV": -21.315,
  "properties": {
    "min_bond_length": 2.73,     // Shortest Mo-Te bond (Å)
    "mean_bond_length": 2.85,    // Average bond length
    "max_force": 0.045,          // Max force on any atom (eV/Å)
    "z_extent": 3.21             // Thickness of layer (Å)
  }
}
```

**What to look for:**
- `converged: true` → Calculation succeeded ✅
- `energy_per_atom < -5 eV` → Likely stable
- `min_bond_length > 1.5 Å` → No atom overlap
- `max_force < 0.05 eV/Å` → Well-relaxed structure

---

## 🔬 Scientific Interpretation

### Mo₂Te₄ (E = -2.722 eV from surrogate)
- **If DFT confirms E < -2 eV:** Exceptionally stable, likely novel phase
- **If DFT gives E ~ 0 eV:** Metastable, may require kinetic stabilization
- **If DFT gives E > +2 eV:** Unstable, decomposes to MoTe₂ + Mo or Te

### CrCuSe₂ (E = +3.458 eV from surrogate)
- **Positive energy expected** (hetero-metallic alloy)
- **If DFT gives E < +2 eV:** Metastable, could be synthesized
- **If structure remains intact:** Local energy minimum exists
- **If atoms separate:** No stable mixed phase

### VTe₂ (E = -1.810 eV from surrogate)
- **Expect E ~ -1.5 to -2.0 eV** (known compound)
- **Compare to 1T-VTe₂ from Materials Project**
- **If match:** Validates surrogate accuracy ✅
- **If different:** Potentially novel polymorph

---

**Pro tip:** Run `fast` mode first to check convergence, then `production` mode for final energies!
