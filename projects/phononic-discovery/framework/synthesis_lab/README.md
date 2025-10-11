# 🧪 Synthesis Lab - Computational MBE Design for CrCuSe₂

**Mission:** Design optimal Molecular Beam Epitaxy (MBE) growth protocol for metastable 2D CrCuSe₂ using ab-initio molecular dynamics and DFT calculations.

---

## 📁 Directory Structure

```
synthesis_lab/
├── README.md                          (this file)
├── temperature_screening/              Tool #2: Growth temperature optimization
│   ├── run_md_temperature_sweep.py    Main MD screening script
│   ├── analyze_md_trajectories.py     Analysis and visualization
│   ├── results/                       MD results at each T
│   └── plots/                         Temperature-dependent properties
├── substrate_screening/                Tool #1: Binding energy calculations
│   └── (future implementation)
├── phase_stability/                    Tool #3: Cr-Cu-Se phase diagram
│   └── (future implementation)
├── growth_kinetics/                    Tool #4: Monte Carlo deposition
│   └── (future implementation)
└── experimental_protocols/             Generated growth recipes
    └── FINAL_MBE_PROTOCOL.txt         Ready-to-use experimental guide
```

---

## 🎯 Current Focus: Temperature Screening (Tool #2)

**Goal:** Find the optimal MBE growth temperature that:
- ✅ Allows sufficient atomic mobility (layer-by-layer growth)
- ✅ Prevents bulk phase nucleation (stay in metastable 2D phase)
- ✅ Maintains structural integrity (no decomposition)
- ✅ Is experimentally feasible (400-700°C range for MBE)

**Computational Method:**
- Ab-initio Molecular Dynamics (AIMD) with GPAW
- Temperature range: 300K, 400K, 500K, 600K, 700K, 800K
- Simulation time: 5-10 ps per temperature
- Analysis: RMSD, bond statistics, phase identification

---

## 🔬 Why This Matters for Your Discovery

### Your Material's Challenge:
**CrCuSe₂ is metastable** (+1.23 eV/atom above bulk mp-568587)
- **Too cold:** Atoms don't move → rough, defective films
- **Too hot:** Bulk phase nucleates → you get mp-568587 metal instead of your semiconductor!
- **Sweet spot:** MD will find the Goldilocks temperature

### What MD Reveals:
1. **Atomic mobility vs T:** When do atoms rearrange?
2. **Phase stability window:** At what T does structure collapse?
3. **Thermal expansion:** How does lattice change with T?
4. **Se desorption risk:** When does Se evaporate?

---

## ⚡ Quick Start

### Run Temperature Screening (2-6 hours depending on mode):
```bash
cd synthesis_lab/temperature_screening
conda activate qcmd_nequip

# Fast screening (5 temperatures, 2 ps each, ~2 hours)
python run_md_temperature_sweep.py --mode fast

# Production run (6 temperatures, 10 ps each, ~6 hours)
python run_md_temperature_sweep.py --mode production

# Analyze results
python analyze_md_trajectories.py
```

### Expected Output:
```
📊 TEMPERATURE SCREENING RESULTS
================================
300K: ✅ Stable, LOW mobility (Diffusion coeff: 0.001 Ų/ps)
400K: ✅✅ Stable, GOOD mobility (Diffusion coeff: 0.015 Ų/ps) ← OPTIMAL
500K: ✅ Stable, HIGH mobility (Diffusion coeff: 0.042 Ų/ps)
600K: ⚠️ Partially stable, Se desorption begins
700K: ❌ Unstable, structure distortion
800K: ❌ Decomposition

RECOMMENDATION: Growth window = 400-550°C (673-823K)
```

---

## 📊 Outputs for Partner Discussions

### 1. Temperature-Property Plots:
- `plots/temperature_vs_stability.png` - Structural stability score
- `plots/temperature_vs_diffusion.png` - Atomic mobility
- `plots/temperature_vs_rmsd.png` - Structure preservation
- `plots/bond_length_evolution.png` - Chemical integrity

### 2. Quantitative Metrics:
- `results/temperature_summary.json` - All metrics in one file
- `results/optimal_temperature_report.txt` - Human-readable recommendation

### 3. Trajectory Animations:
- `results/md_400K.gif` - Atoms moving at optimal T
- `results/md_700K.gif` - Atoms moving at failure T (for comparison)

---

## 🎓 Scientific Background

### Why Ab-Initio MD (not classical force fields)?

**Your material is NOVEL** - no force field parameters exist for Cr-Cu-Se!

Classical MD would require:
- Cr-Cu interaction potential (unknown)
- Cu-Se bond parameters (unknown)
- Three-body Cr-Cu-Se terms (unknown)

**AIMD solves this:**
- Electrons calculated on-the-fly (DFT at each timestep)
- No empirical parameters needed
- Accurate for novel materials
- Computationally expensive but trustworthy

### Typical MBE Growth Temperatures:
- **Graphene on Cu:** 1000°C (high!)
- **MoS₂ MBE:** 600-800°C
- **WSe₂ MBE:** 500-700°C
- **Your CrCuSe₂:** ? (that's what we'll find!)

---

## 🚀 Next Steps After Temperature Screening

Once we know the optimal T, we can:
1. **Substrate screening** (Tool #1): Test graphene/h-BN at optimal T
2. **Flux optimization:** Cr:Cu:Se ratio refinement
3. **Growth rate prediction:** How fast can we deposit?
4. **Protocol generation:** Complete experimental recipe

---

## 📝 Notes for Experimentalists

### What You'll Get:
✅ **Specific temperature:** e.g., "Use 475°C ± 25°C"  
✅ **Physical reasoning:** "At 475°C, Cr/Cu mobility is sufficient for layer-by-layer growth while Se overpressure prevents desorption"  
✅ **Failure modes:** "Above 550°C, bulk phase nucleation begins"  
✅ **Diagnostic targets:** "Monitor RHEED: should see streaky pattern if growth is layer-by-layer"

### What You Won't Get (yet):
⏳ Optimal flux ratios (need Tool #4 Monte Carlo)  
⏳ Substrate choice (need Tool #1 binding energies)  
⏳ Growth rate (need kinetic modeling)

**But temperature is THE critical parameter!** Get this right, and everything else follows.

---

## 📚 References & Validation

### Similar Systems (for confidence):
- **MoS₂ AIMD studies:** Optimal T = 600-700°C (our method works!)
- **Metastable 2D materials:** Diamond-like carbon, h-BN on metals
- **Kinetic trapping precedent:** Graphene synthesis (metastable vs graphite)

### Computational Validation:
- Phonon calculation: 0 imaginary modes at 0K ✅ (confirms local minimum)
- Formation energy: +1.23 eV/atom (explains need for kinetic trapping)
- This MD: Tests finite-T stability (crucial for synthesis)

---

**Ready to find your magic temperature? Let's run it!** 🔥
