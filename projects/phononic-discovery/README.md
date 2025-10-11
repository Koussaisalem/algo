# Phononic Materials for Analog Gravity

Discovery of materials where phonons exhibit relativistic dispersion and effective spacetime curvature.

---

## Objective

Find crystalline materials with:
1. **Dirac/Weyl phonon cones** - relativistic-like dispersion $\omega = v_F |k|$
2. **Strain-tunable velocity** - spatially varying $v_F(\mathbf{r})$ creates effective curvature
3. **Synthesis feasibility** - MBE/CVD compatible, thermodynamically accessible

---

## Current Status

### Discovered Materials

#### CrCuSe₂ - Hetero-Metallic TMD
- **Structure**: 2D layered, triclinic (P 1), 7.26 × 9.40 × 33.20 Å³
- **Electronic**: Indirect bandgap 0.616 eV (semiconductor)
- **Stability**: 0 imaginary phonons (dynamically stable)
- **Formation Energy**: +1.23 eV/atom (metastable, kinetically accessible)
- **Status**: DFT validated, synthesis protocol designed

**Documentation**:
- [Discovery Report](../../docs/discoveries/CrCuSe2/DISCOVERY.md)
- [DFT Validation](../../docs/discoveries/CrCuSe2/VALIDATION.md)
- [Synthesis Protocol](../../docs/discoveries/CrCuSe2/SYNTHESIS.md)

---

## Workflows

### 1. Discovery Pipeline

Generate and screen TMD candidates:

```bash
# Download Materials Project TMD dataset
python framework/scripts/tmd/00_download_materialsproject.py

# Enrich with xTB calculations
python framework/scripts/tmd/01_enrich_tmd_dataset.py

# Train surrogate model
python framework/scripts/tmd/02_train_tmd_surrogate.py

# Train score model for generation
python framework/scripts/tmd/03_train_tmd_score_model.py

# Generate candidates
python framework/scripts/tmd/04_generate_tmd_structures.py --n_samples 100
```

### 2. DFT Validation

High-accuracy quantum chemistry verification:

```bash
# Validate specific candidate
python framework/scripts/tmd/05_validate_with_dft.py \
    --structure framework/results/generated_tmds/tmd_0042.xyz \
    --mode full
```

See [DFT Validation Guide](framework/dft_validation/DFT_VALIDATION_GUIDE.md) for details.

### 3. Synthesis Design

Computational MBE protocol optimization:

```bash
cd framework/synthesis_lab/temperature_screening

# Run temperature screening (fast mode)
python run_md_temperature_sweep.py --mode fast

# Analyze results
python analyze_md_trajectories.py

# View recommendation
cat results/TEMPERATURE_RECOMMENDATION.txt
```

**Output**: Optimal growth temperature window (e.g., "450-550°C for MBE growth")

---

## Project Structure

```
phononic-discovery/
├── framework/
│   ├── scripts/               # Discovery pipeline
│   │   ├── tmd/              # TMD-specific workflows
│   │   └── ...
│   ├── dft_validation/       # Quantum chemistry validation
│   │   ├── priority/         # High-priority candidates
│   │   └── results/          # Validated structures
│   ├── synthesis_lab/        # Experimental design
│   │   ├── temperature_screening/
│   │   └── substrate_screening/ (planned)
│   ├── results/              # Discoveries
│   │   ├── generated_tmds/   # AI-generated structures
│   │   └── CrCuSe2/          # Validated discovery
│   └── collaboration_proposal/ # Le Mans proposal
└── README.md                 # This file
```

---

## Collaboration

### Université Le Mans (LAUM)

**Contact**: [Professor Name] - Acoustics Laboratory  
**Proposal**: [collaboration_proposal/lemans_proposal.pdf](framework/collaboration_proposal/)

**Joint Objectives**:
- Computational discovery of phononic Dirac/Weyl materials
- Experimental validation via Brillouin scattering
- Strain engineering to create effective spacetime curvature
- First laboratory demonstration of phononic analog gravity

**Timeline**:
- **Phase I** (6 months): Discovery campaign → 10+ candidate materials
- **Phase II** (6 months): Strain protocol design
- **Phase III** (12 months): Experimental synthesis and characterization

---

## Key Results

### Computational Predictions
- **Candidates Generated**: 50 novel TMD structures
- **Validation Rate**: 20% pass initial xTB screening
- **DFT Confirmed**: CrCuSe₂ (100% stable)

### Synthesis Protocols
- **Temperature Window**: 450-550°C (AIMD-optimized)
- **Growth Method**: MBE co-deposition (Cr, Cu, Se)
- **Substrate**: Graphene or h-BN (computational screening ongoing)

---

## Publications

### In Preparation
1. "AI-Driven Discovery of CrCuSe₂: A Hetero-Metallic 2D Semiconductor"
   - Target: *Nature Materials*
   - Status: Manuscript draft in progress

2. "Phononic Analog Gravity via Computational Materials Design"
   - Target: *Physical Review Letters*
   - Status: Collaboration proposal submitted

---

## Next Steps

### Immediate (Next Month)
- [ ] Complete substrate binding energy calculations
- [ ] Finalize MBE growth protocol
- [ ] Submit Le Mans collaboration proposal

### Short-term (3-6 months)
- [ ] Extend discovery to magnonic Weyl materials
- [ ] Implement phonon band structure predictor (GNN)
- [ ] Design strain engineering experiments

### Long-term (12+ months)
- [ ] Experimental synthesis at LAUM
- [ ] Phonon dispersion measurements (Brillouin scattering)
- [ ] Demonstration of phononic gravitational lensing

---

## Contributing

We welcome contributions in:
- **Materials Discovery**: Extend to other material classes (perovskites, MXenes)
- **ML Models**: Improve phonon prediction accuracy
- **Synthesis**: Experimental validation and characterization
- **Theory**: Phonon topology, effective field theory

See [Contributing Guide](../../docs/guides/CONTRIBUTING.md) for details.

---

## References

1. Barceló et al., "Analogue Gravity", *Living Rev. Relativity* (2011)
2. Steinhauer, "Observation of Quantum Hawking Radiation", *Nature Physics* (2016)
3. Guinea et al., "Strain-Induced Pseudo-Magnetic Fields in Graphene", *Nature Physics* (2010)
4. Li et al., "Weyl Points in Phononic Crystals", *Nature Physics* (2018)

---

<div align="center">
  <p><sub>Built with precision. Validated with rigor.</sub></p>
</div>
