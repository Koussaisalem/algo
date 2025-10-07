# Multi-Domain Knockout Strategy

## Branch Structure

```
main (protected)
├── ✅ QM9 baseline validated
├── ✅ Score model trained
├── ✅ Manifold ops proven
└── ✅ NEVER modified (safe reference)

operation-magnet-semiconductors (active)
├── Target: 2D TMDs (MoS2, WS2, etc.)
├── Additions: l_max=2, PBC support, C2DB data
└── Goal: Nature Materials paper

operation-cobra-transition-metals (future)
├── Target: Coordination complexes
├── Additions: Metal centers, ligand fields
└── Goal: Nature Chemistry paper

operation-phoenix-drugs (future)
├── Target: Drug-like molecules
├── Additions: Larger molecules, bioactivity
└── Goal: Nature Biotechnology paper

operation-atlas-mofs (future)
├── Target: Metal-organic frameworks
├── Additions: Porous structures, gas adsorption
└── Goal: Nature Materials paper

operation-titan-catalysts (future)
├── Target: Catalytic systems
├── Additions: Reaction pathways, transition states
└── Goal: Nature Catalysis paper
```

## Workflow

### Phase 1: Branch Creation
```bash
git checkout main              # Start from validated baseline
git checkout -b operation-X    # Create domain branch
```

### Phase 2: Domain Adaptation
- Add domain-specific data loaders
- Adapt architecture (e.g., l_max for orbitals)
- Create domain-specific training scripts
- **Never modify core/** (manifold ops stay intact)

### Phase 3: Validation
- Train on domain data
- Benchmark against SOTA
- Generate results
- **If 3-5× improvement → proceed to paper**

### Phase 4: Publication
- Write manuscript on branch
- Submit to journal
- **Keep branch alive for revisions**

### Phase 5: Core Improvements (optional)
- If branch discovers better manifold ops → PR to main
- Main only accepts improvements that help ALL domains
- Branches regularly merge from main (get improvements)

## Strategy Benefits

1. **Risk Mitigation**: If one domain fails, others unaffected
2. **Parallel Progress**: Multiple papers in flight simultaneously  
3. **Clean Architecture**: Core stays clean, domain code isolated
4. **Publication Factory**: 1 branch = 1 paper = 1 knockout
5. **Portfolio Approach**: Diversified bets on different applications

## Current Status

- ✅ **main**: QM9 validated (RMSD 2.56Å, orthogonality 10⁻¹⁶)
- 🚀 **operation-magnet-semiconductors**: ACTIVE (Week 1 starting)
- 📋 **operation-cobra-transition-metals**: Planned (backup if Magnet stalls)
- 📋 **operation-phoenix-drugs**: Planned (high commercial value)
- 📋 **operation-atlas-mofs**: Planned (materials science)
- 📋 **operation-titan-catalysts**: Planned (chemistry core)

## Rules

1. **main is sacred** - only merge proven improvements
2. **Branches are playgrounds** - experiment aggressively
3. **One branch = one domain** - no mixing applications
4. **Branch names = operation codenames** - keep it fun 🎯
5. **Each branch targets one journal** - focused publications

## Next Steps

```bash
# You are now on: operation-magnet-semiconductors
# Safe to experiment - main is protected!

# Start Phase 1: Dataset acquisition
mkdir -p qcmd_hybrid_framework/data/tmd
python qcmd_hybrid_framework/scripts/tmd/00_download_c2db.py
```

🧲 **Operation Magnet is GO!** Main branch is safe. Let's knock out semiconductors!
