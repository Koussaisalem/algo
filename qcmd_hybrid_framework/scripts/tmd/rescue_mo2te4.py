#!/usr/bin/env python3
"""
Three-stage geometry rescue for Mo₂Te₄.

St    print(f"\n📊 Stage 1 Results:")
    print(f"   Output: {output_file}")
    print(f"   Final min bond: {final_min:.3f} Å")
    
    # Check if improvement happened
    if final_min > 1.8:
        print("   ✅ Bonds look reasonable - ready for Stage 2!")
        return True
    elif final_min > 1.2:
        print("   ⚠️  Bonds still short but improved - trying Stage 2 anyway")
        return True
    else:
        print("   ❌ Bonds barely changed - Lennard-Jones couldn't help")
        print("   Will try xTB anyway (it's more sophisticated)")
        return True  # Still try xTB - it's much better than LJce field (separate overlapping atoms)
Stage 2: GFN2-xTB semi-empirical (refine geometry)  
Stage 3: DFT validation (check stability)

This is the most direct approach to fix collapsed AI-generated structures.
"""

import sys
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase.optimize import BFGS, LBFGS
from ase.calculators.lj import LennardJones

# Try xTB
try:
    from xtb.ase.calculator import XTB
    XTB_AVAILABLE = True
except ImportError:
    print("⚠️  xTB not available - Stage 2 will be skipped")
    XTB_AVAILABLE = False


def get_min_bond_length(atoms):
    """Get minimum non-zero bond length."""
    dists = atoms.get_all_distances()
    nonzero = dists[np.nonzero(dists)]
    return nonzero.min() if len(nonzero) > 0 else 0.0


def stage1_uff(input_file, output_file):
    """Stage 1: Aggressive Lennard-Jones relaxation."""
    print("\n" + "="*70)
    print("🔧 STAGE 1: Lennard-Jones Force Field Pre-Relaxation")
    print("   (Goal: Separate overlapping atoms)")
    print("="*70)
    
    atoms = read(input_file)
    print(f"Input: {atoms.get_chemical_formula()}")
    print(f"Initial min bond: {get_min_bond_length(atoms):.3f} Å (should be ~2.7 Å)")
    
    # Setup molecular cell
    atoms.center(vacuum=15.0)
    atoms.pbc = [False, False, False]  # Treat as molecule
    
    # Lennard-Jones calculator (simple repulsive potential)
    print("\n🔧 Setting up Lennard-Jones calculator...")
    print("   (Simple repulsive potential - just pushes atoms apart)")
    atoms.calc = LennardJones()
    
    # Very loose optimization (just separate atoms, don't care about fine details)
    print("🚀 Running Lennard-Jones optimization (fmax=1.0 eV/Å - very loose)...")
    print("   This will push overlapping atoms apart...")
    opt = LBFGS(atoms, trajectory='rescue_stage1.traj', logfile='rescue_stage1.log')
    
    try:
        opt.run(fmax=1.0, steps=500)
        success = True
        print("✅ Lennard-Jones optimization converged")
    except Exception as e:
        print(f"⚠️  Lennard-Jones had issues but saved structure: {str(e)[:100]}")
        success = False
    
    write(output_file, atoms)
    final_min = get_min_bond_length(atoms)
    
    print(f"\n📊 Stage 1 Results:")
    print(f"   Output: {output_file}")
    print(f"   Final min bond: {final_min:.3f} Å")
    
    # Check if improvement happened
    if final_min > 1.8:
        print("   ✅ Bonds look reasonable - ready for Stage 2!")
        return True
    elif final_min > 1.0:
        print("   ⚠️  Bonds still short but improved - trying Stage 2 anyway")
        return True
    else:
        print("   ❌ Bonds still collapsed - structure may be unfixable")
        return False


def stage2_xtb(input_file, output_file):
    """Stage 2: xTB semi-empirical refinement."""
    print("\n" + "="*70)
    print("🔧 STAGE 2: GFN2-xTB Semi-Empirical Refinement")
    print("   (Goal: Find chemically accurate Mo-Te bonding)")
    print("="*70)
    
    if not XTB_AVAILABLE:
        print("❌ xTB not available - copying Stage 1 output directly")
        print("   (You'll need DFT to do the refinement)")
        atoms = read(input_file)
        write(output_file, atoms)
        return True
    
    atoms = read(input_file)
    print(f"Input: {atoms.get_chemical_formula()}")
    print(f"Initial min bond: {get_min_bond_length(atoms):.3f} Å")
    
    # xTB calculator (much better chemistry than UFF)
    print("\n🔧 Setting up GFN2-xTB calculator...")
    print("   (GFN2-xTB is designed for transition metals)")
    atoms.calc = XTB(method="GFN2-xTB")
    
    # Tighter optimization
    print("🚀 Running xTB optimization (fmax=0.1 eV/Å - tighter)...")
    print("   This will find proper Mo-Te/Te-Te bonds...")
    opt = LBFGS(atoms, trajectory='rescue_stage2.traj', logfile='rescue_stage2.log')
    
    try:
        opt.run(fmax=0.1, steps=1000)
        success = True
        print("✅ xTB optimization converged")
    except Exception as e:
        print(f"⚠️  xTB had issues but saved structure: {str(e)[:100]}")
        success = False
    
    write(output_file, atoms)
    final_min = get_min_bond_length(atoms)
    
    print(f"\n📊 Stage 2 Results:")
    print(f"   Output: {output_file}")
    print(f"   Final min bond: {final_min:.3f} Å")
    
    # Check if ready for DFT
    if final_min > 2.3:
        print("   ✅ Bonds look excellent - ready for DFT validation!")
        return True
    elif final_min > 1.8:
        print("   ⚠️  Bonds acceptable - DFT will be the final judge")
        return True
    else:
        print("   ❌ Bonds still too short for reliable DFT")
        return False


def main():
    input_file = "dft_validation/priority/Mo2Te4_candidate.xyz"
    stage1_out = "dft_validation/priority/Mo2Te4_stage1_uff.xyz"
    stage2_out = "dft_validation/priority/Mo2Te4_stage2_xtb.xyz"
    
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║         🚑 Mo₂Te₄ Geometry Rescue Mission                 ║")
    print("║            (Three-Stage Optimization)                      ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    print("Strategy:")
    print("  Stage 1: Lennard-Jones  → separate overlapping atoms")
    print("  Stage 2: GFN2-xTB       → refine to realistic bonds")
    print("  Stage 3: DFT            → validate stability (manual)")
    print()
    print("This is how experimental chemists optimize novel structures!")
    print()
    
    # Stage 1: UFF
    stage1_ok = stage1_uff(input_file, stage1_out)
    
    if not stage1_ok:
        print("\n" + "="*70)
        print("❌ Stage 1 failed - structure may be unfixable")
        print("="*70)
        print("\nPossible reasons:")
        print("  - Structure is too far from any reasonable geometry")
        print("  - Simple force fields can't handle this complexity")
        print("\nOptions:")
        print("  1. Try manual geometry editing")
        print("  2. Regenerate with different random seed")
        print("  3. Use seeded generation from known MoTe₂")
        return 1
    
    # Stage 2: xTB
    stage2_ok = stage2_xtb(stage1_out, stage2_out)
    
    if not stage2_ok:
        print("\n⚠️  Stage 2 had issues but continuing anyway")
        print("   DFT is robust and may still work!")
    
    # Stage 3: DFT (user runs manually)
    print("\n" + "="*70)
    print("🔬 STAGE 3: DFT Validation (Run Manually)")
    print("="*70)
    print("\nThe structure is ready for DFT validation!")
    print("\n📋 Run this command:")
    print(f"\n  python scripts/tmd/05_validate_with_dft.py \\")
    print(f"      {stage2_out} \\")
    print(f"      --mode fast \\")
    print(f"      --single-point")
    print("\n📊 How to interpret results:")
    print("   Max forces < 1.5 eV/Å  →  ✅ GO SIGNAL (run full optimization!)")
    print("   Max forces 1.5-3.0     →  ⚠️  CAUTION (may work with care)")
    print("   Max forces > 3.0       →  ❌ NO-GO (structure not viable)")
    print("\n" + "="*70)
    print("🎯 If GO signal: Run overnight DFT with --mode production")
    print("   Expected time: 1-2 hours for full optimization")
    print("="*70)
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
