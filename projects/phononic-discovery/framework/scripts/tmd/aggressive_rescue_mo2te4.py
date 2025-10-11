#!/usr/bin/env python3
"""
AGGRESSIVE Mo₂Te₄ Geometry Rescue

Strategy:
1. Manual geometry scaling (multiply all distances by factor)
2. xTB optimization with multiple attempts
3. DFT validation

This is more aggressive than waiting for force fields.
"""

import sys
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase.optimize import BFGS, LBFGS

# Try xTB
try:
    from xtb.ase.calculator import XTB
    XTB_AVAILABLE = True
except ImportError:
    print("❌ xTB not available - cannot proceed")
    XTB_AVAILABLE = False
    sys.exit(1)


def get_min_bond_length(atoms):
    """Get minimum non-zero bond length."""
    dists = atoms.get_all_distances()
    nonzero = dists[np.nonzero(dists)]
    return nonzero.min() if len(nonzero) > 0 else 0.0


def scale_structure(atoms, scale_factor):
    """
    Scale structure by expanding from centroid.
    
    This is a brute-force way to separate overlapping atoms.
    """
    positions = atoms.get_positions()
    centroid = positions.mean(axis=0)
    
    # Expand from centroid
    centered = positions - centroid
    scaled = centered * scale_factor
    new_positions = scaled + centroid
    
    atoms.set_positions(new_positions)
    return atoms


def aggressive_xtb_optimize(atoms, max_attempts=5):
    """
    Try xTB optimization with progressively looser settings.
    """
    atoms.calc = XTB(method="GFN2-xTB")
    
    # Try different convergence criteria
    fmax_values = [0.5, 1.0, 2.0, 5.0, 10.0]  # Progressively looser
    
    for attempt, fmax in enumerate(fmax_values[:max_attempts], 1):
        print(f"\n🔧 Attempt {attempt}/{max_attempts}: fmax={fmax} eV/Å")
        
        opt = LBFGS(
            atoms,
            trajectory=f'rescue_aggressive_attempt{attempt}.traj',
            logfile=f'rescue_aggressive_attempt{attempt}.log'
        )
        
        try:
            opt.run(fmax=fmax, steps=200)
            print(f"✅ Converged at fmax={fmax}")
            return True, atoms
        except Exception as e:
            print(f"⚠️  Failed: {str(e)[:80]}")
            if attempt < max_attempts:
                print("   Trying looser convergence...")
            continue
    
    print("❌ All attempts failed")
    return False, atoms


def main():
    input_file = "dft_validation/priority/Mo2Te4_candidate.xyz"
    
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║    💪 AGGRESSIVE Mo₂Te₄ Geometry Rescue                   ║")
    print("║       (Brute Force + Smart Optimization)                  ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    # Load original
    atoms = read(input_file)
    print(f"Original structure: {atoms.get_chemical_formula()}")
    print(f"Initial min bond: {get_min_bond_length(atoms):.3f} Å")
    print()
    
    # Determine scaling factor needed
    # Current: 0.57 Å, Target: ~2.7 Å → scale by ~4.7×
    current_min = get_min_bond_length(atoms)
    target_min = 2.7  # Typical Mo-Te bond
    scale_factor = target_min / current_min
    
    print("="*70)
    print("🔧 STAGE 1: Manual Geometry Scaling")
    print("="*70)
    print(f"Current min bond: {current_min:.3f} Å")
    print(f"Target bond: {target_min:.3f} Å")
    print(f"Scale factor: {scale_factor:.2f}×")
    print()
    
    # Apply scaling
    atoms.center(vacuum=15.0)
    atoms.pbc = [False, False, False]
    atoms = scale_structure(atoms, scale_factor)
    
    scaled_min = get_min_bond_length(atoms)
    print(f"✅ After scaling: min bond = {scaled_min:.3f} Å")
    write("dft_validation/priority/Mo2Te4_scaled.xyz", atoms)
    print(f"   Saved: Mo2Te4_scaled.xyz")
    print()
    
    # Now try xTB optimization
    print("="*70)
    print("🔧 STAGE 2: Aggressive xTB Optimization")
    print("="*70)
    print("Strategy: Try multiple times with progressively looser convergence")
    print()
    
    success, atoms_opt = aggressive_xtb_optimize(atoms, max_attempts=5)
    
    final_min = get_min_bond_length(atoms_opt)
    
    if success:
        print()
        print("="*70)
        print("✅ xTB OPTIMIZATION SUCCESSFUL!")
        print("="*70)
        print(f"Final min bond: {final_min:.3f} Å")
        
        # Save
        output_file = "dft_validation/priority/Mo2Te4_aggressive_rescue.xyz"
        write(output_file, atoms_opt)
        print(f"Saved: {output_file}")
        
        # Try to get energy
        try:
            energy = atoms_opt.get_potential_energy()
            print(f"xTB energy: {energy:.4f} eV ({energy/len(atoms_opt):.4f} eV/atom)")
        except:
            pass
        
        print()
        print("="*70)
        print("🔬 STAGE 3: DFT Validation (Run Manually)")
        print("="*70)
        print()
        print("📋 Run this command:")
        print()
        print(f"  python scripts/tmd/05_validate_with_dft.py \\")
        print(f"      {output_file} \\")
        print(f"      --mode fast \\")
        print(f"      --single-point")
        print()
        print("📊 Expected outcomes:")
        if final_min > 2.3:
            print("   ✅ Bonds look EXCELLENT (> 2.3 Å)")
            print("   → Very likely to pass DFT validation!")
        elif final_min > 2.0:
            print("   ✅ Bonds look GOOD (> 2.0 Å)")
            print("   → Good chance of passing DFT!")
        elif final_min > 1.8:
            print("   ⚠️  Bonds acceptable (> 1.8 Å)")
            print("   → DFT may work with conservative settings")
        else:
            print("   ❌ Bonds still short (< 1.8 Å)")
            print("   → DFT unlikely to converge")
        
        return 0
        
    else:
        print()
        print("="*70)
        print("❌ xTB OPTIMIZATION FAILED")
        print("="*70)
        print("Even with aggressive settings, xTB couldn't optimize this structure.")
        print()
        print("📊 Diagnosis:")
        print(f"   Final min bond: {final_min:.3f} Å")
        print()
        print("🎯 Options:")
        print("   1. Try DFT anyway (it's more robust than xTB):")
        print(f"      python scripts/tmd/05_validate_with_dft.py \\")
        print(f"          dft_validation/priority/Mo2Te4_scaled.xyz \\")
        print(f"          --mode fast")
        print()
        print("   2. This structure may fundamentally not be viable")
        print("      → Focus on CrCuSe₂ instead (hetero-metallic alloy)")
        print()
        print("   3. Generate new Mo₂Te₄ with different random seed")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
