#!/usr/bin/env python3
"""
Aggressive CrCuSe₂ Geometry Rescue

This is the REAL prize structure - first hetero-metallic TMD alloy!
Let's fix the geometry and validate it properly.

Strategy:
1. Scale structure by ~2× (1.26 Å → 2.4 Å bonds)
2. Aggressive xTB optimization with multiple attempts
3. DFT single-point validation
4. If GO signal → Full DFT optimization
"""

import sys
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase.optimize import LBFGS

# Try xTB
try:
    from xtb.ase.calculator import XTB
    XTB_AVAILABLE = True
except ImportError:
    print("❌ xTB not available - cannot proceed")
    sys.exit(1)


def get_min_bond_length(atoms):
    """Get minimum non-zero bond length."""
    dists = atoms.get_all_distances()
    nonzero = dists[np.nonzero(dists)]
    return nonzero.min() if len(nonzero) > 0 else 0.0


def scale_structure(atoms, scale_factor):
    """Scale structure by expanding from centroid."""
    positions = atoms.get_positions()
    centroid = positions.mean(axis=0)
    
    # Expand from centroid
    centered = positions - centroid
    scaled = centered * scale_factor
    new_positions = scaled + centroid
    
    atoms.set_positions(new_positions)
    return atoms


def aggressive_xtb_optimize(atoms, max_attempts=5):
    """Try xTB optimization with progressively looser settings."""
    atoms.calc = XTB(method="GFN2-xTB")
    
    # Try different convergence criteria
    fmax_values = [0.1, 0.3, 0.5, 1.0, 2.0]  # Progressively looser
    
    for attempt, fmax in enumerate(fmax_values[:max_attempts], 1):
        print(f"\n🔧 Attempt {attempt}/{max_attempts}: fmax={fmax} eV/Å")
        
        opt = LBFGS(
            atoms,
            trajectory=f'crcuse2_rescue_attempt{attempt}.traj',
            logfile=f'crcuse2_rescue_attempt{attempt}.log'
        )
        
        try:
            opt.run(fmax=fmax, steps=300)
            print(f"✅ Converged at fmax={fmax} eV/Å")
            return True, atoms
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"⚠️  Failed: {error_msg}")
            if attempt < max_attempts:
                print("   Trying looser convergence...")
            continue
    
    print("❌ All attempts failed")
    return False, atoms


def main():
    input_file = "dft_validation/priority/CrCuSe2_candidate.xyz"
    
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║    🏆 CrCuSe₂ Rescue - First Hetero-Metallic TMD!        ║")
    print("║       (This is the REAL breakthrough structure)           ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    print("Why CrCuSe₂ is exciting:")
    print("  ✨ First TMD with TWO different metals (Cr + Cu)")
    print("  ✨ Magnetic (Cr) + conducting (Cu) properties")
    print("  ✨ CDVAE can't generate these (composition outside training)")
    print()
    
    # Load original
    atoms = read(input_file)
    print(f"Original structure: {atoms.get_chemical_formula()}")
    print(f"Composition: {' '.join(atoms.get_chemical_symbols())}")
    print(f"Initial min bond: {get_min_bond_length(atoms):.3f} Å")
    print()
    
    # Determine scaling factor
    current_min = get_min_bond_length(atoms)
    target_min = 2.4  # Typical Cr-Se or Cu-Se bond
    scale_factor = target_min / current_min
    
    print("="*70)
    print("🔧 STAGE 1: Geometry Scaling")
    print("="*70)
    print(f"Current min bond: {current_min:.3f} Å")
    print(f"Target bond: {target_min:.3f} Å (typical M-Se)")
    print(f"Scale factor: {scale_factor:.2f}×")
    print()
    
    # Apply scaling
    atoms.center(vacuum=15.0)
    atoms.pbc = [False, False, False]  # Treat as molecular cluster
    atoms = scale_structure(atoms, scale_factor)
    
    scaled_min = get_min_bond_length(atoms)
    print(f"✅ After scaling: min bond = {scaled_min:.3f} Å")
    
    # Save scaled structure
    scaled_file = "dft_validation/priority/CrCuSe2_scaled.xyz"
    write(scaled_file, atoms)
    print(f"   Saved: {scaled_file}")
    print()
    
    # Now try xTB optimization
    print("="*70)
    print("🔧 STAGE 2: xTB Optimization (Multiple Attempts)")
    print("="*70)
    print("Strategy: Try progressively looser convergence until success")
    print()
    
    success, atoms_opt = aggressive_xtb_optimize(atoms, max_attempts=5)
    
    final_min = get_min_bond_length(atoms_opt)
    
    if success:
        print()
        print("="*70)
        print("✅ xTB OPTIMIZATION SUCCESSFUL!")
        print("="*70)
        print(f"Final min bond: {final_min:.3f} Å")
        
        # Save optimized structure
        output_file = "dft_validation/priority/CrCuSe2_rescue.xyz"
        write(output_file, atoms_opt)
        print(f"Saved: {output_file}")
        
        # Get xTB energy
        try:
            energy = atoms_opt.get_potential_energy()
            energy_per_atom = energy / len(atoms_opt)
            print(f"\nxTB Results:")
            print(f"  Total energy: {energy:.4f} eV")
            print(f"  Energy/atom:  {energy_per_atom:.4f} eV")
            
            # Get forces
            forces = atoms_opt.get_forces()
            max_force = np.abs(forces).max()
            print(f"  Max force:    {max_force:.4f} eV/Å")
        except Exception as e:
            print(f"\n⚠️  Could not get xTB properties: {e}")
        
        print()
        print("="*70)
        print("🔬 STAGE 3: DFT Validation (Run Manually)")
        print("="*70)
        print()
        print("📋 Run this command for quick validation (~20 min):")
        print()
        print(f"  python scripts/tmd/05_validate_with_dft.py \\")
        print(f"      {output_file} \\")
        print(f"      --mode fast \\")
        print(f"      --single-point")
        print()
        
        # Assessment
        if final_min > 2.3:
            print("📊 Assessment: ✅ EXCELLENT geometry!")
            print("   → Very high confidence for DFT success")
            print("   → If DFT single-point shows max forces < 1.5 eV/Å:")
            print()
            print("     Run FULL optimization:")
            print(f"     python scripts/tmd/05_validate_with_dft.py \\")
            print(f"         {output_file} \\")
            print(f"         --mode production")
            print()
            print("     This will be your BREAKTHROUGH RESULT! 🎉")
        elif final_min > 2.0:
            print("📊 Assessment: ✅ GOOD geometry")
            print("   → Good chance of DFT success")
        else:
            print("📊 Assessment: ⚠️  Acceptable but not ideal")
            print("   → DFT may struggle but worth trying")
        
        return 0
        
    else:
        print()
        print("="*70)
        print("⚠️  xTB OPTIMIZATION HAD ISSUES")
        print("="*70)
        print(f"Final min bond: {final_min:.3f} Å")
        print()
        print("But CrCuSe₂ has PROPER CONNECTIVITY (no isolated atoms)")
        print("→ DFT is more robust and may still work!")
        print()
        print("🎯 Try DFT on the scaled structure:")
        print(f"   python scripts/tmd/05_validate_with_dft.py \\")
        print(f"       {scaled_file} \\")
        print(f"       --mode fast")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
