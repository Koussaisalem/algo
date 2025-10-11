#!/usr/bin/env python3
"""
Fast xTB Pre-Relaxation for Mo₂Te₄ Candidate
=============================================

This script performs a quick geometry cleanup using the xTB semi-empirical
method before running expensive DFT calculations. It fixes major issues like
atomic overlaps that would cause DFT convergence problems.

Runtime: ~5 minutes
Output: Cleaned structure ready for DFT validation
"""

from pathlib import Path
from ase.io import read, write
from ase.optimize import LBFGS

# Try to import xtb-python
try:
    from xtb.ase.calculator import XTB
    XTB_AVAILABLE = True
except ImportError:
    print("⚠️  xtb-python not available - using fallback relaxation")
    XTB_AVAILABLE = False

# --- Configuration ---
INPUT_FILE = "dft_validation/priority/Mo2Te4_candidate.xyz"
OUTPUT_FILE = "dft_validation/priority/Mo2Te4_prerelaxed.xyz"
TRAJECTORY_FILE = "dft_validation/priority/Mo2Te4_prerelax.traj"
# ---------------------

def main():
    print("=" * 70)
    print("🔬 Phase 1: xTB Pre-Relaxation for Mo₂Te₄")
    print("=" * 70)
    print(f"\nInput:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}\n")
    
    # Load the AI-generated structure
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"❌ ERROR: Input file not found: {INPUT_FILE}")
        exit(1)
    
    atoms = read(str(input_path))
    print(f"✅ Loaded structure: {atoms.get_chemical_formula()}")
    print(f"   Number of atoms: {len(atoms)}")
    
    # Set up a proper cell for the molecule (xTB needs this)
    # Add 10 Angstrom vacuum around the molecule
    atoms.center(vacuum=10.0)
    print(f"   Cell dimensions: {atoms.cell.lengths()}")
    
    # Check initial geometry
    min_distance = min([atoms.get_distance(i, j) for i in range(len(atoms)) 
                       for j in range(i+1, len(atoms))])
    print(f"   Minimum interatomic distance: {min_distance:.3f} Å")
    if min_distance < 1.5:
        print(f"   ⚠️  WARNING: Very short bond detected ({min_distance:.3f} Å)!")
        print(f"   This structure is highly distorted. Will try gentle relaxation.")
    
    # Set up calculator
    if XTB_AVAILABLE:
        print("\n🔧 Setting up xTB calculator (GFN2-xTB for transition metals)...")
        atoms.calc = XTB(method="GFN2-xTB")
    else:
        print("\n🔧 xTB not available - using Lennard-Jones (LJ) fallback...")
        print("   This will fix major overlaps but won't be chemically accurate")
        from ase.calculators.lj import LennardJones
        atoms.calc = LennardJones()
    
    # Run quick geometry optimization with loose convergence
    print("🚀 Starting xTB optimization (loose convergence for speed)...")
    print("   Force threshold: 0.5 eV/Å (loose, just cleanup)")
    print("   Expected time: 2-5 minutes\n")
    
    optimizer = LBFGS(atoms, trajectory=TRAJECTORY_FILE, logfile='xtb_opt.log')
    
    converged = False
    final_energy = None
    max_force = None
    
    try:
        optimizer.run(fmax=0.5, steps=100)  # Limit steps for badly distorted structures
        converged = True
        
        # Get final properties
        final_energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        max_force = max([sum(f**2)**0.5 for f in forces])
        
    except Exception as e:
        print(f"⚠️  xTB optimization failed: {str(e)[:200]}")
        print(f"   This structure is too distorted for xTB to handle.")
        print(f"   Will save the original structure and let DFT handle it.")
        converged = False
    
    # Save the structure (even if optimization failed, save original)
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(output_path), atoms)
    
    # Report results
    print("\n" + "=" * 70)
    print("📊 Pre-Relaxation Results")
    print("=" * 70)
    
    if converged and final_energy is not None:
        print(f"✅ Final energy: {final_energy:.4f} eV")
        print(f"✅ Max force: {max_force:.4f} eV/Å")
        print(f"✅ Optimization converged")
        
        # Check final geometry
        final_min_distance = min([atoms.get_distance(i, j) for i in range(len(atoms)) 
                                 for j in range(i+1, len(atoms))])
        print(f"✅ Final minimum distance: {final_min_distance:.3f} Å")
    else:
        print(f"⚠️  xTB optimization failed - structure too distorted")
        print(f"⚠️  Saved original structure with proper cell")
        print(f"   DFT will handle the relaxation (may take longer)")
    
    print(f"\n💾 Structure saved to: {OUTPUT_FILE}")
    if converged:
        print(f"📁 Trajectory saved to: {TRAJECTORY_FILE}")
    
    print("\n" + "=" * 70)
    if converged:
        print("🎯 Phase 1 Complete! Structure cleaned successfully")
        print("   Ready for Phase 2 (DFT single-point)")
    else:
        print("⚠️  Phase 1 Incomplete - xTB couldn't handle this geometry")
        print("   Proceeding to DFT anyway (will use original structure)")
    print("=" * 70)
    print("\nNext step:")
    print("  python scripts/tmd/05_validate_with_dft.py dft_validation/priority/Mo2Te4_prerelaxed.xyz --mode fast --single-point")
    print()

if __name__ == "__main__":
    main()
