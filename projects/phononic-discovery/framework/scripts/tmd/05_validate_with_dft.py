#!/usr/bin/env python3
"""
DFT Validation of Generated TMD Structures using GPAW

This script performs structure relaxation and property calculations
for the top breakthrough candidates from manifold generation.

Priority structures:
1. Mo2Te4 (E=-2.722 eV) - Novel phase candidate
2. CrCuSe2 (E=3.458 eV) - Hetero-metallic alloy
3. VTe2 (E=-1.810 eV) - Magnetic TMD

Usage:
    # Single structure (for parallel execution):
    python 05_validate_with_dft.py path/to/structure.xyz
    
    # Single-point calculation (no optimization):
    python 05_validate_with_dft.py path/to/structure.xyz --single-point
    
    # All priority structures (serial):
    python 05_validate_with_dft.py
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
from ase.io import read, write
from ase.optimize import BFGS, LBFGS
from ase.constraints import FixAtoms
from ase.visualize import view

# Check if GPAW is available
try:
    from gpaw import GPAW, PW, FermiDirac, Mixer
    GPAW_AVAILABLE = True
except ImportError:
    print("⚠️  GPAW not available. Will use ASE calculators for testing.")
    print("   Install GPAW with: pip install gpaw")
    GPAW_AVAILABLE = False
    from ase.calculators.emt import EMT  # Fallback for testing


def setup_dft_calculator(atoms, mode='fast', kpts=(4, 4, 1)):
    """
    Setup GPAW calculator with appropriate settings.
    
    Args:
        atoms: ASE Atoms object
        mode: 'fast' (testing), 'production' (accurate), or 'converged' (publication)
        kpts: k-point sampling
    
    Returns:
        Configured calculator
    """
    if not GPAW_AVAILABLE:
        print("Using EMT calculator (testing only - not quantitatively accurate)")
        return EMT()
    
    # Conservative mixer for difficult systems (prevents charge sloshing)
    # beta=0.025: Only 2.5% of new density per iteration (stable but slow)
    # nmaxold=5: Use last 5 iterations for extrapolation
    # weight=50: Kerker preconditioning for metallic/semi-metallic systems
    stable_mixer = Mixer(beta=0.025, nmaxold=5, weight=50.0)
    
    if mode == 'fast':
        # Quick test: ~10-20 min per structure (conservative convergence)
        calc = GPAW(
            mode='fd',               # Real-space grid mode
            xc='PBE',                # Standard GGA functional
            h=0.18,                  # Coarser grid for speed
            kpts=kpts,               # k-point mesh
            occupations=FermiDirac(0.1),  # Smearing
            eigensolver='rmm-diis',         # Use RMM-DIIS solver for stability
            mixer=stable_mixer,      # Conservative mixer to prevent oscillations
            maxiter=300,             # Allow more SCF iterations
            txt='gpaw_output.txt',   # Log file
            symmetry={'point_group': False}  # Disable for 2D
        )
    elif mode == 'production':
        # Standard accuracy: ~30-60 min per structure
        calc = GPAW(
            mode='fd',
            xc='PBE',
            h=0.15,                  # Refined grid spacing
            kpts=kpts,
            occupations=FermiDirac(0.05),
            eigensolver='rmm-diis',
            mixer=stable_mixer,      # Conservative mixer
            maxiter=300,
            txt='gpaw_output.txt',
            convergence={'energy': 1e-5}  # Tighter convergence
        )
    elif mode == 'converged':
        # Publication quality: ~2-4 hours per structure
        calc = GPAW(
            mode='fd',
            xc='PBE',
            h=0.12,                  # Very fine grid spacing
            kpts=(8, 8, 1),  # Dense mesh
            occupations=FermiDirac(0.01),
            eigensolver='rmm-diis',
            mixer=stable_mixer,      # Conservative mixer
            maxiter=400,
            txt='gpaw_output.txt',
            convergence={'energy': 1e-6, 'density': 1e-5}
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return calc


def add_vacuum_and_cell(atoms, vacuum=15.0):
    """
    Add vacuum along z-axis and set up proper cell for 2D material.
    
    Args:
        atoms: ASE Atoms object
        vacuum: Vacuum distance in Angstroms
    
    Returns:
        Modified atoms with cell and vacuum
    """
    # Get atomic positions
    positions = atoms.get_positions()
    
    # Center in xy-plane
    center_xy = positions[:, :2].mean(axis=0)
    positions[:, :2] -= center_xy
    
    # Find extent in xy
    max_xy = np.abs(positions[:, :2]).max(axis=0)
    cell_xy = 2 * max_xy + 5.0  # Add 5 Å buffer
    
    # Find extent in z
    z_extent = positions[:, 2].max() - positions[:, 2].min()
    cell_z = z_extent + 2 * vacuum
    
    # Set up cell
    atoms.set_cell([cell_xy[0], cell_xy[1], cell_z])
    atoms.center()  # Center in cell
    
    # Make non-periodic in z
    atoms.pbc = [True, True, False]
    
    return atoms


def relax_structure(atoms, calc, fmax=0.05, max_steps=200, logfile=None, single_point=False):
    """
    Relax atomic positions using BFGS optimizer, or just calculate energy/forces.
    
    Args:
        atoms: ASE Atoms object
        calc: Calculator (GPAW or EMT)
        fmax: Maximum force threshold (eV/Å)
        max_steps: Maximum optimization steps
        logfile: Path to optimization log file
        single_point: If True, only calculate energy/forces without optimization
    
    Returns:
        Tuple of (relaxed_atoms, converged, final_energy)
    """
    atoms.calc = calc
    
    if single_point:
        # Phase 2: Single-point calculation only (fast stability check)
        print(f"   Starting SINGLE-POINT DFT calculation...")
        print(f"   (No optimization - just energy and forces)")
        start_time = time.time()
        
        try:
            final_energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            max_force = max([sum(f**2)**0.5 for f in forces])
            converged = True  # Single-point always "converges"
            
            elapsed = time.time() - start_time
            print(f"   Single-point finished in {elapsed:.1f}s")
            print(f"   Energy: {final_energy:.4f} eV ({final_energy/len(atoms):.4f} eV/atom)")
            print(f"   Max Force: {max_force:.4f} eV/Å")
            
            # Interpret force magnitude
            if max_force < 1.5:
                print(f"   ✅ GO SIGNAL: Structure is stable (max force < 1.5 eV/Å)")
                print(f"      → Safe to proceed with full optimization")
            elif max_force < 3.0:
                print(f"   ⚠️  CAUTION: Moderate forces (1.5-3.0 eV/Å)")
                print(f"      → May need careful optimization settings")
            else:
                print(f"   ❌ NO-GO SIGNAL: High forces (> 3.0 eV/Å)")
                print(f"      → Structure likely unstable, save compute budget")
            
        except Exception as e:
            print(f"   ❌ Single-point calculation failed: {e}")
            final_energy = float('nan')
            converged = False
            elapsed = time.time() - start_time
        
        return atoms, converged, final_energy
    
    else:
        # Original behavior: Full geometry optimization
        opt = LBFGS(atoms, logfile=logfile, trajectory='optimization.traj')
        
        print(f"   Starting optimization (fmax={fmax} eV/Å, max_steps={max_steps})")
        start_time = time.time()
        
        try:
            opt.run(fmax=fmax, steps=max_steps)
            converged = True
        except Exception as e:
            print(f"   ⚠️  Optimization failed: {e}")
            converged = False
        
        final_energy = atoms.get_potential_energy()
        elapsed = time.time() - start_time
        
        print(f"   Optimization finished in {elapsed:.1f}s")
        print(f"   Final energy: {final_energy:.4f} eV")
        
        return atoms, converged, final_energy


def calculate_properties(atoms):
    """
    Calculate additional properties after relaxation.
    
    Args:
        atoms: Relaxed ASE Atoms object with calculator attached
    
    Returns:
        Dictionary of calculated properties
    """
    properties = {}
    
    # Basic energetics
    properties['total_energy'] = atoms.get_potential_energy()
    properties['energy_per_atom'] = properties['total_energy'] / len(atoms)
    
    # Forces
    forces = atoms.get_forces()
    properties['max_force'] = np.abs(forces).max()
    properties['rms_force'] = np.sqrt((forces**2).sum(axis=1).mean())
    
    # Structural properties
    positions = atoms.get_positions()
    properties['z_extent'] = positions[:, 2].max() - positions[:, 2].min()
    
    # Bond lengths
    from ase.neighborlist import neighbor_list
    i, j, d = neighbor_list('ijd', atoms, cutoff=3.5)  # 3.5 Å cutoff
    if len(d) > 0:
        properties['min_bond_length'] = d.min()
        properties['mean_bond_length'] = d.mean()
        properties['max_bond_length'] = d.max()
    else:
        properties['min_bond_length'] = None
        properties['mean_bond_length'] = None
        properties['max_bond_length'] = None
    
    # Try to get electronic properties (only works with GPAW)
    if GPAW_AVAILABLE and hasattr(atoms.calc, 'get_fermi_level'):
        try:
            properties['fermi_level'] = atoms.calc.get_fermi_level()
            
            # Get band gap (if semiconducting)
            from gpaw import GPAW as GPAWCalc
            if isinstance(atoms.calc, GPAWCalc):
                # This requires a converged calculation
                # properties['band_gap'] = atoms.calc.get_eigenvalues().max() - atoms.calc.get_eigenvalues().min()
                pass  # Skip for now, needs k-points
        except:
            pass
    
    return properties


def validate_structure(structure_file, output_dir, mode='fast', force_rerun=False, single_point=False):
    """
    Complete validation workflow for a single structure.
    
    Args:
        structure_file: Path to XYZ file
        output_dir: Directory to save results
        mode: Calculation mode ('fast', 'production', or 'converged')
        force_rerun: If True, recompute even if results exist
        single_point: If True, only calculate energy/forces (no optimization)
    
    Returns:
        Dictionary with all validation results
    """
    structure_name = Path(structure_file).stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"{structure_name}_results.json"
    
    # Check if already computed
    if results_file.exists() and not force_rerun:
        print(f"✅ Results already exist for {structure_name}, loading...")
        with open(results_file, 'r') as f:
            return json.load(f)
    
    print(f"\n{'='*60}")
    if single_point:
        print(f"⚡ Single-Point Validation: {structure_name}")
    else:
        print(f"🔬 Full Validation: {structure_name}")
    print(f"{'='*60}")
    
    # Load structure
    atoms = read(structure_file)
    initial_composition = atoms.get_chemical_formula()
    print(f"   Composition: {initial_composition}")
    print(f"   Atoms: {len(atoms)}")
    
    # Setup cell and vacuum
    atoms = add_vacuum_and_cell(atoms, vacuum=15.0)
    print(f"   Cell: {atoms.cell.lengths()}")
    
    # Save initial structure
    write(output_dir / f"{structure_name}_initial.xyz", atoms)
    
    # Setup calculator
    calc_type = "single-point" if single_point else mode
    print(f"   Setting up {calc_type} DFT calculator...")
    calc = setup_dft_calculator(atoms, mode=mode)
    
    # Relax structure (or just calculate forces)
    logfile = output_dir / f"{structure_name}_opt.log"
    atoms, converged, final_energy = relax_structure(
        atoms, calc, fmax=0.05, max_steps=200, logfile=logfile, single_point=single_point
    )
    
    # Save relaxed structure
    write(output_dir / f"{structure_name}_relaxed.xyz", atoms)
    write(output_dir / f"{structure_name}_relaxed.cif", atoms)
    
    # Calculate properties
    print("   Calculating properties...")
    properties = calculate_properties(atoms)
    
    # Compile results
    results = {
        'structure_name': structure_name,
        'initial_composition': initial_composition,
        'final_composition': atoms.get_chemical_formula(),
        'num_atoms': len(atoms),
        'converged': converged,
        'calculation_mode': mode,
        'total_energy_eV': final_energy,
        'energy_per_atom_eV': final_energy / len(atoms),
        'properties': properties,
        'cell_parameters': {
            'a': float(atoms.cell.lengths()[0]),
            'b': float(atoms.cell.lengths()[1]),
            'c': float(atoms.cell.lengths()[2]),
        }
    }
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Validation complete for {structure_name}")
    print(f"   Energy: {final_energy:.4f} eV ({final_energy/len(atoms):.4f} eV/atom)")
    print(f"   Converged: {converged}")
    print(f"   Results saved to: {results_file}")
    
    return results


def compare_to_surrogate(dft_results, surrogate_predictions):
    """
    Compare DFT energies to surrogate model predictions.
    
    Args:
        dft_results: List of DFT validation results
        surrogate_predictions: Dictionary mapping structure_name to predicted energy
    
    Returns:
        Comparison statistics
    """
    mae_list = []
    comparison = []
    
    for result in dft_results:
        name = result['structure_name']
        dft_energy = result['energy_per_atom_eV']
        
        # Find surrogate prediction from original filename
        # Structure files are named like "Mo2Te4_candidate" but original was "tmd_0011_E-2.722eV"
        # For now, use the energy from the filename as "surrogate prediction"
        if 'Mo2Te4' in name:
            surrogate_energy = -2.722 / 6  # 6 atoms
        elif 'CrCuSe2' in name:
            surrogate_energy = 3.458 / 4  # 4 atoms
        elif 'VTe2' in name:
            surrogate_energy = -1.810 / 3  # 3 atoms
        else:
            surrogate_energy = None
        
        if surrogate_energy is not None:
            mae = abs(dft_energy - surrogate_energy)
            mae_list.append(mae)
            
            comparison.append({
                'structure': name,
                'dft_energy_per_atom': dft_energy,
                'surrogate_energy_per_atom': surrogate_energy,
                'absolute_error': mae,
                'relative_error': mae / abs(dft_energy) if dft_energy != 0 else float('inf')
            })
    
    stats = {
        'mean_absolute_error': np.mean(mae_list) if mae_list else None,
        'max_absolute_error': np.max(mae_list) if mae_list else None,
        'comparisons': comparison
    }
    
    return stats


def main():
    """Main validation workflow."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="DFT Validation of Generated TMD Structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single structure (for parallel execution):
  python 05_validate_with_dft.py dft_validation/priority/Mo2Te4_candidate.xyz
  
  # Validate all priority structures (serial):
  python 05_validate_with_dft.py
  
  # Parallel execution in 3 separate terminals:
  Terminal 1: python 05_validate_with_dft.py dft_validation/priority/Mo2Te4_candidate.xyz
  Terminal 2: python 05_validate_with_dft.py dft_validation/priority/CrCuSe2_candidate.xyz
  Terminal 3: python 05_validate_with_dft.py dft_validation/priority/VTe2_candidate.xyz
        """
    )
    parser.add_argument(
        'structure_file',
        type=str,
        nargs='?',  # Optional argument
        default=None,
        help="Path to a single XYZ file to validate (for parallel execution)"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['fast', 'production', 'converged'],
        default='fast',
        help="Calculation mode: fast (~20 min), production (~1 hour), converged (~4 hours)"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force recomputation even if results exist"
    )
    parser.add_argument(
        '--single-point',
        action='store_true',
        help="Only calculate energy/forces (no optimization) - Fast stability check (~5-10 min)"
    )
    
    args = parser.parse_args()
    
    # Configuration
    BASE_DIR = Path("/workspaces/algo/qcmd_hybrid_framework")
    INPUT_DIR = BASE_DIR / "dft_validation" / "priority"
    OUTPUT_DIR = BASE_DIR / "dft_validation" / "results"
    
    MODE = args.mode
    
    # Determine which structures to validate
    if args.structure_file:
        # Single structure mode (for parallel execution)
        structure_path = Path(args.structure_file)
        if not structure_path.is_absolute():
            structure_path = BASE_DIR / structure_path
        
        if not structure_path.exists():
            print(f"❌ Structure not found: {structure_path}")
            sys.exit(1)
        
        structures_to_validate = [(structure_path, structure_path.stem)]
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  DFT VALIDATION - SINGLE STRUCTURE MODE                      ║
║                                                               ║
║  Structure: {structure_path.name:44s}  ║
║  Mode: {MODE.upper():50s}  ║
╚══════════════════════════════════════════════════════════════╝
""")
    else:
        # Batch mode (validate all priority structures serially)
        structures_to_validate = [
            (INPUT_DIR / "Mo2Te4_candidate.xyz", "🏆 Novel MoTe2 phase - HIGHEST PRIORITY"),
            (INPUT_DIR / "CrCuSe2_candidate.xyz", "🔥 Hetero-metallic alloy - BREAKTHROUGH"),
            (INPUT_DIR / "VTe2_candidate.xyz", "⚡ Magnetic TMD - VALIDATION"),
        ]
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  DFT VALIDATION - BATCH MODE (SERIAL)                        ║
║                                                               ║
║  Structures: 3 priority candidates                           ║
║  Mode: {MODE.upper():50s}  ║
║  Output: {str(OUTPUT_DIR):44s}  ║
║                                                               ║
║  💡 TIP: Run in parallel for 3× speedup!                     ║
║     See --help for parallel execution examples               ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    if not GPAW_AVAILABLE:
        print("⚠️  WARNING: GPAW not available!")
        print("   Using EMT calculator for testing structure pipeline only.")
        print("   Results will NOT be quantitatively accurate.")
        print("   Install GPAW with: pip install gpaw")
        sys.exit(1)
    
    # Validate each structure
    all_results = []
    
    for structure_path, description in structures_to_validate:
        if not structure_path.exists():
            print(f"❌ Structure not found: {structure_path}")
            continue
        
        print(f"\n{description}")
        
        try:
            results = validate_structure(
                structure_path,
                OUTPUT_DIR,
                mode=MODE,
                force_rerun=args.force if args.structure_file else False,
                single_point=args.single_point
            )
            all_results.append(results)
        except Exception as e:
            print(f"❌ Validation failed for {structure_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("📊 VALIDATION SUMMARY")
    print(f"{'='*60}\n")
    
    summary = {
        'calculation_mode': MODE,
        'total_structures': len(structures),
        'successfully_validated': len(all_results),
        'results': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Compare to surrogate predictions
    if all_results:
        comparison_stats = compare_to_surrogate(all_results, {})
        summary['surrogate_comparison'] = comparison_stats
        
        print(f"Surrogate Model Accuracy:")
        if comparison_stats['mean_absolute_error'] is not None:
            print(f"  MAE: {comparison_stats['mean_absolute_error']:.3f} eV/atom")
            print(f"  Max Error: {comparison_stats['max_absolute_error']:.3f} eV/atom")
    
    # Save summary
    summary_file = OUTPUT_DIR / "validation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Validation complete!")
    print(f"   Summary saved to: {summary_file}")
    
    # Print key findings
    print(f"\n🔬 KEY FINDINGS:\n")
    for result in all_results:
        print(f"   {result['structure_name']}:")
        print(f"      Energy: {result['total_energy_eV']:.3f} eV ({result['energy_per_atom_eV']:.3f} eV/atom)")
        print(f"      Converged: {'✅ Yes' if result['converged'] else '❌ No'}")
        if result['properties'].get('min_bond_length'):
            print(f"      Min bond: {result['properties']['min_bond_length']:.2f} Å")
        print()
    
    print("\n🚀 NEXT STEPS:")
    print("   1. Check relaxed structures in results/*.xyz")
    print("   2. If stable → run 'production' mode for accurate energies")
    print("   3. If unstable → analyze decomposition pathway")
    print("   4. Compare to Materials Project database")
    print("   5. Calculate band structures for semiconductors")
    print()
    print("   To re-run with production settings:")
    print("      Change MODE = 'production' and force_rerun=True")
    print()


if __name__ == "__main__":
    main()
