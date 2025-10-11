#!/usr/bin/env python3
"""
🔥 Temperature-Dependent MD Screening for CrCuSe₂ MBE Growth

Purpose:
--------
Find the optimal molecular beam epitaxy (MBE) growth temperature for metastable
2D CrCuSe₂ by running ab-initio molecular dynamics at multiple temperatures.

Scientific Question:
-------------------
At what temperature does CrCuSe₂ have:
1. Sufficient atomic mobility for layer-by-layer growth?
2. Structural stability (no decomposition or phase transition)?
3. Resistance to bulk phase (mp-568587) nucleation?

Method:
-------
- AIMD with GPAW (DFT forces at each timestep)
- Temperature range: 300K - 800K (27°C - 527°C)
- Timestep: 1-2 fs
- Ensemble: NVT (constant volume, temperature via Langevin thermostat)
- Duration: 2-10 ps depending on mode

Outputs:
--------
- Trajectory files (.traj) for each temperature
- Stability metrics (RMSD, bond lengths, energy fluctuations)
- Diffusion coefficients (atomic mobility)
- Recommended growth temperature window

Author: QCMD-ECS Synthesis Lab
Date: October 8, 2025
"""

import numpy as np
from pathlib import Path
import json
import time
from ase.io import read, write
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from gpaw import GPAW, PW, FermiDirac
import argparse


class TemperatureScreeningMD:
    """
    Run molecular dynamics at multiple temperatures to find optimal MBE growth conditions.
    """
    
    def __init__(self, structure_path: str, output_dir: str = "results"):
        """
        Initialize MD screening.
        
        Args:
            structure_path: Path to DFT-validated CrCuSe₂ structure (CIF or TRAJ)
            output_dir: Directory for results
        """
        self.structure = read(structure_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        print("=" * 80)
        print("🔥 TEMPERATURE-DEPENDENT MD SCREENING FOR MBE GROWTH OPTIMIZATION")
        print("=" * 80)
        print(f"Structure loaded: {structure_path}")
        print(f"Formula: {self.structure.get_chemical_formula()}")
        print(f"Atoms: {len(self.structure)}")
        print(f"Cell: {self.structure.cell.cellpar()}")
        print(f"Output: {self.output_dir}")
        print("=" * 80)
    
    def setup_calculator(self, mode: str = 'fast'):
        """
        Setup GPAW calculator for MD.
        
        Args:
            mode: 'fast' (screening), 'production' (publication)
        
        Returns:
            GPAW calculator
        """
        if mode == 'fast':
            # Fast screening: Lower accuracy but 3-4x faster
            calc = GPAW(
                mode=PW(300),  # 300 eV cutoff
                xc='PBE',
                kpts=(2, 2, 1),  # Sparse k-points
                occupations=FermiDirac(0.1),
                convergence={'energy': 0.001},
                txt='md_calc.log',
                symmetry='off'  # Important for MD!
            )
            print("⚡ Fast screening mode: 300 eV cutoff, 2×2×1 k-points")
        
        elif mode == 'production':
            # Production: Higher accuracy for publication
            calc = GPAW(
                mode=PW(400),  # 400 eV cutoff
                xc='PBE',
                kpts=(4, 4, 1),
                occupations=FermiDirac(0.05),
                convergence={'energy': 0.0001},
                txt='md_calc.log',
                symmetry='off'
            )
            print("🎯 Production mode: 400 eV cutoff, 4×4×1 k-points")
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return calc
    
    def run_md_at_temperature(self, temperature: float, duration_ps: float, 
                             timestep_fs: float = 1.0, mode: str = 'fast'):
        """
        Run NVT molecular dynamics at specified temperature.
        
        Args:
            temperature: Temperature in Kelvin
            duration_ps: Simulation duration in picoseconds
            timestep_fs: MD timestep in femtoseconds
            mode: Calculator accuracy mode
        
        Returns:
            dict with trajectory path and statistics
        """
        print(f"\n{'=' * 80}")
        print(f"🌡️  MD SIMULATION AT {temperature}K ({temperature - 273.15:.0f}°C)")
        print(f"{'=' * 80}")
        print(f"Duration: {duration_ps} ps")
        print(f"Timestep: {timestep_fs} fs")
        print(f"Mode: {mode}")
        
        # Prepare atoms
        atoms = self.structure.copy()
        
        # Setup calculator
        calc = self.setup_calculator(mode=mode)
        atoms.calc = calc
        
        # Set initial velocities (Maxwell-Boltzmann distribution)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        
        # Remove center-of-mass momentum
        p = atoms.get_momenta()
        p -= p.sum(axis=0) / len(atoms)
        atoms.set_momenta(p)
        
        print(f"\n✅ Initial temperature: {atoms.get_temperature():.1f}K")
        print(f"✅ Initial kinetic energy: {atoms.get_kinetic_energy():.3f} eV")
        
        # Setup Langevin dynamics (NVT ensemble)
        # friction = 0.01 is standard for AIMD (weak coupling to thermostat)
        timestep = timestep_fs * units.fs
        friction = 0.01  # 1/time units
        
        dyn = Langevin(
            atoms,
            timestep=timestep,
            temperature_K=temperature,
            friction=friction,
            trajectory=None  # We'll attach our own
        )
        
        # Trajectory file
        traj_file = self.output_dir / f"md_{int(temperature)}K.traj"
        
        # Attach trajectory writer (save every 10 steps)
        from ase.io.trajectory import Trajectory
        traj = Trajectory(traj_file, 'w', atoms)
        dyn.attach(traj.write, interval=10)
        
        # Attach logger
        log_file = self.output_dir / f"md_{int(temperature)}K.log"
        def log_progress():
            T_inst = atoms.get_temperature()
            E_pot = atoms.get_potential_energy()
            E_kin = atoms.get_kinetic_energy()
            E_tot = E_pot + E_kin
            step = dyn.nsteps
            time_ps = step * timestep_fs / 1000
            
            with open(log_file, 'a') as f:
                f.write(f"{step:6d} {time_ps:6.2f} {T_inst:7.1f} {E_pot:12.4f} "
                       f"{E_kin:10.4f} {E_tot:12.4f}\n")
            
            if step % 50 == 0:
                print(f"  Step {step:5d} | {time_ps:5.2f} ps | "
                      f"T = {T_inst:6.1f}K | E_tot = {E_tot:10.3f} eV")
        
        # Write header
        with open(log_file, 'w') as f:
            f.write("# Step  Time(ps)  T(K)    E_pot(eV)    E_kin(eV)   E_tot(eV)\n")
        
        dyn.attach(log_progress, interval=10)
        
        # Calculate number of steps
        n_steps = int(duration_ps * 1000 / timestep_fs)  # ps to fs, then divide by timestep
        
        print(f"\n🚀 Starting MD: {n_steps} steps ({n_steps * timestep_fs / 1000:.2f} ps)")
        print(f"   Estimated time: {n_steps * 2 / 60:.0f} - {n_steps * 5 / 60:.0f} minutes")
        
        start_time = time.time()
        
        # Run MD
        try:
            dyn.run(n_steps)
            print(f"\n✅ MD completed successfully!")
        except Exception as e:
            print(f"\n❌ MD failed: {e}")
            return {'error': str(e), 'temperature': temperature}
        finally:
            traj.close()
        
        elapsed_time = time.time() - start_time
        print(f"⏱️  Elapsed time: {elapsed_time / 60:.1f} minutes")
        
        # Final statistics
        T_final = atoms.get_temperature()
        T_avg = T_final  # Approximate, will calculate properly in analysis
        
        print(f"\n📊 Final Statistics:")
        print(f"   Temperature: {T_final:.1f}K (target: {temperature}K)")
        print(f"   Deviation: {abs(T_final - temperature):.1f}K")
        
        # Save final structure
        final_structure = self.output_dir / f"md_{int(temperature)}K_final.cif"
        write(final_structure, atoms)
        
        result = {
            'temperature': temperature,
            'duration_ps': duration_ps,
            'timestep_fs': timestep_fs,
            'n_steps': n_steps,
            'traj_file': str(traj_file),
            'log_file': str(log_file),
            'final_structure': str(final_structure),
            'T_final': T_final,
            'elapsed_time_min': elapsed_time / 60,
            'success': True
        }
        
        return result
    
    def screen_temperatures(self, temperatures: list, duration_ps: float, 
                           timestep_fs: float = 1.0, mode: str = 'fast'):
        """
        Screen multiple temperatures sequentially.
        
        Args:
            temperatures: List of temperatures in Kelvin
            duration_ps: Simulation duration per temperature (ps)
            timestep_fs: MD timestep (fs)
            mode: Calculator accuracy
        
        Returns:
            dict with results for all temperatures
        """
        results = {}
        
        print(f"\n{'=' * 80}")
        print(f"🔬 SCREENING {len(temperatures)} TEMPERATURES")
        print(f"{'=' * 80}")
        print(f"Temperatures: {temperatures}")
        print(f"Duration each: {duration_ps} ps")
        print(f"Total estimated time: {len(temperatures) * duration_ps * 2:.0f} - "
              f"{len(temperatures) * duration_ps * 5:.0f} minutes")
        print(f"{'=' * 80}")
        
        for i, T in enumerate(temperatures, 1):
            print(f"\n{'#' * 80}")
            print(f"TEMPERATURE {i}/{len(temperatures)}: {T}K ({T - 273.15:.0f}°C)")
            print(f"{'#' * 80}")
            
            result = self.run_md_at_temperature(T, duration_ps, timestep_fs, mode)
            results[f"{int(T)}K"] = result
            
            # Save intermediate results
            with open(self.output_dir / 'screening_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✅ Completed {i}/{len(temperatures)} temperatures")
        
        print(f"\n{'=' * 80}")
        print(f"🎉 ALL TEMPERATURES COMPLETED!")
        print(f"{'=' * 80}")
        
        return results


def main():
    """
    Main entry point for temperature screening.
    """
    parser = argparse.ArgumentParser(
        description="Temperature-dependent MD screening for MBE growth optimization"
    )
    
    parser.add_argument(
        '--structure',
        type=str,
        default='../../dft_validation/results/CrCuSe2_rescue_relaxed.cif',
        help='Path to DFT-validated structure'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['fast', 'production'],
        default='fast',
        help='Calculation accuracy (fast for screening, production for publication)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='MD duration per temperature in ps (default: 2 ps for fast, 10 ps for production)'
    )
    
    parser.add_argument(
        '--timestep',
        type=float,
        default=1.0,
        help='MD timestep in fs (default: 1.0 fs)'
    )
    
    parser.add_argument(
        '--temperatures',
        type=str,
        default='300,400,500,600,700',
        help='Comma-separated list of temperatures in K (default: 300,400,500,600,700)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Parse temperatures
    temperatures = [float(T.strip()) for T in args.temperatures.split(',')]
    
    # Set default duration based on mode
    if args.duration is None:
        duration_ps = 2.0 if args.mode == 'fast' else 10.0
    else:
        duration_ps = args.duration
    
    # Initialize screener
    screener = TemperatureScreeningMD(
        structure_path=args.structure,
        output_dir=args.output
    )
    
    # Run screening
    results = screener.screen_temperatures(
        temperatures=temperatures,
        duration_ps=duration_ps,
        timestep_fs=args.timestep,
        mode=args.mode
    )
    
    # Summary
    print(f"\n{'=' * 80}")
    print("📊 SCREENING SUMMARY")
    print(f"{'=' * 80}")
    
    for temp_key, result in results.items():
        if result.get('success'):
            print(f"{temp_key:8s}: ✅ Success ({result['elapsed_time_min']:.1f} min)")
        else:
            print(f"{temp_key:8s}: ❌ Failed - {result.get('error', 'Unknown error')}")
    
    print(f"\n✅ Results saved to: {args.output}/")
    print(f"✅ Next step: Run analysis with 'python analyze_md_trajectories.py'")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
