#!/usr/bin/env python3
"""
🧲 Operation Magnet: TMD Structure Generation

Generate novel 2D transition metal dichalcogenide structures using:
- Trained TMD score model (manifold diffusion)
- Trained TMD surrogate (energy guidance via MAECS)
- Stiefel manifold constraints (orthonormality preservation)

This is the critical Week 3 Day 15-17 milestone - make or break for publication!
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qcmd_hybrid_framework.models.tmd_surrogate import TMDSurrogate
from qcmd_hybrid_framework.models.vector_output_model import VectorOutputModel
from qcmd_hybrid_framework.qcmd_ecs.core.dynamics import run_reverse_diffusion
from qcmd_hybrid_framework.qcmd_ecs.core.manifold import retract_to_manifold
from qcmd_hybrid_framework.qcmd_ecs.core.types import DTYPE
from nequip.data import AtomicDataDict

# Atomic number to symbol mapping (extended for TMDs)
ATOMIC_SYMBOLS = {
    1: "H", 3: "Li", 7: "N", 8: "O", 11: "Na", 12: "Mg",
    16: "S", 17: "Cl", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    34: "Se", 35: "Br", 37: "Rb", 41: "Nb", 42: "Mo",
    47: "Ag", 48: "Cd", 49: "In", 52: "Te", 53: "I", 55: "Cs", 56: "Ba",
    73: "Ta", 74: "W", 75: "Re", 80: "Hg", 81: "Tl"
}

DTYPE = torch.float64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🧲 Generate TMD structures with manifold diffusion."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to enriched TMD dataset (templates).",
    )
    parser.add_argument(
        "--score-model-path",
        type=Path,
        default=Path("qcmd_hybrid_framework/models/tmd_score/score_model_state_dict.pt"),
        help="Path to trained score model.",
    )
    parser.add_argument(
        "--surrogate-path",
        type=Path,
        default=Path("qcmd_hybrid_framework/models/tmd_surrogate/surrogate_state_dict.pt"),
        help="Path to trained surrogate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qcmd_hybrid_framework/results/generated_tmds"),
        help="Output directory for generated structures.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of TMD structures to generate.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Reverse diffusion steps.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.5,
        help="Initial noise magnitude.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.05,
        help="Score-based update step size.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.02,
        help="Stochastic noise scale.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.15,
        help="Energy guidance weight (MAECS).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu/cuda).",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=30,
        help="Maximum atoms per structure.",
    )
    return parser.parse_args()


def load_dataset(path: Path, max_atoms: int) -> List[Dict]:
    """Load and filter dataset."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    filtered = [s for s in data if len(s['atomic_numbers']) <= max_atoms]
    print(f"Loaded {len(data)} structures, filtered to {len(filtered)} (≤{max_atoms} atoms)")
    return filtered


def load_models(
    score_path: Path,
    surrogate_path: Path,
    dataset: List[Dict],
    device: torch.device
) -> Tuple[VectorOutputModel, TMDSurrogate]:
    """Load trained models."""
    # Extract atomic numbers from dataset (same as training)
    all_z = set()
    for sample in dataset:
        all_z.update(sample['atomic_numbers'].tolist())
    atomic_numbers = sorted(all_z)
    
    print(f"Found {len(atomic_numbers)} unique atomic numbers in dataset")
    
    # Load score model (VectorOutputModel with TMD config)
    tmd_config = {
        "l_max": 2,  # d-orbital support
        "num_layers": 4,
        "num_features": 64,
        "atomic_numbers": tuple(atomic_numbers),
        "type_names": tuple(f"Z{z}" for z in atomic_numbers),
    }
    score_model = VectorOutputModel(model_config=tmd_config)
    score_state = torch.load(score_path, map_location=device, weights_only=False)
    score_model.load_state_dict(score_state)
    score_model.to(device=device, dtype=DTYPE)
    score_model.eval()
    
    # Load surrogate
    surrogate = TMDSurrogate()
    surr_state = torch.load(surrogate_path, map_location=device, weights_only=False)
    surrogate.load_state_dict(surr_state)
    surrogate.to(device=device, dtype=DTYPE)
    surrogate.eval()
    
    return score_model, surrogate


def create_pyg_data(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor
) -> Data:
    """Create PyG Data object."""
    return Data(
        pos=positions.to(dtype=DTYPE),
        z=atomic_numbers.to(dtype=torch.long)
    )


def reconstruct_positions(
    U: torch.Tensor,
    atomic_numbers: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Reconstruct Cartesian positions from manifold frame U.
    
    Args:
        U: (n_atoms, 3) manifold frame
        atomic_numbers: (n_atoms,) atomic numbers
        device: torch device
    
    Returns:
        positions: (n_atoms, 3) Cartesian coordinates
    """
    # Simple reconstruction: U is already in 3D space
    # For TMDs, we might want to add lattice constraints here
    return U.clone()


def save_xyz(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    energy: float,
    filepath: Path
) -> None:
    """Save structure to XYZ file."""
    n_atoms = len(atomic_numbers)
    
    with open(filepath, 'w') as f:
        f.write(f"{n_atoms}\n")
        f.write(f"Generated TMD structure, Energy: {energy:.6f} eV\n")
        
        for i in range(n_atoms):
            z = int(atomic_numbers[i].item())
            symbol = ATOMIC_SYMBOLS.get(z, f"Z{z}")
            x, y, z_coord = positions[i].tolist()
            f.write(f"{symbol:3s} {x:15.8f} {y:15.8f} {z_coord:15.8f}\n")


def check_validity(positions: torch.Tensor, atomic_numbers: torch.Tensor) -> Dict[str, bool]:
    """
    Quick validity checks for generated structures.
    
    Returns:
        dict with validity flags
    """
    # Check for NaN/Inf
    has_nan = torch.isnan(positions).any().item() or torch.isinf(positions).any().item()
    
    # Check atom distances (too close = invalid)
    if len(positions) > 1:
        dists = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0))[0]
        # Mask diagonal
        mask = ~torch.eye(len(positions), dtype=torch.bool)
        min_dist = dists[mask].min().item()
        too_close = min_dist < 0.5  # Ångströms
    else:
        too_close = False
        min_dist = float('inf')
    
    # Check for reasonable coordinates (not too dispersed)
    coord_std = positions.std().item()
    too_dispersed = coord_std > 50.0  # Ångströms
    
    return {
        'has_nan': has_nan,
        'too_close': too_close,
        'too_dispersed': too_dispersed,
        'valid': not (has_nan or too_close or too_dispersed),
        'min_dist': min_dist,
        'coord_std': coord_std
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🧲 OPERATION MAGNET: TMD STRUCTURE GENERATION")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Diffusion steps: {args.num_steps}")
    print(f"Energy guidance (gamma): {args.gamma}")
    print(f"Device: {device}")
    print("=" * 80)
    print()
    
    # Load data and models
    print("Loading dataset and models...")
    dataset = load_dataset(args.dataset_path, args.max_atoms)
    score_model, surrogate = load_models(
        args.score_model_path,
        args.surrogate_path,
        dataset,
        device
    )
    print(f"✓ Loaded score model and surrogate")
    print()
    
    # Generation statistics
    stats = {
        'generated': 0,
        'valid': 0,
        'invalid_nan': 0,
        'invalid_close': 0,
        'invalid_dispersed': 0,
        'energies': [],
        'orthogonality_errors': []
    }
    
    print("Starting generation...")
    print("=" * 80)
    
    for i in tqdm(range(args.num_samples), desc="Generating TMDs"):
        # Select random template
        template_idx = torch.randint(0, len(dataset), (1,)).item()
        template = dataset[template_idx]
        
        # Extract template data
        positions_ref = template['positions'].to(device=device, dtype=DTYPE)
        atomic_numbers = template['atomic_numbers'].to(device=device)
        U_ref = template['manifold_frame'].to(device=device, dtype=DTYPE)
        
        n_atoms = len(atomic_numbers)
        
        # Add noise to create initial state
        noise = torch.randn_like(U_ref) * args.noise_scale
        U_T = U_ref + noise
        U_T = retract_to_manifold(U_T)  # Project back to manifold
        
        # Define score function wrapper
        def score_fn(U_t: torch.Tensor) -> torch.Tensor:
            """Score function for current noisy state."""
            positions_t = reconstruct_positions(U_t, atomic_numbers, device)
            data = create_pyg_data(positions_t, atomic_numbers)
            
            with torch.no_grad():
                score = score_model(data)
            
            return score.reshape(n_atoms, 3)
        
        # Define energy function wrapper (for MAECS)
        def energy_fn(U_t: torch.Tensor) -> torch.Tensor:
            """Energy function for MAECS guidance."""
            if args.gamma == 0.0:
                return torch.tensor(0.0, device=device, dtype=DTYPE)
            
            positions_t = reconstruct_positions(U_t, atomic_numbers, device)
            data = create_pyg_data(positions_t, atomic_numbers)
            
            with torch.no_grad():
                energy = surrogate(data)
            
            return energy.squeeze()
        
        # Define schedules (constant for simplicity)
        gamma_schedule = lambda t: args.gamma
        eta_schedule = lambda t: args.eta
        tau_schedule = lambda t: args.tau
        
        # Wrap score function to match signature (U_t, t)
        def score_model_wrapper(U_t: torch.Tensor, t: int) -> torch.Tensor:
            """Score model that takes (U_t, t) and returns score."""
            positions_t = reconstruct_positions(U_t, atomic_numbers, device)
            data = create_pyg_data(positions_t, atomic_numbers)
            
            with torch.no_grad():
                score = score_model(data)
            
            return score.reshape(n_atoms, 3)
        
        # Wrap energy gradient function
        def energy_gradient_wrapper(U_t: torch.Tensor) -> torch.Tensor:
            """Energy gradient for MAECS."""
            if args.gamma == 0.0:
                return torch.zeros_like(U_t)
            
            positions_t = reconstruct_positions(U_t, atomic_numbers, device)
            positions_t.requires_grad_(True)
            
            data = create_pyg_data(positions_t, atomic_numbers)
            
            energy = surrogate(data).squeeze()
            
            # Compute gradient
            grad = torch.autograd.grad(energy, positions_t, create_graph=False)[0]
            
            return -grad  # Negative gradient points toward lower energy
        
        # Run reverse diffusion
        try:
            U_final = run_reverse_diffusion(
                U_T=U_T,
                score_model=score_model_wrapper,
                energy_gradient_model=energy_gradient_wrapper,
                gamma_schedule=gamma_schedule,
                eta_schedule=eta_schedule,
                tau_schedule=tau_schedule,
                num_steps=args.num_steps,
                seed=args.seed + i,  # Unique seed per sample
                callback=None
            )
            
            # Reconstruct final positions
            positions_final = reconstruct_positions(U_final, atomic_numbers, device)
            
            # Check orthogonality
            orth_error = torch.norm(U_final.T @ U_final - torch.eye(3, device=device, dtype=DTYPE)).item()
            stats['orthogonality_errors'].append(orth_error)
            
            # Predict energy
            data_final = create_pyg_data(positions_final, atomic_numbers)
            with torch.no_grad():
                energy = surrogate(data_final).item()
            stats['energies'].append(energy)
            
            # Validate structure
            validity = check_validity(positions_final, atomic_numbers)
            
            stats['generated'] += 1
            
            if validity['valid']:
                stats['valid'] += 1
                # Save valid structure
                filename = f"tmd_{i:04d}_E{energy:.3f}eV.xyz"
                save_xyz(positions_final.cpu(), atomic_numbers.cpu(), energy, args.output_dir / filename)
            else:
                if validity['has_nan']:
                    stats['invalid_nan'] += 1
                elif validity['too_close']:
                    stats['invalid_close'] += 1
                elif validity['too_dispersed']:
                    stats['invalid_dispersed'] += 1
                
                # Save invalid structure for debugging
                filename = f"tmd_{i:04d}_INVALID.xyz"
                save_xyz(positions_final.cpu(), atomic_numbers.cpu(), energy, args.output_dir / filename)
        
        except Exception as e:
            print(f"\n⚠️  Generation {i} failed: {e}")
            stats['invalid_nan'] += 1
            continue
    
    print()
    print("=" * 80)
    print("🎉 GENERATION COMPLETE!")
    print("=" * 80)
    print()
    print("📊 GENERATION STATISTICS")
    print("-" * 80)
    print(f"Total generated:       {stats['generated']}")
    if stats['generated'] > 0:
        print(f"Valid structures:      {stats['valid']} ({stats['valid']/stats['generated']*100:.1f}%)")
        print(f"Invalid (NaN/Inf):     {stats['invalid_nan']}")
        print(f"Invalid (too close):   {stats['invalid_close']}")
        print(f"Invalid (dispersed):   {stats['invalid_dispersed']}")
    else:
        print("⚠️  NO STRUCTURES GENERATED!")
    print()
    
    if stats['energies']:
        print("⚡ ENERGY STATISTICS")
        print("-" * 80)
        energies = np.array(stats['energies'])
        print(f"Mean energy:           {energies.mean():.3f} eV")
        print(f"Std energy:            {energies.std():.3f} eV")
        print(f"Min energy:            {energies.min():.3f} eV")
        print(f"Max energy:            {energies.max():.3f} eV")
        print()
    
    if stats['orthogonality_errors']:
        print("🔧 MANIFOLD CONSTRAINTS")
        print("-" * 80)
        orth_errors = np.array(stats['orthogonality_errors'])
        print(f"Mean orth error:       {orth_errors.mean():.2e}")
        print(f"Max orth error:        {orth_errors.max():.2e}")
        if orth_errors.max() < 1e-9:
            print(f"Status:                ✅ EXCELLENT (< 1e-9)")
        elif orth_errors.max() < 1e-6:
            print(f"Status:                ✅ GOOD (< 1e-6)")
        else:
            print(f"Status:                ⚠️  MODERATE (> 1e-6)")
        print()
    
    # Save statistics
    stats_file = args.output_dir / "generation_stats.json"
    with open(stats_file, 'w') as f:
        # Convert Path objects to strings for JSON serialization
        config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        json.dump({
            'config': config_dict,
            'statistics': {
                'total_generated': stats['generated'],
                'valid': stats['valid'],
                'validity_rate': stats['valid'] / stats['generated'] if stats['generated'] > 0 else 0,
                'invalid_breakdown': {
                    'nan_inf': stats['invalid_nan'],
                    'too_close': stats['invalid_close'],
                    'dispersed': stats['invalid_dispersed']
                },
                'energy': {
                    'mean': float(np.mean(stats['energies'])) if stats['energies'] else None,
                    'std': float(np.std(stats['energies'])) if stats['energies'] else None,
                    'min': float(np.min(stats['energies'])) if stats['energies'] else None,
                    'max': float(np.max(stats['energies'])) if stats['energies'] else None,
                },
                'orthogonality': {
                    'mean_error': float(np.mean(stats['orthogonality_errors'])) if stats['orthogonality_errors'] else None,
                    'max_error': float(np.max(stats['orthogonality_errors'])) if stats['orthogonality_errors'] else None,
                }
            }
        }, f, indent=2)
    
    print(f"✓ Statistics saved to {stats_file}")
    print()
    print("=" * 80)
    print("🎯 NEXT STEPS:")
    print("   1. Inspect generated .xyz files")
    print("   2. Run quick validation: scripts/tmd/05_quick_validate.py")
    print("   3. DFT validation on top candidates: scripts/tmd/06_dft_validate.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
