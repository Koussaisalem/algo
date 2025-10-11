#!/usr/bin/env python3
"""
Download REAL TMD data from Materials Project database.

Materials Project contains 150,000+ DFT-calculated materials including
comprehensive coverage of 2D transition metal dichalcogenides (TMDs).

All data is from published DFT calculations (VASP), not synthetic!

Data includes:
- Crystal structures (experimental + DFT-relaxed)
- Band gaps (PBE functional)
- Formation energies
- Electronic band structures
- Magnetic properties

Website: https://materialsproject.org/
API Docs: https://api.materialsproject.org/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from ase import Atoms
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

# Target TMD formulas
TMD_FORMULAS = [
    # Group 6 TMDs (semiconducting)
    "MoS2", "MoSe2", "MoTe2",
    "WS2", "WSe2", "WTe2",
    "CrS2", "CrSe2", "CrTe2",
    
    # Group 7 TMDs  
    "ReS2", "ReSe2", "ReTe2",
    
    # Group 5 TMDs
    "VS2", "VSe2", "VTe2",
    "NbS2", "NbSe2", "NbTe2",
    "TaS2", "TaSe2", "TaTe2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download real TMD data from Materials Project."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tmd"),
        help="Output directory.",
    )
    parser.add_argument(
        "--formulas",
        nargs="+",
        default=TMD_FORMULAS,
        help="TMD formulas to download.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="FDKf2QDmlclbUtwODgZku3Kh0ESiqWY8",
        help="Materials Project API key",
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=20,
        help="Max structures per formula.",
    )
    return parser.parse_args()


def query_materials_project(
    formula: str,
    api_key: str = None,
    max_structures: int = 20
) -> List[Dict]:
    """
    Query Materials Project for TMD structures.
    
    Returns list of structures with properties.
    """
    results = []
    
    try:
        # Initialize MP API client
        with MPRester(api_key) as mpr:
            # Search for materials with this formula
            docs = mpr.materials.summary.search(
                formula=formula,
                num_elements=(2, 3),  # Binary or ternary
                fields=[
                    "material_id",
                    "formula_pretty",
                    "structure",
                    "band_gap",
                    "formation_energy_per_atom",
                    "energy_above_hull",
                    "is_stable",
                    "symmetry",
                    "ordering",  # Magnetic ordering (was 'magnetic_ordering')
                    "total_magnetization",
                ]
            )
            
            if not docs:
                return []
            
            print(f"    Found {len(docs)} structures for {formula}")
            
            for doc in docs[:max_structures]:
                # Extract structure
                structure = doc.structure
                
                # Convert to ASE Atoms
                adaptor = AseAtomsAdaptor()
                atoms = adaptor.get_atoms(structure)
                
                # Extract properties
                props = {
                    "material_id": doc.material_id,
                    "formula": doc.formula_pretty,
                    "gap": doc.band_gap,  # eV (PBE functional)
                    "formation_energy": doc.formation_energy_per_atom,  # eV/atom
                    "ehull": doc.energy_above_hull,  # eV/atom
                    "is_stable": doc.is_stable,
                    "spacegroup": doc.symmetry.symbol if doc.symmetry else "unknown",
                    "magnetic": doc.ordering != "NM" if hasattr(doc, 'ordering') and doc.ordering else False,
                    "magmom": doc.total_magnetization if doc.total_magnetization else 0.0,
                    "source": "Materials Project",
                    "dft_functional": "PBE",
                }
                
                results.append((atoms, props))
                
    except Exception as e:
        print(f"    Error querying {formula}: {e}")
        return []
    
    return results


def process_structure(atoms: Atoms, props: Dict) -> Dict:
    """Convert ASE Atoms to our format."""
    data = {
        "positions": torch.tensor(atoms.get_positions(), dtype=torch.float64),
        "atomic_numbers": torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
        "cell": torch.tensor(atoms.get_cell().array, dtype=torch.float64),
        "pbc": torch.tensor(atoms.get_pbc(), dtype=torch.bool),
        "properties": props,
    }
    return data


def main():
    args = parse_args()
    
    print("=" * 70)
    print("🧲 Operation Magnet: Materials Project TMD Download")
    print("=" * 70)
    print(f"Target formulas: {len(args.formulas)}")
    print(f"Max structures per formula: {args.max_structures}")
    print(f"Output: {args.output_dir}")
    print(f"Data source: Materials Project (REAL DFT data!)")
    print("=" * 70)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    all_structures = []
    
    # Query for each formula
    print("\nQuerying Materials Project...")
    for formula in tqdm(args.formulas, desc="Downloading"):
        structures = query_materials_project(
            formula,
            api_key=args.api_key,
            max_structures=args.max_structures
        )
        all_structures.extend(structures)
    
    print(f"\n✓ Downloaded {len(all_structures)} real TMD structures!")
    
    if len(all_structures) == 0:
        print("\n⚠️  No structures found!")
        print("This might be due to:")
        print("  1. Network issues")
        print("  2. API rate limiting")
        print("  3. Need API key for full access")
        print("\nGet free API key at: https://materialsproject.org/api")
        return
    
    # Process structures
    print("\nProcessing structures...")
    dataset = []
    for atoms, props in tqdm(all_structures, desc="Converting"):
        try:
            sample = process_structure(atoms, props)
            dataset.append(sample)
        except Exception as e:
            print(f"  Warning: Failed to process {props.get('formula', 'unknown')}: {e}")
            continue
    
    # Save dataset
    output_file = args.output_dir / "tmd_materialsproject.pt"
    torch.save(dataset, output_file)
    print(f"\n✓ Saved to {output_file}")
    
    # Statistics
    formulas = {}
    stable_count = 0
    semiconductor_count = 0
    
    for sample in dataset:
        formula = sample["properties"]["formula"]
        formulas[formula] = formulas.get(formula, 0) + 1
        
        if sample["properties"]["is_stable"]:
            stable_count += 1
        
        gap = sample["properties"]["gap"]
        if gap is not None and 0.1 < gap < 4.0:
            semiconductor_count += 1
    
    # Summary
    summary = {
        "n_samples": len(dataset),
        "n_stable": stable_count,
        "n_semiconductors": semiconductor_count,
        "formulas": formulas,
        "source": "Materials Project",
        "data_type": "REAL DFT (VASP/PBE)",
    }
    
    summary_file = args.output_dir / "dataset_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print("Dataset Summary:")
    print(f"  Total structures: {len(dataset)}")
    print(f"  Stable structures: {stable_count} ({100*stable_count/len(dataset):.1f}%)")
    print(f"  Semiconductors: {semiconductor_count} ({100*semiconductor_count/len(dataset):.1f}%)")
    print(f"  Unique formulas: {len(formulas)}")
    print(f"\nFormula distribution:")
    for formula, count in sorted(formulas.items(), key=lambda x: -x[1]):
        print(f"    {formula}: {count}")
    
    print(f"\n{'=' * 70}")
    print("✓ Real TMD dataset complete!")
    print("  Source: Materials Project (materialsproject.org)")
    print("  Data type: DFT-calculated (VASP, PBE functional)")
    print("  Quality: Peer-reviewed, published data")
    print(f"{'=' * 70}\n")
    print("Next step: xTB enrichment")
    print(f"  python scripts/tmd/01_enrich_tmd_dataset.py \\")
    print(f"    --input-path {output_file}")


if __name__ == "__main__":
    main()
