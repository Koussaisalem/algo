import argparse
import torch
from xtb.interface import Calculator, Param
from rdkit import Chem
from rdkit.Chem import AllChem

DTYPE = torch.float32

def main():
    parser = argparse.ArgumentParser(description="Enrich dataset with GFN2-xTB calculations.")
    parser.add_argument("--input_path", type=str, default="data/qm9_micro_raw.pt",
                        help="Path to the input raw dataset (.pt file).")
    parser.add_argument("--output_path", type=str, default="data/qm9_micro_enriched.pt",
                        help="Path to save the enriched dataset (.pt file).")
    args = parser.parse_args()

    print(f"Loading raw dataset from {args.input_path}")
    raw_dataset = torch.load(args.input_path)
    print(f"Loaded {len(raw_dataset)} molecules.")

    enriched_dataset = []
    for i, (atomic_numbers, positions) in enumerate(raw_dataset):
        if i % 100 == 0:
            print(f"Processing molecule {i+1}/{len(raw_dataset)}")

        # Convert to RDKit molecule
        mol = Chem.Mol()
        editable_mol = Chem.EditableMol(mol)
        for atom_num in atomic_numbers:
            editable_mol.AddAtom(Chem.Atom(int(atom_num)))
        mol = editable_mol.GetMol()

        conf = Chem.Conformer(mol.GetNumAtoms())
        for j, pos in enumerate(positions):
            conf.SetAtomPosition(j, [float(pos[0]), float(pos[1]), float(pos[2])])
        mol.AddConformer(conf)

        # Perform GFN2-xTB calculation
        res = Calculator(Param.GFN2xTB, atomic_numbers.numpy(), positions.numpy()).singlepoint()
        energy = res.get_energy()
        orbitals = torch.tensor(res.get_orbital_coefficients(), dtype=DTYPE)

        enriched_dataset.append({
            'atomic_numbers': atomic_numbers,
            'positions': positions,
            'energy': energy,
            'orbitals': orbitals
        })

    print(f"Saving enriched dataset to {args.output_path}")
    torch.save(enriched_dataset, args.output_path)
    print("Phase 1.2 Complete: 2_enrich_dataset.py created successfully.")

if __name__ == "__main__":
    main()