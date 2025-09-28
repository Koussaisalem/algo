import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
import os
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

def load_xyz_molecules(filepath):
    """Loads molecules from an .xyz file and returns a list of RDKit Mol objects."""
    molecules = []
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by molecule (assuming each molecule starts with a line containing the number of atoms)
    # and then by lines
    mol_blocks = content.strip().split('\n\n')
    
    for block in mol_blocks:
        if not block.strip():
            continue
        
        lines = block.strip().split('\n')
        if len(lines) < 2: # Needs at least atom count and comment line
            continue
        
        try:
            num_atoms = int(lines[0].strip())
        except ValueError:
            continue # Skip if atom count is not an integer

        # Reconstruct the XYZ block for RDKit
        xyz_block = f"{num_atoms}\n{lines[1]}\n" + "\n".join(lines[2:2+num_atoms])
        
        mol = Chem.MolFromXYZBlock(xyz_block)
        if mol is not None:
            molecules.append(mol)
    return molecules

def check_validity(molecules):
    """Checks the chemical validity of RDKit molecules."""
    valid_molecules = []
    for mol in molecules:
        # Basic validity check: can it be sanitized without errors?
        # Also check for a reasonable number of atoms and bonds
        try:
            Chem.SanitizeMol(mol)
            if mol.GetNumAtoms() > 0 and mol.GetNumBonds() > 0:
                valid_molecules.append(mol)
        except Exception:
            pass # Molecule is invalid
    return valid_molecules

def predict_energies(valid_molecules, surrogate_model_path):
    """Predicts energies for valid molecules using the frozen surrogate model."""
    # Load the frozen surrogate model
    surrogate_model = torch.jit.load(surrogate_model_path)
    surrogate_model.eval() # Set to evaluation mode

    energies = []
    for mol in valid_molecules:
        # Extract atomic numbers and positions
        atomic_numbers = []
        positions = []
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            atomic_numbers.append(atom.GetAtomicNum())
            pos = conf.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
        
        x = torch.tensor(atomic_numbers, dtype=torch.long).unsqueeze(1) # Node features (atomic numbers)
        pos = torch.tensor(positions, dtype=torch.float) # Node positions

        # Create edge_index using radius_graph
        # Assuming a reasonable radius for molecular graphs, e.g., 1.5 Angstroms
        edge_index = radius_graph(pos, r=1.5, loop=False)

        # Create torch_geometric.data.Data object
        data = Data(x=x, pos=pos, edge_index=edge_index)
        
        # The surrogate model expects a batch of Data objects, so we create a batch of one
        # If the model expects a single Data object, this might need adjustment.
        # For now, assuming it can handle a single Data object directly or via a DataLoader.
        # If the model expects a batch, you might need to collect multiple Data objects
        # and then use torch_geometric.data.Batch.from_data_list([data])
        
        with torch.no_grad():
            # The surrogate model expects a batch of graphs, so we need to ensure the input is batched.
            # For a single molecule, we can create a batch of size 1.
            predicted_energy = surrogate_model(data).item()
            energies.append(predicted_energy)
    return np.array(energies)

def main():
    qcmd_ecs_filepath = 'results/generated_qcmd_ecs.xyz'
    euclidean_filepath = 'results/generated_euclidean.xyz'
    surrogate_model_path = 'models/surrogate_frozen.pt'
    report_filepath = 'results/final_report.md'

    # --- QCMD-ECS Analysis ---
    print(f"Analyzing QCMD-ECS molecules from {qcmd_ecs_filepath}...")
    qcmd_ecs_mols = load_xyz_molecules(qcmd_ecs_filepath)
    qcmd_ecs_valid_mols = check_validity(qcmd_ecs_mols)
    qcmd_ecs_validity_percent = (len(qcmd_ecs_valid_mols) / len(qcmd_ecs_mols) * 100) if qcmd_ecs_mols else 0

    qcmd_ecs_mean_energy = 0
    qcmd_ecs_std_dev_energy = 0
    if qcmd_ecs_valid_mols:
        qcmd_ecs_energies = predict_energies(qcmd_ecs_valid_mols, surrogate_model_path)
        qcmd_ecs_mean_energy = np.mean(qcmd_ecs_energies)
        qcmd_ecs_std_dev_energy = np.std(qcmd_ecs_energies)

    # --- Euclidean Baseline Analysis ---
    print(f"Analyzing Euclidean Baseline molecules from {euclidean_filepath}...")
    euclidean_mols = load_xyz_molecules(euclidean_filepath)
    euclidean_valid_mols = check_validity(euclidean_mols)
    euclidean_validity_percent = (len(euclidean_valid_mols) / len(euclidean_mols) * 100) if euclidean_mols else 0

    euclidean_mean_energy = 0
    euclidean_std_dev_energy = 0
    if euclidean_valid_mols:
        euclidean_energies = predict_energies(euclidean_valid_mols, surrogate_model_path)
        euclidean_mean_energy = np.mean(euclidean_energies)
        euclidean_std_dev_energy = np.std(euclidean_energies)

    # --- Generate Report ---
    print(f"Generating final report to {report_filepath}...")
    with open(report_filepath, 'w') as f:
        f.write("| Metric                        | QCMD-ECS      | Euclidean Baseline |\n")
        f.write("|-------------------------------|---------------|--------------------|\n")
        f.write(f"| Validity (%)                  | {qcmd_ecs_validity_percent:.2f}      | {euclidean_validity_percent:.2f}         |\n")
        f.write(f"| Mean Energy (Valid Molecules) | {qcmd_ecs_mean_energy:.2f}      | {euclidean_mean_energy:.2f}         |\n")
        f.write(f"| Std Dev Energy (Valid Molecules)| {qcmd_ecs_std_dev_energy:.2f}      | {euclidean_std_dev_energy:.2f}         |\n")
    
    print("Analysis complete and report generated.")

if __name__ == '__main__':
    main()