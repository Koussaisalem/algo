import argparse
import torch
import os
from models.score_model import ScoreModel
from models.surrogate import SurrogateModel # Import SurrogateModel
from qcmd_ecs.core.dynamics import run_reverse_diffusion
from qcmd_ecs.core.types import DTYPE, StiefelManifold
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

# --- Helper functions and classes ---

# Placeholder schedules for run_reverse_diffusion
def gamma_schedule(t):
    return 0.1 # Constant for now

def eta_schedule(t):
    return 0.01 # Constant for now

def tau_schedule(t):
    return 0.001 # Constant for now

# Placeholder energy gradient model
# Wrapper for energy gradient model
class EnergyGradientModelWrapper:
    def __init__(self, surrogate_model: SurrogateModel, atomic_numbers: torch.Tensor, cutoff: float = 5.0):
        self.surrogate_model = surrogate_model
        self.atomic_numbers = atomic_numbers
        self.cutoff = cutoff

    def __call__(self, U_t: StiefelManifold) -> torch.Tensor:
        # U_t is (num_atoms * 3, 1)
        num_atoms = len(self.atomic_numbers)
        pos = U_t.view(num_atoms, 3)
        pos.requires_grad_(True) # Enable gradient computation for positions

        # Create edge_index using radius_graph
        edge_index = radius_graph(pos, r=self.cutoff, batch=None, loop=False)

        data = Data(z=self.atomic_numbers, pos=pos, edge_index=edge_index, batch=torch.zeros(num_atoms, dtype=torch.long))
        
        # Get energy from the surrogate model
        energy = self.surrogate_model(data)
        
        # Compute gradient of energy with respect to positions
        grad_pos = torch.autograd.grad(outputs=energy, inputs=pos, grad_outputs=torch.ones_like(energy), retain_graph=True)[0]
        
        return grad_pos.view(-1).unsqueeze(1).to(DTYPE) # Flatten and ensure DTYPE

# Wrapper for ScoreModel to adapt to run_reverse_diffusion signature
class ScoreModelWrapper:
    def __init__(self, score_model: ScoreModel, atomic_numbers: torch.Tensor, cutoff: float = 5.0):
        self.score_model = score_model
        self.atomic_numbers = atomic_numbers
        self.cutoff = cutoff

    def __call__(self, U_t: StiefelManifold, t: int) -> torch.Tensor:
        # U_t is (num_atoms * 3) or (num_atoms, 3)
        # Reshape U_t to (num_atoms, 3) if it's flattened
        num_atoms = len(self.atomic_numbers)
        pos = U_t.view(num_atoms, 3)

        # Create edge_index using radius_graph
        edge_index = radius_graph(pos, r=self.cutoff, batch=None, loop=False)

        data = Data(z=self.atomic_numbers, pos=pos, edge_index=edge_index, batch=torch.zeros(num_atoms, dtype=torch.long))
        
        # The score model returns scores for each atom's position
        scores = self.score_model(data)
        
        # Reshape scores back to the shape of U_t if needed by run_reverse_diffusion
        return scores.view(-1).to(DTYPE) # Flatten and ensure DTYPE

def save_xyz(filename: str, positions: torch.Tensor, atomic_numbers: torch.Tensor):
    """
    Saves molecular positions to an .xyz file.
    positions: (num_molecules, num_atoms, 3)
    atomic_numbers: (num_atoms,)
    """
    num_molecules, num_atoms, _ = positions.shape
    atom_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'} # Example mapping

    with open(filename, 'w') as f:
        for i in range(num_molecules):
            f.write(f"{num_atoms}\n")
            f.write(f"Molecule {i+1}\n")
            for j in range(num_atoms):
                atom_type = atom_map.get(atomic_numbers[j].item(), 'X')
                x, y, z = positions[i, j].tolist()
                f.write(f"{atom_type} {x:.6f} {y:.6f} {z:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate molecules using a trained score model.")
    parser.add_argument('--mode', type=str, required=True, choices=['qcmd_ecs', 'euclidean'],
                        help="Mode for molecule generation: 'qcmd_ecs' or 'euclidean'.")
    args = parser.parse_args()

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Define common parameters
    num_molecules_to_generate = 500
    num_atoms_per_molecule = 3 # Example: H2O
    atomic_numbers = torch.tensor([1, 8, 1], dtype=torch.long) # H, O, H
    diffusion_steps = 1000
    seed = 42

    model_path = ""
    if args.mode == 'qcmd_ecs':
        model_path = "models/score_model_qcmd_ecs.pt"
        print(f"Loading QCMD-ECS model from {model_path}...")
        
        # Initialize ScoreModel (adjust parameters as per your trained model)
        score_model = ScoreModel(hidden_channels=128, num_filters=128, num_interactions=3, cutoff=10.0)
        score_model.load_state_dict(torch.load(model_path))
        score_model.eval()

        # Load the surrogate model
        surrogate_model_path = "models/surrogate_model.pt" # Assuming a default path
        print(f"Loading Surrogate model from {surrogate_model_path}...")
        surrogate_model = SurrogateModel(hidden_channels=128, num_filters=128, num_interactions=3, cutoff=10.0) # Adjust params as needed
        surrogate_model.load_state_dict(torch.load(surrogate_model_path))
        surrogate_model.eval()
        
        # Wrap the score model for run_reverse_diffusion
        wrapped_score_model = ScoreModelWrapper(score_model, atomic_numbers)

        # Wrap the energy gradient model for run_reverse_diffusion
        wrapped_energy_gradient_model = EnergyGradientModelWrapper(surrogate_model, atomic_numbers)

        generated_molecules_qcmd_ecs = []
        for i in range(num_molecules_to_generate):
            print(f"Generating QCMD-ECS molecule {i+1}/{num_molecules_to_generate}...")
            # Initial noisy state on the manifold (random positions)
            # U_T should be (num_atoms * 3, 1) or (num_atoms, 3) and then flattened
            initial_pos = torch.randn(num_atoms_per_molecule, 3, dtype=DTYPE)
            # Project to manifold if necessary, for now, assume it's handled by run_reverse_diffusion
            # For simplicity, let's assume U_T is just the flattened positions for now
            U_T = initial_pos.view(-1).unsqueeze(1) # (num_atoms * 3, 1)
            
            # Run reverse diffusion
            final_U = run_reverse_diffusion(
                U_T=U_T,
                score_model=wrapped_score_model,
                energy_gradient_model=wrapped_energy_gradient_model,
                gamma_schedule=gamma_schedule,
                eta_schedule=eta_schedule,
                tau_schedule=tau_schedule,
                num_steps=diffusion_steps,
                seed=seed + i # Use different seed for each molecule
            )
            generated_molecules_qcmd_ecs.append(final_U.view(num_atoms_per_molecule, 3).cpu())
        
        # Save generated molecules
        save_xyz("results/generated_qcmd_ecs.xyz", torch.stack(generated_molecules_qcmd_ecs), atomic_numbers)
        print(f"Generated QCMD-ECS molecules saved to results/generated_qcmd_ecs.xyz")

    elif args.mode == 'euclidean':
        model_path = "models/score_model_euclidean.pt"
        print(f"Loading Euclidean model from {model_path}...")
        
        # Initialize ScoreModel (adjust parameters as per your trained model)
        score_model = ScoreModel(hidden_channels=128, num_filters=128, num_interactions=3, cutoff=10.0)
        score_model.load_state_dict(torch.load(model_path))
        score_model.eval()

        generated_molecules_euclidean = []
        for i in range(num_molecules_to_generate):
            print(f"Generating Euclidean molecule {i+1}/{num_molecules_to_generate}...")
            # Initial random noise for positions
            pos_t = torch.randn(num_atoms_per_molecule, 3, dtype=DTYPE)
            
            # Implement the reverse diffusion loop (Langevin dynamics)
            # Loop from t=T to 1
            for t in range(diffusion_steps, 0, -1):
                # Placeholder for time-dependent coefficients (e.g., from a noise schedule)
                # In a real scenario, these would come from the diffusion process definition
                alpha_t = 1.0 / diffusion_steps # Simple linear decay for demonstration
                sigma_t = 0.1 # Constant noise scale for demonstration

                # Create Data object for the score model
                edge_index = radius_graph(pos_t, r=10.0, batch=None, loop=False) # Use model's cutoff
                data = Data(z=atomic_numbers, pos=pos_t, edge_index=edge_index, batch=torch.zeros(num_atoms_per_molecule, dtype=torch.long))
                
                # Get score from the model
                score_t = score_model(data)
                
                # Langevin update step
                # x_t-1 = x_t + alpha_t * score_t + sqrt(2 * alpha_t) * noise
                noise = torch.randn_like(pos_t, dtype=DTYPE)
                pos_t = pos_t + alpha_t * score_t + torch.sqrt(2 * alpha_t) * noise
            
            generated_molecules_euclidean.append(pos_t.cpu())
        
        # Save generated molecules
        save_xyz("results/generated_euclidean.xyz", torch.stack(generated_molecules_euclidean), atomic_numbers)
        print(f"Generated Euclidean molecules saved to results/generated_euclidean.xyz")

if __name__ == "__main__":
    main()