import torch
from torch_geometric.datasets import QM9
import random
import os

# Ensure reproducibility
random.seed(42)
torch.manual_seed(42)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download the raw QM9 dataset
dataset = QM9(root='./data/QM9')

# Create a random subsample of 2,000 molecules
num_molecules = len(dataset)
indices = random.sample(range(num_molecules), 10)
subsample = [dataset[i] for i in indices]

# Extract atomic numbers (z) and 3D positions (pos)
extracted_data = []
for molecule in subsample:
    atomic_numbers = molecule.z
    positions = molecule.pos
    extracted_data.append((atomic_numbers, positions))

# Save the extracted data
torch.save(extracted_data, './data/qm9_micro_raw.pt')

print("Dataset preparation complete. Data saved to ./data/qm9_micro_raw.pt")