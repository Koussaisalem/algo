import os
from typing import List, Dict, Any

import torch
from torch_geometric.datasets import QM9

from nequip.data import AtomicDataDict

# Configuration
DATA_ROOT = "data/qm9"
OUTPUT_PATH = "data/qm9_micro_5k.pt"
NUM_SAMPLES = 5000
SEED = 42

# According to the PyG QM9 docs, index 8 corresponds to the U0 atomization energy (eV)
U0_ENERGY_TARGET_INDEX = 8


def convert_sample(data) -> Dict[str, Any]:
    """Convert a PyG QM9 sample into an AtomicDataDict-compatible payload."""

    sample: Dict[str, Any] = {
        AtomicDataDict.POSITIONS_KEY: data.pos.to(torch.float32),
        AtomicDataDict.ATOM_TYPE_KEY: data.z.to(torch.long),
        AtomicDataDict.TOTAL_ENERGY_KEY: data.y[0, U0_ENERGY_TARGET_INDEX].view(1).to(torch.float32),
    }

    if getattr(data, "force", None) is not None:
        sample[AtomicDataDict.FORCE_KEY] = data.force.to(torch.float32)

    return sample


def main():
    print(f"--- Preparing QM9-Micro Dataset ({NUM_SAMPLES} molecules) ---")

    print("Loading full QM9 dataset from torch_geometric (will download if necessary)...")
    full_dataset = QM9(root=DATA_ROOT)
    dataset_size = len(full_dataset)
    print(f"Loaded QM9 with {dataset_size} molecules.")

    if NUM_SAMPLES > dataset_size:
        raise ValueError(f"Requested {NUM_SAMPLES} samples but dataset only has {dataset_size} entries.")

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(dataset_size, generator=generator)[:NUM_SAMPLES]

    print(f"Subsampling {NUM_SAMPLES} molecules...")
    processed_samples: List[Dict[str, Any]] = []
    for idx in indices.tolist():
        data = full_dataset[int(idx)]
        processed_samples.append(convert_sample(data))

    os.makedirs("data", exist_ok=True)
    torch.save(processed_samples, OUTPUT_PATH)

    print(f"--- Success! QM9-Micro dataset saved to {OUTPUT_PATH} ---")


if __name__ == "__main__":
    main()
