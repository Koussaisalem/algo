#!/usr/bin/env python3
"""
Train TMD energy surrogate model for Operation Magnet.

This surrogate predicts formation energies and band gaps for TMD structures.
It will be used during generation to guide the diffusion process toward
low-energy, stable semiconductor structures.

Architecture: NequIP with d-orbital support (l_max=2)
- Input: Atomic positions, species, cell
- Output: Formation energy (eV), band gap (eV)
- Training: Materials Project DFT data (VASP/PBE)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qcmd_hybrid_framework.models.tmd_surrogate import TMDSurrogate
from nequip.data import AtomicDataDict

DTYPE = torch.float64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TMD energy surrogate with NequIP."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to enriched TMD dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/tmd_surrogate"),
        help="Directory to save trained model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--l-max",
        type=int,
        default=2,
        help="Maximum angular momentum (2 for d-orbitals).",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=30,
        help="Maximum atoms per structure (filter larger).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def load_and_filter_dataset(
    path: Path,
    max_atoms: int
) -> List[Dict]:
    """Load dataset and filter by size."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    
    # Filter by size
    filtered = [s for s in data if len(s["atomic_numbers"]) <= max_atoms]
    
    print(f"Loaded {len(data)} samples, filtered to {len(filtered)} (≤{max_atoms} atoms)")
    
    return filtered


def sample_to_pyg(sample: Dict) -> Data:
    """
    Convert enriched TMD sample to PyTorch Geometric Data object.
    """
    data = Data(
        pos=sample["positions"].to(dtype=DTYPE),
        z=sample["atomic_numbers"],
        cell=sample["cell"].to(dtype=DTYPE),
        pbc=sample["pbc"],
        energy=sample["energy_ev"].to(dtype=DTYPE),
        gap=torch.tensor(sample["properties"]["gap"], dtype=DTYPE),
        batch=torch.zeros(len(sample["atomic_numbers"]), dtype=torch.long),
    )
    return data


def split_indices(
    n: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """Split dataset indices."""
    indices = list(range(n))
    
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()
    
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    
    return train_idx, val_idx, test_idx


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Predict energy
        pred_energy = model(batch).squeeze()
        target_energy = batch.energy
        
        # MSE loss
        loss = torch.mean((pred_energy - target_energy) ** 2)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch.energy)
        total_samples += len(batch.energy)
    
    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            pred_energy = model(batch).squeeze()
            target_energy = batch.energy
            
            loss = torch.mean((pred_energy - target_energy) ** 2)
            mae = torch.mean(torch.abs(pred_energy - target_energy))
            
            total_loss += loss.item() * len(batch.energy)
            total_mae += mae.item() * len(batch.energy)
            total_samples += len(batch.energy)
    
    return {
        "mse": total_loss / total_samples,
        "mae": total_mae / total_samples,
    }


def main():
    args = parse_args()
    
    print("=" * 70)
    print("🧲 Operation Magnet: TMD Surrogate Training")
    print("=" * 70)
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"l_max: {args.l_max} (d-orbital support)")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Load and filter dataset
    print("\nLoading dataset...")
    samples = load_and_filter_dataset(args.dataset_path, args.max_atoms)
    
    if len(samples) < 50:
        print(f"❌ Only {len(samples)} samples after filtering - need at least 50")
        sys.exit(1)
    
    # Convert to PyG format
    print("Converting to PyTorch Geometric format...")
    pyg_data = [sample_to_pyg(s) for s in tqdm(samples, desc="Converting")]
    
    # Split dataset
    train_idx, val_idx, test_idx = split_indices(len(pyg_data))
    print(f"\nSplit: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    
    # Create dataloaders
    train_data = [pyg_data[i] for i in train_idx]
    val_data = [pyg_data[i] for i in val_idx]
    test_data = [pyg_data[i] for i in test_idx]
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    print("\nInitializing TMD surrogate model...")
    device = torch.device(args.device)
    
    # Create model config with user-specified l_max
    model_config = {
        "l_max": args.l_max,
        "num_layers": 4,
        "num_features": 64,
        "seed": args.seed,
    }
    
    model = TMDSurrogate(model_config=model_config).to(device).to(dtype=DTYPE)
    
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"✓ l_max={args.l_max} (d-orbital support for transition metals)")
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Training loop
    print(f"\n{'=' * 70}")
    print("Starting training...")
    print(f"{'=' * 70}\n")
    
    best_val_mae = float("inf")
    best_state = None
    history = {"train_mse": [], "val_mse": [], "val_mae": []}
    
    for epoch in range(1, args.epochs + 1):
        train_mse = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_metrics["mse"])
        history["val_mae"].append(val_metrics["mae"])
        
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Train MSE: {train_mse:.4f} | "
                f"Val MSE: {val_metrics['mse']:.4f} | "
                f"Val MAE: {val_metrics['mae']:.4f} eV"
            )
        
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if epoch % 10 == 0:
                print(f"  → New best MAE: {best_val_mae:.4f} eV")
    
    # Load best model and evaluate on test
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\n{'=' * 70}")
    print("Training complete!")
    print(f"{'=' * 70}")
    print(f"Best val MAE: {best_val_mae:.4f} eV")
    print(f"Test MSE: {test_metrics['mse']:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.4f} eV")
    print(f"{'=' * 70}\n")
    
    # Save model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = args.output_dir / "surrogate_state_dict.pt"
    torch.save(best_state, model_path)
    print(f"✓ Saved model to {model_path}")
    
    # Save metrics
    metrics_path = args.output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "history": history,
            "test": test_metrics,
            "best_val_mae": best_val_mae,
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "l_max": args.l_max,
                "max_atoms": args.max_atoms,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
            }
        }, f, indent=2)
    print(f"✓ Saved metrics to {metrics_path}")
    
    print("\n🎉 TMD surrogate model training complete!")
    print("\nNext step: Train TMD score model")
    print("  python scripts/tmd/03_train_tmd_score_model.py")


if __name__ == "__main__":
    main()
