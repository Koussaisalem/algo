#!/usr/bin/env python3
"""
Train TMD score model for manifold-constrained diffusion generation.

Position-based denoising approach:
1. Reconstruct positions from noisy manifold frames
2. Train model to predict direction from noisy → clean positions  
3. Leverages NEquIP's force prediction architecture
4. At inference, project position updates to tangent space

This enables MAECS (Manifold-Aware Energy-Constrained Sampling) for
2D transition metal dichalcogenides with d-orbital support.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qcmd_hybrid_framework.models.vector_output_model import VectorOutputModel  # noqa: E402
from qcmd_hybrid_framework.qcmd_ecs.core.manifold import retract_to_manifold  # noqa: E402
from nequip.data import AtomicDataDict  # noqa: E402

DTYPE = torch.float64


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    lr: float
    weight_decay: float
    epochs: int
    train_fraction: float
    val_fraction: float
    seed: int
    noise_levels: List[float]
    l_max: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TMD score model for manifold diffusion."
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
        default=Path("models/tmd_score"),
        help="Directory for trained model.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="L2 penalty.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Train fraction.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Val fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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
        help="Maximum atoms per structure.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.15, 0.2],
        help="Noise scales for multi-noise training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_filter_dataset(path: Path, max_atoms: int) -> List[Dict]:
    """Load dataset and filter by size."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    
    if not isinstance(data, list):
        raise TypeError("Expected dataset to be a list")
    
    # Filter by atom count
    filtered = []
    for sample in data:
        # Try different possible keys
        if "positions" in sample:
            n_atoms = sample["positions"].shape[0]
        elif AtomicDataDict.POSITIONS_KEY in sample:
            n_atoms = sample[AtomicDataDict.POSITIONS_KEY].shape[0]
        else:
            continue
        
        if n_atoms <= max_atoms:
            filtered.append(sample)
    
    print(f"Loaded {len(data)} samples, filtered to {len(filtered)} (≤{max_atoms} atoms)")
    return filtered


def frame_to_positions(
    frame: torch.Tensor,
    centroid: torch.Tensor,
    mass_weights: torch.Tensor,
    reference_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct atomic positions from manifold frame.
    
    Frame is (m, k) on Stiefel manifold.
    Positions are weighted reconstruction: pos = centroid + frame @ weights
    """
    m, k = frame.shape
    n_atoms = reference_positions.shape[0]
    
    # For simplicity, use direct projection
    # In practice, frame encodes low-rank structure
    centered = reference_positions - centroid.unsqueeze(0)
    flat_centered = centered.reshape(-1)
    
    # Project to frame basis
    if flat_centered.shape[0] >= k:
        coords = frame.T @ flat_centered[:m]
        reconstructed_flat = frame @ coords
        
        # Pad if needed
        if reconstructed_flat.shape[0] < flat_centered.shape[0]:
            padding = torch.zeros(
                flat_centered.shape[0] - reconstructed_flat.shape[0],
                dtype=DTYPE,
                device=reconstructed_flat.device,
            )
            reconstructed_flat = torch.cat([reconstructed_flat, padding])
        else:
            reconstructed_flat = reconstructed_flat[:flat_centered.shape[0]]
        
        positions = reconstructed_flat.reshape(n_atoms, 3) + centroid.unsqueeze(0)
    else:
        # Fallback: use reference positions with small perturbation
        positions = reference_positions
    
    return positions


class PositionScoreDataset(torch.utils.data.Dataset):
    """
    Dataset for position-based score matching.
    
    Samples noisy manifold frames, reconstructs positions,
    trains model to predict clean position direction.
    """
    
    def __init__(
        self,
        samples: List[Dict],
        noise_levels: List[float],
        seed: int,
    ) -> None:
        self.samples = samples
        self.noise_levels = noise_levels
        self.rng = torch.Generator().manual_seed(seed)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Data:
        sample = self.samples[idx]
        
        # Extract data
        clean_frame = sample["manifold_frame"].to(dtype=DTYPE)
        clean_positions = sample.get("positions", sample.get(AtomicDataDict.POSITIONS_KEY)).to(dtype=DTYPE)
        atomic_numbers = sample.get("atomic_numbers", sample.get(AtomicDataDict.ATOM_TYPE_KEY)).to(dtype=torch.long)
        
        # Compute centroid and mass weights
        masses = sample.get("masses", torch.ones(clean_positions.shape[0], dtype=DTYPE))
        centroid = (clean_positions * masses.unsqueeze(1)).sum(0) / masses.sum()
        mass_weights = masses / masses.sum()
        
        # Random noise level
        noise_idx = torch.randint(len(self.noise_levels), (1,), generator=self.rng).item()
        noise_scale = self.noise_levels[noise_idx]
        
        # Create noisy frame
        noise = torch.randn_like(clean_frame, dtype=DTYPE)
        noisy_ambient = clean_frame + noise_scale * noise
        noisy_frame = retract_to_manifold(noisy_ambient)
        
        # Reconstruct noisy positions
        noisy_positions = frame_to_positions(
            noisy_frame, centroid, mass_weights, clean_positions
        )
        
        # Score target: direction from noisy to clean
        score_target = clean_positions - noisy_positions
        
        # Pack into PyG Data
        data = Data(
            pos=noisy_positions,
            z=atomic_numbers,
            score_target=score_target,
            noise_scale=torch.tensor([noise_scale], dtype=DTYPE),
        )
        
        return data


def collate_fn(batch: List[Data]) -> Data:
    return Batch.from_data_list(batch)


def split_indices(
    n_items: int, train_frac: float, val_frac: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("Invalid split fractions")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_items, generator=generator).tolist()

    train_end = math.floor(train_frac * n_items)
    val_end = train_end + math.floor(val_frac * n_items)

    return permutation[:train_end], permutation[train_end:val_end], permutation[val_end:]


def subset_dataset(
    dataset: PositionScoreDataset, indices: Iterable[int]
) -> PositionScoreDataset:
    subset_samples = [dataset.samples[i] for i in indices]
    return PositionScoreDataset(
        subset_samples, dataset.noise_levels, dataset.rng.initial_seed() + len(indices)
    )


def build_dataloaders(
    dataset: PositionScoreDataset,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(
            subset_dataset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        ),
        DataLoader(
            subset_dataset(dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        ),
        DataLoader(
            subset_dataset(dataset, test_idx),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        ),
    )


def score_loss(pred_score: torch.Tensor, target_score: torch.Tensor) -> torch.Tensor:
    """MSE loss on score vectors."""
    return nn.functional.mse_loss(pred_score, target_score)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        pred_score = model(batch).to(dtype=DTYPE)
        target_score = batch.score_target.to(dtype=DTYPE)
        
        loss = score_loss(pred_score, target_score)
        
        batch_size = batch.num_graphs
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return {"mse": total_loss / total_samples}


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred_score = model(batch).to(dtype=DTYPE)
        target_score = batch.score_target.to(dtype=DTYPE)
        
        loss = score_loss(pred_score, target_score)
        loss.backward()
        optimizer.step()

        batch_size = batch.num_graphs
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def train(
    model: nn.Module,
    loaders: Tuple[DataLoader, DataLoader, DataLoader],
    config: TrainConfig,
    device: torch.device,
) -> Dict[str, List[float]]:
    train_loader, val_loader, _ = loaders
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history: Dict[str, List[float]] = {"train_mse": [], "val_mse": []}
    best_val_mse = float("inf")
    best_state = None

    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")

    for epoch in range(1, config.epochs + 1):
        train_mse = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_metrics["mse"])

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Train MSE: {train_mse:.6f} | "
                f"Val MSE: {val_metrics['mse']:.6f}"
            )

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if epoch % 10 == 0:
                print(f"  → New best val MSE: {best_val_mse:.6f}")

    if best_state is None:
        raise RuntimeError("Training failed")

    model.load_state_dict(best_state)
    return history


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("="*70)
    print("🧲 Operation Magnet: TMD Score Model Training")
    print("="*70)
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"l_max: {args.l_max} (d-orbital support)")
    print(f"Noise levels: {args.noise_levels}")
    print(f"Device: {args.device}")
    print("="*70 + "\n")

    # Load dataset
    print("Loading dataset...")
    samples = load_and_filter_dataset(args.dataset_path, args.max_atoms)

    # Create dataset
    config = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        noise_levels=args.noise_levels,
        l_max=args.l_max,
    )

    dataset = PositionScoreDataset(samples, config.noise_levels, config.seed)
    train_idx, val_idx, test_idx = split_indices(
        len(dataset), config.train_fraction, config.val_fraction, config.seed
    )

    print(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test\n")

    loaders = build_dataloaders(dataset, train_idx, val_idx, test_idx, config.batch_size)

    # Initialize model
    print("Initializing TMD score model...")
    device = torch.device(args.device)
    
    # Get unique atomic numbers from dataset
    all_z = set()
    for sample in samples:
        z_data = sample.get("atomic_numbers", sample.get(AtomicDataDict.ATOM_TYPE_KEY))
        all_z.update(z_data.tolist())
    atomic_numbers = sorted(all_z)
    
    model_config = {
        "l_max": config.l_max,
        "num_layers": 4,
        "num_features": 64,
        "atomic_numbers": tuple(atomic_numbers),
        "type_names": tuple(f"Z{z}" for z in atomic_numbers),
    }
    
    model = VectorOutputModel(model_config=model_config).to(device).to(dtype=DTYPE)
    
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"✓ l_max={config.l_max} (d-orbital support)\n")

    # Train
    history = train(model, loaders, config, device)

    # Test
    test_metrics = evaluate(model, loaders[2], device)
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)
    print(f"Best val MSE: {min(history['val_mse']):.6f}")
    print(f"Test MSE: {test_metrics['mse']:.6f}")
    print("="*70 + "\n")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = args.output_dir / "score_model_state_dict.pt"
    torch.save(model.state_dict(), weights_path)

    metrics_path = args.output_dir / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "history": history,
                "test": test_metrics,
                "config": {
                    "batch_size": config.batch_size,
                    "lr": config.lr,
                    "weight_decay": config.weight_decay,
                    "epochs": config.epochs,
                    "noise_levels": config.noise_levels,
                    "l_max": config.l_max,
                    "seed": config.seed,
                },
            },
            f,
            indent=2,
        )

    print(f"✓ Saved model to {weights_path}")
    print(f"✓ Saved metrics to {metrics_path}")
    print("\n🎉 TMD score model training complete!")
    print("\nNext step: Generate TMD structures")
    print("  python scripts/tmd/04_generate_tmd_structures.py")


if __name__ == "__main__":
    main()
