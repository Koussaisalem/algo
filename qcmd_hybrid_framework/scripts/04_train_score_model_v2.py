#!/usr/bin/env python3
"""
Simplified score model training using position-based denoising.

Instead of trying to train directly on manifold frames (which has architecture
challenges with NequIP), we:

1. Reconstruct positions from noisy manifold frames
2. Train the model to predict the direction from noisy to clean positions
3. This leverages NequIP's native force prediction architecture
4. At inference, we project position updates to the tangent space

This is mathematically sound because:
- Manifold frames encode positions via weighted reconstruction
- Position-space scores can be projected to tangent space
- The model learns physically meaningful denoising directions
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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qcmd_hybrid_framework.models.vector_output_model import VectorOutputModel  # noqa: E402
from qcmd_hybrid_framework.qcmd_ecs.core.manifold import (  # noqa: E402
    retract_to_manifold,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train position-based score model for manifold diffusion."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/qm9_micro_5k_enriched.pt"),
        help="Path to the enriched dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/score_model"),
        help="Directory for trained model and metrics.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="L2 penalty.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Training fraction.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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
        default=[0.1, 0.2, 0.3, 0.5],
        help="Noise scales for training robustness.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: Path) -> Sequence[Dict[str, torch.Tensor]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, Sequence):
        raise TypeError("Expected dataset to be a sequence")
    return data


def frame_to_positions(
    frame: torch.Tensor,
    centroid: torch.Tensor,
    mass_weights: torch.Tensor,
    positions_ref: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct positions from manifold frame."""
    sqrt_weights = torch.sqrt(mass_weights).unsqueeze(-1).clamp_min(1e-12)
    
    # Get components by projecting reference positions onto frame
    centered_ref = positions_ref - centroid
    weighted_ref = centered_ref * sqrt_weights
    components = frame.transpose(0, 1) @ weighted_ref
    
    # Reconstruct
    weighted = frame @ components
    centered = weighted / sqrt_weights
    return centered + centroid


class PositionScoreDataset(torch.utils.data.Dataset):
    """
    Dataset that generates noisy positions and their denoising targets.
    
    Strategy:
    1. Add noise to manifold frame and retract
    2. Reconstruct positions from noisy frame
    3. Target = clean_positions - noisy_positions (score direction)
    """
    
    def __init__(
        self,
        samples: Sequence[Dict[str, torch.Tensor]],
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
        frame_info = sample["manifold_frame"]
        clean_frame = frame_info["frame"].to(dtype=DTYPE)
        centroid = frame_info["centroid"].to(dtype=DTYPE)
        mass_weights = frame_info["mass_weights"].to(dtype=DTYPE)
        
        clean_positions = sample[AtomicDataDict.POSITIONS_KEY].to(dtype=DTYPE)
        atomic_numbers = sample[AtomicDataDict.ATOM_TYPE_KEY].to(dtype=torch.long)
        
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
    from torch_geometric.data import Batch
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

    train_idx = permutation[:train_end]
    val_idx = permutation[train_end:val_end]
    test_idx = permutation[val_end:]

    if not train_idx or not val_idx or not test_idx:
        raise ValueError("Empty split - adjust fractions")

    return train_idx, val_idx, test_idx


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


def score_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss on position-space scores."""
    return nn.functional.mse_loss(pred, target)


@torch.no_grad()
def evaluate(model: VectorOutputModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        target = batch.score_target.to(dtype=DTYPE)
        
        # Model predicts vectors (scores)
        pred = model(batch).to(dtype=DTYPE)
        
        # Compute loss
        loss = score_loss(pred, target)
        mse = torch.mean((pred - target) ** 2)
        
        n_atoms = target.shape[0]
        total_loss += loss.item() * n_atoms
        total_mse += mse.item() * n_atoms
        total_samples += n_atoms

    return {"loss": total_loss / total_samples, "mse": total_mse / total_samples}


def train(
    model: VectorOutputModel,
    loaders: Tuple[DataLoader, DataLoader, DataLoader],
    config: TrainConfig,
    device: torch.device,
) -> Dict[str, List[float]]:
    train_loader, val_loader, _ = loaders
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_mse": []}
    best_val_loss = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None

    print(f"\n{'='*60}")
    print(f"Training Position-Based Score Model")
    print(f"{'='*60}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")
    print(f"Epochs: {config.epochs}")
    print(f"Noise levels: {config.noise_levels}")
    print(f"{'='*60}\n")

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_atoms = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            target = batch.score_target.to(dtype=DTYPE)
            pred = model(batch).to(dtype=DTYPE)
            
            loss = score_loss(pred, target)
            loss.backward()
            optimizer.step()

            n_atoms = target.shape[0]
            epoch_loss += loss.item() * n_atoms
            total_atoms += n_atoms

        train_loss = epoch_loss / total_atoms
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mse"].append(val_metrics["mse"])

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_metrics['loss']:.6f} | "
                f"Val MSE: {val_metrics['mse']:.6f}"
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if epoch % 10 == 0:
                print(f"  → New best: {best_val_loss:.6f}")

    if best_state is None:
        raise RuntimeError("Training failed")

    model.load_state_dict(best_state)
    return history


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        noise_levels=args.noise_levels,
    )

    print(f"Loading dataset from {args.dataset_path}...")
    dataset_samples = load_dataset(args.dataset_path)
    print(f"✓ Loaded {len(dataset_samples)} samples")

    score_dataset = PositionScoreDataset(dataset_samples, config.noise_levels, config.seed)

    train_idx, val_idx, test_idx = split_indices(
        len(score_dataset), config.train_fraction, config.val_fraction, config.seed
    )
    print(f"✓ Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    loaders = build_dataloaders(score_dataset, train_idx, val_idx, test_idx, config.batch_size)

    device = torch.device(args.device)
    print(f"✓ Using device: {device}")
    
    model = VectorOutputModel().to(device).to(dtype=DTYPE)

    history = train(model, loaders, config, device)

    test_metrics = evaluate(model, loaders[2], device)
    print(f"\n{'='*60}")
    print(f"Final Test Loss: {test_metrics['loss']:.6f} | MSE: {test_metrics['mse']:.6f}")
    print(f"{'='*60}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = args.output_dir / "score_model_state_dict.pt"
    torch.save(model.state_dict(), weights_path)

    metrics_path = args.output_dir / "training_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(
            {
                "history": history,
                "test": test_metrics,
                "config": {
                    "batch_size": config.batch_size,
                    "lr": config.lr,
                    "weight_decay": config.weight_decay,
                    "epochs": config.epochs,
                    "train_fraction": config.train_fraction,
                    "val_fraction": config.val_fraction,
                    "seed": config.seed,
                    "noise_levels": config.noise_levels,
                    "dataset_path": str(args.dataset_path),
                },
            },
            f,
            indent=2,
        )

    print(f"✓ Saved weights to {weights_path}")
    print(f"✓ Saved metrics to {metrics_path}")
    print("\n🎉 Score model training complete!\n")


if __name__ == "__main__":
    main()
