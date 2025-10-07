#!/usr/bin/env python3
"""
Train a NequIP-based score model for manifold-constrained diffusion.

The score model learns to predict the direction toward the data manifold
from noisy samples. During training, we:
1. Take clean manifold frames from the enriched dataset
2. Add controlled Gaussian noise and project/retract to manifold
3. Train the model to predict (clean - noisy) as the score
4. Supervise with MSE loss in the tangent space

This script mirrors 03_train_surrogate.py but targets score prediction
instead of energy regression.
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

from qcmd_hybrid_framework.models.score_model import ScoreModel  # noqa: E402
from qcmd_hybrid_framework.qcmd_ecs.core.manifold import (  # noqa: E402
    project_to_tangent_space,
    retract_to_manifold,
)
from qcmd_hybrid_framework.qcmd_ecs.core.types import DTYPE as CORE_DTYPE  # noqa: E402
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
    noise_levels: List[float]  # Multiple noise scales for robust training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the NequIP score model for manifold diffusion."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/qm9_micro_5k_enriched.pt"),
        help="Path to the enriched dataset produced by 02_enrich_dataset.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/score_model"),
        help="Directory where the trained score model and metrics will be saved.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size."
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate for Adam."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="L2 penalty."
    )
    parser.add_argument(
        "--train-fraction", type=float, default=0.8, help="Fraction of data for training."
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.1, help="Fraction of data for validation."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splits and initialization."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device (e.g. cuda, cuda:1, cpu).",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.5],
        help="Noise scales to use during training for robustness.",
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
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, Sequence):
        raise TypeError("Expected the enriched dataset to be a sequence of dict samples")
    return data


class ScoreDataset(torch.utils.data.Dataset):
    """Dataset that generates noisy samples and their score targets on-the-fly."""
    
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
    
    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor]:
        sample = self.samples[idx]
        
        # Extract clean manifold frame
        frame_info = sample["manifold_frame"]
        clean_frame = frame_info["frame"].to(dtype=DTYPE)  # (n_atoms, k)
        
        # Get atomic information for NequIP
        positions = sample[AtomicDataDict.POSITIONS_KEY].to(dtype=DTYPE)
        numbers = sample[AtomicDataDict.ATOM_TYPE_KEY].to(dtype=torch.long)
        
        # Randomly select noise level
        noise_idx = torch.randint(len(self.noise_levels), (1,), generator=self.rng).item()
        noise_scale = self.noise_levels[noise_idx]
        
        # Generate noisy frame: add Gaussian noise and retract
        noise = torch.randn_like(clean_frame, dtype=DTYPE)
        noisy_ambient = clean_frame + noise_scale * noise
        noisy_frame = retract_to_manifold(noisy_ambient)
        
        # Score target: direction from noisy to clean, projected to tangent space
        raw_score = clean_frame - noisy_frame
        score_target = project_to_tangent_space(noisy_frame, raw_score)
        
        # Pack into PyG Data object
        # We'll store the noisy frame as a flattened vector in the data object
        # The model will reconstruct it as (n_atoms, k) internally
        # IMPORTANT: positions must require grad for NequIP force computation
        positions_grad = positions.clone().requires_grad_(True)
        data = Data(
            pos=positions_grad,
            z=numbers,
            noisy_frame=noisy_frame,  # (n_atoms, k)
            score_target=score_target,  # (n_atoms, k)
            noise_scale=torch.tensor([noise_scale], dtype=DTYPE),
        )
        
        return data


def collate_score_batch(batch: List[Data]) -> Data:
    """Custom collate to handle manifold frame data."""
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


def split_indices(
    n_items: int, train_frac: float, val_frac: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    if not 0 < train_frac < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= val_frac < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if train_frac + val_frac >= 1:
        raise ValueError("train_fraction + val_fraction must be < 1")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_items, generator=generator).tolist()

    train_end = math.floor(train_frac * n_items)
    val_end = train_end + math.floor(val_frac * n_items)

    train_idx = permutation[:train_end]
    val_idx = permutation[train_end:val_end]
    test_idx = permutation[val_end:]

    if not train_idx:
        raise ValueError("Training split is empty; adjust train_fraction")
    if not val_idx:
        raise ValueError("Validation split is empty; adjust val_fraction")
    if not test_idx:
        raise ValueError("Test split is empty; reduce train/val fractions")

    return train_idx, val_idx, test_idx


def subset_dataset(dataset: ScoreDataset, indices: Iterable[int]) -> ScoreDataset:
    subset_samples = [dataset.samples[i] for i in indices]
    # Create new dataset with same noise levels but different seed offset
    return ScoreDataset(subset_samples, dataset.noise_levels, dataset.rng.initial_seed() + len(indices))


def build_dataloaders(
    dataset: ScoreDataset,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = subset_dataset(dataset, train_idx)
    val_ds = subset_dataset(dataset, val_idx)
    test_ds = subset_dataset(dataset, test_idx)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_score_batch),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_score_batch),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_score_batch),
    )


def score_loss(pred_score: torch.Tensor, target_score: torch.Tensor) -> torch.Tensor:
    """MSE loss in the tangent space."""
    return nn.functional.mse_loss(pred_score, target_score)


class ScoreModelWrapper(nn.Module):
    """
    Wrapper that adapts the ScoreModel to work with our training setup.
    
    The ScoreModel returns forces (n_atoms, 3) but we need scores on manifold frames (n_atoms, k).
    For now, we'll use the force predictions as-is since they represent per-atom vectors.
    """
    
    def __init__(self):
        super().__init__()
        self.score_model = ScoreModel()
    
    def forward(self, batch: Data) -> torch.Tensor:
        """
        Args:
            batch: PyG batch with pos, z, and noisy_frame
        
        Returns:
            Predicted score on the manifold frame (n_atoms, k)
        """
        # Get force predictions (these represent per-atom vector directions)
        forces = self.score_model(batch)  # (total_atoms_in_batch, 3)
        
        # For the score model, we interpret forces as tangent space directions
        # In practice, we'd need to adapt this to work with arbitrary k
        # For now, assume k=3 (the spatial dimensions)
        return forces


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        
        # Extract ground truth scores
        target_scores = batch.score_target.to(dtype=DTYPE)  # (total_atoms, k)
        
        # Get predictions
        pred_scores = model(batch).to(dtype=DTYPE)
        
        # Compute loss
        loss = score_loss(pred_scores, target_scores)
        mse = torch.mean((pred_scores - target_scores) ** 2)
        
        # Accumulate
        n_atoms = target_scores.shape[0]
        total_loss += loss.item() * n_atoms
        total_mse += mse.item() * n_atoms
        total_samples += n_atoms

    return {
        "loss": total_loss / total_samples,
        "mse": total_mse / total_samples,
    }


def train(
    model: nn.Module,
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
    print(f"Training Score Model")
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
            
            # Extract targets
            target_scores = batch.score_target.to(dtype=DTYPE)
            
            # Forward pass
            pred_scores = model(batch).to(dtype=DTYPE)
            
            # Compute loss
            loss = score_loss(pred_scores, target_scores)
            
            # Backward
            loss.backward()
            optimizer.step()

            # Accumulate
            n_atoms = target_scores.shape[0]
            epoch_loss += loss.item() * n_atoms
            total_atoms += n_atoms

        train_loss = epoch_loss / total_atoms
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mse"].append(val_metrics["mse"])

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val MSE: {val_metrics['mse']:.6f}"
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"  → New best validation loss: {best_val_loss:.6f}")

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state")

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
    print(f"Loaded {len(dataset_samples)} enriched samples")

    # Create score dataset
    score_dataset = ScoreDataset(dataset_samples, config.noise_levels, config.seed)

    # Split indices
    train_idx, val_idx, test_idx = split_indices(
        len(score_dataset), config.train_fraction, config.val_fraction, config.seed
    )
    
    print(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    # Build loaders
    loaders = build_dataloaders(
        score_dataset, train_idx, val_idx, test_idx, config.batch_size
    )

    # Initialize model
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    model = ScoreModelWrapper().to(device)
    model = model.to(dtype=DTYPE)

    # Train
    history = train(model, loaders, config, device)

    # Test evaluation
    test_metrics = evaluate(model, loaders[2], device)
    print(f"\n{'='*60}")
    print(f"Test Loss: {test_metrics['loss']:.6f} | Test MSE: {test_metrics['mse']:.6f}")
    print(f"{'='*60}\n")

    # Save artifacts
    args.output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = args.output_dir / "score_model_state_dict.pt"
    torch.save(model.state_dict(), weights_path)

    metrics_path = args.output_dir / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
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
                    "output_dir": str(args.output_dir),
                },
            },
            handle,
            indent=2,
        )

    print(f"✓ Saved best score model weights to {weights_path}")
    print(f"✓ Training metrics logged to {metrics_path}")
    print("\nScore model training complete!")


if __name__ == "__main__":
    main()
