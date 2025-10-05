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

from qcmd_hybrid_framework.models.surrogate import Surrogate  # noqa: E402
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
    target_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the NequIP surrogate on the enriched hybrid dataset."
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
        default=Path("models/surrogate"),
        help="Directory where the trained surrogate and metrics will be saved.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for Adam."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="L2 penalty."
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
        "--target",
        type=str,
        default="energy_ev",
        choices=["energy_ev", "energy_hartree"],
        help="Supervision target from the enriched dataset.",
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


class EnrichedDataset(torch.utils.data.Dataset):
    def __init__(self, samples: Sequence[Dict[str, torch.Tensor]], target_key: str) -> None:
        self.samples = samples
        self.target_key = target_key

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        sample = self.samples[idx]
        pos = sample[AtomicDataDict.POSITIONS_KEY].to(dtype=DTYPE)
        numbers = sample[AtomicDataDict.ATOM_TYPE_KEY].to(dtype=torch.long)
        target = sample[self.target_key].to(dtype=DTYPE)
        if target.ndim == 0:
            target = target.view(1)
        data = Data(pos=pos, z=numbers, y=target)
        return data


def split_indices(n_items: int, train_frac: float, val_frac: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
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


def subset_dataset(dataset: EnrichedDataset, indices: Iterable[int]) -> EnrichedDataset:
    subset_samples = [dataset.samples[i] for i in indices]
    return EnrichedDataset(subset_samples, dataset.target_key)


def build_dataloaders(
    dataset: EnrichedDataset,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = subset_dataset(dataset, train_idx)
    val_ds = subset_dataset(dataset, val_idx)
    test_ds = subset_dataset(dataset, test_idx)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred.view(-1), target.view(-1))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        target = batch.y.to(dtype=DTYPE)
        pred = model(batch).to(dtype=DTYPE)
        loss = mse_loss(pred, target)
        mae = torch.mean(torch.abs(pred.view(-1) - target.view(-1)))

        batch_size = target.shape[0]
        total_loss += loss.item() * batch_size
        total_mae += mae.item() * batch_size
        total_samples += batch_size

    return {
        "mse": total_loss / total_samples,
        "mae": total_mae / total_samples,
    }


def train(
    model: nn.Module,
    loaders: Tuple[DataLoader, DataLoader, DataLoader],
    config: TrainConfig,
    device: torch.device,
) -> Dict[str, List[float]]:
    train_loader, val_loader, _ = loaders
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history: Dict[str, List[float]] = {"train_mse": [], "val_mse": [], "val_mae": []}
    best_val_mse = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            target = batch.y.to(dtype=DTYPE)
            pred = model(batch).to(dtype=DTYPE)
            loss = mse_loss(pred, target)
            loss.backward()
            optimizer.step()

            batch_size = target.shape[0]
            epoch_loss += loss.item() * batch_size
            total_samples += batch_size

        train_mse = epoch_loss / total_samples
        val_metrics = evaluate(model, val_loader, device)

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_metrics["mse"])
        history["val_mae"].append(val_metrics["mae"])

        print(
            f"Epoch {epoch:03d} | Train MSE: {train_mse:.6f} | "
            f"Val MSE: {val_metrics['mse']:.6f} | Val MAE: {val_metrics['mae']:.6f}"
        )

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

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
        target_key=args.target,
    )

    dataset_samples = load_dataset(args.dataset_path)
    enriched_dataset = EnrichedDataset(dataset_samples, config.target_key)

    train_idx, val_idx, test_idx = split_indices(
        len(enriched_dataset), config.train_fraction, config.val_fraction, config.seed
    )

    loaders = build_dataloaders(
        enriched_dataset, train_idx, val_idx, test_idx, config.batch_size
    )

    device = torch.device(args.device)
    model = Surrogate().to(device)
    model = model.to(dtype=DTYPE)

    history = train(model, loaders, config, device)

    test_metrics = evaluate(model, loaders[2], device)
    print(
        f"Test MSE: {test_metrics['mse']:.6f} | Test MAE: {test_metrics['mae']:.6f}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = args.output_dir / "surrogate_state_dict.pt"
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
                    "target": config.target_key,
                    "dataset_path": str(args.dataset_path),
                    "output_dir": str(args.output_dir),
                },
            },
            handle,
            indent=2,
        )

    print(f"Saved best surrogate weights to {weights_path}")
    print(f"Training metrics logged to {metrics_path}")


if __name__ == "__main__":
    main()
