from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect an enriched QM9 dataset and optionally drop invalid samples.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/qm9_micro_5k_enriched.pt"),
        help="Path to the enriched dataset produced by 02_enrich_dataset.py.",
    )
    parser.add_argument(
        "--clean-output",
        type=Path,
        default=None,
        help="If provided, write a filtered dataset without NaNs to this file.",
    )
    parser.add_argument(
        "--drop-nan-gradients",
        action="store_true",
        help="Exclude samples whose gradients or forces contain NaNs/inf.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to save the computed statistics as JSON.",
    )
    return parser.parse_args()


def tensor_nan_inf_counts(tensor: torch.Tensor) -> tuple[int, int]:
    return (
        torch.isnan(tensor).sum().item(),
        torch.isinf(tensor).sum().item(),
    )


def describe_tensor(name: str, tensor: torch.Tensor) -> Dict[str, Any]:
    t = tensor.detach()
    nan_count, inf_count = tensor_nan_inf_counts(t)
    entry: Dict[str, Any] = {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "nan": int(nan_count),
        "inf": int(inf_count),
    }
    if t.numel() and nan_count == 0 and inf_count == 0:
        entry["min"] = float(t.min().item())
        entry["max"] = float(t.max().item())
        entry["mean"] = float(t.mean().item())
        entry["std"] = float(t.std(unbiased=False).item())
    return entry


def describe_dataset(entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    invalid_count = 0
    invalid_indices: list[int] = []
    stats: Dict[str, Dict[str, Any]] = {}
    atom_count_histogram: Dict[int, int] = {}

    for idx, entry in enumerate(entries):
        total += 1
        atom_count = int(entry.get("num_atoms", 0))
        atom_count_histogram[atom_count] = atom_count_histogram.get(atom_count, 0) + 1

        entry_invalid = False

        for key, value in entry.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if torch.is_tensor(sub_value):
                        stat_key = f"{key}.{sub_key}"
                        stats.setdefault(stat_key, {"count": 0, "nan": 0, "inf": 0})
                        stats[stat_key]["count"] += sub_value.numel()
                        nan, inf = tensor_nan_inf_counts(sub_value)
                        stats[stat_key]["nan"] += nan
                        stats[stat_key]["inf"] += inf
                        if nan or inf:
                            entry_invalid = True
            elif torch.is_tensor(value):
                stat_key = key
                stats.setdefault(stat_key, {"count": 0, "nan": 0, "inf": 0})
                stats[stat_key]["count"] += value.numel()
                nan, inf = tensor_nan_inf_counts(value)
                stats[stat_key]["nan"] += nan
                stats[stat_key]["inf"] += inf
                if nan or inf:
                    entry_invalid = True

        if entry_invalid:
            invalid_count += 1
            invalid_indices.append(idx)

    for key, aggregate in stats.items():
        count = aggregate["count"] or 1
        aggregate["nan_ratio"] = aggregate["nan"] / count
        aggregate["inf_ratio"] = aggregate["inf"] / count

    return {
        "total_entries": total,
        "invalid_entries": invalid_count,
        "invalid_indices": invalid_indices,
        "field_stats": stats,
        "atom_count_histogram": atom_count_histogram,
    }


def filter_entries(
    entries: Iterable[Dict[str, Any]],
    *,
    drop_nan_gradients: bool,
) -> list[Dict[str, Any]]:
    cleaned: list[Dict[str, Any]] = []
    for entry in entries:
        keep = True
        if drop_nan_gradients:
            for key in ("gradient_hartree_per_bohr", "forces_ev_per_angstrom"):
                tensor = entry.get(key)
                if torch.is_tensor(tensor):
                    nan, inf = tensor_nan_inf_counts(tensor)
                    if nan or inf:
                        keep = False
                        break
        if keep:
            cleaned.append(entry)
    return cleaned


def main() -> None:
    args = parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.input_path}")

    entries = torch.load(args.input_path, map_location="cpu")
    if not isinstance(entries, list):
        raise TypeError("Expected the enriched dataset to be a list of dictionaries")

    report = describe_dataset(entries)

    print(f"Entries: {report['total_entries']}")
    print(f"Invalid entries (NaN/Inf detected): {report['invalid_entries']}")

    print("\nPer-field NaN statistics:")
    for key, stats in sorted(report["field_stats"].items()):
        count = stats["count"]
        nan = stats["nan"]
        inf = stats["inf"]
        print(
            f"  {key:30s} count={count:8d} nan={nan:6d} ({stats['nan_ratio']*100:6.2f}%)"
            f" inf={inf:6d} ({stats['inf_ratio']*100:6.2f}%)"
        )

    print("\nAtom count histogram (num_atoms -> frequency):")
    for num_atoms in sorted(report["atom_count_histogram"].keys()):
        freq = report["atom_count_histogram"][num_atoms]
        print(f"  {num_atoms:2d}: {freq}")

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Detailed report saved to {args.report_json}")

    if args.clean_output:
        cleaned = filter_entries(entries, drop_nan_gradients=args.drop_nan_gradients)
        args.clean_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cleaned, args.clean_output)
        print(
            f"Cleaned dataset written to {args.clean_output} ({len(cleaned)} entries,"
            f" removed {len(entries) - len(cleaned)} samples)"
        )


if __name__ == "__main__":
    main()