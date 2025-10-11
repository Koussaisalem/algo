from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch_geometric.data import Data

from nequip.data import AtomicDataDict, compute_neighborlist_, from_dict
from nequip.utils.global_state import set_global_state

DEFAULT_TYPE_NAMES: Tuple[str, ...] = ("H", "C", "N", "O", "F")
DEFAULT_ATOMIC_NUMBERS: Tuple[int, ...] = (1, 6, 7, 8, 9)

DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "num_layers": 3,
    "l_max": 1,
    "parity": True,
    "num_features": 64,
    "radial_mlp_depth": 2,
    "radial_mlp_width": 64,
    "r_max": 5.0,
    "avg_num_neighbors": 15.0,
    "seed": 0,
    "model_dtype": "float64",
    "type_names": DEFAULT_TYPE_NAMES,
    "atomic_numbers": DEFAULT_ATOMIC_NUMBERS,
    "per_type_energy_shifts": {name: 0.0 for name in DEFAULT_TYPE_NAMES},
    "per_type_energy_scales": {name: 1.0 for name in DEFAULT_TYPE_NAMES},
}


def ensure_global_state() -> None:
    """Initialise NequIP's global state if needed."""

    set_global_state()


def ensure_batch(data: Data) -> torch.Tensor:
    batch = getattr(data, "batch", None)
    if batch is None:
        batch = data.pos.new_zeros(data.pos.size(0), dtype=torch.long)
    return batch


def to_type_indices(atomic_numbers: torch.Tensor, supported_numbers: Sequence[int]) -> torch.Tensor:
    mapped = torch.full_like(atomic_numbers, fill_value=-1)
    for type_index, number in enumerate(supported_numbers):
        mask = atomic_numbers == number
        if mask.any():
            mapped[mask] = type_index
    if (mapped < 0).any():
        missing = atomic_numbers[mapped < 0].unique().tolist()
        raise ValueError(
            f"Encountered atomic numbers {missing} outside supported set {list(supported_numbers)}"
        )
    return mapped


def _normalize_per_type_mapping(
    values: Union[Sequence[float], Dict[str, float], None],
    type_names: Sequence[str],
    default: float,
) -> Dict[str, float]:
    if values is None:
        values = {}
    if isinstance(values, dict):
        mapping = {name: float(values.get(name, default)) for name in type_names}
    else:
        seq = list(values)
        if len(seq) != len(type_names):
            raise ValueError(
                "Per-type parameter sequences must match the number of type names"
            )
        mapping = {name: float(seq[idx]) for idx, name in enumerate(type_names)}
    return mapping


def consume_model_config(
    model_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Tuple[int, ...], Tuple[str, ...], Dict[str, float], Dict[str, float], int, str]:
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    if model_config:
        config.update(model_config)

    atomic_numbers = tuple(config.pop("atomic_numbers"))
    type_names = tuple(config.pop("type_names"))
    if len(atomic_numbers) != len(type_names):
        raise ValueError("atomic_numbers and type_names must have the same length")

    per_type_energy_shifts = _normalize_per_type_mapping(
        config.pop("per_type_energy_shifts", None), type_names, default=0.0
    )
    per_type_energy_scales = _normalize_per_type_mapping(
        config.pop("per_type_energy_scales", None), type_names, default=1.0
    )

    seed = int(config.pop("seed", 0))
    model_dtype = config.pop("model_dtype", "float64")

    return (
        config,
        atomic_numbers,
        type_names,
        per_type_energy_shifts,
        per_type_energy_scales,
        seed,
        model_dtype,
    )


def prepare_atomic_inputs(
    data: Data,
    atomic_numbers: Sequence[int],
    r_max: float,
) -> AtomicDataDict.Type:
    positions = data.pos
    batch = ensure_batch(data)
    type_indices = to_type_indices(data.z.to(torch.long), atomic_numbers)

    if batch.numel() == 0:
        counts = torch.tensor([positions.size(0)], device=positions.device, dtype=torch.long)
    else:
        counts = torch.bincount(batch, minlength=int(batch.max().item()) + 1)

    pbc = torch.zeros((counts.shape[0], 3), dtype=torch.bool, device=positions.device)

    atomic_dict = from_dict(
        {
            AtomicDataDict.POSITIONS_KEY: positions,
            AtomicDataDict.ATOM_TYPE_KEY: type_indices,
            AtomicDataDict.BATCH_KEY: batch,
            AtomicDataDict.NUM_NODES_KEY: counts,
            AtomicDataDict.PBC_KEY: pbc,
        }
    )
    atomic_dict = AtomicDataDict.with_batch_(atomic_dict)
    atomic_dict = compute_neighborlist_(atomic_dict, r_max=r_max)
    return atomic_dict