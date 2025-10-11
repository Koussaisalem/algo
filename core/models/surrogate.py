from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from nequip.data import AtomicDataDict
from nequip.model import NequIPGNNEnergyModel

from .nequip_common import (
    DEFAULT_MODEL_CONFIG,
    consume_model_config,
    ensure_global_state,
    prepare_atomic_inputs,
)


class Surrogate(nn.Module):
    """NequIP surrogate used throughout the hybrid pipeline."""

    def __init__(self, model_config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        (
            config,
            atomic_numbers,
            type_names,
            per_type_energy_shifts,
            per_type_energy_scales,
            seed,
            model_dtype,
        ) = consume_model_config(model_config)

        self._atomic_numbers = atomic_numbers
        self._r_max = float(config.get("r_max", DEFAULT_MODEL_CONFIG["r_max"]))

        ensure_global_state()
        self.model = NequIPGNNEnergyModel(
            seed=seed,
            model_dtype=model_dtype,
            type_names=type_names,
            per_type_energy_shifts=per_type_energy_shifts,
            per_type_energy_scales=per_type_energy_scales,
            **config,
        )

    def forward(self, data: Data) -> torch.Tensor:
        nequip_ready = prepare_atomic_inputs(
            data,
            self._atomic_numbers,
            self._r_max,
        )

        results = self.model(nequip_ready)
        total_energy = results.get(AtomicDataDict.TOTAL_ENERGY_KEY)
        if total_energy is None:
            raise KeyError("NequIP model did not return TOTAL_ENERGY_KEY")
        return total_energy.view(-1)
