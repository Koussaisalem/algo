from __future__ import annotations

from typing import Dict, Optional

import torch.nn as nn
from torch_geometric.data import Data

from nequip.data import AtomicDataDict
from nequip.model import NequIPGNNModel

from .nequip_common import (
    DEFAULT_MODEL_CONFIG,
    consume_model_config,
    ensure_global_state,
    prepare_atomic_inputs,
)


class ScoreModel(nn.Module):
    """NequIP-based score predictor returning per-atom vectors."""

    def __init__(self, model_config: Optional[Dict[str, any]] = None) -> None:
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
        self._r_max = float(config.get("r_max", 5.0))

        ensure_global_state()
        self.model = NequIPGNNModel(
            seed=seed,
            model_dtype=model_dtype,
            type_names=type_names,
            per_type_energy_shifts=per_type_energy_shifts,
            per_type_energy_scales=per_type_energy_scales,
            **config,
        )

    def forward(self, data: Data):
        nequip_ready = prepare_atomic_inputs(
            data,
            self._atomic_numbers,
            self._r_max,
        )

        results = self.model(nequip_ready)
        forces = results.get(AtomicDataDict.FORCE_KEY)
        if forces is None:
            raise KeyError("NequIP model did not return FORCE_KEY")
        return forces
