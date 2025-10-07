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


class TMDSurrogate(nn.Module):
    """
    NequIP surrogate for TMD materials with d-orbital support (l_max=2).
    
    Predicts formation energy for 2D transition metal dichalcogenides.
    Trained on Materials Project VASP PBE data.
    """

    def __init__(self, model_config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        
        # Default TMD config with d-orbital support
        # Support all elements found in Materials Project TMD database
        atomic_numbers = (1, 3, 7, 8, 11, 12, 16, 17, 23, 24, 25, 26, 27, 28, 29, 
                         34, 35, 37, 41, 42, 47, 48, 49, 52, 53, 55, 56, 73, 74, 75, 80, 81)
        type_names = tuple(f"Z{z}" for z in atomic_numbers)
        
        tmd_config = {
            "num_layers": 4,
            "l_max": 2,  # d-orbital support for transition metals!
            "parity": True,
            "num_features": 64,
            "radial_mlp_depth": 2,
            "radial_mlp_width": 64,
            "r_max": 5.0,
            "avg_num_neighbors": 15.0,
            "seed": 42,
            "model_dtype": "float64",
            "atomic_numbers": atomic_numbers,
            "type_names": type_names,
            "per_type_energy_shifts": {name: 0.0 for name in type_names},
            "per_type_energy_scales": {name: 1.0 for name in type_names},
        }
        
        # Override with user config
        if model_config:
            tmd_config.update(model_config)
        
        (
            config,
            atomic_numbers,
            type_names,
            per_type_energy_shifts,
            per_type_energy_scales,
            seed,
            model_dtype,
        ) = consume_model_config(tmd_config)

        self._atomic_numbers = atomic_numbers
        self._r_max = float(config.get("r_max", 5.0))

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
        """
        Forward pass through NequIP model.
        
        Args:
            data: PyG Data object with pos (positions) and z (atomic numbers)
        
        Returns:
            Formation energy in eV (shape: [batch_size])
        """
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
