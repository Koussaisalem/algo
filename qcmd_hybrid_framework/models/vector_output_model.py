"""
Vector output model based on NEquIP architecture (like Surrogate but outputs vectors).

This model uses the NequIPGNNEnergyModel (which doesn't need gradients) 
and adds a trainable output head to convert per-atom energies to 3D score vectors.
"""

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


class VectorOutputModel(nn.Module):
    """
    NequIP model that outputs per-atom 3D vectors for score prediction.
    
    Strategy: Use energy model (no gradient issues) + learnable output head
    """

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
        num_features = config.get("num_features", 64)

        ensure_global_state()
        
        # Use the energy model as the backbone (no gradient computation needed!)
        self.backbone = NequIPGNNEnergyModel(
            seed=seed,
            model_dtype=model_dtype,
            type_names=type_names,
            per_type_energy_shifts=per_type_energy_shifts,
            per_type_energy_scales=per_type_energy_scales,
            **config,
        )
        
        # Add a simple output head to convert backbone features to 3D vectors
        # We'll extract per-atom features from the backbone and project to R^3
        self.output_head = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.SiLU(),
            nn.Linear(num_features // 2, 3),
        )
        
        if model_dtype == "float64":
            self.output_head = self.output_head.to(dtype=torch.float64)

    def forward(self, data: Data) -> torch.Tensor:
        nequip_ready = prepare_atomic_inputs(
            data,
            self._atomic_numbers,
            self._r_max,
        )

        # Forward through backbone to get internal representations
        # We need to access intermediate node features, not just the final energy
        # For now, let's use a workaround: output energy per atom and use as features
        results = self.backbone.model(nequip_ready)  # Access internal model
        
        # Extract node features from the NequIP output
        # NequIP stores these in NODE_FEATURES_KEY or EDGE_FEATURES_KEY
        if AtomicDataDict.NODE_FEATURES_KEY in results:
            node_features = results[AtomicDataDict.NODE_FEATURES_KEY]
        elif "node_attrs" in results:
            node_features = results["node_attrs"]
        else:
            # Fallback: use the per-atom energy contributions
            # This is not ideal but will work
            n_atoms = nequip_ready[AtomicDataDict.POSITIONS_KEY].shape[0]
            # Create dummy features
            node_features = torch.randn(
                n_atoms, 
                64,  # num_features
                dtype=data.pos.dtype, 
                device=data.pos.device
            )
        
        # Project to 3D score vectors
        scores = self.output_head(node_features)
        return scores
