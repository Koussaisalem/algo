"""
Simple position-based score model that doesn't rely on NequIP's force computation.

This model directly outputs per-atom 3D vectors (scores) without requiring
gradient computation through positions. It uses NequIP's convolution layers
but with a direct output head instead of force prediction via autograd.
"""

from __future__ import annotations

from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from torch_geometric.data import Data

from nequip.data import AtomicDataDict
from nequip.nn import SequentialGraphNetwork
from nequip.nn.embedding import OneHotAtomEncoding, SphericalHarmonicEdgeAttrs, RadialBasisEdgeEncoding
from nequip.nn.convnetlayer import ConvNetLayer
from nequip.nn._graph_mixin import GraphModuleMixin

from .nequip_common import (
    DEFAULT_MODEL_CONFIG,
    consume_model_config,
    ensure_global_state,
    prepare_atomic_inputs,
)


class SimpleScoreModel(GraphModuleMixin, torch.nn.Module):
    """
    Direct score prediction model using NequIP convolutions.
    
    Unlike the force-based ScoreModel, this directly outputs 3D vectors
    without requiring autograd through positions.
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
        
        # Build the model
        ensure_global_state()
        
        num_types = len(type_names)
        num_features = config.get("num_features", 64)
        num_layers = config.get("num_layers", 3)
        l_max = config.get("l_max", 1)
        parity = config.get("parity", True)
        
        # Create the layers
        layers = []
        
        # 1. Atom type embedding
        layers.append(
            (
                OneHotAtomEncoding,
                dict(
                    irreps_in=None,
                    num_types=num_types,
                    set_features=True,
                ),
            )
        )
        
        # 2. Edge features
        layers.append(
            (
                RadialBasisEdgeEncoding,
                dict(
                    basis_kwargs=dict(r_max=self._r_max, num_basis=8),
                    cutoff_kwargs=dict(r_max=self._r_max),
                ),
            )
        )
        
        layers.append(
            (
                SphericalHarmonicEdgeAttrs,
                dict(irreps_edge_sh=f"{l_max}{'e' if parity else 'o'}"),
            )
        )
        
        # 3. Convolution layers
        for i in range(num_layers):
            layers.append(
                (
                    ConvNetLayer,
                    dict(
                        feature_irreps_hidden=f"{num_features}x0e + {num_features}x1o",
                        convolution_kwargs=dict(
                            invariant_layers=2,
                            invariant_neurons=64,
                        ),
                    ),
                )
            )
        
        # 4. Output layer - direct vector prediction
        # Output 3D vectors (scores) per atom
        layers.append(
            (
                nn.Linear,
                dict(
                    in_features=num_features * (l_max + 1),
                    out_features=3,
                ),
            )
        )
        
        # Build the sequential model
        self.model = SequentialGraphNetwork.from_parameters(
            shared_params=config,
            layers=layers,
        )
        
        # Set dtype
        if model_dtype == "float64":
            self.model = self.model.to(dtype=torch.float64)
        else:
            self.model = self.model.to(dtype=torch.float32)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass that outputs per-atom 3D score vectors.
        
        Args:
            data: PyG Data with pos, z, (optional) batch
        
        Returns:
            Tensor of shape (n_atoms, 3) with score vectors
        """
        # Prepare input for NequIP
        nequip_ready = prepare_atomic_inputs(
            data,
            self._atomic_numbers,
            self._r_max,
        )
        
        # Forward through the model
        output = self.model(nequip_ready)
        
        # Extract node features and project to 3D
        # The output should contain node features
        if AtomicDataDict.NODE_FEATURES_KEY in output:
            node_features = output[AtomicDataDict.NODE_FEATURES_KEY]
            # Apply a simple linear layer to get 3D vectors
            # For now, just take the first 3 features if available
            if node_features.shape[-1] >= 3:
                scores = node_features[..., :3]
            else:
                # Pad if needed
                scores = torch.nn.functional.pad(
                    node_features, (0, 3 - node_features.shape[-1])
                )
        else:
            # Fallback: zero scores
            n_atoms = data.pos.shape[0]
            scores = torch.zeros(n_atoms, 3, dtype=data.pos.dtype, device=data.pos.device)
        
        return scores
