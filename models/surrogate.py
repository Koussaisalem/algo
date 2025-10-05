from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, radius_graph

from qcmd_hybrid_framework.models.surrogate import (
    DEFAULT_MODEL_CONFIG as HYBRID_DEFAULT_CONFIG,
    Surrogate as HybridSurrogate,
    _ensure_batch as _hybrid_ensure_batch,
)


class Surrogate(HybridSurrogate):
    """Alias for the hybrid NequIP surrogate to keep architecture consistent."""

    def __init__(self, model_config: Optional[Dict[str, Any]] = None) -> None:
        config = dict(HYBRID_DEFAULT_CONFIG)
        if model_config:
            config.update(model_config)
        super().__init__(model_config=config)


class SimpleSurrogate(nn.Module):
    """Lightweight GCN fallback when NequIP is unavailable."""

    def __init__(
        self,
        hidden_channels: int = 128,
        cutoff: float = 4.5,
        max_atomic_num: int = 100,
    ) -> None:
        super().__init__()
        if hidden_channels < 16:
            raise ValueError("hidden_channels must be >= 16 for stable training")
        self.cutoff = cutoff
        self.embedding = nn.Embedding(max_atomic_num, hidden_channels)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        pos = data.pos
        z = data.z
        batch = _hybrid_ensure_batch(data)
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch, loop=False)

        x = self.embedding(z)
        x = x.to(dtype=pos.dtype)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        pooled = global_mean_pool(x, batch)
        energy = self.readout(pooled).view(-1)
        return energy


