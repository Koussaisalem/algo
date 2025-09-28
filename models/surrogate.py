import torch
import torch.nn as nn
import numpy as np
import schnetpack as spk
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

model = spk.representation.SchNet(
    n_atom_basis=64,
    n_interactions=3,
    radial_basis=spk.nn.radial.GaussianRBF(n_rbf=50, cutoff=5.0),
    cutoff_fn=spk.nn.CosineCutoff(5.0)
)


