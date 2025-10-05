import torch
from typing import TypeAlias

# The required data type for all high-precision floating point operations.
DTYPE: TypeAlias = torch.float64

# A tensor representing a point on the Stiefel manifold St(m,k).
StiefelManifold: TypeAlias = torch.Tensor