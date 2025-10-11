from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch

__all__ = [
    "HARTREE_TO_EV",
    "EV_TO_HARTREE",
    "BOHR_TO_ANGSTROM",
    "ANGSTROM_TO_BOHR",
    "ATOMIC_MASSES_AMU",
    "ManifoldFrame",
    "hartree_to_ev",
    "ev_to_hartree",
    "gradient_hartree_per_bohr_to_force_ev_per_ang",
    "get_atomic_masses",
    "compute_manifold_frame",
]

# --- Physical constants ---
HARTREE_TO_EV: float = 27.211386245988
BOHR_TO_ANGSTROM: float = 0.529177210903
EV_TO_HARTREE: float = 1.0 / HARTREE_TO_EV
ANGSTROM_TO_BOHR: float = 1.0 / BOHR_TO_ANGSTROM


# --- Atomic reference data ---
# Values (in atomic mass units) sourced from NIST; truncated to the precision needed for
# molecular modelling. Extend this dictionary as new atomic species are introduced.
ATOMIC_MASSES_AMU: Dict[int, float] = {
    1: 1.00784,  # H
    2: 4.002602,  # He
    3: 6.938,  # Li
    4: 9.0121831,  # Be
    5: 10.806,  # B
    6: 12.0096,  # C
    7: 14.00643,  # N
    8: 15.99903,  # O
    9: 18.998403163,  # F
    10: 20.1797,  # Ne
    11: 22.98976928,  # Na
    12: 24.304,  # Mg
    13: 26.9815385,  # Al
    14: 28.085,  # Si
    15: 30.973761998,  # P
    16: 32.06,  # S
    17: 35.45,  # Cl
    18: 39.948,  # Ar
}


@dataclass
class ManifoldFrame:
    """Container for the information required to initialise Stiefel manifold states."""

    frame: torch.Tensor
    centroid: torch.Tensor
    mass_weights: torch.Tensor
    rank: int

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Export a serialisable representation compatible with ``torch.save``."""

        return {
            "frame": self.frame,
            "centroid": self.centroid,
            "mass_weights": self.mass_weights,
            "rank": torch.tensor(self.rank, dtype=torch.int64, device=self.frame.device),
        }


def hartree_to_ev(value: torch.Tensor | float) -> torch.Tensor:
    """Convert Hartree units to electron volts (float64)."""

    tensor = torch.as_tensor(value, dtype=torch.float64)
    return tensor * HARTREE_TO_EV


def ev_to_hartree(value: torch.Tensor | float) -> torch.Tensor:
    """Convert electron volts to Hartree (float64)."""

    tensor = torch.as_tensor(value, dtype=torch.float64)
    return tensor * EV_TO_HARTREE


def gradient_hartree_per_bohr_to_force_ev_per_ang(gradient: torch.Tensor) -> torch.Tensor:
    """Convert an energy gradient (Hartree/Bohr) into a force (eV/Å)."""

    gradient = torch.as_tensor(gradient, dtype=torch.float64)
    return -gradient * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)


def get_atomic_masses(
    atomic_numbers: Iterable[int] | torch.Tensor, *, device: torch.device | None = None
) -> torch.Tensor:
    """Return atomic masses (amu) for the provided atomic numbers."""

    if isinstance(atomic_numbers, torch.Tensor):
        numbers = atomic_numbers.to(dtype=torch.int64).tolist()
        inferred_device = atomic_numbers.device
    else:
        numbers = [int(z) for z in atomic_numbers]
        inferred_device = torch.device("cpu")

    try:
        masses = [ATOMIC_MASSES_AMU[z] for z in numbers]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Atomic mass for Z={exc.args[0]} is not defined") from exc

    target_device = device if device is not None else inferred_device
    return torch.tensor(masses, dtype=torch.float64, device=target_device)


def compute_manifold_frame(
    positions: torch.Tensor, atomic_numbers: Iterable[int] | torch.Tensor
) -> ManifoldFrame:
    """
    Construct a mass-weighted orthonormal frame anchored to a molecular geometry.

    Args:
        positions: Cartesian coordinates in Å with shape ``(n_atoms, 3)``.
        atomic_numbers: Atomic numbers with shape ``(n_atoms,)``.

    Returns:
        A :class:`ManifoldFrame` containing the orthonormal basis and metadata.
    """

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")

    positions_64 = positions.to(dtype=torch.float64)
    device = positions_64.device
    n_atoms = positions_64.shape[0]

    if isinstance(atomic_numbers, torch.Tensor):
        numbers = atomic_numbers.to(dtype=torch.int64, device=device)
    else:
        numbers = torch.tensor(list(atomic_numbers), dtype=torch.int64, device=device)

    if numbers.ndim != 1 or numbers.shape[0] != n_atoms:
        raise ValueError("atomic_numbers must have shape (n_atoms,)")

    masses = get_atomic_masses(numbers, device=device)
    total_mass = masses.sum()
    if not torch.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("total mass must be positive and finite")

    weights = masses / total_mass
    centroid = (weights[:, None] * positions_64).sum(dim=0)
    centered = positions_64 - centroid

    sqrt_weights = torch.sqrt(weights).unsqueeze(-1)
    weighted = centered * sqrt_weights

    q, r = torch.linalg.qr(weighted, mode="reduced")
    if q.numel() == 0:
        raise ValueError("unable to compute orthonormal frame for empty geometry")

    sign = torch.sign(torch.diag(r))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign

    frame_rank = min(3, q.shape[1])
    frame = q[:, :frame_rank].contiguous()

    if frame_rank == 0:
        raise ValueError("unable to compute orthonormal frame for the given geometry")

    if frame_rank == 3:
        det = torch.linalg.det(frame[:frame_rank, :])
        if det < 0:
            frame[:, -1] = -frame[:, -1]

    return ManifoldFrame(
        frame=frame,
        centroid=centroid.contiguous(),
        mass_weights=weights.contiguous(),
        rank=int(frame_rank),
    )
