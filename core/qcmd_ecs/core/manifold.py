import torch
from .types import DTYPE, StiefelManifold

def sym(A: torch.Tensor) -> torch.Tensor:
    """
    Symmetrizes a matrix A.

    Args:
        A (torch.Tensor): The input matrix. Dtype: torch.float64.

    Returns:
        torch.Tensor: The symmetrized matrix. Dtype: torch.float64.
    """
    return 0.5 * (A + A.transpose(-1, -2))

def project_to_tangent_space(U: StiefelManifold, Z: torch.Tensor) -> torch.Tensor:
    """
    Projects a matrix Z onto the tangent space of the Stiefel manifold at U.

    This implements the formula: Pi_T_U(Z) = Z - U * sym(U.T @ Z), which is
    derived from Eq. (14) in the QCMD-ECS paper.

    Args:
        U (StiefelManifold): A point on the Stiefel manifold.
                             Shape: (m, k). Dtype: torch.float64.
        Z (torch.Tensor): An arbitrary matrix in the ambient space to be projected.
                          Shape: (m, k). Dtype: torch.float64.

    Returns:
        torch.Tensor: The projection of Z onto the tangent space at U.
                      Shape: (m, k). Dtype: torch.float64.
    """
    return Z - U @ sym(U.transpose(-1, -2) @ Z)

def retract_to_manifold(M: torch.Tensor) -> StiefelManifold:
    """
    Retracts a matrix M from the ambient space back to the Stiefel manifold
    using the QR decomposition, as specified in Eq. (10) of the QCMD-ECS paper.

    This operation re-orthonormalizes the columns of M, ensuring the result
    strictly satisfies the U.T @ U = I constraint.

    Args:
        M (torch.Tensor): The matrix in the ambient space to be retracted.
                          Shape: (m, k). Dtype: torch.float64.

    Returns:
        StiefelManifold: The new point on the Stiefel manifold.
                         Shape: (m, k). Dtype: torch.float64.
    """
    Q, R = torch.linalg.qr(M)
    # The QR decomposition is made unique by ensuring the diagonal elements of R are positive.
    # If R_ii < 0, we flip the sign of the i-th column of Q.
    signs = torch.diag(R).sign().to(dtype=DTYPE)
    return Q @ torch.diag(signs)
