import torch
import numpy as np

# --- FIX 1: Define a global data type for high precision ---
DTYPE = torch.float64

# Mock variables for a single generation instance
m = 50
k = 10
T = 1000
tolerance = 1e-6
# --- FIX 2: Initialize all tensors with the high-precision data type ---
U_T = torch.linalg.qr(torch.randn(m, k, dtype=DTYPE))[0]


# Placeholder (dummy) functions for neural network outputs
def mock_score_model(U, t):
    """
    Simulates the output of the score model s_theta,U(X_t, t).
    Returns a random tensor of shape (m, k).
    """
    return torch.randn(m, k, dtype=DTYPE)

def mock_energy_gradient(U):
    """
    Simulates the output of the energy gradient nabla_U E_phi(X_t).
    Returns a random tensor of shape (m, k).
    """
    return torch.randn(m, k, dtype=DTYPE)

# Placeholder functions for gamma(t), eta_t, and tau_t
def gamma_t(t):
    """
    Placeholder for the gamma(t) function.
    Returns a small constant value.
    """
    # --- FIX 3: Ensure schedule functions use the correct data type ---
    return torch.tensor(1e-4, dtype=DTYPE)

def eta_t(t):
    """
    Placeholder for the eta_t function.
    Returns a small constant value.
    """
    return torch.tensor(1e-4, dtype=DTYPE)

def tau_t(t):
    """
    Placeholder for the tau_t function.
    Returns a small constant value.
    """
    return torch.tensor(1e-4, dtype=DTYPE)

# Helper functions for Stiefel manifold operations
def sym(A):
    """
    Implements the symmetric part of a matrix: 0.5 * (A + A^T).
    """
    return 0.5 * (A + A.T)

def project_to_tangent_space(U, Z):
    """
    Projects a matrix Z onto the tangent space of the Stiefel manifold at U.
    Pi_T_U(Z) = Z - U * sym(U^T @ Z).
    """
    return Z - U @ sym(U.T @ Z)

def retract_to_manifold(M):
    """
    Retracts a matrix M to the Stiefel manifold using QR decomposition.
    Returns the orthogonal matrix Q from M = QR.
    """
    Q, R = torch.linalg.qr(M)
    # Ensure the diagonal elements of R are positive for a unique retraction
    # This is important for consistency, as QR decomposition is not unique.
    # If R_ii < 0, then negate the corresponding column in Q.
    signs = torch.diag(R).sign().to(dtype=DTYPE)
    # Ensure signs is a diagonal matrix for correct broadcasting
    Q = Q @ torch.diag(signs)
    return Q

# Manifold Constraint Integrity Test
print("Starting Manifold Constraint Integrity Test...")

U_t = U_T
for t in range(T, 0, -1):
    # a. Get outputs from mock functions
    s_t = mock_score_model(U_t, t)
    grad_E_t = mock_energy_gradient(U_t)

    # b. Form the MAECS score for the orbitals
    S_MAE_U = project_to_tangent_space(U_t, s_t + gamma_t(t) * grad_E_t)

    # c. Generate random noise Z_prime
    Z_prime = torch.randn(m, k, dtype=DTYPE)

    # d. Perform the tangent space update step
    U_tilde = U_t - eta_t(t) * S_MAE_U + tau_t(t) * project_to_tangent_space(U_t, Z_prime)

    # e. Perform the retraction step
    U_t_minus_1 = retract_to_manifold(U_tilde)

    # f. VERIFICATION: Calculate the orthonormality error
    # --- FIX 4: Ensure the identity matrix for comparison uses the correct data type ---
    identity = torch.eye(k, dtype=DTYPE)
    error = torch.linalg.norm(U_t_minus_1.T @ U_t_minus_1 - identity, ord='fro')

    # g. ASSERT:
    assert error < tolerance, f"Orthonormality error {error} exceeded tolerance {tolerance} at step {t}"

    # h. Update U_t for the next iteration
    U_t = U_t_minus_1

print("Manifold Constraint Integrity Test passed!")

# Energy Gradient Directionality Test
print("\nStarting Energy Gradient Directionality Test...")

# 1. Setup
# --- FIX 5: Ensure all tensors in this test use the high-precision data type ---
U_t = torch.linalg.qr(torch.randn(m, k, dtype=DTYPE))[0]
s_t = torch.zeros(m, k, dtype=DTYPE)
grad_E_t = torch.randn(m, k, dtype=DTYPE)

# 2. Calculations
physics_direction = -1 * project_to_tangent_space(U_t, grad_E_t)
S_MAE_with_grad = project_to_tangent_space(U_t, gamma_t(t=0) * grad_E_t)
update_vector_with_grad = -eta_t(t=0) * S_MAE_with_grad

# 3. Verification
expected_update_vector = eta_t(t=0) * gamma_t(t=0) * physics_direction
assert torch.allclose(update_vector_with_grad, expected_update_vector, atol=tolerance), \
    "The update vector is not a positive scalar multiple of the projected negative gradient."

print("Energy Gradient Directionality Test passed!")
