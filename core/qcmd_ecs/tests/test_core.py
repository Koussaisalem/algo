import torch
import pytest
from qcmd_ecs.core.manifold import project_to_tangent_space, retract_to_manifold, sym
from qcmd_ecs.core.dynamics import run_reverse_diffusion
from qcmd_ecs.core.types import DTYPE, StiefelManifold

# Ensure all computations are done in float64 for numerical stability
torch.set_default_dtype(torch.float64)

# --- Mock Models and Schedules (for testing purposes) ---
class MockScoreModel(torch.nn.Module):
    def __init__(self, m, k):
        super().__init__()
        self.linear = torch.nn.Linear(m * k, m * k, bias=False, dtype=DTYPE)
        # Initialize weights to something deterministic for reproducibility
        torch.nn.init.eye_(self.linear.weight.data.view(m * k, m * k))

    def forward(self, U_t, t):
        # Simple mock: returns a fixed gradient direction
        # In a real scenario, this would be a neural network output
        return torch.ones_like(U_t) * 0.01

class MockEnergyGradientModel(torch.nn.Module):
    def __init__(self, m, k):
        super().__init__()
        self.linear = torch.nn.Linear(m * k, m * k, bias=False, dtype=DTYPE)
        torch.nn.init.eye_(self.linear.weight.data.view(m * k, m * k))

    def forward(self, U_t):
        # Simple mock: returns a fixed energy gradient
        return -torch.ones_like(U_t) * 0.005

def mock_gamma_schedule(t):
    return 0.1

def mock_eta_schedule(t):
    return 0.05

def mock_tau_schedule(t):
    return 0.01

# --- Verification Tests ---
class TestQCMDECSMechanics:
    @pytest.fixture
    def setup_manifold_params(self):
        m, k = 10, 5
        U = torch.randn(m, k, dtype=DTYPE)
        U = torch.linalg.qr(U)[0] # Project to Stiefel manifold
        Z = torch.randn(m, k, dtype=DTYPE)
        return U, Z, m, k

    def test_sym(self, setup_manifold_params):
        U, Z, m, k = setup_manifold_params
        A = torch.randn(m, m, dtype=DTYPE)
        symmetrized_A = sym(A)
        assert torch.allclose(symmetrized_A, symmetrized_A.transpose(-1, -2))
        assert torch.allclose(symmetrized_A, 0.5 * (A + A.transpose(-1, -2)))

    def test_project_to_tangent_space_orthogonality(self, setup_manifold_params):
        U, Z, m, k = setup_manifold_params
        projected_Z = project_to_tangent_space(U, Z)
        # Check if U.T @ projected_Z is skew-symmetric (tangent space property)
        UT_projected_Z = U.transpose(-1, -2) @ projected_Z
        assert torch.allclose(UT_projected_Z, -UT_projected_Z.transpose(-1, -2), atol=1e-9)

    def test_retract_to_manifold_constraint(self, setup_manifold_params):
        U, Z, m, k = setup_manifold_params
        # Create a tangent vector G
        G = project_to_tangent_space(U, Z)
        epsilon = 0.1
        U_new = retract_to_manifold(U + epsilon * G)
        # Check if U_new is on the Stiefel manifold (U_new.T @ U_new = I)
        identity_matrix = torch.eye(k, dtype=DTYPE, device=U_new.device)
        assert torch.allclose(U_new.transpose(-1, -2) @ U_new, identity_matrix, atol=1e-9)

    def test_manifold_constraint_integrity(self, setup_manifold_params):
        U_T, _, m, k = setup_manifold_params

        score_model = MockScoreModel(m, k)
        energy_gradient_model = MockEnergyGradientModel(m, k)
        gamma_schedule = mock_gamma_schedule
        eta_schedule = mock_eta_schedule
        tau_schedule = mock_tau_schedule
        num_steps = 10
        seed = 42

        torch.manual_seed(seed) # Ensure reproducibility of noise

        # Define the callback function
        def manifold_constraint_callback(t: int, U_t: StiefelManifold):
            identity_matrix = torch.eye(k, dtype=DTYPE, device=U_t.device)
            assert torch.allclose(U_t.transpose(-1, -2) @ U_t, identity_matrix, atol=1e-9), \
                f"Manifold constraint violated at step {t}"

        # Run reverse diffusion with the callback
        U_final = run_reverse_diffusion(
            U_T,
            score_model,
            energy_gradient_model,
            gamma_schedule,
            eta_schedule,
            tau_schedule,
            num_steps,
            seed,
            callback=manifold_constraint_callback
        )

        # Final check (already done by the callback, but good for explicit final assertion)
        identity_matrix = torch.eye(k, dtype=DTYPE, device=U_final.device)
        assert torch.allclose(U_final.transpose(-1, -2) @ U_final, identity_matrix, atol=1e-9), \
            "Manifold constraint violated at the end of reverse diffusion"

    def test_energy_gradient_directionality(self, setup_manifold_params):
        U, Z, m, k = setup_manifold_params
        # Mock an energy gradient
        mock_grad_E = torch.randn_like(U, dtype=DTYPE)
        # Project it to the tangent space
        projected_grad_E = project_to_tangent_space(U, mock_grad_E)

        # The projected gradient should be "tangent" to the manifold at U.
        # This means U.T @ projected_grad_E should be skew-symmetric.
        UT_projected_grad_E = U.transpose(-1, -2) @ projected_grad_E
        assert torch.allclose(UT_projected_grad_E, -UT_projected_grad_E.transpose(-1, -2), atol=1e-9)

        # Also, the component of mock_grad_E orthogonal to the tangent space should be removed.
        # The component removed is U @ sym(U.T @ mock_grad_E)
        removed_component = U @ sym(U.transpose(-1, -2) @ mock_grad_E)
        assert torch.allclose(projected_grad_E, mock_grad_E - removed_component, atol=1e-9)
