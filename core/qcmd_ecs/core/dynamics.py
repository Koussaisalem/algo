import torch
from typing import Callable, Optional
from .types import StiefelManifold, DTYPE
from .manifold import project_to_tangent_space, retract_to_manifold

def run_reverse_diffusion(
    U_T: StiefelManifold,
    score_model: Callable,
    energy_gradient_model: Callable,
    gamma_schedule: Callable,
    eta_schedule: Callable,
    tau_schedule: Callable,
    num_steps: int,
    seed: int,
    callback: Optional[Callable] = None
) -> StiefelManifold:
    """
    Executes the full reverse diffusion process on the Stiefel manifold, strictly
    following Algorithm 1 in the QCMD-ECS paper.

    Args:
        U_T (StiefelManifold): The initial noisy state on the manifold at t=T.
        score_model (Callable): The trained score model s_theta.
        energy_gradient_model (Callable): The energy gradient model.
        gamma_schedule (Callable): Schedule for the energy gradient weight.
        eta_schedule (Callable): Schedule for the main update step size.
        tau_schedule (Callable): Schedule for the noise magnitude.
        num_steps (int): The total number of reverse diffusion steps (T).
        seed (int): An integer for the random number generator to ensure determinism.
        callback (Callable, optional): An optional callback for verification.

    Returns:
        StiefelManifold: The final generated state on the manifold, U_0.
    """
    torch.manual_seed(seed)
    U_t = U_T.clone()

    for t in range(num_steps, 0, -1):
        # Step a) Compute learned score
        s_t = score_model(U_t, t)
        
        # Step b) Compute energy gradients
        grad_E_t = energy_gradient_model(U_t)

        # Step c) Form MAECS score
        # The paper combines these, but projecting first is cleaner.
        # S_MAE_U = project_to_tangent_space(U_t, s_t + gamma_schedule(t) * grad_E_t)
        # For clarity and strict adherence, we follow the paper's update formula directly.
        
        # Step d) Sample noise and project it
        Z_prime = torch.randn_like(U_t, dtype=DTYPE)
        projected_noise = project_to_tangent_space(U_t, Z_prime)

        # Step f) Update orbitals (manifold) - This combines steps c, d, and f from Algorithm 1
        S_MAE_U = project_to_tangent_space(U_t, s_t + gamma_schedule(t) * grad_E_t)
        
        U_tilde = U_t - eta_schedule(t) * S_MAE_U + tau_schedule(t) * projected_noise
        
        U_t = retract_to_manifold(U_tilde) # This is now the QR retraction

        if callback:
            # The callback should check the state *after* the update, which is now U_t
            callback(t, U_t)
    
    return U_t
