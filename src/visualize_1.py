"""version 0.1.0"""
import torch
from .forward_process import forward_process
from .reverse_process import reverse_process


def visualize_1(latent_data: torch.tensor, alphas_bar, betas, alphas, max_timesteps, diffusion_model, encoder, decoder, device):
    """
    Temp function for visualization results
    """
    timestep = max_timesteps
    min_timesteps = 2
    mu_q, sigma_q, xt = forward_process(
            timestep,
            alphas_bar,
            latent_data,
            betas,
            alphas
        )

    timesteps_to_save = [max_timesteps, max_timesteps // 2, min_timesteps]
    saved_tensors = []

    for t in range(max_timesteps, min_timesteps - 1, -1):
        with torch.no_grad():
            _, _, xt = reverse_process(diffusion_model, t, xt, device)

        if t in timesteps_to_save:
            saved_tensors.append(xt)

    decoder.eval()
    with torch.no_grad():
        output_data = decoder(xt)

    decoder.train()

    return saved_tensors
