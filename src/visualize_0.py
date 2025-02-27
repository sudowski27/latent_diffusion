"""version 0.1.0"""
import torch
from .forward_process import forward_process
from .reverse_process import reverse_process


def visualize_0(input_tensor: torch.tensor, alphas_bar, betas, alphas, max_timesteps, diffusion_model, encoder, decoder, device):
    """
    Temp function for visualization results
    """
    encoder.eval()
    with torch.no_grad():
        latent_data = encoder(input_tensor)
    encoder.train()

    timestep = max_timesteps
    min_timesteps = 2
    mu_q, sigma_q, xt = forward_process(
            timestep,
            alphas_bar,
            latent_data,
            betas,
            alphas
        )

    xt = torch.randn_like(xt)

    for t in range(max_timesteps, min_timesteps - 1, -1):
        with torch.no_grad():
            _, _, xt = reverse_process(diffusion_model, t, xt, device)

    xt = torch.ones_like(xt) * 0
    decoder.eval()
    with torch.no_grad():
        output_data = decoder(xt)

    decoder.train()

    return output_data
