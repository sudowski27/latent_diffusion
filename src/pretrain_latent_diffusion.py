"""version 0.1.0"""
import torch
import tqdm
from .forward_process import forward_process
import matplotlib.pyplot as plt


def pretrain_latent_diffusion(diffusion_model: torch.nn.Module, encoder: torch.nn.Module, input_tensor: torch.tensor, optimizer, epochs, betas, alphas, alphas_bar):
    """
    """
    encoder.eval()
    with torch.no_grad():
        latent_data = encoder(input_tensor)
    encoder.train()

    for epoch in tqdm.tqdm(range(epochs)):
        timestep = 2
        mu_q, sigma_q, xt = forward_process(
            timestep,
            alphas_bar,
            latent_data,
            betas,
            alphas
        )
    plt.plot(xt)
    plt.show()