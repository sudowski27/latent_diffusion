"""version 0.1.0"""
import torch
import tqdm
# import matplotlib.pyplot as plt
import numpy as np
from .forward_process import forward_process
from .reverse_process import reverse_process


def pretrain_latent_diffusion(diffusion_model: torch.nn.Module, encoder: torch.nn.Module, input_tensor: torch.tensor, optimizer, epochs, betas, alphas, alphas_bar, max_timesteps, logger):
    """
    """
    min_timesteps = 2
    encoder.eval()
    with torch.no_grad():
        latent_data = encoder(input_tensor)
    encoder.train()

    for epoch in tqdm.tqdm(range(epochs)):
        timestep = np.random.randint(min_timesteps, max_timesteps + 1)
        mu_q, sigma_q, xt = forward_process(
            timestep,
            alphas_bar,
            latent_data,
            betas,
            alphas
        )

        for net in diffusion_model.network_tail:
            for param in net.parameters():
                param.requires_grad = False

        for param in diffusion_model.network_tail[timestep - 1].parameters():
            param.requires_grad = True

        mu_p, sigma_p, xt_minus1 = reverse_process(diffusion_model, timestep, xt)

        KL = torch.log(sigma_p) - torch.log(sigma_q) + (
            sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2)
        K = - KL.mean()
        loss = -K

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            logger.info(f"PRETRAIN ERROR: EPOCH: {epoch} ERROR: {loss.item()}")
