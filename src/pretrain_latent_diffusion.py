"""version 0.1.0"""
import torch
import tqdm
# import matplotlib.pyplot as plt
import numpy as np
from .forward_process import forward_process
from .reverse_process import reverse_process
from .debug_plot import debug_plot


def pretrain_latent_diffusion(diffusion_model: torch.nn.Module, encoder: torch.nn.Module, input_tensor: torch.tensor, optimizer, epochs, betas, alphas, alphas_bar, max_timesteps, logger, device, train_loader, min_val, max_val, decoder):
    """
    """
    min_timesteps = 2
    encoder.eval()

    with torch.no_grad():
        latent_data = encoder(input_tensor)

    for epoch in tqdm.tqdm(range(epochs)):
        for _, dev_data in enumerate(train_loader):
            optimizer.zero_grad()
            timestep = np.random.randint(min_timesteps, max_timesteps + 1)

            device_data = dev_data.to(torch.float32)
            device_data = device_data.to(device)

            with torch.no_grad():
                latent_data = encoder(device_data)

            mean = latent_data.mean(dim=1, keepdim=True)
            std = latent_data.std(dim=1, keepdim=True)
    
            # latent_data = (latent_data - mean) / (std + 1e-8)

            mu_q, sigma_q, xt = forward_process(
                timestep,
                alphas_bar,
                latent_data,
                betas,
                alphas
            )

            # for net in diffusion_model.network_tail:
                # for param in net.parameters():
                    # param.requires_grad = False

            # for param in diffusion_model.network_tail[timestep - 1].parameters():
                # param.requires_grad = True

            mu_p, sigma_p, xt_minus1 = reverse_process(diffusion_model, timestep, xt, device)

            sigma_q = torch.ones_like(sigma_p) * sigma_q

            KL = torch.log(sigma_p) - torch.log(sigma_q) + (
                sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2)
            K = - KL.mean()
            loss = -K

            loss.backward()
            optimizer.step()

        if epoch % 1000 == 0:
            logger.info(f"PRETRAIN ERROR: EPOCH: {epoch} ERROR: {loss.item()}")

        if (epoch % 2000 == 0) and (epoch != 0):
            debug_plot(latent_data, device, logger, epoch, alphas_bar, betas, alphas, max_timesteps, diffusion_model, encoder, decoder)

    encoder.train()
