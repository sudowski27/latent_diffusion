"""version 0.1.0"""
import torch
import matplotlib.pyplot as plt
from .visualize_1 import visualize_1


def debug_plot(latent_data: torch.Tensor, device: str, logger, epoch, alphas_bar, betas, alphas, TIMESTEPS, mlp_model, encoder, decoder):
    """"""
    if device == "cuda":
        latent_numpy = latent_data.cpu().numpy()
    if device == "cpu":
        latent_numpy = latent_data.numpy()

    # print(latent_numpy.shape[0])
    for i in range(latent_numpy.shape[0]):
        plt.plot(latent_numpy[i])
        plt.savefig(f"latent_diffusion_plots/latent_numpy_epoch_{epoch}_{i}.png")
        plt.close()

    for i in range(latent_numpy.shape[0]):
        dev_data = latent_numpy[i]
        dev_data = torch.tensor(dev_data, dtype=torch.float32, device=device)
        dev_data = dev_data.unsqueeze(0)
        # print(dev_data)
        out_tensor = visualize_1(
            dev_data,
            alphas_bar,
            betas,
            alphas,
            TIMESTEPS,
            mlp_model,
            encoder,
            decoder,
            device
        )

        # print(out_tensor)
        plots_count = 3
        fig, axs = plt.subplots(1, plots_count, figsize=(10, 4))
        for i in range(plots_count):
            data_to_plot = out_tensor[i]
            data_to_plot = data_to_plot.cpu().numpy()
            data_to_plot = data_to_plot.squeeze(0)
            axs[i].plot(data_to_plot)
        plt.tight_layout()
        plt.savefig(f"latent_diffusion_plots/latent_numpy_epoch_{epoch}_{i}_0.png")
        plt.close()
        plt.plot(dev_data.squeeze(0).cpu().numpy())
        plt.savefig(f"latent_diffusion_plots/latent_numpy_epoch_{epoch}_{i}_1.png")
        plt.close()
        # raise KeyboardInterrupt
        # plt.plot(out_tensor[i])
        # plt.savefig(f"latent_diffusion_plots/latent_numpy_epoch_{epoch}_{i}.png")
        # plt.close()

    logger.info(f"latent_numpy shape: {latent_numpy.shape}")
