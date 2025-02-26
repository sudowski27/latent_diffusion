"""version 0.1.0"""
import torch


def reverse_process(diffusion_model: torch.nn.Module, time: int, x_t: torch.tensor):
    """
    """
    t = time - 1
    assert t >= 0, f"t= {t} reverse_process"

    mu, std = diffusion_model(x_t, t)
    epsilon = torch.randn_like(x_t)

    return mu, std, mu + epsilon * std
