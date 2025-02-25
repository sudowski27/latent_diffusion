"""version 0.1.0"""
import torch


def forward_process(time: int, alphas_bar: torch.tensor, x_0: torch.tensor, betas: torch.tensor, alphas: torch.tensor):
    """
    """
    assert time > 0
    t = time - 1
    assert t - 1 >= 0
    mu = torch.sqrt(alphas_bar[t]) * x_0
    std = torch.sqrt(1 - alphas_bar[t])
    epsilon = torch.randn_like(x_0)
    x_t = mu + epsilon * std

    std_q = torch.sqrt((1 - alphas_bar[t - 1])/ (1 - alphas_bar[t]) * betas[t])
    m1 = torch.sqrt(alphas_bar[t-1]) * betas[t] / (1 - alphas_bar[t])
    m2 = torch.sqrt(alphas[t]) * (1 - alphas_bar[t-1]) / (1 - alphas_bar[t])

    mu_q = m1 * x_0 + m2 * x_t

    return mu_q, std_q, x_t
