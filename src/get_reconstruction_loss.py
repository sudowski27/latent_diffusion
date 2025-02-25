"""version 0.1.0"""
import torch


def get_reconstruction_loss(input_tensor: torch.tensor, target_tensor: torch.tensor):
    """
    """
    norm_value = 1e-14

    input_tensor_log = torch.log(input_tensor / norm_value)
    target_tensor_log = torch.log(target_tensor / norm_value)

    mse_loss = torch.nn.MSELoss()

    loss = mse_loss(input_tensor_log, target_tensor_log)

    return loss
