"""version 0.1.0"""
import torch


class Mlp(torch.nn.Module):
    """
    """
    def __init__(self, n=40):
        super().__init__()
        self.activation = torch.nn.Sigmoid()

        features_0 = 30
        features_1 = 40
        features_2 = 40
        features_3 = 40

        features_4 = 60
        features_5 = 60
        features_6 = 60

        self.layer_0 = torch.nn.Linear(features_0, features_1)
        self.layer_1 = torch.nn.Linear(features_1, features_2)
        self.layer_2 = torch.nn.Linear(features_2, features_3)

        self.network_tail = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(features_3, features_4),
                self.activation,
                torch.nn.Linear(features_4, features_5),
                self.activation,
                torch.nn.Linear(features_5, features_6)
            ) for _ in range(n)
        ])

    def forward(self, x, t):
        """
        """
        output_0 = self.layer_0(x)
        output_1 = self.activation(output_0)
        output_2 = self.layer_1(output_1)
        output_3 = self.activation(output_2)
        output_4 = self.layer_2(output_3)

        output_5 = self.network_tail[t - 1](output_4)
        mu, h = torch.chunk(output_5, 2, dim=0)

        var = torch.exp(h)
        std = torch.sqrt(var)

        return mu, std
