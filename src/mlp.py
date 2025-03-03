"""version 0.1.0"""
import torch


class Mlp(torch.nn.Module):
    """
    """
    def __init__(self, n=40):
        super().__init__()
        # self.activation = torch.nn.SiLU()
        self.activation = torch.nn.Sigmoid()

        features_0 = 30
        features_1 = 960
        features_2 = 100
        features_3 = 40

        features_4 = 960
        features_5 = 60
        features_6 = 60

        self.layer_0 = torch.nn.Linear(features_0, features_1)
        self.batchnorm_0 = torch.nn.BatchNorm1d(features_1)
        self.layer_1 = torch.nn.Linear(features_1, features_1)
        # self.batchnorm_1 = torch.nn.BatchNorm1d(features_2)

        self.network_tail = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(features_1, features_4),
                self.activation,
                torch.nn.Linear(features_4, features_5)
            ) for _ in range(n)
        ])

    def forward(self, x, t):
        """
        """
        output_0 = self.layer_0(x)
        # output_1 = self.batchnorm_0(output_0)
        output_2 = self.activation(output_0)
        output_3 = self.layer_1(output_2)
        output_5 = self.activation(output_3)
        # output_4 = self.batchnorm_1(output_3)
        # output_5 = self.activation(output_4)
        # output_4 = self.layer_2(output_3)

        output_6 = self.network_tail[t - 1](output_5)
        mu, h = torch.chunk(output_6, 2, dim=1)

        var = torch.exp(h)
        std = torch.sqrt(var)

        return mu, std
