"""version 0.1.0"""
from torch import nn


class Decoder(nn.Module):
    """
    """
    def __init__(self):
        super().__init__()
        features_0 = 30
        features_1 = 300
        features_2 = 1378
        self.layer_0 = nn.Linear(features_0, features_1)
        self.layer_1 = nn.Linear(features_1, features_2)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        """
        output_0 = self.layer_0(x)
        output_1 = self.activation(output_0)
        output_2 = self.layer_1(output_1)
        output_3 = self.activation(output_2)

        output = output_3
        return output
