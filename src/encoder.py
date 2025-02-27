"""version 0.1.0"""
from torch import nn


class Encoder(nn.Module):
    """
    """
    def __init__(self):
        super().__init__()
        features_0 = 1378
        features_1 = 300
        features_2 = 30
        self.layer_0 = nn.Linear(features_0, features_1)
        self.layer_1 = nn.Linear(features_1, features_2)

        self.activation = nn.ReLU()
        self.droput = nn.Dropout(p=0.1)

    def forward(self, x):
        """
        """
        output_0 = self.layer_0(x)
        output_1 = self.activation(output_0)
        output_2 = self.droput(output_1)
        output_3 = self.layer_1(output_2)
        output_4 = self.activation(output_3)
        output_5 = self.droput(output_4)

        output = output_5
        return output
