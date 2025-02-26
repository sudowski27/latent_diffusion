"""version 0.1.0"""
from torch import nn


class Decoder(nn.Module):
    """
    """
    def __init__(self):
        super().__init__()
        features_0 = 100
        features_1 = 800
        features_2 = 1378
        features_3 = 800
        features_4 = 1000
        features_5 = 1200
        features_6 = 1378
        self.layer_0 = nn.Linear(features_0, features_1)
        self.layer_1 = nn.Linear(features_1, features_2)
        # self.layer_2 = nn.Linear(features_2, features_3)
        # self.layer_3 = nn.Linear(features_3, features_4)
        # self.layer_4 = nn.Linear(features_4, features_5)
        # self.layer_5 = nn.Linear(features_5, features_6)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        """
        output_0 = self.layer_0(x)
        output_1 = self.activation(output_0)
        output_2 = self.layer_1(output_1)
        output_3 = self.activation(output_2)
        # output_4 = self.layer_2(output_3)
        # output_5 = self.activation(output_4)
        # output_6 = self.layer_3(output_5)
        # output_7 = self.activation(output_6)
        # output_8 = self.layer_4(output_7)
        # output_9 = self.activation(output_8)
        # output_10 = self.layer_5(output_9)
        # output_11 = self.activation(output_10)

        output = output_3
        return output
