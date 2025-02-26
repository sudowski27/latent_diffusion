"""version 0.1.0"""
import torch


class TorchDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, dataframe, current_col, device_dim):
        super().__init__()
        current_data_0 = dataframe[current_col].iloc[:].to_numpy()
        current_data_1 = current_data_0.reshape(-1, device_dim)
        assert current_data_1.shape[1] == device_dim, f"current_data_1.shape[1]: {current_data_1.shape[1]}"

        self.data = current_data_1
        self.length = self.data.shape[0]

    def __len__(self):
        """
        """
        return self.length

    def __getitem__(self, key):
        """
        """
        return self.data[key]
