"""version 0.1.0"""
import pandas as pd
import numpy as np


def get_20_devices(dataframe: pd.DataFrame, device_dim: int, current_col: int) -> np.ndarray:
    """
    """
    devices = 20
    current_data = dataframe[current_col].iloc[:device_dim * devices].to_numpy()

    current_data = current_data.reshape(-1, device_dim)
    assert current_data.shape[0] == devices

    return current_data
