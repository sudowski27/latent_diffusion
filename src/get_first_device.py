"""version 0.1.0"""
import pandas as pd
import numpy as np


def get_first_device(dataframe: pd.DataFrame, device_dim: int, current_col: int) -> np.ndarray:
    """
    """
    current_data = dataframe[current_col].iloc[:device_dim].to_numpy()

    return current_data
