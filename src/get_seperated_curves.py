"""version 0.1.0"""
import numpy as np


def get_seperated_curves(data: np.ndarray, dim_0: int, dim_1: int) -> np.ndarray:
    """
    """
    curves_list = []
    single_curve_list = []
    offset = 0
    current_offset = None
    dtype = np.float32

    for _ in range(dim_0):
        current_offset = offset
        for _ in range(dim_1):
            single_curve_list.append(data[current_offset])
            current_offset = current_offset + dim_0
        curves_list.append(single_curve_list.copy())
        single_curve_list.clear()
        offset = offset + 1

    curves_list = np.array(curves_list, dtype=dtype)

    return curves_list
