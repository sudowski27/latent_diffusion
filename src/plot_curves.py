"""version 0.1.0"""
import numpy as np
import matplotlib.pyplot as plt


def plot_curves(seperated_curves: np.ndarray):
    """
    """
    number_of_plots =  seperated_curves.shape[0]

    for i in range(number_of_plots):
        plt.plot(seperated_curves[i], 'b')

    plt.ylabel("Current [A]")

    plt.show()
