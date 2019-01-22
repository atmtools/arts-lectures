"""Perform an OEM retrieval and plot the results. """
import numpy as np
import matplotlib.pyplot as plt
from typhon.arts import xml
from typhon.plots import (cmap2rgba, profile_z, styles)
from scipy.linalg import inv


styles.use()


def retrieve(y, K, xa, ya, Sa, Sy):
    """Perform an OEM retrieval.

    Parameters:
        y (np.ndarray): Measuremed brightness temperature [K].
        K (np.ndarray): Jacobians [K/1].
        xa (np.ndarray): A priori state [VMR].
        ya (np.ndarray): Forward simulation of a priori state ``F(xa)`` [K].
        Sa (np.ndarray): A priori error covariance matrix.
        Sy (np.ndarray): Measurement covariance matrix

    Returns:
        np.ndarray: Retrieved atmospheric state.

    """
    raise NotImplementedError


def averaging_kernel_matrix(K, Sa, Sy):
    """Calculate the averaging kernel matrix.

    Parameters:
        K (np.ndarray): Simulated Jacobians.
        Sa (np.ndarray): A priori error covariance matrix.
        Sy (np.ndarray): Measurement covariance matrix.

    Returns:
        np.ndarray: Averaging kernel matrix.
    """
    raise NotImplementedError


# Load a priori information.
f_grid = xml.load('input/f_grid.xml')
x_apriori = xml.load('input/x_apriori.xml').get('abs_species-H2O', keep_dims=False)
y_apriori = xml.load('results/y_apriori.xml')
S_x = xml.load('input/S_x.xml')
S_y = xml.load('input/S_y.xml') * np.eye(f_grid.size)

# Load ARTS results.
z = xml.load('results/z_field.xml')
K = xml.load('results/jacobian.xml')

# Load y measurement.
y_measure = xml.load('input/y_measurement.xml')

# Plot the y measurement alongside the simulated y for the a priori.

# Plot the Jacobians.

# Plot the OEM result next to the true atmospheric state and the a priori.
x_oem = retrieve(y_measure, K, x_apriori, y_apriori, S_x, S_y)

# Plot the averaging kernels and the measurement response.
A = averaging_kernel_matrix(K, S_x, S_y)

plt.show()
