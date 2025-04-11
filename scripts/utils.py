import numpy as np


RANDOM_SEED = 23502
NUM_TEST = 1000000
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'


def runge_vec(xx: np.ndarray, parameter: float) -> float:
    """Compute the M-dimensional Runge function (vectorized)."""
    return 1 / (1 + parameter**2 * np.sum(xx**2, axis=1))




def compute_linf_error(yy_test, yy_interpolation):
    """Compute the l-inf of errors."""

    return np.max(np.abs(yy_test - yy_interpolation))

