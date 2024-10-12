import numpy as np


def _soft_max_(x):
    """
    Soft max function with softmax-trick.

    Parameters:
    -----------
    x: ndarray
        One dimensional ndarray of values.
    Returns:
    --------
    ndarray
    """
    e = np.exp
    max_x = np.max(x)
    return e(x - max_x) / (np.sum(e(x - max_x)))
