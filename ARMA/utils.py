import numpy as np

def is_stationary(ar):
    """Returns true if AR coefficients give a stationary series

    Args:
        ar: List of weights in the AR

    Returns:
        bool: True if weights lead to a stationary series
    """
    ar = np.asarray(ar, dtype=float)
    # roots: [1, -phi_1, -phi_2, ..., -phi_p]
    roots = np.roots(np.concatenate(([1.0], -ar)))
    return np.all(np.abs(roots) < 1)