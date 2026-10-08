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

def generate_arma(ar=(), ma=(), n=400, n_series=1, c=0.0, sigma=1.0,
                  burn_in=100, seed=None):
    """Generates an ARMA series
    
        Args:
            ar: list of floats representing AR coefficients
            ma: list of floats representing MA coefficients
            n (int): Length of each returned series (after burn-in is removed)
            n_series (int): How many independent series to generate differing only in random shocks
            c (float): Constant term
            sigma (float): Standard deviation of the shocks
            burn_in (int): Number of initial steps to simulate and then throw away
            seed(int | None): Random seed
    
        Returns:
            np.ndarray: Shape (n_series, n) array of generated series
    """
    ar = np.asarray(ar, dtype=float)
    ma = np.asarray(ma, dtype=float)
    p, q = ar.size, ma.size
    if n < 1 or n_series < 1:
        raise ValueError("n and n_series must be at least 1.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if burn_in < 0:
        raise ValueError("burn_in cannot be negative.")
    if not is_stationary(ar):
        raise ValueError(
            f"AR coefficients {ar.tolist()} are non-stationary: the series would explode. All roots of z^p - phi_1 z^(p-1) - ... - phi_p must have absolute value < 1."
        )

    rng = np.random.default_rng(seed)
    m = max(p, q)
    total = m + burn_in + n
    mu = c / (1.0 - ar.sum())
    y = np.full((n_series, total), mu)
    eps = np.zeros((n_series, total))
    eps[:, m:] = sigma * rng.standard_normal((n_series, total - m))

    for t in range(m, total):
        ar_part = sum(ar[i] * y[:, t - 1 - i] for i in range(p))
        ma_part = sum(ma[j] * eps[:, t - 1 - j] for j in range(q))
        y[:, t] = c + ar_part + eps[:, t] + ma_part

    return y[:, m + burn_in:]