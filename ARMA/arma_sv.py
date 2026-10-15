import numpy as np

from utils import is_stationary


def generate_arma_sv(ar=(), ma=(), n=400, n_series=1, c=0.0,
                     mu_x=0.0, rho=0.9, sigma_eta=0.3,
                     burn_in=100, seed=None):
    """Generates an ARMA-SV series

        Args:
            ar: list of floats representing AR coefficients
            ma: list of floats representing MA coefficients
            n (int): Length of each returned series (after burn-in is removed)
            n_series (int): How many independent series to generate differing only in random shocks
            c (float): Constant term
            mu_x (float): Long-run mean of the log-variance
            rho (float): Persistence of the log-variance (|rho| < 1)
            sigma_eta (float): Standard deviation of the log-variance shocks (> 0)
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
    if burn_in < 0:
        raise ValueError("burn_in cannot be negative.")
    if not is_stationary(ar):
        raise ValueError(
            f"AR coefficients {ar.tolist()} are non-stationary: the series would explode. All roots of z^p - phi_1 z^(p-1) - ... - phi_p must have absolute value < 1."
        )
    if abs(rho) >= 1:
        raise ValueError(f"|rho| = {abs(rho)} must be < 1, otherwise the volatility would wander off.")
    if sigma_eta <= 0:
        raise ValueError("sigma_eta must be positive.")

    rng = np.random.default_rng(seed)
    m = max(p, q, 1)
    total = m + burn_in + n
    mu = c / (1.0 - ar.sum())

    y = np.full((n_series, total), mu)
    x = np.full((n_series, total), mu_x) 
    eps = np.zeros((n_series, total))
    z = rng.standard_normal((n_series, total))
    eta = sigma_eta * rng.standard_normal((n_series, total))

    for t in range(m, total):
        x[:, t] = mu_x + rho * (x[:, t - 1] - mu_x) + eta[:, t]
        eps[:, t] = np.exp(x[:, t] / 2) * z[:, t]

        ar_part = sum(ar[i] * y[:, t - 1 - i] for i in range(p))
        ma_part = sum(ma[j] * eps[:, t - 1 - j] for j in range(q))
        y[:, t] = c + ar_part + eps[:, t] + ma_part

    return y[:, m + burn_in:]