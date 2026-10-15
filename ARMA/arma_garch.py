import numpy as np

from utils import is_stationary


def generate_arma_garch(ar=(), ma=(), n=400, n_series=1, c=0.0,
                        omega=0.1, alpha=0.1, beta=0.8,
                        burn_in=100, seed=None):
    """Generates an ARMA-GARCH(1,1) series

        Args:
            ar: list of floats representing AR coefficients
            ma: list of floats representing MA coefficients
            n (int): Length of each returned series (after burn-in is removed)
            n_series (int): How many independent series to generate differing only in random shocks
            c (float): Constant term
            omega (float): Baseline variance (must be > 0)
            alpha (float): Reaction of variance to last period's squared shock (>= 0)
            beta (float): Persistence of variance (>= 0)
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
    if omega <= 0 or alpha < 0 or beta < 0:
        raise ValueError("Need omega > 0, alpha >= 0 and beta >= 0.")
    if alpha + beta >= 1:
        raise ValueError(
            f"alpha + beta = {alpha + beta} must be < 1, otherwise the variance would explode."
        )

    rng = np.random.default_rng(seed)
    m = max(p, q, 1)
    total = m + burn_in + n
    mu = c / (1.0 - ar.sum())
    h_bar = omega / (1.0 - alpha - beta)

    y = np.full((n_series, total), mu)
    h = np.full((n_series, total), h_bar)
    eps = np.zeros((n_series, total))
    z = rng.standard_normal((n_series, total))

    for t in range(m, total):
        h[:, t] = omega + alpha * eps[:, t - 1] ** 2 + beta * h[:, t - 1]
        eps[:, t] = np.sqrt(h[:, t]) * z[:, t]

        # ARMA: identical to generate_arma
        ar_part = sum(ar[i] * y[:, t - 1 - i] for i in range(p))
        ma_part = sum(ma[j] * eps[:, t - 1 - j] for j in range(q))
        y[:, t] = c + ar_part + eps[:, t] + ma_part

    return y[:, m + burn_in:]