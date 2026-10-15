"""Bayesian VAR generator with stochastic volatility."""

import numpy as np

from bvar import BVAR


class BVAR_SV(BVAR):
    """BVAR whose correlated innovations have persistent stochastic scales."""

    def __init__(self, *args, sv_persistence=0.95, sv_vol_of_vol=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        if abs(sv_persistence) >= 1:
            raise ValueError("abs(sv_persistence) must be less than 1.")
        if sv_vol_of_vol <= 0:
            raise ValueError("sv_vol_of_vol must be positive.")
        self.sv_persistence = float(sv_persistence)
        self.sv_vol_of_vol = float(sv_vol_of_vol)

    def _draw_innovations(self, total, rng):
        covariance = self.bundle["covariance"]
        base_variance = np.clip(np.diag(covariance), 1e-12, None)
        standard_deviation = np.sqrt(base_variance)
        correlation = covariance / np.outer(standard_deviation, standard_deviation)
        correlation = (correlation + correlation.T) / 2
        try:
            cholesky = np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError:
            cholesky = np.linalg.cholesky(correlation + np.eye(len(base_variance)) * 1e-8)

        log_variance_mean = np.log(base_variance)
        log_variance = log_variance_mean.copy()
        innovations = np.empty((total, len(base_variance)))
        for time in range(total):
            log_variance = (
                log_variance_mean
                + self.sv_persistence * (log_variance - log_variance_mean)
                + self.sv_vol_of_vol * rng.standard_normal(len(base_variance))
            )
            correlated_shock = cholesky @ rng.standard_normal(len(base_variance))
            innovations[time] = np.exp(log_variance / 2) * correlated_shock
        return innovations