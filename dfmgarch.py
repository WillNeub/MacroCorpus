"""
Code for DFM-GARCH synthetic data generator

Design:
DFM -> Base class (constant-variance factor shocks)
DFM_GARCH -> DFM with GARCH(1,1)
"""
import numpy as np
from dfm import DFM


class DFM_GARCH(DFM):
    def __init__(self, k_factors = 3, factor_order = 2, maxiter = 80, garch_alpha = 0.10, garch_beta = 0.85):
        """
        Parameters
        ----------
        k_factors, factor_order, maxiter
            Same as DFM.
        garch_alpha: float
            Reaction to recent shocks: how much a big shock today raises tomorrow's variance
        garch_beta: float
            Persistence of variance: how much of yesterday's variance carries forward.
        """
        super().__init__(k_factors, factor_order, maxiter)
        self.garch_alpha = garch_alpha
        self.garch_beta = garch_beta

    def _draw_factor_innovations(self, total, rng):
        cov = np.asarray(self.bundle['var_meta']['cov'], dtype = float)
        k = cov.shape[0]

        # Split cov into base variances (size) + correlation (shape).
        # Grabs the variation size
        base_var = np.clip(np.diag(cov), 1e-12, None)
        d = np.sqrt(base_var)
        # The pure correlation matrix
        corr = cov / np.outer(d, d)
        corr = np.clip(corr, -1.0, 1.0)
        try:
            L = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            L = np.eye(k)

        # base_var: base_var = omega / (1 - alpha - beta).
        alpha = self.garch_alpha
        beta = self.garch_beta
        omega = base_var * max(1.0 - alpha - beta, 1e-6)

        # Roll the variance recursion forward and draw each step's shock
        u = np.zeros((total, k))
        var = base_var.copy()
        prev_shock2 = base_var.copy()
        for t in range(total):
            var = omega + alpha * prev_shock2 + beta * var
            z = L @ rng.standard_normal(k)
            eps = np.sqrt(var) * z
            u[t] = eps
            prev_shock2 = eps ** 2
        return u