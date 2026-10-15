"""
Code for DFM-SV synthetic data generator

Design:
DFM    -> Base class (constant-variance factor shocks)
DFM_SV -> DFM with stochastic volatility 
"""
import numpy as np
from dfm import DFM

class DFM_SV(DFM):
    def __init__(self, k_factors = 3, factor_order = 2, maxiter = 80, sv_persistence = 0.95, sv_vol_of_vol = 0.2):
        """
        Parameters
        ----------
        k_factors, factor_order, maxiter
            Same as DFM
        sv_persistence: float
            phi of the log-volatility AR(1): how sticky volatility is.
        sv_vol_of_vol: float
            sigma of the log-volatility AR(1): how violently volatility itself moves. 
        """
        super().__init__(k_factors, factor_order, maxiter)
        self.sv_persistence = sv_persistence
        self.sv_vol_of_vol = sv_vol_of_vol

    def _draw_factor_innovations(self, total, rng):
        # Overrides the _draw_factor_innovations in DFM

        # Grab the found cov of the factors
        cov = np.asarray(self.bundle['var_meta']['cov'], dtype = float)
        # Grab number of factors
        k = cov.shape[0]

        # Split cov into base variances + correlation
        # We only want to change the size of each cov, not the correlation

        # The variation size
        base_var = np.clip(np.diag(cov), 1e-12, None)
        d = np.sqrt(base_var)
        # The pure correlation matrix
        corr = cov / np.outer(d, d)
        corr = np.clip(corr, -1.0, 1.0)
        try:
            L = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            L = np.eye(k)

        # Log-vol AR(1) settings
        # Tracking the variation in log space to avoid negative
        mu = np.log(base_var)
        phi = self.sv_persistence
        sig = self.sv_vol_of_vol

        # Roll the vol path forward and draw each step's shock
        u = np.zeros((total, k))
        h = mu.copy()
        for t in range(total):
            # Update the vol
            h = mu + phi * (h - mu) + sig * rng.standard_normal(k)
            # Draw the cov shock
            z = L @ rng.standard_normal(k)
            # Scale by shock value
            u[t] = np.exp(h / 2.0) * z
        return u