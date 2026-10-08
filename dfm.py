"""
Code for DFM synthetic data generator

Design:
DFM -> Base class

"""

# Imports
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA

class DFM:
    def __init__(self, k_factors = 3, factor_order = 2 , maxiter = 80):
        """
        Parameters
        ----------
        k_factors : int
            Number of hidden common factors to extract
            MACROCAST uses 3 factors .
        factor_order : int
            Number of lags in the factor dynamics — the p in the VAR(p)/AR(p) that
            governs how the factors evolve over time (their momentum). MACROCAST uses 2.
        maxiter : int, default 80
            Cap on EM iterations when fitting DynamicFactorMQ (speed vs convergence
            trade-off). Raise it if you see non-convergence warnings
        """

        self.k_factors = k_factors
        self.factor_order = factor_order
        self.maxiter = maxiter
        self.bundle = None

    """
    Fitting the Dynamic Factor Model

    Parameters
    ----------
    y_df: pd.DataFrame
        The input data panel: rows = time (months), columns = variables/series.
        The data the DFM is fit on.

    Returns
    -------
    dict:
        Parameter bundle: loadings (B), idiosyncratic noise sizes (resid_std),
        factor dynamics (var_meta: coefs + shock covariance), historical factor
        path (factors_hist), and dimensions (k, p).
    """
    def fit_dfm(self, y_df):
        model = DynamicFactorMQ(y_df, factors = self.k_factors, factor_orders = self.factor_order)
        res = model.fit(disp = False, maxiter = self.maxiter)
        self.bundle = res

        factors_df = res.factors.smoothed
        if isinstance(factors_df, pd.Series):
            factors_df = factors_df.to_frame("f1")
        F = np.asarray(factors_df, dtype = float)

        Y = y_df.values.astype(float)

        B = np.linalg.lstsq(F, Y, rcond = None)[0]
        resid = Y - F @ B
        resid_std = np.std(resid, axis = 0)
        resid_std = np.where(resid_std < 1e-8, 1e-8, resid_std)

        if F.shape[1] == 1:
            f = pd.Series(F[:, 0])
            ar_fit = ARIMA(f, order = (self.factor_order, 0, 0), trend = 'n').fit()
            var_meta = {
                'kind': 'ar1d',
                'phi': np.asarray(getattr(ar_fit, 'arparams', np.array([])), dtype = float),
                'cov': np.array([[float(max(ar_fit.scale, 1e-6))]])
            }
        else:
            var_fit = VAR(pd.DataFrame(F)).fit(maxlags = self.factor_order, trend = 'n')
            var_meta = {
                'kind': 'var',
                'coefs': var_fit.coefs,
                'cov': var_fit.sigma_u.values
            }

        self.bundle = {
            'k': self.k_factors,
            'p': self.factor_order,
            'B': B,
            'resid_std': resid_std,
            'var_meta': var_meta,
            'factors_hist': F
        }

        return self


    """Generate Synthetic Data Pannels.
 
        Parameters
        ----------
        n_steps : int
            Length of the output panel.
        burn_in : int
            Extra steps rolled at the start and then thrown away.
            Used so the factors forget the all-zeros initial state.
        rng : np.random.Generator
            Random source.
        shock_scale : float
            Multiplier on the factor innovations. 
            How much the shared factors move around, which makes the dynamics up or down.
        resid_scale : float
            Multiplier on the per-series noise.
            How much each series wobbles on its own, independent of the shared factors.
        """
    def generate(self, n_steps, burn_in, rng, shock_scale = 1.0, resid_scale = 1.0):
        b = self.bundle
        B = b['B']
        k = b['k']
        p = b['p']
        resid_std = b['resid_std']
        var_meta = b['var_meta']
        total = n_steps + burn_in

        u = self._draw_factor_innovations(total, rng) * float(shock_scale)

        factors = np.zeros((total, k))
        if var_meta['kind'] == 'ar1d':
            phi = var_meta['phi']
            for t in range(total):
                v = u[t, 0]
                for lag in range(1, min(p, len(phi) + 1)):
                    if t - lag >= 0:
                        v += phi[lag - 1] * factors[t - lag, 0]
                factors[t, 0] = v
        else:
            A = var_meta['coefs']
            for t in range(total):
                v = u[t].copy()
                for lag in range(1, p + 1):
                    if t - lag >= 0:
                        v += A[lag - 1] @ factors[t - lag]
                factors[t] = v

        factors = factors[burn_in:]
        eps = rng.normal(0.0, resid_std, size = (n_steps, len(resid_std))) * float(resid_scale)
        return factors @ B + eps

    def generate_df(self, n_panels, n_steps = 400, burn_in = 60, rng = 60, seed = 0):
        rng = np.random.default_rng(seed)
        return [self.generate(n_steps, burn_in, rng) for _ in range(n_panels)]