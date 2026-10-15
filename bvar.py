"""Bayesian vector autoregression synthetic panel generator."""

import math

import numpy as np


def _multivariate_log_gamma(value, dimension):
    return (
        dimension * (dimension - 1) * math.log(math.pi) / 4
        + sum(math.lgamma(value - index / 2) for index in range(dimension))
    )


class BVAR:
    """Fit a conjugate Minnesota-prior BVAR and generate synthetic panels.

    Inputs are centered and scaled before fitting, so the prior is comparable
    across series. Generated panels are transformed back to the input scale.
    """

    def __init__(
        self,
        lags=6,
        prior_scale=0.2,
        lag_decay=1.0,
        intercept_scale=10.0,
        optimize_prior=True,
    ):
        if lags < 1:
            raise ValueError("lags must be at least 1.")
        if prior_scale <= 0 or lag_decay <= 0 or intercept_scale <= 0:
            raise ValueError("Prior scales and lag_decay must be positive.")
        self.lags = int(lags)
        self.prior_scale = float(prior_scale)
        self.lag_decay = float(lag_decay)
        self.intercept_scale = float(intercept_scale)
        self.optimize_prior = bool(optimize_prior)
        self.bundle = None

    def _design(self, values):
        rows = len(values) - self.lags
        regressors = np.ones((rows, 1 + self.lags * values.shape[1]))
        targets = values[self.lags:]
        for lag in range(1, self.lags + 1):
            start = 1 + (lag - 1) * values.shape[1]
            stop = start + values.shape[1]
            regressors[:, start:stop] = values[self.lags - lag : len(values) - lag]
        return regressors, targets

    def _prior(self, n_variables, scale):
        n_coefficients = 1 + self.lags * n_variables
        variances = np.empty(n_coefficients)
        variances[0] = self.intercept_scale**2
        mean = np.zeros((n_coefficients, n_variables))
        for lag in range(1, self.lags + 1):
            start = 1 + (lag - 1) * n_variables
            stop = start + n_variables
            variances[start:stop] = scale**2 / lag ** (2 * self.lag_decay)
            if lag == 1:
                mean[start:stop, :] = np.eye(n_variables)
        return variances, mean

    def _posterior(self, regressors, targets, scale, prior_df):
        n_variables = targets.shape[1]
        variances, prior_mean = self._prior(n_variables, scale)
        precision = regressors.T @ regressors + np.diag(1.0 / variances)
        right_hand = regressors.T @ targets + prior_mean / variances[:, None]
        coefficients = np.linalg.solve(precision, right_hand)
        residual_scale = (
            targets.T @ targets
            + prior_mean.T @ (prior_mean / variances[:, None])
            - coefficients.T @ precision @ coefficients
        )
        residual_scale = (residual_scale + residual_scale.T) / 2
        residual_scale += np.eye(n_variables) * 1e-10
        posterior_df = prior_df + len(targets)
        sign_precision, logdet_precision = np.linalg.slogdet(precision)
        sign_scale, logdet_scale = np.linalg.slogdet(residual_scale)
        if sign_precision <= 0 or sign_scale <= 0:
            raise np.linalg.LinAlgError("BVAR posterior scale is not positive definite.")
        logdet_prior_covariance = np.log(variances).sum()
        log_marginal = (
            -len(targets) * n_variables * math.log(math.pi) / 2
            + n_variables * (-logdet_precision - logdet_prior_covariance) / 2
            - posterior_df * logdet_scale / 2
            + _multivariate_log_gamma(posterior_df / 2, n_variables)
            - _multivariate_log_gamma(prior_df / 2, n_variables)
        )
        return coefficients, residual_scale, posterior_df, log_marginal

    def fit(self, data):
        """Fit the BVAR to a time-by-variable array or DataFrame."""
        values = np.asarray(data, dtype=float)
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError("data must be a two-dimensional time-by-variable panel.")
        if values.shape[0] <= self.lags + 1:
            raise ValueError("data must contain more observations than lags + 1.")
        if not np.all(np.isfinite(values)):
            raise ValueError("data must contain only finite values.")

        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_ = np.where(self.scale_ < 1e-8, 1.0, self.scale_)
        standardized = (values - self.mean_) / self.scale_
        regressors, targets = self._design(standardized)
        prior_df = values.shape[1] + 2.0

        if self.optimize_prior:
            candidates = np.geomspace(0.01, 2.0, 32)
            fits = [
                self._posterior(regressors, targets, candidate, prior_df)
                for candidate in candidates
            ]
            best = int(np.argmax([result[3] for result in fits]))
            self.prior_scale_ = float(candidates[best])
            coefficients, posterior_scale, posterior_df, _ = fits[best]
        else:
            self.prior_scale_ = self.prior_scale
            coefficients, posterior_scale, posterior_df, _ = self._posterior(
                regressors, targets, self.prior_scale_, prior_df
            )

        covariance = posterior_scale / (posterior_df - values.shape[1] - 1)
        covariance = (covariance + covariance.T) / 2
        covariance += np.eye(values.shape[1]) * 1e-10
        self.bundle = {
            "coefficients": coefficients,
            "covariance": covariance,
            "n_variables": values.shape[1],
            "lags": self.lags,
        }
        return self

    def _draw_innovations(self, total, rng):
        """Draw constant-covariance VAR innovations; overridable by BVAR-SV."""
        return rng.multivariate_normal(
            np.zeros(self.bundle["n_variables"]),
            self.bundle["covariance"],
            size=total,
        )

    def generate(self, n_steps, burn_in=60, rng=None, seed=None):
        """Generate one panel with shape ``(n_steps, n_variables)``."""
        if self.bundle is None:
            raise RuntimeError("Call fit(data) before generate().")
        if n_steps < 1 or burn_in < 0:
            raise ValueError("n_steps must be positive and burn_in cannot be negative.")
        if rng is not None and seed is not None:
            raise ValueError("Pass either rng or seed, not both.")
        rng = rng if rng is not None else np.random.default_rng(seed)
        total = int(n_steps) + int(burn_in)
        innovations = self._draw_innovations(total, rng)
        coefficients = self.bundle["coefficients"]
        history = np.zeros((self.lags, self.bundle["n_variables"]))
        generated = np.empty((total, self.bundle["n_variables"]))

        for time in range(total):
            regressors = np.concatenate(
                [np.ones(1)] + [history[-lag] for lag in range(1, self.lags + 1)]
            )
            generated[time] = regressors @ coefficients + innovations[time]
            history[:-1] = history[1:]
            history[-1] = generated[time]

        generated = generated[int(burn_in) :]
        return generated * self.scale_ + self.mean_

    def generate_panels(self, n_panels, n_steps=400, burn_in=60, seed=0):
        """Generate multiple panels from a shared reproducible random stream."""
        if n_panels < 1:
            raise ValueError("n_panels must be at least 1.")
        rng = np.random.default_rng(seed)
        return [
            self.generate(n_steps, burn_in=burn_in, rng=rng)
            for _ in range(n_panels)
        ]