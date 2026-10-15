"""
Tests and visualizations for arma_garch.py.

    pytest test_arma_garch.py -v    # run tests
    python test_arma_garch.py       # show plots
"""

import numpy as np
import pytest

from ARMA.arma_garch import generate_arma_garch


# ---------------------------------------------------------------------------
# Theory helpers
# ---------------------------------------------------------------------------

def psi_weights(ar, ma, k_max=2000):
    """Impulse response of the ARMA mean: effect of a unit shock at lags 0..k_max-1."""
    psi = np.zeros(k_max)
    psi[0] = 1.0
    for k in range(1, k_max):
        theta_k = ma[k - 1] if k <= len(ma) else 0.0
        psi[k] = theta_k + sum(ar[i] * psi[k - 1 - i] for i in range(len(ar)) if k - 1 - i >= 0)
    return psi


def garch_variance(omega, alpha, beta):
    """Long-run shock variance: omega / (1 - alpha - beta)."""
    return omega / (1 - alpha - beta)


def garch_kurtosis(alpha, beta):
    """Kurtosis of GARCH(1,1) shocks (needs 1 - (alpha+beta)^2 - 2 alpha^2 > 0)."""
    s2 = (alpha + beta) ** 2
    return 3 * (1 - s2) / (1 - s2 - 2 * alpha**2)


def garch_sq_acf(alpha, beta, max_lag):
    """Autocorrelation of squared shocks: rho_1 * (alpha+beta)^(k-1)."""
    rho1 = alpha * (1 - alpha * beta - beta**2) / (1 - 2 * alpha * beta - beta**2)
    return np.array([1.0] + [rho1 * (alpha + beta) ** (k - 1) for k in range(1, max_lag + 1)])


def arma_acf(ar, ma, max_lag):
    psi = psi_weights(ar, ma)
    gamma = np.array([np.sum(psi[: len(psi) - h] * psi[h:]) for h in range(max_lag + 1)])
    return gamma / gamma[0]


def sample_acf(data, max_lag):
    x = data - data.mean(axis=1, keepdims=True)
    denom = np.sum(x**2, axis=1)
    return np.array([
        np.mean(np.sum(x[:, h:] * x[:, : x.shape[1] - h], axis=1) / denom)
        for h in range(max_lag + 1)
    ])


def sample_kurtosis(data):
    x = data - data.mean()
    return np.mean(x**4) / np.mean(x**2) ** 2


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# GARCH-only cases (no AR/MA, so the output IS the shock series): (name, omega, alpha, beta)
GARCH_CASES = [
    ("base",       0.10, 0.10, 0.80),
    ("persistent", 0.05, 0.05, 0.90),
    ("reactive",   0.20, 0.15, 0.60),
]
GARCH_IDS = [case[0] for case in GARCH_CASES]

# Full ARMA-GARCH cases: (name, ar, ma, c, omega, alpha, beta)
ARMA_GARCH_CASES = [
    ("ar1",    [0.5],       [],     1.0, 0.10, 0.10, 0.80),
    ("ar2",    [0.5, 0.2],  [],     0.5, 0.10, 0.10, 0.80),
    ("ma1",    [],          [0.6],  2.0, 0.10, 0.10, 0.80),
    ("arma11", [0.7],       [0.4],  0.0, 0.20, 0.15, 0.60),
]
ARMA_GARCH_IDS = [case[0] for case in ARMA_GARCH_CASES]

N, N_SERIES, SEED = 5000, 200, 123


@pytest.fixture(scope="module")
def garch_sim():
    return {
        name: generate_arma_garch(omega=om, alpha=a, beta=b, n=N, n_series=N_SERIES, seed=SEED)
        for name, om, a, b in GARCH_CASES
    }


@pytest.fixture(scope="module")
def arma_garch_sim():
    return {
        name: generate_arma_garch(ar=ar, ma=ma, c=c, omega=om, alpha=a, beta=b,
                                  n=N, n_series=N_SERIES, seed=SEED)
        for name, ar, ma, c, om, a, b in ARMA_GARCH_CASES
    }


# ---------------------------------------------------------------------------
# Output behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, n_series", [(1, 1), (10, 1), (400, 5), (50, 100)])
def test_output_shape(n, n_series):
    data = generate_arma_garch(ar=[0.5], ma=[0.3], n=n, n_series=n_series, seed=0)
    assert data.shape == (n_series, n)


def test_pure_garch_shape():
    """No AR/MA terms still works (m is forced to at least 1)."""
    assert generate_arma_garch(n=10, seed=0).shape == (1, 10)


@pytest.mark.parametrize("name, ar, ma, c, om, a, b", ARMA_GARCH_CASES, ids=ARMA_GARCH_IDS)
def test_output_is_finite(arma_garch_sim, name, ar, ma, c, om, a, b):
    assert np.all(np.isfinite(arma_garch_sim[name]))


def test_same_seed_reproducible():
    a = generate_arma_garch(ar=[0.5], ma=[0.3], n=100, n_series=3, seed=7)
    b = generate_arma_garch(ar=[0.5], ma=[0.3], n=100, n_series=3, seed=7)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    a = generate_arma_garch(ar=[0.5], n=100, seed=1)
    b = generate_arma_garch(ar=[0.5], n=100, seed=2)
    assert not np.allclose(a, b)


def test_series_in_batch_differ():
    data = generate_arma_garch(ar=[0.5], n=100, n_series=2, seed=0)
    assert not np.allclose(data[0], data[1])


# ---------------------------------------------------------------------------
# GARCH properties (shock series)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, om, a, b", GARCH_CASES, ids=GARCH_IDS)
def test_shock_variance(garch_sim, name, om, a, b):
    assert garch_sim[name].var() == pytest.approx(garch_variance(om, a, b), rel=0.05)


@pytest.mark.parametrize("name, om, a, b", GARCH_CASES, ids=GARCH_IDS)
def test_fat_tails(garch_sim, name, om, a, b):
    k = sample_kurtosis(garch_sim[name])
    assert k > 3.0
    assert k == pytest.approx(garch_kurtosis(a, b), abs=0.15)


@pytest.mark.parametrize("name, om, a, b", GARCH_CASES, ids=GARCH_IDS)
def test_shocks_uncorrelated(garch_sim, name, om, a, b):
    """Shocks themselves have no autocorrelation; only their size clusters."""
    np.testing.assert_allclose(sample_acf(garch_sim[name], 5)[1:], 0.0, atol=0.02)


@pytest.mark.parametrize("name, om, a, b", GARCH_CASES, ids=GARCH_IDS)
def test_squared_shock_acf(garch_sim, name, om, a, b):
    """Volatility clustering: squared shocks are positively autocorrelated, as theory predicts."""
    sample = sample_acf(garch_sim[name] ** 2, 5)[1:]
    assert np.all(sample > 0)
    np.testing.assert_allclose(sample, garch_sq_acf(a, b, 5)[1:], atol=0.02)


def test_alpha_zero_is_constant_variance():
    """alpha = 0 removes the feedback: normal shocks, no clustering."""
    e = generate_arma_garch(omega=0.2, alpha=0.0, beta=0.8, n=N, n_series=N_SERIES, seed=SEED)
    assert e.var() == pytest.approx(1.0, rel=0.02)
    assert sample_kurtosis(e) == pytest.approx(3.0, abs=0.05)
    np.testing.assert_allclose(sample_acf(e**2, 3)[1:], 0.0, atol=0.01)


# ---------------------------------------------------------------------------
# ARMA mean dynamics are unchanged by GARCH
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, ar, ma, c, om, a, b", ARMA_GARCH_CASES, ids=ARMA_GARCH_IDS)
def test_mean(arma_garch_sim, name, ar, ma, c, om, a, b):
    data = arma_garch_sim[name]
    long_run_var = garch_variance(om, a, b) * (1 + np.sum(ma)) ** 2 / (1 - np.sum(ar)) ** 2
    tol = 5 * np.sqrt(long_run_var / data.size)
    assert abs(data.mean() - c / (1 - np.sum(ar))) < tol


@pytest.mark.parametrize("name, ar, ma, c, om, a, b", ARMA_GARCH_CASES, ids=ARMA_GARCH_IDS)
def test_variance(arma_garch_sim, name, ar, ma, c, om, a, b):
    """Var(y) = average shock variance * sum(psi^2)."""
    expected = garch_variance(om, a, b) * np.sum(psi_weights(ar, ma) ** 2)
    assert arma_garch_sim[name].var(axis=1).mean() == pytest.approx(expected, rel=0.05)


@pytest.mark.parametrize("name, ar, ma, c, om, a, b", ARMA_GARCH_CASES, ids=ARMA_GARCH_IDS)
def test_acf_matches_plain_arma(arma_garch_sim, name, ar, ma, c, om, a, b):
    np.testing.assert_allclose(
        sample_acf(arma_garch_sim[name], 5)[1:], arma_acf(ar, ma, 5)[1:], atol=0.02
    )


def test_theory_helpers_match_closed_forms():
    assert garch_variance(0.1, 0.1, 0.8) == pytest.approx(1.0)
    assert garch_kurtosis(0.0, 0.8) == pytest.approx(3.0)
    np.testing.assert_allclose(arma_acf([0.8], [], 3), 0.8 ** np.arange(4))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(ar=[1.0]),
    dict(ar=[0.6, 0.5]),
    dict(alpha=0.5, beta=0.5),
    dict(alpha=0.3, beta=0.8),
    dict(omega=0.0),
    dict(omega=-1.0),
    dict(alpha=-0.1),
    dict(beta=-0.1),
    dict(n=0),
    dict(n_series=0),
    dict(burn_in=-1),
])
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        generate_arma_garch(**kwargs)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def rolling_std(x, window=30):
    """Centered rolling standard deviation; makes volatility clusters visible."""
    return np.sqrt(np.convolve(x**2, np.ones(window) / window, mode="same"))


def plot_examples():
    import matplotlib.pyplot as plt

    examples = [
        ("Constant variance (α=0)",       dict(omega=0.2, alpha=0.0,  beta=0.8)),
        ("GARCH α=0.1, β=0.8",            dict(omega=0.1, alpha=0.1,  beta=0.8)),
        ("Reactive α=0.3, β=0.6",         dict(omega=0.1, alpha=0.3,  beta=0.6)),
        ("Persistent α=0.1, β=0.88",      dict(omega=0.02, alpha=0.1,  beta=0.88)),
        ("AR(1) φ=0.5 + GARCH",           dict(ar=[0.5], omega=0.1, alpha=0.1, beta=0.8)),
        ("ARMA(1,1) c=1 + GARCH",         dict(ar=[0.7], ma=[0.4], c=1.0,
                                               omega=0.1, alpha=0.15, beta=0.8)),
    ]

    # Figure 1: sample paths. All shock-only panels share a long-run variance of 1,
    # so differences in "burstiness" come purely from alpha and beta.
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for ax, (title, kwargs) in zip(axes.flat, examples):
        y = generate_arma_garch(n=1000, n_series=1, seed=4, **kwargs)[0]
        ar, c = kwargs.get("ar", []), kwargs.get("c", 0.0)
        mu = c / (1 - np.sum(ar))
        ax.plot(y, linewidth=0.7, color="steelblue")
        ax.plot(mu + 2 * rolling_std(y - mu), color="crimson", linewidth=1.2)
        ax.plot(mu - 2 * rolling_std(y - mu), color="crimson", linewidth=1.2)
        ax.axhline(mu, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("t")
    fig.suptitle("ARMA-GARCH series (dashed = theoretical mean, red = ±2 × rolling 30-step std)")
    fig.tight_layout()

    # Figure 2: the GARCH fingerprint. Shocks are uncorrelated, squared shocks are not.
    om, a, b = 0.1, 0.1, 0.8
    e = generate_arma_garch(omega=om, alpha=a, beta=b, n=5000, n_series=100, seed=5)
    lags = np.arange(16)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(lags, sample_acf(e, 15), color="lightsteelblue")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("ACF of shocks ε (≈ 0 beyond lag 0)")

    axes[1].bar(lags, sample_acf(e**2, 15), color="lightsteelblue", label="sample")
    axes[1].plot(lags, garch_sq_acf(a, b, 15), "o", color="crimson", markersize=4, label="theory")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("ACF of squared shocks ε² (clustering)")
    axes[1].legend()

    x = np.linspace(-6, 6, 400)
    axes[2].hist(e.ravel() / e.std(), bins=200, range=(-6, 6), density=True,
                 color="lightsteelblue", label=f"GARCH (kurtosis {sample_kurtosis(e):.2f})")
    axes[2].plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color="crimson", label="normal (kurtosis 3)")
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-5, 1)
    axes[2].set_title("Standardized shocks: fat tails")
    axes[2].legend()

    for ax in axes[:2]:
        ax.set_xlabel("lag")
    fig.suptitle(f"GARCH(1,1) fingerprint (ω={om}, α={a}, β={b})")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_examples()