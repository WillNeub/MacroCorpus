"""
Tests and visualizations for arma_sv.py.

    pytest test_arma_sv.py -v    # run tests
    python test_arma_sv.py       # show plots
"""

import numpy as np
import pytest

from arma_sv import generate_arma_sv


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


def log_var_variance(rho, sigma_eta):
    """Variance of the log-variance x_t: V = sigma_eta^2 / (1 - rho^2)."""
    return sigma_eta**2 / (1 - rho**2)


def sv_variance(mu_x, rho, sigma_eta):
    """Average shock variance: exp(mu_x + V / 2)."""
    return np.exp(mu_x + log_var_variance(rho, sigma_eta) / 2)


def sv_kurtosis(rho, sigma_eta):
    """Kurtosis of SV shocks: 3 * exp(V)."""
    return 3 * np.exp(log_var_variance(rho, sigma_eta))


def sv_sq_acf(rho, sigma_eta, max_lag):
    """Autocorrelation of squared shocks: (exp(V rho^k) - 1) / (3 exp(V) - 1)."""
    V = log_var_variance(rho, sigma_eta)
    return np.array([1.0] + [(np.exp(V * rho**k) - 1) / (3 * np.exp(V) - 1)
                             for k in range(1, max_lag + 1)])


def unit_variance_mu_x(rho, sigma_eta):
    """mu_x that makes the average shock variance exactly 1."""
    return -log_var_variance(rho, sigma_eta) / 2


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

# SV-only cases (no AR/MA, so the output IS the shock series): (name, mu_x, rho, sigma_eta)
SV_CASES = [
    ("base",       0.0,  0.90, 0.30),
    ("persistent", 0.0,  0.97, 0.15),
    ("volatile",  -0.5,  0.80, 0.50),
]
SV_IDS = [case[0] for case in SV_CASES]

# Full ARMA-SV cases: (name, ar, ma, c, mu_x, rho, sigma_eta)
ARMA_SV_CASES = [
    ("ar1",    [0.5],       [],     1.0, 0.0, 0.90, 0.30),
    ("ar2",    [0.5, 0.2],  [],     0.5, 0.0, 0.90, 0.30),
    ("ma1",    [],          [0.6],  2.0, 0.0, 0.90, 0.30),
    ("arma11", [0.7],       [0.4],  0.0, 0.5, 0.80, 0.50),
]
ARMA_SV_IDS = [case[0] for case in ARMA_SV_CASES]

N, N_SERIES, SEED = 5000, 200, 123


@pytest.fixture(scope="module")
def sv_sim():
    return {
        name: generate_arma_sv(mu_x=mx, rho=r, sigma_eta=s, n=N, n_series=N_SERIES, seed=SEED)
        for name, mx, r, s in SV_CASES
    }


@pytest.fixture(scope="module")
def arma_sv_sim():
    return {
        name: generate_arma_sv(ar=ar, ma=ma, c=c, mu_x=mx, rho=r, sigma_eta=s,
                               n=N, n_series=N_SERIES, seed=SEED)
        for name, ar, ma, c, mx, r, s in ARMA_SV_CASES
    }


# ---------------------------------------------------------------------------
# Output behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, n_series", [(1, 1), (10, 1), (400, 5), (50, 100)])
def test_output_shape(n, n_series):
    data = generate_arma_sv(ar=[0.5], ma=[0.3], n=n, n_series=n_series, seed=0)
    assert data.shape == (n_series, n)


def test_pure_sv_shape():
    """No AR/MA terms still works (m is forced to at least 1)."""
    assert generate_arma_sv(n=10, seed=0).shape == (1, 10)


@pytest.mark.parametrize("name, ar, ma, c, mx, r, s", ARMA_SV_CASES, ids=ARMA_SV_IDS)
def test_output_is_finite(arma_sv_sim, name, ar, ma, c, mx, r, s):
    assert np.all(np.isfinite(arma_sv_sim[name]))


def test_same_seed_reproducible():
    a = generate_arma_sv(ar=[0.5], ma=[0.3], n=100, n_series=3, seed=7)
    b = generate_arma_sv(ar=[0.5], ma=[0.3], n=100, n_series=3, seed=7)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    a = generate_arma_sv(ar=[0.5], n=100, seed=1)
    b = generate_arma_sv(ar=[0.5], n=100, seed=2)
    assert not np.allclose(a, b)


def test_series_in_batch_differ():
    data = generate_arma_sv(ar=[0.5], n=100, n_series=2, seed=0)
    assert not np.allclose(data[0], data[1])


# ---------------------------------------------------------------------------
# SV properties (shock series)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, mx, r, s", SV_CASES, ids=SV_IDS)
def test_shock_variance(sv_sim, name, mx, r, s):
    assert sv_sim[name].var() == pytest.approx(sv_variance(mx, r, s), rel=0.05)


@pytest.mark.parametrize("name, mx, r, s", SV_CASES, ids=SV_IDS)
def test_fat_tails(sv_sim, name, mx, r, s):
    k = sample_kurtosis(sv_sim[name])
    assert k > 3.0
    assert k == pytest.approx(sv_kurtosis(r, s), rel=0.05)


@pytest.mark.parametrize("name, mx, r, s", SV_CASES, ids=SV_IDS)
def test_shocks_uncorrelated(sv_sim, name, mx, r, s):
    """Shocks themselves have no autocorrelation; only their size clusters."""
    np.testing.assert_allclose(sample_acf(sv_sim[name], 5)[1:], 0.0, atol=0.02)


@pytest.mark.parametrize("name, mx, r, s", SV_CASES, ids=SV_IDS)
def test_squared_shock_acf(sv_sim, name, mx, r, s):
    """Volatility clustering: squared shocks are positively autocorrelated, as theory predicts."""
    sample = sample_acf(sv_sim[name] ** 2, 5)[1:]
    assert np.all(sample > 0)
    np.testing.assert_allclose(sample, sv_sq_acf(r, s, 5)[1:], atol=0.02)


def test_tiny_sigma_eta_is_constant_variance():
    """sigma_eta -> 0 freezes the volatility at exp(mu_x): normal shocks, no clustering."""
    e = generate_arma_sv(mu_x=np.log(2.0), rho=0.9, sigma_eta=1e-6,
                         n=N, n_series=N_SERIES, seed=SEED)
    assert e.var() == pytest.approx(2.0, rel=0.02)
    assert sample_kurtosis(e) == pytest.approx(3.0, abs=0.05)
    np.testing.assert_allclose(sample_acf(e**2, 3)[1:], 0.0, atol=0.01)


def test_unit_variance_mu_x():
    """Choosing mu_x = -V/2 gives an average shock variance of 1."""
    r, s = 0.9, 0.4
    e = generate_arma_sv(mu_x=unit_variance_mu_x(r, s), rho=r, sigma_eta=s,
                         n=N, n_series=N_SERIES, seed=SEED)
    assert e.var() == pytest.approx(1.0, rel=0.05)


# ---------------------------------------------------------------------------
# ARMA mean dynamics are unchanged by SV
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, ar, ma, c, mx, r, s", ARMA_SV_CASES, ids=ARMA_SV_IDS)
def test_mean(arma_sv_sim, name, ar, ma, c, mx, r, s):
    data = arma_sv_sim[name]
    long_run_var = sv_variance(mx, r, s) * (1 + np.sum(ma)) ** 2 / (1 - np.sum(ar)) ** 2
    tol = 5 * np.sqrt(long_run_var / data.size)
    assert abs(data.mean() - c / (1 - np.sum(ar))) < tol


@pytest.mark.parametrize("name, ar, ma, c, mx, r, s", ARMA_SV_CASES, ids=ARMA_SV_IDS)
def test_variance(arma_sv_sim, name, ar, ma, c, mx, r, s):
    """Var(y) = average shock variance * sum(psi^2)."""
    expected = sv_variance(mx, r, s) * np.sum(psi_weights(ar, ma) ** 2)
    assert arma_sv_sim[name].var(axis=1).mean() == pytest.approx(expected, rel=0.05)


@pytest.mark.parametrize("name, ar, ma, c, mx, r, s", ARMA_SV_CASES, ids=ARMA_SV_IDS)
def test_acf_matches_plain_arma(arma_sv_sim, name, ar, ma, c, mx, r, s):
    np.testing.assert_allclose(
        sample_acf(arma_sv_sim[name], 5)[1:], arma_acf(ar, ma, 5)[1:], atol=0.02
    )


def test_theory_helpers_match_closed_forms():
    assert sv_variance(0.0, 0.0, 1e-9) == pytest.approx(1.0)
    assert sv_kurtosis(0.0, 1e-9) == pytest.approx(3.0)
    assert sv_variance(unit_variance_mu_x(0.9, 0.3), 0.9, 0.3) == pytest.approx(1.0)
    np.testing.assert_allclose(arma_acf([0.8], [], 3), 0.8 ** np.arange(4))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(ar=[1.0]),
    dict(ar=[0.6, 0.5]),
    dict(rho=1.0),
    dict(rho=-1.0),
    dict(rho=1.5),
    dict(sigma_eta=0.0),
    dict(sigma_eta=-0.1),
    dict(n=0),
    dict(n_series=0),
    dict(burn_in=-1),
])
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        generate_arma_sv(**kwargs)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def rolling_std(x, window=30):
    """Centered rolling standard deviation; makes volatility clusters visible."""
    return np.sqrt(np.convolve(x**2, np.ones(window) / window, mode="same"))


def unit_var(rho, sigma_eta):
    """SV kwargs with an average shock variance of 1."""
    return dict(mu_x=unit_variance_mu_x(rho, sigma_eta), rho=rho, sigma_eta=sigma_eta)


def plot_examples():
    import matplotlib.pyplot as plt

    examples = [
        ("Near-constant variance (σ_η=0.01)", unit_var(0.9, 0.01)),
        ("SV ρ=0.9, σ_η=0.3",                 unit_var(0.9, 0.3)),
        ("Persistent ρ=0.98, σ_η=0.15",       unit_var(0.98, 0.15)),
        ("Volatile ρ=0.9, σ_η=0.5",           unit_var(0.9, 0.5)),
        ("AR(1) φ=0.5 + SV",                  dict(ar=[0.5], **unit_var(0.95, 0.25))),
        ("ARMA(1,1) c=1 + SV",                dict(ar=[0.7], ma=[0.4], c=1.0, **unit_var(0.95, 0.25))),
    ]

    # Figure 1: sample paths. All shock-only panels share an average variance of 1,
    # so differences in "burstiness" come purely from rho and sigma_eta.
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for ax, (title, kwargs) in zip(axes.flat, examples):
        y = generate_arma_sv(n=1000, n_series=1, seed=4, **kwargs)[0]
        ar, c = kwargs.get("ar", []), kwargs.get("c", 0.0)
        mu = c / (1 - np.sum(ar))
        ax.plot(y, linewidth=0.7, color="steelblue")
        ax.plot(mu + 2 * rolling_std(y - mu), color="crimson", linewidth=1.2)
        ax.plot(mu - 2 * rolling_std(y - mu), color="crimson", linewidth=1.2)
        ax.axhline(mu, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("t")
    fig.suptitle("ARMA-SV series (dashed = theoretical mean, red = ±2 × rolling 30-step std)")
    fig.tight_layout()

    # Figure 2: the SV fingerprint. Shocks are uncorrelated, squared shocks are not.
    r, s = 0.9, 0.3
    mx = unit_variance_mu_x(r, s)
    e = generate_arma_sv(mu_x=mx, rho=r, sigma_eta=s, n=5000, n_series=100, seed=5)
    lags = np.arange(16)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(lags, sample_acf(e, 15), color="lightsteelblue")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("ACF of shocks ε (≈ 0 beyond lag 0)")

    axes[1].bar(lags, sample_acf(e**2, 15), color="lightsteelblue", label="sample")
    axes[1].plot(lags, sv_sq_acf(r, s, 15), "o", color="crimson", markersize=4, label="theory")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("ACF of squared shocks ε² (clustering)")
    axes[1].legend()

    x = np.linspace(-6, 6, 400)
    axes[2].hist(e.ravel() / e.std(), bins=200, range=(-6, 6), density=True,
                 color="lightsteelblue", label=f"SV (kurtosis {sample_kurtosis(e):.2f})")
    axes[2].plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color="crimson", label="normal (kurtosis 3)")
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-5, 1)
    axes[2].set_title("Standardized shocks: fat tails")
    axes[2].legend()

    for ax in axes[:2]:
        ax.set_xlabel("lag")
    fig.suptitle(f"SV fingerprint (μ_x={mx:.3f}, ρ={r}, σ_η={s})")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_examples()