"""
Tests and visualizations for arma_generator.py.

    pytest test_arma_generator.py -v    # run tests
    python test_arma_generator.py       # show plots
"""

import numpy as np
import pytest

from arma import generate_arma, is_stationary


# ---------------------------------------------------------------------------
# Theory helpers
# ---------------------------------------------------------------------------

def psi_weights(ar, ma, k_max=2000):
    """Impulse response: effect of a unit shock at lags 0..k_max-1."""
    psi = np.zeros(k_max)
    psi[0] = 1.0
    for k in range(1, k_max):
        theta_k = ma[k - 1] if k <= len(ma) else 0.0
        psi[k] = theta_k + sum(ar[i] * psi[k - 1 - i] for i in range(len(ar)) if k - 1 - i >= 0)
    return psi


def theoretical_mean(ar, c):
    return c / (1.0 - np.sum(ar))


def theoretical_variance(ar, ma, sigma):
    return sigma**2 * np.sum(psi_weights(ar, ma) ** 2)


def theoretical_acf(ar, ma, max_lag):
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


# ---------------------------------------------------------------------------
# Test cases: (name, ar, ma, c, sigma)
# ---------------------------------------------------------------------------

CASES = [
    ("white_noise",    [],          [],          0.0, 1.0),
    ("ar1",            [0.5],       [],          0.5, 1.0),
    ("ar1_negative",   [-0.5],      [],          0.0, 1.0),
    ("ar1_persistent", [0.9],       [],          1.0, 0.5),
    ("ar2",            [0.5, 0.2],  [],          0.5, 1.0),
    ("ar2_cycle",      [1.0, -0.5], [],          0.0, 1.0),
    ("ma1",            [],          [0.6],       2.0, 1.0),
    ("ma2",            [],          [0.5, -0.3], 0.0, 2.0),
    ("arma11",         [0.7],       [0.4],       0.0, 1.0),
    ("arma21",         [0.5, 0.2],  [0.4],       0.5, 1.0),
]
IDS = [case[0] for case in CASES]
N, N_SERIES, SEED = 2000, 200, 123


@pytest.fixture(scope="module")
def simulated():
    return {
        name: generate_arma(ar=ar, ma=ma, c=c, sigma=sigma, n=N, n_series=N_SERIES, seed=SEED)
        for name, ar, ma, c, sigma in CASES
    }


# ---------------------------------------------------------------------------
# Output behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, n_series", [(1, 1), (10, 1), (400, 5), (50, 100)])
def test_output_shape(n, n_series):
    data = generate_arma(ar=[0.5, 0.2], ma=[0.4], n=n, n_series=n_series, seed=0)
    assert data.shape == (n_series, n)


@pytest.mark.parametrize("name, ar, ma, c, sigma", CASES, ids=IDS)
def test_output_is_finite(simulated, name, ar, ma, c, sigma):
    assert np.all(np.isfinite(simulated[name]))


def test_same_seed_reproducible():
    a = generate_arma(ar=[0.5], ma=[0.3], n=100, n_series=3, seed=7)
    b = generate_arma(ar=[0.5], ma=[0.3], n=100, n_series=3, seed=7)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    a = generate_arma(ar=[0.5], n=100, seed=1)
    b = generate_arma(ar=[0.5], n=100, seed=2)
    assert not np.allclose(a, b)


def test_series_in_batch_differ():
    data = generate_arma(ar=[0.5], n=100, n_series=2, seed=0)
    assert not np.allclose(data[0], data[1])


def test_sigma_scales_output():
    a = generate_arma(ar=[0.5, 0.2], ma=[0.4], sigma=1.0, n=200, seed=3)
    b = generate_arma(ar=[0.5, 0.2], ma=[0.4], sigma=2.0, n=200, seed=3)
    np.testing.assert_allclose(b, 2 * a)


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, ar, ma, c, sigma", CASES, ids=IDS)
def test_mean(simulated, name, ar, ma, c, sigma):
    data = simulated[name]
    long_run_var = sigma**2 * (1 + np.sum(ma)) ** 2 / (1 - np.sum(ar)) ** 2
    tol = 5 * np.sqrt(long_run_var / data.size)
    assert abs(data.mean() - theoretical_mean(ar, c)) < tol


@pytest.mark.parametrize("name, ar, ma, c, sigma", CASES, ids=IDS)
def test_variance(simulated, name, ar, ma, c, sigma):
    sample_var = simulated[name].var(axis=1).mean()
    assert sample_var == pytest.approx(theoretical_variance(ar, ma, sigma), rel=0.05)


@pytest.mark.parametrize("name, ar, ma, c, sigma", CASES, ids=IDS)
def test_acf(simulated, name, ar, ma, c, sigma):
    np.testing.assert_allclose(
        sample_acf(simulated[name], 5)[1:], theoretical_acf(ar, ma, 5)[1:], atol=0.02
    )


def test_ma_acf_cuts_off_after_q():
    data = generate_arma(ma=[0.5, -0.3], n=N, n_series=N_SERIES, seed=SEED)
    np.testing.assert_allclose(sample_acf(data, 5)[3:], 0.0, atol=0.02)


def test_ar_coefficients_recovered_by_ols():
    true_phi = np.array([0.5, 0.2])
    y = generate_arma(ar=true_phi, n=20000, seed=SEED)[0]
    X = np.column_stack([y[1:-1], y[:-2], np.ones(len(y) - 2)])
    coef, *_ = np.linalg.lstsq(X, y[2:], rcond=None)
    np.testing.assert_allclose(coef[:2], true_phi, atol=0.02)


def test_theory_helpers_match_closed_forms():
    phi, theta = 0.8, 0.6
    assert theoretical_variance([phi], [], 1.0) == pytest.approx(1 / (1 - phi**2))
    np.testing.assert_allclose(theoretical_acf([phi], [], 3), phi ** np.arange(4))
    assert theoretical_variance([], [theta], 1.0) == pytest.approx(1 + theta**2)
    np.testing.assert_allclose(
        theoretical_acf([], [theta], 2), [1, theta / (1 + theta**2), 0], atol=1e-12
    )


# ---------------------------------------------------------------------------
# Stationarity and validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ar, expected", [
    ([],          True),
    ([0.5],       True),
    ([-0.5],      True),
    ([0.99],      True),
    ([1.0],       False),
    ([1.5],       False),
    ([-1.5],      False),
    ([0.5, 0.2],  True),
    ([0.6, 0.5],  False),
    ([1.0, -0.5], True),
])
def test_is_stationary(ar, expected):
    assert is_stationary(ar) == expected


@pytest.mark.parametrize("kwargs", [
    dict(ar=[1.0]),
    dict(ar=[0.6, 0.5]),
    dict(sigma=0.0),
    dict(sigma=-1.0),
    dict(n=0),
    dict(n_series=0),
    dict(burn_in=-1),
])
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        generate_arma(**kwargs)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_examples():
    import matplotlib.pyplot as plt

    examples = [
        ("White noise",               dict()),
        ("AR(1) φ=0.5",               dict(ar=[0.5])),
        ("AR(1) φ=0.95",              dict(ar=[0.95])),
        ("AR(1) φ=-0.5",              dict(ar=[-0.5])),
        ("AR(2) φ=[1.0, -0.5]",       dict(ar=[1.0, -0.5])),
        ("MA(2) θ=[0.5, -0.3]",       dict(ma=[0.5, -0.3])),
        ("ARMA(2,1) c=0.5",           dict(ar=[0.5, 0.2], ma=[0.4], c=0.5)),
        ("ARMA(1,1) σ=3",             dict(ar=[0.7], ma=[0.4], sigma=3.0)),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    for ax, (title, kwargs) in zip(axes.flat, examples):
        data = generate_arma(n=200, n_series=3, seed=1, **kwargs)
        for row in data:
            ax.plot(row, linewidth=0.9, alpha=0.8)
        ax.axhline(theoretical_mean(kwargs.get("ar", []), kwargs.get("c", 0.0)),
                   color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("t")
    fig.suptitle("Generated ARMA series (dashed line = theoretical mean)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_examples()