import numpy as np
import pytest

from bvar import BVAR
from bvar_sv import BVAR_SV


@pytest.fixture
def panel():
    rng = np.random.default_rng(14)
    values = np.zeros((180, 3))
    for time in range(1, len(values)):
        values[time] = np.array(
            [[0.55, 0.08, 0.0], [0.05, 0.4, 0.1], [0.0, 0.05, 0.3]]
        ) @ values[time - 1] + rng.normal(0, 1, 3)
    return values


def test_bvar_fit_generate_shape_finite_reproducible(panel):
    model = BVAR(lags=2).fit(panel)
    first = model.generate(80, seed=7)
    second = model.generate(80, seed=7)
    assert first.shape == (80, 3)
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)
    assert 0.01 <= model.prior_scale_ <= 2.0


def test_bvar_generates_multiple_panels(panel):
    model = BVAR(lags=2).fit(panel)
    panels = model.generate_panels(3, n_steps=25, seed=2)
    assert len(panels) == 3
    assert all(values.shape == (25, 3) for values in panels)
    assert not np.array_equal(panels[0], panels[1])


def test_bvar_sv_overrides_innovation_draw_and_varies_volatility(panel):
    model = BVAR_SV(lags=2, sv_vol_of_vol=0.8).fit(panel)
    shocks = model._draw_innovations(300, np.random.default_rng(3))
    assert shocks.shape == (300, 3)
    assert np.all(np.isfinite(shocks))
    first_half = shocks[:150].std(axis=0)
    second_half = shocks[150:].std(axis=0)
    assert np.any(np.abs(first_half - second_half) > 0.05)


@pytest.mark.parametrize("data", [np.ones(5), np.array([[1.0, np.nan]] * 8)])
def test_fit_rejects_invalid_panel(data):
    with pytest.raises(ValueError):
        BVAR(lags=2).fit(data)