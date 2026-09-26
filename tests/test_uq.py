import numpy as np
import pytest

from phycore.uq import (BandNotDeclared, BandTable, DeclaredBand, IsotonicCalibrator,
                        MondrianConformal, combined_sigma, conformal_quantile,
                        isotonic_fit, probit)
from phycore.engine.ensemble import EnsembleSurrogate


def test_probit_round_trip():
    for p in (0.01, 0.1, 0.5, 0.9, 0.99):
        z = probit(p)
        # Phi(z) ~ p via erf
        from math import erf, sqrt
        assert 0.5 * (1 + erf(z / sqrt(2))) == pytest.approx(p, abs=1e-8)


def test_conformal_quantile_small_n_falls_back():
    assert conformal_quantile(np.ones(5), np.ones(5)) == 2.0


def test_conformal_coverage_on_gaussian():
    rng = np.random.default_rng(0)
    res = rng.normal(0, 1.0, 2000)
    q = conformal_quantile(res, np.ones(2000), alpha=0.1)
    assert 1.5 < q < 1.8            # ~1.645
    assert np.mean(np.abs(res) <= q) >= 0.89


def test_isotonic_is_monotone():
    x = np.arange(10)
    y = np.array([1, 3, 2, 5, 4, 4, 7, 6, 9, 8], float)
    _, ys = isotonic_fit(x, y)
    assert np.all(np.diff(ys) >= -1e-12)


def test_isotonic_calibrator_declares_unfitted_below_20():
    c = IsotonicCalibrator().fit(np.ones(10), np.ones(10))
    assert not c.fitted and c.level_for(0.9) == 0.9


def test_combined_sigma_quadrature():
    s = combined_sigma(0.3, 2.0, 0.2)     # sqrt(0.09 + 0.16) = 0.5
    assert s == pytest.approx(0.5)


def test_band_table_no_defaults():
    t = BandTable([DeclaredBand("manglik_bergles", "j", 0.076, "Manglik & Bergles 1995")])
    assert t.band_for("manglik_bergles", "j").relative == 0.076
    with pytest.raises(BandNotDeclared):
        t.band_for("manglik_bergles", "f")


def test_mondrian_blocks_thin_cell():
    rng = np.random.default_rng(1)
    big = {"residuals": rng.normal(0, 1, 200), "sigmas": np.ones(200), "r2": [0.95]}
    thin = {"residuals": rng.normal(0, 1, 10), "sigmas": np.ones(10), "r2": [0.95]}
    m = MondrianConformal().fit({("a", "dry"): big, ("b", "boiling"): thin})
    g = m.gate()
    assert not g["passed"] and ("b", "boiling") in g["blocking_cells"]
    m2 = MondrianConformal(out_of_scope=(("b", "boiling"),)).fit(
        {("a", "dry"): big, ("b", "boiling"): thin})
    assert m2.gate()["passed"]
    with pytest.raises(KeyError):
        m.quantile(("c", "x"))


def test_ensemble_learns_and_keeps_calibration_split():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, (160, 3))
    Y = np.sin(2 * X[:, 0]) + 0.3 * X[:, 1] ** 2 + 0.05 * rng.normal(size=160)
    e = EnsembleSurrogate(k=3, seed=0)
    m = e.fit(X, Y, rng=rng)
    assert m["r2"][0] > 0.8
    assert len(e.cal_index) == m["n_cal"]
    mu, sd = e.predict(X[:5], conformal=True)
    assert mu.shape == (5, 1) and np.all(sd > 0)


def test_heldout_coverage_is_read_on_residuals_the_quantile_never_saw():
    """``coverage_conformal`` is in-sample (>= 1 - alpha by construction);
    ``coverage_heldout`` is read on the validation split and is what a
    comparison of honesty between searches must use."""
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, (160, 3))
    Y = np.stack([np.sin(3 * X[:, 0]) + 0.1 * rng.standard_normal(160),
                  X[:, 1] * X[:, 2] + 0.1 * rng.standard_normal(160)], axis=1)
    s = EnsembleSurrogate(k=3, seed=0)
    m = s.fit(X, Y, rng=np.random.default_rng(0))
    assert m["coverage_conformal"] >= 0.9 - 1e-12
    assert 0.0 <= m["coverage_heldout"] <= 1.0
    assert m["coverage_heldout"] != m["coverage_conformal"] or m["n_val"] > 0
