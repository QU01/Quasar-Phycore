"""The AR1 discrepancy: what it must reproduce exactly, what it must
refuse, and the closed forms that would otherwise be believed on faith."""
import math

import numpy as np
import pytest

from phycore.plugin import ContractError
from phycore.uq import PAIRS_MIN, DiscrepancyGP
from phycore.uq.discrepancy import _matern52, _matern52_dr2


def _sample(n, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, d))
    y_lf = 1.0 + 0.5 * X[:, 0] + 0.2 * X[:, 1] ** 2
    return X, y_lf


# =============================================================================
# 1. Kernel and gradient: the analytic pieces the likelihood rests on
# =============================================================================


def test_kernel_and_its_derivative_are_consistent():
    # central differences away from r = 0, where dk/d(r^2) is finite but
    # its own derivative is not (the kernel is a function of r, not of r^2)
    r2 = np.linspace(0.02, 4.0, 60)
    h = 1e-7
    num = (_matern52(r2 + h, 1.0) - _matern52(r2 - h, 1.0)) / (2.0 * h)
    assert np.max(np.abs(num - _matern52_dr2(r2))) < 1e-6
    assert _matern52(np.array([0.0]), 1.0)[0] == pytest.approx(1.0)


def test_gradient_of_the_log_marginal_likelihood_matches_finite_differences():
    X, f = _sample(20, 3, seed=1)
    y = 1.2 * f + 0.05 * np.sin(3.0 * X[:, 0])
    gp = DiscrepancyGP(prior_rho_sd=0.2)
    gp.X_mean, gp.X_std = X.mean(0), X.std(0) + 1e-12
    Xn = (X - gp.X_mean) / gp.X_std
    p = np.array([0.1, -0.2, 0.3, math.log(0.4), math.log(0.03)])
    _, g = gp._nlml(p, Xn, f, y)
    num = np.zeros_like(p)
    for i in range(len(p)):
        e = np.zeros_like(p); e[i] = 1e-6
        num[i] = (gp._nlml(p + e, Xn, f, y, grad=False)
                  - gp._nlml(p - e, Xn, f, y, grad=False)) / 2e-6
    assert np.max(np.abs(num - g) / (np.abs(num) + 1.0)) < 1e-5


# =============================================================================
# 2. Identities: a pure scale, and leave-one-out without refitting
# =============================================================================


def test_a_pure_scale_is_recovered_as_rho_with_no_discrepancy():
    """y_HF = 1.3 y_LF exactly: rho -> 1.3 (the prior must not stop it when
    it is wide) and the discrepancy is numerically zero."""
    X, f = _sample(30, 3, seed=2)
    gp = DiscrepancyGP(prior_rho_sd=0.15, seed=0).fit(X, f, 1.3 * f)
    assert gp.rho == pytest.approx(1.3, abs=5e-3)
    scale = float(np.mean(np.abs(1.3 * f)))
    assert float(np.sqrt(np.mean(gp.delta(X, f) ** 2))) / scale < 5e-3
    mu, sd = gp.predict(X, f)
    assert np.max(np.abs(mu - 1.3 * f)) / scale < 5e-3
    rep = gp.verdict(declared_band=0.05)
    assert rep.verdict == "CALIBRAR: rho" and rep.rho_sigmas_from_prior > 1.0


def test_the_prior_holds_rho_when_the_evidence_is_thin():
    """Ten NOISY pairs of the same 1.3 scaling: with a [V]-narrow prior the
    posterior of rho stays much closer to 1 than with an [M]-wide one. That
    the prior width does the work when the data cannot is the point of
    marginalising rho instead of fitting it - and it is why a [?] mark is
    refused rather than given a wide prior."""
    X, f = _sample(10, 2, seed=12)
    rng = np.random.default_rng(5)
    y = 1.3 * f + rng.normal(0.0, 0.25 * float(np.std(f)), len(f))
    narrow = DiscrepancyGP(prior_rho_sd=0.03, seed=0).fit(X, f, y)
    wide = DiscrepancyGP(prior_rho_sd=0.40, seed=0).fit(X, f, y)
    assert abs(narrow.rho - 1.0) < abs(wide.rho - 1.0)
    # the posterior of rho is never wider than its prior (A = f' C^-1 f + 1/s^2)
    for gp in (narrow, wide):
        assert math.sqrt(gp.rho_var) <= gp.prior_rho_sd


def test_leave_one_out_closed_form_equals_the_brute_force_one():
    """The LOO used by the verdict is a formula, not a loop; if the formula
    is wrong every coverage number afterwards is a lie. Here it is checked
    against n explicit fits AT THE SAME hyperparameters."""
    X, f = _sample(16, 2, seed=3)
    y = 1.1 * f + 0.08 * np.cos(2.5 * X[:, 1])
    gp = DiscrepancyGP(prior_rho_sd=0.15, seed=0).fit(X, f, y)
    mu, sd = gp.loo()
    n = len(f)
    S = (gp.sigma_f ** 2 * _matern52(gp._sqdist(gp.Xn, gp.Xn, gp.lengthscales), gp.lengthscales)
         + (gp.sigma_n ** 2 + 1e-10) * np.eye(n)
         + gp.prior_rho_sd ** 2 * np.outer(gp.f, gp.f))
    z = gp.y - gp.prior_rho_mean * gp.f
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        Sii = S[np.ix_(idx, idx)]
        k = S[i, idx]
        m = gp.prior_rho_mean * gp.f[i] + float(k @ np.linalg.solve(Sii, z[idx]))
        v = float(S[i, i] - k @ np.linalg.solve(Sii, k))
        assert m * gp.y_scale == pytest.approx(mu[i], rel=1e-9, abs=1e-12)
        assert math.sqrt(v) * gp.y_scale == pytest.approx(sd[i], rel=1e-9)


# =============================================================================
# 3. It has to be better than doing nothing, and honest about it
# =============================================================================


def test_it_beats_the_uncorrected_low_fidelity_and_covers():
    """A structured discrepancy over 40 pairs: the LOO error must be a
    fraction of the raw L1-L2 gap and the 90 % interval must cover ~90 %."""
    X, f = _sample(40, 3, seed=4)
    rng = np.random.default_rng(11)
    y = 1.05 * f + 0.15 * np.sin(2.0 * X[:, 0]) * np.exp(0.3 * X[:, 1]) + rng.normal(0, 0.004, 40)
    gp = DiscrepancyGP(prior_rho_sd=0.15, seed=0).fit(X, f, y)
    rep = gp.verdict(declared_band=0.03)
    assert rep.loo_rmse < 0.5 * rep.loo_rmse_naive
    assert rep.loo_coverage >= 0.80
    assert rep.n_pairs == 40 and rep.sigma_n > 0.0


def test_structural_discrepancy_is_named_as_such():
    X, f = _sample(35, 2, seed=5)
    y = f + 0.25 * np.tanh(3.0 * X[:, 0])                 # rho = 1, delta big
    rep = DiscrepancyGP(prior_rho_sd=0.15, seed=0).fit(X, f, y).verdict(declared_band=0.02)
    assert rep.verdict == "DISCREPANCIA ESTRUCTURAL" and rep.delta_rms_over_band > 1.0


def test_no_calibrar_when_the_expensive_level_says_nothing_new():
    X, f = _sample(35, 2, seed=6)
    rng = np.random.default_rng(7)
    y = f * 1.002 + rng.normal(0.0, 0.002, len(f))
    rep = DiscrepancyGP(prior_rho_sd=0.15, seed=0).fit(X, f, y).verdict(declared_band=0.05)
    assert rep.verdict == "NO CALIBRAR" and "inside the declared band" in rep.reason


# =============================================================================
# 4. Refusals: what is not a model
# =============================================================================


def test_refusals():
    X, f = _sample(30, 2, seed=8)
    with pytest.raises(ContractError, match=r"BLOCKED"):
        DiscrepancyGP(prior_rho_sd=None)
    with pytest.raises(ContractError, match=r"at least"):
        DiscrepancyGP().fit(X[:PAIRS_MIN - 1], f[:PAIRS_MIN - 1], f[:PAIRS_MIN - 1])
    with pytest.raises(ContractError, match=r"non-finite"):
        y = f.copy(); y[3] = np.nan
        DiscrepancyGP().fit(X, f, y)
    with pytest.raises(ContractError, match=r"not identifiable"):
        DiscrepancyGP().fit(X, np.ones_like(f), f)
    with pytest.raises(ContractError, match=r"has not been fitted"):
        DiscrepancyGP().predict(X, f)


def test_deterministic_given_the_seed():
    X, f = _sample(24, 2, seed=9)
    y = 1.1 * f + 0.05 * X[:, 0] ** 2
    a = DiscrepancyGP(seed=3).fit(X, f, y)
    b = DiscrepancyGP(seed=3).fit(X, f, y)
    assert a.rho == b.rho and np.array_equal(a.lengthscales, b.lengthscales)
