"""AR1 discrepancy: the model that stands between the cheap ladder and the
expensive one, ``y_HF(x) = rho y_LF(x) + delta(x)``.

Step 3 of the architecture plan, the half the ensemble cannot cover. The
ensemble (``engine.ensemble``) learns the L1 - L0 residual and needs
hundreds of points; it is the right tool there because L1 costs
milliseconds. The L2/L3 gap is the opposite regime: ten to fifty pairs,
each costing seconds to hours. The evidence collected in the architecture
report is unambiguous for that regime - AR1 co-kriging beats a
multi-fidelity network on a hundred low- plus ten high-fidelity points
(Schouler, Belme & Cinnella 2025), and deep ensembles do not diversify
with little data (Li, Rudner & Wilson, ICLR 2024). So:

    y_HF(x) = rho y_LF(x) + delta(x) + eps,
    rho ~ N(mu_rho, s_rho^2),  delta ~ GP(0, sigma_f^2 k_ARD),  eps ~ N(0, sigma_n^2)

Three things make this more than a GP with a linear term:

  * **rho is marginalised, not fitted.** With a Gaussian prior on rho the
    marginal covariance picks up ``s_rho^2 f f^T`` and the posterior of
    rho is closed form, so ten pairs give a rho with an honest error bar
    instead of a point estimate that the next pair moves.
  * **The prior width comes from the mark.** ``[V]`` 0.05, ``[S]`` 0.15,
    ``[M]`` 0.30 (``contracts.marks.PRIOR_WIDTH``): the same table the
    hierarchical calibration reads. A hook with ``[?]`` has no prior and
    the fit is refused, which is what BLOCKED means.
  * **The binary verdict becomes a test.** ``verdict()`` reports how many
    prior sigmas rho moved and how big delta is against the declared band
    of the cheap level. "NO CALIBRAR" stops being a decision somebody
    made and becomes a sentence with two numbers in it.

Everything is NumPy: Cholesky of an n x n matrix with n <= ~100, analytic
gradients of the log marginal likelihood, Adam on the log hyperparameters
from several starts. No scipy, no autodiff, deterministic given a seed.

    from phycore.uq import DiscrepancyGP
    gp = DiscrepancyGP(prior_rho_sd=0.15).fit(X, y_lf, y_hf)
    mu, sd = gp.predict(X_new, y_lf_new)
    gp.verdict(declared_band=0.05)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..plugin import ContractError

#: Below this many pairs there is no discrepancy to estimate: two
#: hyperparameters per dimension and a rho cannot come out of six points,
#: and returning a number computed from six points would be worse than
#: refusing. The literature's own regime is 10-50 pairs.
PAIRS_MIN = 8

#: Hard ceiling of the exact solve. Above this the Cholesky stops being
#: free and the honest move is a different model, not a slower one.
PAIRS_MAX = 400

#: Adam on the log hyperparameters: steps, rate, restarts.
OPT_STEPS = 250
OPT_RATE = 0.05
OPT_RESTARTS = 3

#: Declared bounds of the hyperparameters, in units of the standardised
#: inputs and outputs. The lengthscale floor keeps the kernel from
#: interpolating noise as structure; the noise floor keeps the Cholesky
#: conditioned.
LS_BOUNDS = (0.05, 20.0)
SF_BOUNDS = (1e-3, 10.0)
SN_BOUNDS = (1e-4, 2.0)


def _matern52(D2, ls):
    """Matern 5/2 with ARD from the squared scaled distance. Chosen over
    the squared exponential because a discrepancy is not analytic: a
    separation that appears at one angle of attack is not infinitely
    differentiable in it."""
    r = np.sqrt(np.maximum(D2, 0.0))
    s5 = math.sqrt(5.0)
    return (1.0 + s5 * r + 5.0 / 3.0 * r * r) * np.exp(-s5 * r)


def _matern52_dr2(D2):
    """d k / d(r^2) for the Matern 5/2 (k written in terms of r^2), used by
    the analytic gradient of the log marginal likelihood."""
    r = np.sqrt(np.maximum(D2, 1e-300))
    s5 = math.sqrt(5.0)
    e = np.exp(-s5 * r)
    # dk/dr = -5/3 r (1 + sqrt5 r) exp(-sqrt5 r);  dk/d(r^2) = dk/dr / (2 r)
    return -5.0 / 6.0 * (1.0 + s5 * r) * e


@dataclass
class DiscrepancyReport:
    """What the fit says, in the family's vocabulary."""
    n_pairs: int
    rho: float
    rho_sd: float
    rho_sigmas_from_prior: float
    delta_rms: float
    delta_rms_over_band: float
    loo_rmse: float
    loo_rmse_naive: float
    loo_coverage: float
    lengthscales: tuple
    sigma_f: float
    sigma_n: float
    verdict: str
    reason: str

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["lengthscales"] = list(self.lengthscales)
        return d


class DiscrepancyGP:
    """``y_HF = rho y_LF + delta(x)`` with rho marginalised and delta a
    Matern 5/2 GP with ARD.

    ``prior_rho_sd`` is the half-width of the prior on rho and comes from
    the mark of the cheap level (``contracts.marks.PRIOR_WIDTH``);
    ``prior_rho_mean`` is 1 because the cheap level is supposed to be the
    same physics, not a different one.
    """

    def __init__(self, prior_rho_sd: float = 0.15, prior_rho_mean: float = 1.0,
                 seed: int = 0, standardise: bool = True):
        if prior_rho_sd is None:
            raise ContractError(
                "a discrepancy with no prior on rho is a BLOCKED hook, not a wide one: "
                "the mark [?] has no prior width (contracts.marks.PRIOR_WIDTH)")
        if not (prior_rho_sd > 0.0):
            raise ContractError(f"prior_rho_sd must be positive, got {prior_rho_sd}")
        self.prior_rho_sd = float(prior_rho_sd)
        self.prior_rho_mean = float(prior_rho_mean)
        self.seed = int(seed)
        self.standardise = bool(standardise)
        self.fitted = False

    # ---- kernel and likelihood ------------------------------------------

    def _sqdist(self, A, Bm, ls):
        A = A / ls
        Bm = Bm / ls
        return (np.sum(A * A, 1)[:, None] + np.sum(Bm * Bm, 1)[None, :]
                - 2.0 * A @ Bm.T)

    def _cov(self, X, ls, sf2, sn2):
        D2 = self._sqdist(X, X, ls)
        K = sf2 * _matern52(D2, ls)
        return K + (sn2 + 1e-10) * np.eye(len(X))

    def _nlml(self, p, X, f, y, grad: bool = True):
        """Negative log marginal likelihood of y ~ N(mu_rho f, Sigma) with
        Sigma = s_rho^2 f f^T + K + sn^2 I, and its gradient in the log
        hyperparameters p = [log ls (D), log sf, log sn]."""
        d = X.shape[1]
        ls = np.exp(p[:d])
        sf2 = np.exp(2.0 * p[d])
        sn2 = np.exp(2.0 * p[d + 1])
        D2 = self._sqdist(X, X, ls)
        Kbase = _matern52(D2, ls)
        S = sf2 * Kbase + (sn2 + 1e-10) * np.eye(len(X)) \
            + (self.prior_rho_sd ** 2) * np.outer(f, f)
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            return (1e12, np.zeros_like(p)) if grad else 1e12
        z = y - self.prior_rho_mean * f
        a = np.linalg.solve(L.T, np.linalg.solve(L, z))
        nll = 0.5 * float(z @ a) + float(np.sum(np.log(np.diag(L)))) \
            + 0.5 * len(X) * math.log(2.0 * math.pi)
        if not grad:
            return nll
        Sinv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(X))))
        W = np.outer(a, a) - Sinv                     # dNLML/dS = -1/2 W
        g = np.zeros_like(p)
        dk = _matern52_dr2(D2)
        for i in range(d):
            Xi = X[:, i:i + 1] / ls[i]
            dD2 = -2.0 * (Xi - Xi.T) ** 2             # d(r^2)/d(log ls_i)
            g[i] = -0.5 * float(np.sum(W * (sf2 * dk * dD2)))
        g[d] = -0.5 * float(np.sum(W * (2.0 * sf2 * Kbase)))
        g[d + 1] = -0.5 * float(np.sum(W * (2.0 * sn2 * np.eye(len(X)))))
        return nll, g

    # ---- fit -------------------------------------------------------------

    def fit(self, X, y_lf, y_hf) -> "DiscrepancyGP":
        """X: (n, d) the embedding of each pair; y_lf / y_hf: the cheap and
        expensive magnitude of the SAME point. Refuses fewer than
        ``PAIRS_MIN`` pairs, non-finite entries and a constant y_lf that
        makes rho unidentifiable."""
        X = np.asarray(X, dtype=float)
        f = np.asarray(y_lf, dtype=float).ravel()
        y = np.asarray(y_hf, dtype=float).ravel()
        if X.ndim != 2:
            raise ContractError(f"X must be (n, d), got shape {X.shape}")
        n, d = X.shape
        if not (len(f) == len(y) == n):
            raise ContractError(f"{n} rows of X, {len(f)} low- and {len(y)} high-fidelity values")
        if n < PAIRS_MIN:
            raise ContractError(
                f"{n} pairs: a discrepancy needs at least {PAIRS_MIN}. Fewer points is not a "
                "wider band, it is no model")
        if n > PAIRS_MAX:
            raise ContractError(f"{n} pairs exceed the exact solve's ceiling ({PAIRS_MAX})")
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(f)) and np.all(np.isfinite(y))):
            raise ContractError("non-finite entries in the pairs: a failed run is not a datum")
        scale_f = float(np.std(f))
        if scale_f <= 1e-12 * (abs(float(np.mean(f))) + 1e-30):
            raise ContractError(
                "y_lf is constant over the pairs: rho and delta are not identifiable "
                "(any rho is compensated by a constant discrepancy)")

        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-12
        self.Xn = (X - self.X_mean) / self.X_std if self.standardise else X.copy()
        # y and y_lf share the scale: rho must stay dimensionless
        self.y_scale = float(np.std(np.concatenate([f, y]))) + 1e-30
        self.f = f / self.y_scale
        self.y = y / self.y_scale
        self.d = d

        rng = np.random.default_rng(self.seed)
        best, best_p = np.inf, None
        p0s = [np.concatenate([np.zeros(d), [math.log(0.3)], [math.log(0.05)]])]
        for _ in range(OPT_RESTARTS - 1):
            p0s.append(np.concatenate([rng.normal(0.0, 0.7, d),
                                       [math.log(float(np.exp(rng.normal(-1.0, 0.5))))],
                                       [math.log(float(np.exp(rng.normal(-3.0, 0.5))))]]))
        lo = np.concatenate([np.full(d, math.log(LS_BOUNDS[0])), [math.log(SF_BOUNDS[0])],
                             [math.log(SN_BOUNDS[0])]])
        hi = np.concatenate([np.full(d, math.log(LS_BOUNDS[1])), [math.log(SF_BOUNDS[1])],
                             [math.log(SN_BOUNDS[1])]])
        for p0 in p0s:
            p = np.clip(np.asarray(p0, float), lo, hi)
            m = np.zeros_like(p); v = np.zeros_like(p)
            for t in range(1, OPT_STEPS + 1):
                nll, g = self._nlml(p, self.Xn, self.f, self.y)
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * g * g
                mh = m / (1.0 - 0.9 ** t)
                vh = v / (1.0 - 0.999 ** t)
                p = np.clip(p - OPT_RATE * mh / (np.sqrt(vh) + 1e-8), lo, hi)
            nll = self._nlml(p, self.Xn, self.f, self.y, grad=False)
            if nll < best:
                best, best_p = nll, p.copy()
        self.p = best_p
        self.nlml = float(best)
        self.lengthscales = np.exp(best_p[:d])
        self.sigma_f = float(np.exp(best_p[d]))
        self.sigma_n = float(np.exp(best_p[d + 1]))

        # posterior of rho and of delta, in closed form, at the MAP hyperparameters
        C = self._cov(self.Xn, self.lengthscales, self.sigma_f ** 2, self.sigma_n ** 2)
        self.L = np.linalg.cholesky(C)
        Cinv_f = self._chol_solve(self.f)
        Cinv_y = self._chol_solve(self.y)
        A = float(self.f @ Cinv_f) + 1.0 / self.prior_rho_sd ** 2
        self.rho = (float(self.f @ Cinv_y) + self.prior_rho_mean / self.prior_rho_sd ** 2) / A
        self.rho_var = 1.0 / A
        self.Cinv_f = Cinv_f
        self.alpha = self._chol_solve(self.y - self.rho * self.f)
        self.fitted = True
        return self

    def _chol_solve(self, b):
        return np.linalg.solve(self.L.T, np.linalg.solve(self.L, b))

    # ---- predict ---------------------------------------------------------

    def predict(self, X, y_lf, with_noise: bool = False):
        """Posterior mean and standard deviation of y_HF at new points.
        The variance carries three terms: the GP's own, the uncertainty of
        rho (which is why rho was marginalised) and, if asked, the
        observation noise."""
        self._check()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        f = np.atleast_1d(np.asarray(y_lf, dtype=float)).ravel() / self.y_scale
        Xn = (X - self.X_mean) / self.X_std if self.standardise else X
        D2 = self._sqdist(Xn, self.Xn, self.lengthscales)
        Ks = self.sigma_f ** 2 * _matern52(D2, self.lengthscales)
        mu = self.rho * f + Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = self.sigma_f ** 2 - np.sum(v * v, axis=0)
        # rho's contribution: the residual of y_lf after the GP has explained it
        h = f - Ks @ self.Cinv_f
        var = np.maximum(var, 0.0) + self.rho_var * h * h
        if with_noise:
            var = var + self.sigma_n ** 2
        return mu * self.y_scale, np.sqrt(np.maximum(var, 0.0)) * self.y_scale

    def delta(self, X, y_lf):
        """The discrepancy alone, ``y_HF - rho y_LF``, in the original units."""
        mu, _ = self.predict(X, y_lf)
        return mu - self.rho * np.atleast_1d(np.asarray(y_lf, float)).ravel()

    # ---- leave-one-out, in closed form ----------------------------------

    def loo(self) -> tuple:
        """Leave-one-out mean and variance of the training pairs WITHOUT
        refitting, from the marginal covariance (Rasmussen & Williams 5.4.2
        applied to Sigma = s_rho^2 f f^T + C, so the rho prior is inside).
        The hyperparameters stay at their MAP: this measures the model, not
        the search."""
        self._check()
        S = (self.sigma_f ** 2 * _matern52(self._sqdist(self.Xn, self.Xn, self.lengthscales),
                                           self.lengthscales)
             + (self.sigma_n ** 2 + 1e-10) * np.eye(len(self.Xn))
             + self.prior_rho_sd ** 2 * np.outer(self.f, self.f))
        Sinv = np.linalg.inv(S)
        z = self.y - self.prior_rho_mean * self.f
        a = Sinv @ z
        dg = np.diag(Sinv)
        mu = (self.prior_rho_mean * self.f + z - a / dg) * self.y_scale
        var = (1.0 / dg) * self.y_scale ** 2
        return mu, np.sqrt(var)

    # ---- verdict ---------------------------------------------------------

    def verdict(self, declared_band: float, alpha: float = 0.10) -> DiscrepancyReport:
        """The continuous replacement of "NO CALIBRAR".

        ``declared_band`` is the relative band the cheap level already
        declares (the polar band, the closure's band...). The verdict is
        NO CALIBRAR when rho has not moved more than one prior sigma from
        1 AND the discrepancy is inside that band: then the expensive
        level says nothing the cheap one had not already admitted.
        """
        self._check()
        band = float(declared_band)
        rho_sd = math.sqrt(self.rho_var)
        n_sig = abs(self.rho - self.prior_rho_mean) / self.prior_rho_sd
        dl = self.delta(self.Xn * self.X_std + self.X_mean if self.standardise else self.Xn,
                        self.f * self.y_scale)
        scale = float(np.mean(np.abs(self.y * self.y_scale))) + 1e-30
        delta_rms = float(np.sqrt(np.mean(dl * dl)))
        loo_mu, loo_sd = self.loo()
        y = self.y * self.y_scale
        loo_rmse = float(np.sqrt(np.mean((loo_mu - y) ** 2)))
        naive = float(np.sqrt(np.mean((self.prior_rho_mean * self.f * self.y_scale - y) ** 2)))
        k = 1.6448536269514722 if abs(alpha - 0.10) < 1e-12 else float(
            math.sqrt(2.0) * _erfinv(1.0 - alpha))
        cov = float(np.mean(np.abs(loo_mu - y) <= k * loo_sd))
        moved = n_sig > 1.0
        big = delta_rms / scale > band
        if not moved and not big:
            v, why = "NO CALIBRAR", (
                f"rho moved {n_sig:.2f} prior sigmas from {self.prior_rho_mean:g} and the "
                f"discrepancy is {delta_rms / scale:.1%} of the magnitude, inside the declared "
                f"band {band:.1%}")
        elif moved and not big:
            v, why = "CALIBRAR: rho", (
                f"rho = {self.rho:.4f} +- {rho_sd:.4f}, {n_sig:.2f} prior sigmas from "
                f"{self.prior_rho_mean:g}, with a discrepancy inside the band: a SCALE error")
        elif big and not moved:
            v, why = "DISCREPANCIA ESTRUCTURAL", (
                f"rho stayed at {self.rho:.4f} but the discrepancy is {delta_rms / scale:.1%} "
                f"of the magnitude against a declared band of {band:.1%}: physics that is "
                "missing, not a constant that is wrong")
        else:
            v, why = "CALIBRAR: rho + discrepancia", (
                f"rho = {self.rho:.4f} ({n_sig:.2f} prior sigmas) AND a discrepancy of "
                f"{delta_rms / scale:.1%} over a band of {band:.1%}")
        return DiscrepancyReport(
            n_pairs=len(self.Xn), rho=float(self.rho), rho_sd=float(rho_sd),
            rho_sigmas_from_prior=float(n_sig), delta_rms=float(delta_rms),
            delta_rms_over_band=float(delta_rms / scale / band) if band > 0 else float("inf"),
            loo_rmse=loo_rmse, loo_rmse_naive=naive, loo_coverage=cov,
            lengthscales=tuple(float(v_) for v_ in self.lengthscales),
            sigma_f=float(self.sigma_f * self.y_scale), sigma_n=float(self.sigma_n * self.y_scale),
            verdict=v, reason=why)

    def _check(self):
        if not self.fitted:
            raise ContractError("the discrepancy has not been fitted")


def _erfinv(x: float) -> float:
    """Inverse error function (Giles' rational approximation, 1e-9); used
    only for a non-default confidence level."""
    w = -math.log((1.0 - x) * (1.0 + x))
    if w < 5.0:
        w -= 2.5
        p = 2.81022636e-08
        for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087,
                  -0.00125372503, -0.00417768164, 0.246640727, 1.50140941):
            p = p * w + c
    else:
        w = math.sqrt(w) - 3.0
        p = -0.000200214257
        for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773,
                  -0.0076224613, 0.00943887047, 1.00167406, 2.83297682):
            p = p * w + c
    return p * x
