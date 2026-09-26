"""The residual ensemble: K small MLPs on bootstrap resamples, conformal on top.

Ported from Phy-HX. The only addition is that the calibration residuals,
sigmas and indices are KEPT (``cal_residuals``, ``cal_sigmas``,
``cal_index``) so a Mondrian gate can be built per cell afterwards
without retraining - nothing about training or the RNG order changes.
"""

from __future__ import annotations

import numpy as np

from ..plugin import ContractError
from ..uq.conformal import (ALPHA, IsotonicCalibrator, conformal_quantile,
                            split_reliability_gap)
from .mlp import MLP

K_ENSEMBLE = 5
CAL_FRAC = 0.25

#: Below this many converged expensive evaluations there is no residual
#: to learn, and the honest move is to say so and keep searching on the
#: exact cheap physics rather than to fit five points and call it a
#: model.
RESIDUAL_MIN_SAMPLES = 24


class EnsembleSurrogate:
    """K small MLPs on bootstrap resamples. Mean is mu, spread is sigma."""

    def __init__(self, k: int = K_ENSEMBLE, seed: int = 0,
                 alpha: float = ALPHA, hidden=(64, 64)):
        self.k = int(k)
        self.seed = int(seed)
        self.alpha = float(alpha)
        self.hidden = tuple(hidden)
        self.members: list = []
        self.x_mean = self.x_std = None
        self.y_mean = self.y_std = None
        self.q_conformal = 2.0
        self.iso = IsotonicCalibrator()
        self.metrics: dict = {}
        self.cal_index = np.zeros(0, dtype=int)
        self.cal_residuals = np.zeros((0, 1))
        self.cal_sigmas = np.zeros((0, 1))

    def fit(self, X, Y, rng=None) -> dict:
        r = rng if rng is not None else np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        if Y.shape[0] != X.shape[0]:
            Y = Y.T
        # Standardise the input. Without it one feature in large units
        # saturates the first layer and the ensemble learns nothing.
        self.x_mean, self.x_std = X.mean(axis=0), X.std(axis=0) + 1e-9
        Xn = (X - self.x_mean) / self.x_std
        self.y_mean, self.y_std = Y.mean(axis=0), Y.std(axis=0) + 1e-9
        Yn = (Y - self.y_mean) / self.y_std

        n = Xn.shape[0]
        n_cal = max(int(CAL_FRAC * n), 8)
        n_val = max(int(0.15 * n), 6)
        perm = r.permutation(n)
        ci, vi, ti = (perm[:n_cal], perm[n_cal:n_cal + n_val],
                      perm[n_cal + n_val:])
        if len(ti) < 10:
            raise ContractError(
                f"{n} points cannot be split into calibration, validation "
                "and training sets; the surrogate refuses rather than "
                "training on the set it will be judged on")

        self.members = []
        for kk in range(self.k):
            boot = r.integers(0, len(ti), len(ti))
            m = MLP(Xn.shape[1], Yn.shape[1], hidden=self.hidden,
                    seed=self.seed + kk)
            m.train(Xn[ti][boot], Yn[ti][boot], Xn[vi], Yn[vi],
                    seed=self.seed + kk)
            self.members.append(m)

        mu_c, sd_c = self._predict_norm_raw(Xn[ci])
        res_c = Yn[ci] - mu_c
        self.q_conformal = conformal_quantile(res_c, sd_c, self.alpha)
        self.iso.fit(res_c, sd_c)
        gap_cal, gap_raw, _ok = split_reliability_gap(res_c, sd_c)
        self.cal_index, self.cal_residuals, self.cal_sigmas = ci, res_c, sd_c

        mu_v, sd_v = self._predict_norm_raw(Xn[vi])
        ss_res = np.sum((Yn[vi] - mu_v) ** 2, axis=0)
        ss_tot = np.sum((Yn[vi] - Yn[vi].mean(axis=0)) ** 2, axis=0) + 1e-30
        cov = float(np.mean(np.abs(res_c) <= self.q_conformal * sd_c))
        # ``cov`` is read on the SAME residuals the quantile was taken from:
        # >= 1 - alpha by construction. The held-out figure uses the
        # validation split, which the quantile never saw (it did steer the
        # members' early stopping, so it is, if anything, optimistic).
        cov_heldout = float(np.mean(np.abs(Yn[vi] - mu_v) <= self.q_conformal * sd_v))
        self.metrics = {
            "n": n, "n_cal": len(ci), "n_val": len(vi), "n_train": len(ti),
            "r2": [float(v) for v in (1.0 - ss_res / ss_tot)],
            "q_conformal": float(self.q_conformal),
            "coverage_conformal": cov,
            "coverage_heldout": cov_heldout,
            "reliability_gap_raw": float(gap_raw),
            "reliability_gap_calibrated": float(gap_cal),
        }
        return self.metrics

    def _predict_norm_raw(self, Xn):
        preds = np.stack([m.forward(np.asarray(Xn, dtype=float))
                          for m in self.members])
        return preds.mean(axis=0), preds.std(axis=0) + 1e-12

    def predict(self, X, conformal: bool = False, level=None):
        Xn = (np.atleast_2d(np.asarray(X, dtype=float)) - self.x_mean) \
            / self.x_std
        mu_n, sd_n = self._predict_norm_raw(Xn)
        if level is not None:
            sd_n = sd_n * self.iso.z_for(float(level))
        elif conformal:
            sd_n = sd_n * self.q_conformal
        return mu_n * self.y_std + self.y_mean, sd_n * self.y_std


def residual_matrix(plugin, recs: list, spec) -> tuple:
    """The training set: only records whose expensive truth EXISTS.

    A design whose L1 refused to converge keeps its L0 answer and still
    counts for the constraints and for the verified front. It does not
    count here, because feeding it in as a zero residual would teach the
    surrogate that the L1 agrees with the L0 exactly where the L1 could
    not be computed.
    """
    X, Y = [], []
    for rec in recs:
        r = plugin.residual(rec, spec)
        if r is None:
            continue
        X.append(plugin.embedding(rec))
        Y.append(r)
    return (np.array(X, dtype=float) if X else np.zeros((0, 1)),
            np.array(Y, dtype=float) if Y else np.zeros((0, 2)))
