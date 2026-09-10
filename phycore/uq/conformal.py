"""Split conformal, isotonic recalibration and the Mondrian partition.

Ported from Phy-HX ``optimizer_hx`` (sections 2 and 3) without changing a
number. What Phy-HX added over the rest of the family and what this
module makes standard for all of it:

  * the conformal quantile is per Mondrian cell, because a global one is
    set by the worst cell and hands every family the boiling
    correlation's band;
  * the quality gate is per cell too, and it BLOCKS on an under-populated
    cell rather than averaging it away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ALPHA = 0.10

#: Minimum calibration points in a Mondrian cell before its quantile is
#: allowed to mean anything.
CELL_N_MIN = 25


def conformal_quantile(residuals, sigmas, alpha: float = ALPHA) -> float:
    """Split-conformal quantile of the NORMALISED scores.

        s_i = |y_i - mu(x_i)| / sigma(x_i)
        q = the ceil((n+1)(1-alpha))/n quantile of {s_i}

    Below eight points it returns 2.0 - the +-2 sigma heuristic - because
    the quantile is not defined, and returning a number computed from six
    points would be worse than admitting that.
    """
    s = np.abs(np.asarray(residuals, dtype=float).ravel()) / np.maximum(
        np.asarray(sigmas, dtype=float).ravel(), 1e-9)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 8:
        return 2.0
    k = math.ceil((n + 1) * (1.0 - alpha))
    return float(np.max(s)) if k > n else float(np.sort(s)[k - 1])


def isotonic_fit(x, y) -> tuple:
    """Pool-adjacent-violators. Kuleshov, Fenner & Ermon (2018)."""
    order = np.argsort(x)
    xs, ys = np.asarray(x)[order], np.asarray(y)[order].astype(float)
    w = np.ones_like(ys)
    i = 0
    while i < len(ys) - 1:
        if ys[i] > ys[i + 1] + 1e-15:
            tw = w[i] + w[i + 1]
            av = (ys[i] * w[i] + ys[i + 1] * w[i + 1]) / tw
            ys[i] = ys[i + 1] = av
            w[i] = w[i + 1] = tw
            j = i
            while j > 0 and ys[j - 1] > ys[j] + 1e-15:
                t2 = w[j - 1] + w[j]
                a2 = (ys[j - 1] * w[j - 1] + ys[j] * w[j]) / t2
                ys[j - 1] = ys[j] = a2
                w[j - 1] = w[j] = t2
                j -= 1
            i = max(j - 1, 0)
        else:
            i += 1
    return xs, ys


# Acklam's inverse normal CDF. |rel err| < 1.15e-9, and it keeps the
# "pure NumPy in the core" invariant: SciPy is not a dependency.
_A = (-3.969683028665376e+01, 2.209460984245205e+02,
      -2.759285104469687e+02, 1.383577518672690e+02,
      -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02,
      -1.556989798598866e+02, 6.680131188771972e+01,
      -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01,
      -2.400758277161838e+00, -2.549732539343734e+00,
      4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01,
      2.445134137142996e+00, 3.754408661907416e+00)
_PLOW, _PHIGH = 0.02425, 1.0 - 0.02425


def probit(p: float) -> float:
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
    if p < _PLOW:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q
                 + _C[4]) * q + _C[5]) / ((((_D[0] * q + _D[1]) * q
                                            + _D[2]) * q + _D[3]) * q + 1.0)
    if p > _PHIGH:
        return -probit(1.0 - p)
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r
             + _A[4]) * r + _A[5]) * q / (((((_B[0] * r + _B[1]) * r
                                             + _B[2]) * r + _B[3]) * r
                                           + _B[4]) * r + 1.0)


class IsotonicCalibrator:
    """Recalibrates every level, not only 1 - alpha.

    The conformal quantile guarantees marginal coverage at ONE level. A
    report that draws bars at 50 %, 68 % and 95 % has no guarantee at any
    of them. Below twenty calibration points it declares ``fitted =
    False`` and returns the identity - announced degradation rather than
    a calibration invented from four numbers.
    """

    LEVELS = (0.30, 0.50, 0.68, 0.80, 0.90, 0.95, 0.99)

    def __init__(self):
        self.fitted = False
        self.nominal = np.array(self.LEVELS, dtype=float)
        self.observed = np.array(self.LEVELS, dtype=float)
        self.n_cal = 0

    def fit(self, residuals, sigmas) -> "IsotonicCalibrator":
        s = np.abs(np.asarray(residuals).ravel()) / np.maximum(
            np.asarray(sigmas).ravel(), 1e-9)
        s = s[np.isfinite(s)]
        self.n_cal = int(len(s))
        if self.n_cal < 20:
            self.fitted = False
            return self
        obs = np.array([float(np.mean(s <= probit(0.5 * (1.0 + lv))))
                        for lv in self.LEVELS])
        _, obs_iso = isotonic_fit(self.nominal, obs)
        self.observed = np.asarray(obs_iso, dtype=float)
        self.fitted = True
        return self

    def level_for(self, target: float) -> float:
        if not self.fitted:
            return float(target)
        return float(np.interp(target, self.observed, self.nominal,
                               left=self.nominal[0], right=self.nominal[-1]))

    def z_for(self, target: float) -> float:
        return probit(0.5 * (1.0 + self.level_for(target)))

    def report(self) -> dict:
        return {"fitted": self.fitted, "n_cal": self.n_cal,
                "levels_nominal": [float(x) for x in self.nominal],
                "coverage_observed": [float(x) for x in self.observed],
                "max_gap": (float(np.max(np.abs(self.observed
                                                - self.nominal)))
                            if self.fitted else None)}


def reliability_curve(residuals, sigmas,
                      levels=(0.5, 0.68, 0.8, 0.9, 0.95, 0.99)) -> dict:
    s = np.abs(np.asarray(residuals).ravel()) / np.maximum(
        np.asarray(sigmas).ravel(), 1e-9)
    s = s[np.isfinite(s)]
    return {float(lv): (float(np.mean(s <= probit(0.5 * (1.0 + lv))))
                        if len(s) else float("nan")) for lv in levels}


def split_reliability_gap(residuals, sigmas) -> tuple:
    """Reliability of the RECALIBRATED interval, measured out of sample.

    The isotonic map is fitted on one half of the calibration residuals
    and the coverage it delivers is measured on the other. Below forty
    points there are not two halves worth having, and the function says
    so by returning the raw gap and ``False``.
    """
    res = np.asarray(residuals, dtype=float).ravel()
    sig = np.asarray(sigmas, dtype=float).ravel()
    raw = max(abs(v - lv) for lv, v in reliability_curve(res, sig).items())
    n = len(res)
    if n < 40:
        return raw, raw, False
    half = n // 2
    iso = IsotonicCalibrator().fit(res[:half], sig[:half])
    if not iso.fitted:
        return raw, raw, False
    s_test = np.abs(res[half:]) / np.maximum(sig[half:], 1e-9)
    gap = max(abs(float(np.mean(s_test <= iso.z_for(lv))) - lv)
              for lv in IsotonicCalibrator.LEVELS)
    return gap, raw, True


@dataclass
class CellReport:
    cell: tuple
    n: int
    q_conformal: float
    r2: list
    coverage: float
    reliability_gap: float          # AFTER recalibration, out of sample
    reliability_gap_raw: float      # the ensemble sigma on its own
    status: str          # "green" | "thin" | "blocked" | "out_of_scope"
    note: str = ""


class MondrianConformal:
    """One conformal quantile per cell, and a gate per cell.

    The gate BLOCKS rather than averages: a cell with fewer than
    ``n_min`` calibration points is reported as such and stops the gate
    unless it has been declared out of scope by name.
    """

    R2_MIN = 0.85
    Q_MAX = 6.0
    GAP_MAX = 0.15

    def __init__(self, alpha: float = ALPHA, n_min: int = CELL_N_MIN,
                 out_of_scope: tuple = ()):
        self.alpha = float(alpha)
        self.n_min = int(n_min)
        self.out_of_scope = tuple(out_of_scope)
        self.cells: dict = {}
        self.reports: dict = {}
        self.global_quantile = 2.0

    def fit(self, per_cell: dict) -> "MondrianConformal":
        """``per_cell`` maps a cell key to ``{"residuals", "sigmas",
        "r2" (optional)}`` of that cell's calibration points."""
        all_res, all_sig = [], []
        for cell, blob in per_cell.items():
            res = np.asarray(blob["residuals"], dtype=float).ravel()
            sig = np.asarray(blob["sigmas"], dtype=float).ravel()
            all_res.append(res)
            all_sig.append(sig)
            q = conformal_quantile(res, sig, self.alpha)
            cov = float(np.mean(np.abs(res) <= q * np.maximum(sig, 1e-9))) if len(res) else float("nan")
            gap, gap_raw, _split_ok = split_reliability_gap(res, sig) if len(res) else (1.0, 1.0, False)
            r2 = list(blob.get("r2", []))
            n = int(len(res))
            if cell in self.out_of_scope:
                status = "out_of_scope"
            elif n < self.n_min:
                status = "blocked"
            elif (r2 and min(r2) < self.R2_MIN) or q > self.Q_MAX \
                    or gap > self.GAP_MAX:
                status = "thin"
            else:
                status = "green"
            self.cells[cell] = q
            self.reports[cell] = CellReport(
                cell=cell, n=n, q_conformal=q, r2=r2, coverage=cov,
                reliability_gap=gap, reliability_gap_raw=gap_raw,
                status=status,
                note=("declared out of scope" if status == "out_of_scope"
                      else f"n < n_min={self.n_min}" if status == "blocked"
                      else ""))
        if all_res:
            self.global_quantile = conformal_quantile(
                np.concatenate(all_res), np.concatenate(all_sig), self.alpha)
        return self

    def quantile(self, cell) -> float:
        if cell not in self.cells:
            raise KeyError(
                f"no conformal quantile for cell {cell!r}: a design in an "
                "uncalibrated cell cannot be given a band, and giving it "
                "the global one would hand it the worst cell's width")
        return self.cells[cell]

    def gate(self) -> dict:
        blocking = [c for c, r in self.reports.items()
                    if r.status in ("blocked", "thin")]
        return {
            "passed": not blocking,
            "blocking_cells": blocking,
            "cells": {str(c): {"n": r.n, "q": r.q_conformal, "r2": r.r2,
                               "coverage": r.coverage,
                               "reliability_gap": r.reliability_gap,
                               "reliability_gap_raw": r.reliability_gap_raw,
                               "status": r.status, "note": r.note}
                      for c, r in self.reports.items()},
            "global_quantile": self.global_quantile,
            "note": ("an aggregate can hide an empty cell: the gate "
                     "reports every cell with its n and its R2 and "
                     "blocks on any under-populated one that has not "
                     "been declared out of scope"),
        }

    def dilution(self) -> dict:
        """How much a cell would be over-charged by a global quantile."""
        return {str(c): self.global_quantile / q if q > 0 else float("inf")
                for c, q in self.cells.items()}
