"""Phy-Bench problems (ToyHX, MWShift, ...) as plugins.

The record layout is the one ``Phy-Bench/baseline.py`` builds for
``optimizer_hx``, so the Phy-1 loop here and the family's loop there see
the same numbers in the same order - which is what the parity test
checks bit for bit.
"""

from __future__ import annotations

import numpy as np

from ..plugin import repair


class ProblemPlugin:
    model_revision = 1
    integer_dims = ()

    def __init__(self, problem, band_frac: float = 0.03):
        self.problem = problem
        self.plugin_id = str(problem.name)
        self.bounds = np.asarray(problem.bounds, dtype=float)
        self.objective_names = tuple(f"f{i}" for i in range(int(problem.n_obj)))
        self.band_frac = float(band_frac)
        if getattr(problem, "has_torch", False):
            self.torch_cheap = problem.torch_cheap

    def spec_vec(self, spec):
        return np.asarray(self.problem.spec_vec(spec), dtype=float)

    def evaluate(self, theta, spec, fidelity: int = 0) -> dict:
        th = repair(self, theta)
        F0, G = self.problem.eval_cheap(th, spec)
        if not (np.all(np.isfinite(F0)) and np.all(np.isfinite(G))):
            raise ValueError("non-finite")
        rec = {"theta": th, "f_cheap": np.asarray(F0, float), "g": np.asarray(G, float),
               "f": np.asarray(F0, float), "fidelity_used": "L0"}
        if int(fidelity) >= 1:
            F1, _, ok = self.problem.eval_exp(th, spec)
            if ok:
                rec["f_exp"] = np.asarray(F1, float)
                rec["f"] = np.asarray(F1, float)
                rec["fidelity_used"] = "L1"
        return rec

    def objectives(self, rec, spec):
        return np.asarray(rec["f"], float)

    def objective_bands(self, rec, spec):
        f = np.asarray(rec["f"], float)
        b = np.zeros_like(f)
        b[0] = abs(f[0]) * self.band_frac
        return b

    def constraints(self, rec, spec):
        return np.asarray(rec["g"], float)

    def embedding(self, rec):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        thn = (rec["theta"] - lo) / (hi - lo)
        return np.concatenate([thn, rec["f_cheap"], rec["g"]])

    def residual(self, rec, spec):
        if rec.get("fidelity_used") != "L1":
            return None
        f0 = max(abs(rec["f_cheap"][0]), 1e-9)
        return np.array([rec["f_exp"][0] / f0 - 1.0])

    def apply_residual(self, rec, spec, res):
        out = dict(rec)
        f = np.array(rec["f_cheap"], float)
        f[0] = f[0] * (1.0 + float(np.asarray(res).ravel()[0]))
        out["f"] = f
        out["corrected_by_surrogate"] = True
        return out
