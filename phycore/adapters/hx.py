"""Phy-HX as a plugin: ``design_space_hx`` behind the contract, unchanged.

Phy-HX already had the plugin shape (a family object plus module-level
``evaluate / objectives / objective_bands / constraints / embedding /
residual / apply_residual``); this wraps it, translates its exceptions
into the engine's, and gives it a ``spec_vec`` so Phy-2 can carry a
prior between specs. The Mondrian cell is (family, mode), as report 04
defined it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..plugin import ContractError, InfeasibleDesign

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "Phy-HX"

#: The window a spec moves along, normalised to [0, 1]. Same knobs as
#: Phy-Bench's HXReal; a product may pass its own.
DEFAULT_KNOBS = (("m_dot_hot", 0.55, 1.00), ("t_hot_in", 850.0, 950.0),
                 ("dp_budget_hot", None, None), ("mass_max", 180.0, 300.0))


def _load(root):
    root = Path(root or DEFAULT_ROOT)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import design_space_hx as ds                      # noqa: E402
    import interfaces_hx as ifc                       # noqa: E402
    return ds, ifc


class HXPlugin:
    integer_dims: tuple

    def __init__(self, family=None, root=None, knobs=DEFAULT_KNOBS):
        self.ds, self.ifc = _load(root)
        self.family = family if family is not None else self.ds.PlateFinOSF()
        self.plugin_id = f"phy_hx:{self.family.family_id}:{getattr(self.family, 'mode_hot', 'dry')}"
        self.model_revision = int(getattr(self.ds, "MODEL_REVISION", 0))
        self.bounds = np.asarray(self.family.bounds, dtype=float)
        self.integer_dims = tuple(getattr(self.family, "integer_dims", ()))
        self.objective_names = tuple(self.ds.OBJECTIVE_NAMES)
        self.residual_dim = len(getattr(self.ds, "RESIDUAL_NAMES", ("ua_ratio", "pinch_shift")))
        self.knobs = tuple(knobs)
        self._errors = (self.ifc.InfeasibleDesign, self.ifc.ContractError, ValueError)
        self._base = None

    # --- spec ---------------------------------------------------------------
    def spec_vec(self, spec) -> np.ndarray:
        if self._base is None:
            self._base = self.ds.recuperator_spec()
        out = []
        for name, lo, hi in self.knobs:
            v = float(getattr(spec, name))
            if lo is None or hi is None:
                base = float(getattr(self._base, name))
                lo, hi = 0.7 * base, 1.3 * base
            out.append(np.clip((v - lo) / max(hi - lo, 1e-12), -0.5, 1.5))
        return np.asarray(out, dtype=float)

    # --- the eight calls ------------------------------------------------------
    def evaluate(self, theta, spec, fidelity: int = 0) -> dict:
        try:
            return self.ds.evaluate(self.family, theta, spec, fidelity=int(fidelity))
        except self.ifc.ContractError as exc:
            raise ContractError(str(exc)) from exc
        except self.ifc.InfeasibleDesign as exc:
            raise InfeasibleDesign(str(exc), where=getattr(exc, "where", ""),
                                   value=getattr(exc, "value", 0.0)) from exc

    def objectives(self, rec, spec):
        return self.ds.objectives(rec, spec)

    def objective_bands(self, rec, spec):
        return self.ds.objective_bands(rec, spec)

    def constraints(self, rec, spec):
        return self.ds.constraints(self.family, rec, spec)

    def embedding(self, rec):
        return self.ds.embedding(rec)

    def residual(self, rec, spec):
        return self.ds.residual(rec, spec)

    def apply_residual(self, rec, spec, res):
        return self.ds.apply_residual(rec, spec, res)

    # --- optional -------------------------------------------------------------
    def violation(self, g):
        return self.ds.violation(g)

    def cell(self, rec):
        return (str(rec.get("family", self.family.family_id)),
                str(rec.get("mode", getattr(self.family, "mode_hot", "dry"))))

    def label(self, rec):
        return str(rec.get("family", self.family.family_id))

    def constraint_names(self, spec):
        return self.ds.constraint_names(self.family, spec)
