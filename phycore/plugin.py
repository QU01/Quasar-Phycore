"""The plugin contract: what a Phy model must give the engine, and nothing else.

The engine never sees physics. It sees a box, a record, three vectors
(objectives, their declared half-bands, exact constraints), an embedding
to learn on and a residual to learn. Everything a plugin does inside
``evaluate`` is its own business; everything the engine does with the
result is the same for every plugin.

Three invariants the family settled on and this contract enforces:

  * **constraints are exact and never predicted** - ``constraints`` is
    algebra on the record, so the search can aim at g ~ 0 for free;
  * **the surrogate learns the L1 - L0 residual**, not L1 - hence the
    ``residual`` / ``apply_residual`` pair, which must be inverses;
  * **only real evaluations reach the front** - the record says which
    fidelity it actually reached (``fidelity_used``) and the engine
    believes the record, not the request.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


class PhyError(Exception):
    """Base of every error the engine catches around a plugin call."""


class InfeasibleDesign(PhyError):
    """A theta that cannot be built or evaluated. The engine skips it."""

    def __init__(self, message: str, *, where: str = "", value: float = 0.0):
        super().__init__(message)
        self.where, self.value = where, float(value)


class ContractError(PhyError):
    """A record, contract or dataset that violates its schema."""


#: What ``evaluate`` may raise to say "no design here". Anything else
#: propagates, because a bug in the physics must not be counted as an
#: infeasible corner of the box.
EVALUATION_ERRORS = (InfeasibleDesign, ContractError, ValueError)


@runtime_checkable
class PhyPlugin(Protocol):
    """One model = one plugin. Attributes first, then the eight calls."""

    #: Identifies the plugin in records, cache keys and reports.
    plugin_id: str
    #: The plugin's own revision, bumped when its numbers can move.
    model_revision: int
    #: (d, 2) array of physical bounds; the engine searches inside it.
    bounds: np.ndarray
    #: Indices of theta repaired to integers before evaluation.
    integer_dims: tuple
    #: Names of the objectives, all MINIMISED.
    objective_names: tuple

    def spec_vec(self, spec: Any) -> np.ndarray:
        """The design specification as a vector in roughly [0, 1]^k.
        It conditions the learned front (Phy-2) and labels data pools;
        the Phy-1 loop only stores it."""
        ...

    def evaluate(self, theta: np.ndarray, spec: Any, fidelity: int = 0) -> dict:
        """The physics. Returns a record with at least ``theta`` (repaired)
        and ``fidelity_used`` ("L0", "L1", ...). Raises one of
        ``EVALUATION_ERRORS`` for a design that cannot exist."""
        ...

    def objectives(self, rec: dict, spec: Any) -> np.ndarray: ...

    def objective_bands(self, rec: dict, spec: Any) -> np.ndarray:
        """Absolute half-widths, same order as the objectives. Zero for
        an axis that is geometry rather than a correlation."""
        ...

    def constraints(self, rec: dict, spec: Any) -> np.ndarray:
        """g <= 0 feasible, continuous, EXACT."""
        ...

    def embedding(self, rec: dict) -> np.ndarray:
        """What the residual surrogate learns on. Dimensionless, spec-
        comparable, and without the objective itself in it."""
        ...

    def residual(self, rec: dict, spec: Any) -> np.ndarray | None:
        """The L1 - L0 gap of this record, or None when the expensive
        fidelity was not reached (a missing truth is not a zero)."""
        ...

    def apply_residual(self, rec: dict, spec: Any, res) -> dict:
        """The inverse of ``residual`` on an L0 record. Must set
        ``corrected_by_surrogate`` so the corrected copy never reaches
        the verified front."""
        ...


# Optional members the engine looks for with getattr:
#   cell(rec) -> tuple          Mondrian cell for the per-cell gate
#   violation(g) -> float       a plugin-specific violation aggregate
#   torch_cheap(theta_t, s_t)   differentiable L0 for the learned front
#   constraint_names(spec)      for reports
#   label(rec) -> str           what the front colours by (family, mode...)


def repair(plugin, theta) -> np.ndarray:
    """Clip to the box and round the integer dimensions.

    Integers are repaired rather than searched over, the choice Phy-CT
    made first: the optimizer sees a continuous box and the physics sees
    a buildable design.
    """
    b = np.asarray(plugin.bounds, dtype=float)
    th = np.clip(np.asarray(theta, dtype=float), b[:, 0], b[:, 1])
    for i in getattr(plugin, "integer_dims", ()):
        th[i] = float(round(th[i]))
    return th


def violation(g) -> float:
    """Sum of positive parts. The one aggregate Deb dominance needs."""
    return float(np.sum(np.maximum(np.asarray(g, dtype=float), 0.0)))


def plugin_violation(plugin, g) -> float:
    fn = getattr(plugin, "violation", None)
    return float(fn(g)) if fn is not None else violation(g)


def fidelity_label(fidelity: int) -> str:
    return f"L{int(fidelity)}"


def reached(rec: dict, fidelity: int) -> bool:
    """Did the record reach the fidelity that was asked for?"""
    return str(rec.get("fidelity_used", "L0")) == fidelity_label(fidelity)


def plugin_label(plugin, rec: dict) -> str:
    fn = getattr(plugin, "label", None)
    return str(fn(rec)) if fn is not None else str(plugin.plugin_id)


def plugin_cell(plugin, rec: dict) -> tuple:
    fn = getattr(plugin, "cell", None)
    return tuple(fn(rec)) if fn is not None else ("all",)


def spec_vector(plugin, spec) -> np.ndarray:
    return np.asarray(plugin.spec_vec(spec), dtype=float).ravel()


# =============================================================================
# The validator: a smoke test a plugin runs before the engine trusts it
# =============================================================================


def validate_plugin(plugin, spec, theta=None, fidelity: int = 0,
                    n_probe: int = 16, seed: int = 0) -> list:
    """Return the list of contract violations (empty = valid).

    It probes the box (centre first, then a few Latin-hypercube points)
    until one design builds, and checks shapes, finiteness and the
    residual / apply_residual inverse on that record. It does not judge
    the physics - it judges whether the engine can hold it.
    """
    errs: list = []
    for name in ("plugin_id", "model_revision", "bounds", "integer_dims",
                 "objective_names"):
        if not hasattr(plugin, name):
            errs.append(f"falta el atributo '{name}'")
    if errs:
        return errs
    b = np.asarray(plugin.bounds, dtype=float)
    if b.ndim != 2 or b.shape[1] != 2 or not np.all(b[:, 1] > b[:, 0]):
        errs.append(f"bounds debe ser (d, 2) con hi > lo; hay {b.shape}")
        return errs
    d = b.shape[0]
    for i in plugin.integer_dims:
        if not 0 <= int(i) < d:
            errs.append(f"integer_dims contiene {i} fuera de [0, {d})")
    if errs:
        return errs
    try:
        s = spec_vector(plugin, spec)
        if not np.all(np.isfinite(s)):
            errs.append("spec_vec devuelve valores no finitos")
    except Exception as exc:                                 # noqa: BLE001
        errs.append(f"spec_vec falla: {exc!r}")
        return errs

    from .engine.sampling import lhs
    probes = [0.5 * (b[:, 0] + b[:, 1])] if theta is None else [np.asarray(theta, float)]
    probes += list(b[:, 0] + lhs(n_probe, d, np.random.default_rng(seed)) * (b[:, 1] - b[:, 0]))
    rec = None
    for th in probes:
        try:
            rec = plugin.evaluate(repair(plugin, th), spec, fidelity)
            break
        except EVALUATION_ERRORS:
            continue
        except Exception as exc:                             # noqa: BLE001
            errs.append(f"evaluate lanza {type(exc).__name__} en vez de "
                        f"InfeasibleDesign/ContractError/ValueError: {exc}")
            return errs
    if rec is None:
        errs.append(f"ningún diseño construible en {len(probes)} sondeos de la caja")
        return errs
    if not isinstance(rec, dict):
        errs.append(f"evaluate debe devolver un dict, devuelve {type(rec).__name__}")
        return errs
    if "theta" not in rec:
        errs.append("el record no lleva 'theta'")
    elif np.asarray(rec["theta"]).shape != (d,):
        errs.append(f"rec['theta'] tiene forma {np.asarray(rec['theta']).shape}, no ({d},)")
    if "fidelity_used" not in rec:
        errs.append("el record no lleva 'fidelity_used'")

    n_obj = len(plugin.objective_names)
    checks = (("objectives", lambda: plugin.objectives(rec, spec), (n_obj,)),
              ("objective_bands", lambda: plugin.objective_bands(rec, spec), (n_obj,)),
              ("constraints", lambda: plugin.constraints(rec, spec), None),
              ("embedding", lambda: plugin.embedding(rec), None))
    for name, fn, shape in checks:
        try:
            v = np.asarray(fn(), dtype=float)
        except Exception as exc:                             # noqa: BLE001
            errs.append(f"{name} falla sobre un record válido: {exc!r}")
            continue
        if v.ndim != 1 or (shape is not None and v.shape != shape):
            errs.append(f"{name} devuelve forma {v.shape}, se esperaba "
                        f"{shape if shape else '(n,)'}")
        if not np.all(np.isfinite(v)):
            errs.append(f"{name} devuelve valores no finitos")
        if name == "objective_bands" and v.size and np.any(v < 0):
            errs.append("objective_bands debe ser >= 0 (semianchuras absolutas)")
    try:
        r = plugin.residual(rec, spec)
    except Exception as exc:                                 # noqa: BLE001
        errs.append(f"residual falla: {exc!r}")
        r = None
    if r is not None:
        if not reached(rec, max(fidelity, 1)) and fidelity >= 1:
            errs.append("residual devuelve un valor sobre un record que no "
                        "alcanzó la fidelidad cara")
        try:
            corr = plugin.apply_residual(rec, spec, np.asarray(r, float))
            if not corr.get("corrected_by_surrogate"):
                errs.append("apply_residual no marca 'corrected_by_surrogate'")
        except Exception as exc:                             # noqa: BLE001
            errs.append(f"apply_residual falla: {exc!r}")
    elif fidelity == 0:
        # No expensive truth here, so the inverse is probed with a null
        # residual. Its width comes from ``residual_dim`` when the plugin
        # declares it, otherwise the first width that the plugin accepts.
        dims = [int(getattr(plugin, "residual_dim"))] if hasattr(plugin, "residual_dim")             else [1, 2, 3, 4]
        last = None
        for nd in dims:
            try:
                corr = plugin.apply_residual(rec, spec, np.zeros(nd))
                if not corr.get("corrected_by_surrogate"):
                    errs.append("apply_residual no marca 'corrected_by_surrogate'")
                last = None
                break
            except Exception as exc:                         # noqa: BLE001
                last = exc
        if last is not None:
            errs.append(f"apply_residual falla con residuo nulo: {last!r}")
    return errs
