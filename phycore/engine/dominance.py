"""Deb dominance, NSGA-II helpers, the pessimistic merit and the fronts.

Ported from Phy-HX. The dominance is evaluated on the PESSIMISTIC merit:
each objective at the unfavourable edge of its band. The reason is
arithmetic: order a joint front on point estimates and the wide-banded
closure wins or loses on where its correlation happens to be optimistic,
not on its physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def dominates(f1, v1, f2, v2) -> bool:
    """Constrained dominance (Deb 2002). Feasible beats infeasible;
    between infeasibles the smaller violation wins; between feasibles it
    is ordinary Pareto."""
    ok1, ok2 = v1 <= 1e-9, v2 <= 1e-9
    if ok1 and not ok2:
        return True
    if ok2 and not ok1:
        return False
    if not ok1 and not ok2:
        return v1 < v2
    return bool(np.all(f1 <= f2) and np.any(f1 < f2))


def fast_nondominated_sort(F, V):
    n = len(F)
    S = [[] for _ in range(n)]
    nd = np.zeros(n, dtype=int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], V[p], F[q], V[q]):
                S[p].append(q)
            elif dominates(F[q], V[q], F[p], V[p]):
                nd[p] += 1
        if nd[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                nd[q] -= 1
                if nd[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding(F, idxs):
    d = np.zeros(len(idxs))
    idxs = np.asarray(idxs)
    for j in range(F.shape[1]):
        order = np.argsort(F[idxs, j])
        d[order[0]] = d[order[-1]] = np.inf
        span = F[idxs[order[-1]], j] - F[idxs[order[0]], j] + 1e-12
        for k in range(1, len(idxs) - 1):
            d[order[k]] += (F[idxs[order[k + 1]], j]
                            - F[idxs[order[k - 1]], j]) / span
    return d


def pessimistic(F, B):
    """Every objective at the unfavourable edge (all minimised: upper)."""
    return np.asarray(F, dtype=float) + np.asarray(B, dtype=float)


def optimistic(F, B):
    return np.asarray(F, dtype=float) - np.asarray(B, dtype=float)


@dataclass
class Individual:
    theta: np.ndarray
    rec: dict
    f: np.ndarray
    b: np.ndarray
    g: np.ndarray
    v: float
    label: str = ""              # what the front colours by (family, mode, ...)
    cell: tuple = ("all",)       # Mondrian cell
    feasible: bool = field(init=False)

    def __post_init__(self):
        self.feasible = self.v <= 1e-9

    # Phy-HX vocabulary, kept so its report code reads the same object.
    @property
    def family(self) -> str:
        return self.label

    @property
    def mode(self) -> str:
        return str(self.cell[1]) if len(self.cell) > 1 else ""


def scalar_fitness(ind: Individual) -> tuple:
    """The incumbent rule, and it is DEB's, not a linear penalty: any
    feasible design beats any infeasible one; among infeasible ones the
    smaller violation wins; among feasible ones the first objective."""
    return (0 if ind.feasible else 1,
            ind.v if not ind.feasible else 0.0,
            float(ind.f[0]))


def joint_front(individuals: list, use_pessimistic: bool = True) -> list:
    """One non-dominated sort over the union. The label travels with each
    point and colours the front; it does not enter the comparison."""
    if not individuals:
        return []
    F = np.array([pessimistic(i.f, i.b) if use_pessimistic else i.f
                  for i in individuals])
    V = np.array([i.v for i in individuals])
    return [individuals[k] for k in fast_nondominated_sort(F, V)[0]]


def front_comparison(individuals: list) -> dict:
    """Does the pessimistic front REORDER the labels or only shift it?"""
    point = joint_front(individuals, use_pessimistic=False)
    pess = joint_front(individuals, use_pessimistic=True)

    def composition(front):
        out: dict = {}
        for i in front:
            out[i.label] = out.get(i.label, 0) + 1
        total = max(len(front), 1)
        return {k: v / total for k, v in out.items()}

    def best_label(front, use_pess):
        if not front:
            return None
        F = np.array([pessimistic(i.f, i.b) if use_pess else i.f
                      for i in front])
        span = F.max(axis=0) - F.min(axis=0) + 1e-30
        score = ((F - F.min(axis=0)) / span).sum(axis=1)
        return front[int(np.argmin(score))].label

    ids_point = {id(i) for i in point}
    ids_pess = {id(i) for i in pess}
    overlap = len(ids_point & ids_pess) / max(len(ids_point | ids_pess), 1)
    return {
        "n_point": len(point), "n_pessimistic": len(pess),
        "composition_point": composition(point),
        "composition_pessimistic": composition(pess),
        "jaccard": overlap,
        "winner_point": best_label(point, False),
        "winner_pessimistic": best_label(pess, True),
        "reordered": best_label(point, False) != best_label(pess, True),
    }


def value_of_reducing_uncertainty(individuals: list, objective_names=()) -> dict:
    """The distance between the optimistic and pessimistic fronts: how
    much merit is being surrendered to not knowing, per objective, and
    attributed per closure when the records carry ``bands.attribution``."""
    if not individuals:
        return {}
    pess = joint_front(individuals, use_pessimistic=True)
    F_p = np.array([pessimistic(i.f, i.b) for i in pess])
    F_o = np.array([optimistic(i.f, i.b) for i in pess])
    gap = F_p - F_o
    scale = np.maximum(np.abs(F_p).mean(axis=0), 1e-30)
    attribution: dict = {}
    for i in pess:
        for k, v in (i.rec.get("bands", {}) or {}).get("attribution", {}).items():
            attribution[k] = attribution.get(k, 0.0) + v
    tot = sum(attribution.values()) or 1.0
    return {
        "objectives": list(objective_names),
        "absolute_gap": [float(x) for x in gap.mean(axis=0)],
        "relative_gap": [float(x) for x in (gap.mean(axis=0) / scale)],
        "band_attribution": {k: v / tot for k, v in attribution.items()},
    }
