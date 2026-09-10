"""Hypervolume of a minimised front, in NumPy.

Exact: a sweep for two objectives and the slicing recursion (HSO) for
more. The fronts of this family are tens of points in two or three
objectives, where this is instant; the point is not needing pymoo in the
core. The self-test compares against pymoo when it is importable.
"""

from __future__ import annotations

import numpy as np


def _nondominated(F: np.ndarray) -> np.ndarray:
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if not keep[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dom.any():
            keep[i] = False
    return F[keep]


def _hv_rec(F: np.ndarray, ref: np.ndarray) -> float:
    d = F.shape[1]
    if d == 1:
        return float(ref[0] - F[:, 0].min())
    if d == 2:
        order = np.argsort(F[:, 0])
        P = F[order]
        hv, prev1 = 0.0, ref[1]
        for p in P:
            if p[1] < prev1:
                hv += (ref[0] - p[0]) * (prev1 - p[1])
                prev1 = p[1]
        return float(hv)
    order = np.argsort(F[:, -1])
    P = F[order]
    hv = 0.0
    for i in range(len(P)):
        upper = P[i + 1, -1] if i + 1 < len(P) else ref[-1]
        depth = upper - P[i, -1]
        if depth <= 0:
            continue
        slab = _nondominated(P[: i + 1, :-1])
        hv += depth * _hv_rec(slab, ref[:-1])
    return float(hv)


def hypervolume(F, ref) -> float:
    """Volume dominated by ``F`` (minimised) below ``ref``. Points that do
    not dominate the reference contribute nothing."""
    F = np.atleast_2d(np.asarray(F, dtype=float))
    ref = np.asarray(ref, dtype=float).ravel()
    if F.size == 0:
        return 0.0
    F = F[np.all(np.isfinite(F), axis=1)]
    F = F[np.all(F <= ref, axis=1)]
    if len(F) == 0:
        return 0.0
    return _hv_rec(_nondominated(F), ref)


def reference_point(fronts: list, n_obj: int, margin: float = 0.1) -> np.ndarray:
    """Common reference for one spec: ``margin`` beyond the nadir of the
    UNION of every feasible verified point any method found."""
    pts = [np.atleast_2d(np.asarray(p, float)) for p in fronts if len(p)]
    if not pts:
        return np.ones(n_obj)
    U = np.vstack(pts)
    z, nad = U.min(0), U.max(0)
    return nad + margin * np.maximum(nad - z, 1e-9 * np.maximum(np.abs(nad), 1e-9))


def hv_curve(F, feasible_mask, cuts, ref) -> list:
    """Hypervolume after each cut of the evaluation order."""
    F = np.asarray(F, float)
    m = np.asarray(feasible_mask, bool)
    out = []
    for c in cuts:
        c = int(min(c, len(F)))
        out.append((c, hypervolume(F[:c][m[:c]], ref)))
    return out
