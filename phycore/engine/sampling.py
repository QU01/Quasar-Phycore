"""Latin hypercube, the family's initialisation, with the family's RNG order."""

from __future__ import annotations

import numpy as np


def lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """n points in [0, 1]^d. Consumes the generator exactly as
    ``optimizer_hx.Designer._lhs`` does (one ``random`` then d shuffles),
    which is what keeps the Phy-1 loop bit-exact against Phy-HX."""
    cut = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        rng.shuffle(cut[:, j])
    return cut


def lhs_box(n: int, bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    b = np.asarray(bounds, dtype=float)
    lo, hi = b[:, 0], b[:, 1]
    return lo + lhs(n, len(lo), rng) * (hi - lo)
