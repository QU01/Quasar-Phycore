"""Which designs to spend the expensive fidelity on next. Ported from Phy-HX.

Three pools, then a greedy k-means de-duplication so the batch does not
spend ten evaluations on ten neighbours:

  * EXPLOIT  - the predicted front, best violation first;
  * EXPLORE  - where the band is widest among the near-feasible;
  * FRONTIER - where g is nearest zero. Cheap because g is EXACT.
"""

from __future__ import annotations

import numpy as np

from .dominance import fast_nondominated_sort, pessimistic

EXPLORE_FRAC = 0.30
BOUNDARY_FRAC = 0.25


def select_acquisition_batch(population: list, batch_size: int,
                             seed: int = 0,
                             explore_frac: float = EXPLORE_FRAC,
                             boundary_frac: float = BOUNDARY_FRAC) -> list:
    if len(population) <= batch_size:
        return list(population)
    rng = np.random.default_rng(seed)
    n_exp = max(1, int(batch_size * explore_frac))
    n_bnd = max(1, int(batch_size * boundary_frac))

    merit = np.array([pessimistic(i.f, i.b) for i in population])
    viol = np.array([i.v for i in population])
    front = np.asarray(fast_nondominated_sort(merit, viol)[0], dtype=int)
    exploit = [int(k) for k in front[np.argsort(viol[front])]]

    band = np.array([float(np.sum(np.abs(i.b))) for i in population])
    near = np.where(viol <= np.quantile(viol, 0.5))[0]
    explore = [int(k) for k in near[np.argsort(-band[near])][:4 * n_exp]]

    frontier = [int(k) for k in np.argsort(np.abs(viol))[:4 * n_bnd]]

    pool = list(dict.fromkeys(exploit + explore + frontier))
    if len(pool) <= batch_size:
        return [population[k] for k in pool]

    # Greedy k-means over theta, normalised on the box, as a cheap
    # approximation to a determinantal point process.
    thetas = np.array([population[k].theta for k in pool], dtype=float)
    span = thetas.max(axis=0) - thetas.min(axis=0) + 1e-12
    Z = (thetas - thetas.min(axis=0)) / span
    centres = Z[rng.choice(len(Z), size=batch_size, replace=False)]
    labels = np.zeros(len(Z), dtype=int)
    for _ in range(12):
        d = np.linalg.norm(Z[:, None, :] - centres[None], axis=2)
        labels = d.argmin(axis=1)
        for j in range(batch_size):
            pts = Z[labels == j]
            if len(pts):
                centres[j] = pts.mean(axis=0)
    chosen: list[int] = []
    for j in range(batch_size):
        members = np.where(labels == j)[0]
        if len(members):
            chosen.append(pool[int(members[0])])
    for k in pool:
        if len(chosen) >= batch_size:
            break
        if k not in chosen:
            chosen.append(k)
    return [population[k] for k in chosen[:batch_size]]
