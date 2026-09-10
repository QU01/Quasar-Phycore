"""NSGA-II on the pessimistic merit, generic over the plugin. Ported from Phy-HX.

The inner loop is ALWAYS cheap. With a trained surrogate it runs on the
L0 physics plus the learned L1 residual; without one it runs on the L0
physics itself. What it never does is call the expensive fidelity.
"""

from __future__ import annotations

import numpy as np

from ..plugin import (EVALUATION_ERRORS, InfeasibleDesign, plugin_cell,
                      plugin_label, plugin_violation)
from .dominance import Individual, crowding, fast_nondominated_sort, pessimistic


class NSGA2:
    def __init__(self, plugin, spec, pop: int = 40, generations: int = 20,
                 seed: int = 0, use_pessimistic: bool = True,
                 on_generation=None, surrogate=None, fidelity: int = 0):
        self.plugin = plugin
        self.spec = spec
        self.pop = int(pop)
        self.generations = int(generations)
        self.rng = np.random.default_rng(seed)
        self.use_pessimistic = bool(use_pessimistic)
        self.evaluations = 0
        #: Called as ``on_generation(dict)`` after the sample and after
        #: every generation. A callback, not a print: the optimizer is
        #: also imported by the gate, the campaign and the test suite.
        self.on_generation = on_generation
        self.history: list = []
        self.surrogate = surrogate
        self.fidelity = int(fidelity)
        self.corrections = 0

    def _evaluate(self, theta) -> Individual | None:
        p = self.plugin
        try:
            rec = p.evaluate(theta, self.spec)
        except EVALUATION_ERRORS:
            return None
        self.evaluations += 1
        sigma_extra = None
        if self.surrogate is not None:
            try:
                x = p.embedding(rec)
                mu, sd = self.surrogate.predict([x], conformal=True)
                rec = p.apply_residual(rec, self.spec, mu[0])
                sigma_extra = np.asarray(sd[0], dtype=float)
                self.corrections += 1
            except Exception:                            # noqa: BLE001
                # A surrogate that cannot speak about a point does not get
                # to guess about it: the L0 physics stands, unlabelled as
                # corrected, and the point still competes.
                sigma_extra = None
        f = p.objectives(rec, self.spec)
        b = p.objective_bands(rec, self.spec)
        if sigma_extra is not None:
            # The surrogate's own disagreement widens the band on the
            # objective it corrected, and on that one only.
            b = b.copy()
            b[0] = float(np.hypot(b[0], abs(f[0]) * sigma_extra[0]))
        g = p.constraints(rec, self.spec)
        return Individual(theta=rec["theta"], rec=rec, f=f, b=b, g=g,
                          v=plugin_violation(p, g), label=plugin_label(p, rec),
                          cell=plugin_cell(p, rec))

    def _merit(self, ind: Individual) -> np.ndarray:
        return pessimistic(ind.f, ind.b) if self.use_pessimistic else ind.f

    def _bounds(self):
        b = np.asarray(self.plugin.bounds, dtype=float)
        return b[:, 0], b[:, 1]

    def _initial(self) -> list:
        lo, hi = self._bounds()
        d = len(lo)
        cut = (np.arange(self.pop)[:, None] + self.rng.random((self.pop, d))) \
            / self.pop
        for j in range(d):
            self.rng.shuffle(cut[:, j])
        out = []
        for row in cut:
            ind = self._evaluate(lo + row * (hi - lo))
            if ind is not None:
                out.append(ind)
        return out

    def _vary(self, parents: list) -> list:
        lo, hi = self._bounds()
        kids = []
        for _ in range(self.pop):
            a, b = self.rng.integers(0, len(parents), 2)
            pa, pb = parents[a].theta, parents[b].theta
            # SBX-flavoured blend plus a polynomial-ish mutation. Kept
            # deliberately simple: the expensive part is the physics.
            w = self.rng.random(len(lo))
            child = np.where(self.rng.random(len(lo)) < 0.5,
                             w * pa + (1.0 - w) * pb, pa)
            mask = self.rng.random(len(lo)) < 0.25
            child = np.where(mask,
                             child + 0.12 * (hi - lo)
                             * self.rng.normal(size=len(lo)), child)
            ind = self._evaluate(np.clip(child, lo, hi))
            if ind is not None:
                kids.append(ind)
        return kids

    def _report(self, stage: str, gen: int, popn: list) -> None:
        feas = [i for i in popn if i.feasible]
        best = min(feas, key=lambda i: i.f[0]) if feas else None
        row = {"stage": stage, "gen": gen, "generations": self.generations,
               "population": len(popn), "feasible": len(feas),
               "evaluations": self.evaluations,
               "plugin": self.plugin.plugin_id,
               "best_f0": float(best.f[0]) if best else None,
               "best_f": [float(x) for x in best.f] if best else None,
               "min_violation": float(min(i.v for i in popn)) if popn
               else None}
        self.history.append(row)
        if self.on_generation is not None:
            self.on_generation(row)

    def run(self) -> dict:
        popn = self._initial()
        if len(popn) < 4:
            raise InfeasibleDesign(
                f"plugin {self.plugin.plugin_id} could not produce four "
                "buildable designs anywhere in its box", where="bounds")
        self._report("sample", 0, popn)
        for _gen in range(self.generations):
            popn = popn + self._vary(popn)
            F = np.array([self._merit(i) for i in popn])
            V = np.array([i.v for i in popn])
            fronts = fast_nondominated_sort(F, V)
            chosen = []
            for fr in fronts:
                if len(chosen) + len(fr) <= self.pop:
                    chosen += list(fr)
                else:
                    d = crowding(F, fr)
                    order = np.argsort(-d)
                    chosen += [fr[k] for k in order[:self.pop - len(chosen)]]
                    break
            popn = [popn[k] for k in chosen]
            self._report("generation", _gen + 1, popn)
        F = np.array([self._merit(i) for i in popn])
        V = np.array([i.v for i in popn])
        front = fast_nondominated_sort(F, V)[0]
        return {"population": popn, "front": [popn[k] for k in front],
                "evaluations": self.evaluations, "history": self.history,
                "merit": "pessimistic" if self.use_pessimistic else "point"}
