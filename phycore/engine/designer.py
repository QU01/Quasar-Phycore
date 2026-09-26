"""The design loop, with the search chosen by one argument.

    search="nsga2"  Phy-1: LHS at the expensive fidelity, an ensemble on the
                    L1-L0 residual, a cheap inner NSGA-II on L0 + residual,
                    an exploit/explore/frontier batch, a VERIFIED front.
                    Ported from Phy-HX and bit-exact against it for the
                    same seed (the test suite checks this on Phy-Bench).
    search="psl"    Phy-2: the same money spent through an amortised actor
                    pi(theta | spec, lambda) trained by pathwise gradients
                    through an anchored critic; carries a ``state`` from
                    one spec to the next. Needs torch.

Both return a :class:`DesignResult` with the same fields, so a report, a
gate or a benchmark reads either without knowing which one ran.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..plugin import (EVALUATION_ERRORS, fidelity_label, plugin_cell, plugin_label,
                      plugin_violation, spec_vector)
from ..uq.conformal import ALPHA, MondrianConformal
from .acquisition import select_acquisition_batch
from .dominance import Individual, fast_nondominated_sort, pessimistic, scalar_fitness
from .ensemble import K_ENSEMBLE, RESIDUAL_MIN_SAMPLES, EnsembleSurrogate, residual_matrix
from .nsga2 import NSGA2
from .sampling import lhs

SEED = 20260827
SEARCHES = ("nsga2", "psl")


@dataclass
class DesignResult:
    population: list                      # every real evaluation, as Individuals
    front: list                           # the VERIFIED pessimistic front
    history: list
    evaluations: int                      # expensive evaluations
    n_cheap: int                          # exact L0 evaluations spent searching
    fidelity: str
    surrogate: dict
    trained: bool
    merit: str
    search: str
    seconds: float = 0.0
    state: object = None                  # Phy-2: what the next spec inherits
    honesty: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    batch_log: list = field(default_factory=list)
    cell_gate: dict = field(default_factory=dict)

    def arrays(self) -> dict:
        """The verified data as arrays, in evaluation order (benchmarks)."""
        P = self.population
        return {
            "theta": np.array([np.asarray(i.theta, float) for i in P]) if P else np.zeros((0, 1)),
            "F": np.array([np.asarray(i.f, float) for i in P]) if P else np.zeros((0, 1)),
            "G": np.array([np.asarray(i.g, float) for i in P]) if P else np.zeros((0, 1)),
            "ok": np.array([not i.rec.get("corrected_by_surrogate")
                            and str(i.rec.get("fidelity_used", "L0")) == self.fidelity
                            for i in P], bool),
            "n_exp": len(P), "n_cheap": self.n_cheap,
        }


def _make_individual(plugin, spec, rec) -> Individual:
    f = plugin.objectives(rec, spec)
    b = plugin.objective_bands(rec, spec)
    g = plugin.constraints(rec, spec)
    return Individual(theta=rec["theta"], rec=rec, f=f, b=b, g=g,
                      v=plugin_violation(plugin, g), label=plugin_label(plugin, rec),
                      cell=plugin_cell(plugin, rec))


def verified_front(individuals: list) -> list:
    """Only evaluations that really happened. A predicted point has
    never been built and never will be."""
    if not individuals:
        return []
    real = [i for i in individuals if not i.rec.get("corrected_by_surrogate")]
    if not real:
        return []
    F = np.array([pessimistic(i.f, i.b) for i in real])
    V = np.array([i.v for i in real])
    return [real[k] for k in fast_nondominated_sort(F, V)[0]]


class Designer:
    """One plugin, one spec, one search."""

    def __init__(self, plugin, spec, fidelity: int = 0, seed: int = SEED,
                 log=None, k: int = K_ENSEMBLE, search: str = "nsga2",
                 psl_config=None, state=None):
        if search not in SEARCHES:
            raise ValueError(f"search debe ser uno de {SEARCHES}, no {search!r}")
        self.plugin = plugin
        self.spec = spec
        self.fidelity = int(fidelity)
        self.seed = int(seed)
        self.search = search
        self.rng = np.random.default_rng(self.seed)
        self.log = log if log is not None else (lambda *_a, **_k: None)
        self.surrogate = EnsembleSurrogate(k=k, seed=self.seed)
        self.trained = False
        self.records: list = []
        self.individuals: list = []
        self.history: list = []
        self.gate: dict = {}
        self.n_cheap = 0
        self.psl_config = psl_config
        self.state = state
        self._train_recs: list = []

    # --- sampling ---------------------------------------------------------
    def _lhs(self, n: int) -> np.ndarray:
        b = np.asarray(self.plugin.bounds, dtype=float)
        lo, hi = b[:, 0], b[:, 1]
        return lo + lhs(n, len(lo), self.rng) * (hi - lo)

    def _evaluate(self, theta) -> Individual | None:
        try:
            rec = self.plugin.evaluate(theta, self.spec, fidelity=self.fidelity)
        except EVALUATION_ERRORS:
            return None
        ind = _make_individual(self.plugin, self.spec, rec)
        self.records.append(rec)
        self.individuals.append(ind)
        return ind

    def _spend(self, thetas) -> list:
        out = []
        for t in thetas:
            ind = self._evaluate(t)
            if ind is not None:
                out.append(ind)
        return out

    # --- the surrogate ----------------------------------------------------
    def train(self) -> bool:
        if self.fidelity < 1:
            self.log("surrogate", "not built: with L0 as the truth the "
                     "residual is zero by construction, so the search "
                     "runs on the exact physics instead of on a model "
                     "of it")
            return True
        X, Y = residual_matrix(self.plugin, self.records, self.spec)
        self._train_recs = [r for r in self.records
                            if self.plugin.residual(r, self.spec) is not None]
        n_l1 = len(X)
        n_all = len(self.records)
        if n_l1 < RESIDUAL_MIN_SAMPLES:
            self.log("surrogate", f"only {n_l1}/{n_all} evaluations "
                     f"reached {fidelity_label(self.fidelity)}: too few to "
                     "learn a residual, the search stays on the exact L0 "
                     "this round")
            self.trained = False
            return False
        try:
            metrics = self.surrogate.fit(X, Y, rng=self.rng)
        except EVALUATION_ERRORS as exc:
            self.log("surrogate", f"refused to train: {exc}")
            self.trained = False
            return False
        self.trained = True
        cover = metrics["coverage_conformal"]
        self.gate = {
            "n": n_l1, "n_all": n_all, "r2": metrics["r2"],
            "coverage": cover, "q_conformal": metrics["q_conformal"],
            "coverage_heldout": metrics.get("coverage_heldout"),
            "reliability_gap": metrics["reliability_gap_calibrated"],
            "passed": bool(cover >= 1.0 - ALPHA - 0.05),
        }
        self.log("surrogate",
                 f"trained on {n_l1}/{n_all} samples "
                 f"({100 * n_l1 / max(n_all, 1):.0f} % reached "
                 f"{fidelity_label(self.fidelity)}): "
                 f"R2 {np.round(metrics['r2'], 3).tolist()}, conformal "
                 f"coverage {100 * cover:.1f} % at {100 * (1 - ALPHA):.0f} "
                 f"% nominal, reliability gap "
                 f"{metrics['reliability_gap_calibrated']:.3f}")
        if not self.gate["passed"]:
            self.log("surrogate", "QUALITY GATE FAILED: one more sampling "
                     "round before the search trusts it")
        return self.gate["passed"]

    def cell_gate(self, n_min: int | None = None, out_of_scope=()) -> dict:
        """The Mondrian gate over the surrogate's calibration split,
        grouped by the plugin's ``cell``. Uses what ``fit`` kept: no
        retraining, no RNG consumed. Empty when nothing is trained."""
        if not self.trained or not len(self.surrogate.cal_index):
            return {}
        per_cell: dict = {}
        res, sig = self.surrogate.cal_residuals, self.surrogate.cal_sigmas
        for row, idx in enumerate(self.surrogate.cal_index):
            c = plugin_cell(self.plugin, self._train_recs[int(idx)])
            blob = per_cell.setdefault(c, {"residuals": [], "sigmas": [],
                                           "r2": self.gate.get("r2", [])})
            blob["residuals"].extend(np.asarray(res[row]).ravel().tolist())
            blob["sigmas"].extend(np.asarray(sig[row]).ravel().tolist())
        kw = {"out_of_scope": tuple(out_of_scope)}
        if n_min is not None:
            kw["n_min"] = int(n_min)
        return MondrianConformal(**kw).fit(per_cell).gate()

    # --- the loop ---------------------------------------------------------
    def run(self, n_init: int = 60, rounds: int = 3, batch_size: int = 12,
            pop: int = 40, generations: int = 20,
            on_generation=None) -> DesignResult:
        if self.search == "psl":
            return self._run_psl(n_init, rounds, batch_size)
        t0 = time.perf_counter()
        label = fidelity_label(self.fidelity)
        self.log("sample", f"initial Latin hypercube: {n_init} physical "
                 f"evaluations at {label}")
        first = self._spend(self._lhs(n_init))
        n_feas = sum(1 for i in first if i.feasible)
        self.log("sample", f"{len(first)}/{n_init} buildable, "
                 f"{n_feas} feasible")
        if self.fidelity >= 1:
            n_l1 = sum(1 for r in self.records
                       if str(r.get("fidelity_used")) == label)
            self.log("sample", f"{label} converged on {n_l1}/{len(self.records)} "
                     f"({100 * n_l1 / max(len(self.records), 1):.0f} %); "
                     "the rest kept their L0 value, labelled")
        gate = self.train()

        for rd in range(rounds):
            self.log("round", f"{rd + 1}/{rounds}")
            if not gate and self.fidelity >= 1:
                self.log("round", "the surrogate is not trusted yet: "
                         f"{2 * batch_size} more samples instead of a "
                         "search on a model that has not earned it")
                self._spend(self._lhs(2 * batch_size))
                gate = self.train()
                continue
            ga = NSGA2(self.plugin, self.spec, pop=pop,
                       generations=generations, seed=self.seed + rd,
                       use_pessimistic=True, on_generation=on_generation,
                       surrogate=self.surrogate if self.trained else None,
                       fidelity=self.fidelity)
            res = ga.run()
            self.n_cheap += int(res["evaluations"])
            self.log("search", f"{res['evaluations']} cheap evaluations, "
                     f"{len(res['front'])} on the predicted front"
                     + (" (L0 + learned residual)" if self.trained
                        else " (exact L0)"))
            if self.fidelity < 1:
                # At L0 the cheap physics IS the truth: every point the
                # search touched is a real evaluation and goes straight
                # onto the verified set. The rounds are still spent - each
                # is a fresh multi-start with its own seed.
                self.individuals.extend(res["population"])
                self.records.extend(i.rec for i in res["population"])
                self.history.append({"round": rd + 1,
                                     "evaluations": res["evaluations"],
                                     "verified": len(self.individuals)})
                continue
            batch = select_acquisition_batch(res["population"], batch_size,
                                             seed=self.seed + 100 * rd)
            self.log("acquire", f"{len(batch)} designs to the expensive "
                     "fidelity: predicted front, widest band, and the "
                     "feasibility frontier")
            spent = self._spend([i.theta for i in batch])
            best = min(spent, key=scalar_fitness) if spent else None
            self.history.append({
                "round": rd + 1, "evaluations": res["evaluations"],
                "verified": len(self.individuals),
                "surrogate": dict(self.gate),
                "best_f0": float(best.f[0]) if best else None,
                "best_feasible": bool(best.feasible) if best else False})
            if best is not None:
                self.log("round", f"best of the round: f0 {best.f[0]:+.4f}, "
                         + ("feasible" if best.feasible
                            else f"violation {best.v:.4g}"))
            gate = self.train()

        verified = verified_front(self.individuals)
        dt = time.perf_counter() - t0
        self.log("done", f"{len(self.individuals)} physical evaluations in "
                 f"{dt:.1f} s, {len(verified)} on the VERIFIED front")
        return DesignResult(
            population=list(self.individuals), front=verified,
            history=self.history, evaluations=len(self.individuals),
            n_cheap=self.n_cheap + len(self.individuals), fidelity=label,
            surrogate=dict(self.gate), trained=self.trained,
            merit="pessimistic", search="nsga2", seconds=dt,
            cell_gate=self.cell_gate() if self.trained else {})

    def _run_psl(self, n_init: int, rounds: int, batch_size: int) -> DesignResult:
        from .psl import PSL, PSLConfig
        t0 = time.perf_counter()
        cfg = self.psl_config or PSLConfig()
        engine = PSL(self.plugin, self.spec, seed=self.seed, cfg=cfg,
                     state=self.state, fidelity=max(self.fidelity, 1)
                     if self.fidelity >= 1 else 0, log=self.log)
        out = engine.run(n_init, rounds, batch_size)
        self.individuals = list(out["individuals"])
        self.records = [i.rec for i in self.individuals]
        self.state = engine.st
        verified = verified_front(self.individuals)
        dt = time.perf_counter() - t0
        self.log("done", f"{len(self.individuals)} physical evaluations in "
                 f"{dt:.1f} s, {len(verified)} on the VERIFIED front "
                 f"(device {engine.dev})")
        gate = dict(out.get("surrogate_gate", {}))
        return DesignResult(
            population=self.individuals, front=verified,
            history=list(out.get("history", [])), evaluations=len(self.individuals),
            n_cheap=int(out.get("n_cheap", 0)), fidelity=fidelity_label(engine.fidelity),
            surrogate=gate, trained=bool(gate), merit="pessimistic",
            search="psl", seconds=dt, state=engine.st,
            honesty=out.get("honesty", np.zeros((0, 3))),
            batch_log=list(out.get("batch_log", [])))

    def verified_front(self) -> list:
        return verified_front(self.individuals)


def design(plugin, spec, fidelity: int = 0, n_init: int = 60,
           rounds: int = 3, batch_size: int = 12, pop: int = 40,
           generations: int = 20, seed: int = SEED, log=None,
           on_generation=None, search: str = "nsga2", psl_config=None,
           state=None) -> DesignResult:
    """One plugin, one spec, designed. The entry point a product calls."""
    d = Designer(plugin, spec, fidelity=fidelity, seed=seed, log=log,
                 search=search, psl_config=psl_config, state=state)
    return d.run(n_init=n_init, rounds=rounds, batch_size=batch_size,
                 pop=pop, generations=generations,
                 on_generation=on_generation)


def design_sequence(plugin, specs: list, fidelity: int = 1, search: str = "psl",
                    n_init: int = 60, n_init_next: int | None = None,
                    rounds: int = 3, batch_size: int = 12, seed: int = SEED,
                    log=None, psl_config=None, **kw) -> list:
    """A sequence of specs. With ``search="psl"`` the actor, critics and
    data pools of one spec become the prior of the next, and
    ``n_init_next`` (default: ``batch_size``) is the short initial design
    the benchmark showed buys 40-60 % of the expensive budget. With
    ``search="nsga2"`` every spec starts cold, which is what the family
    does today."""
    out, state = [], None
    n_next = batch_size if n_init_next is None else int(n_init_next)
    for k, spec in enumerate(specs):
        n0 = n_init if (k == 0 or search != "psl") else n_next
        res = design(plugin, spec, fidelity=fidelity, n_init=n0, rounds=rounds,
                     batch_size=batch_size, seed=seed + k, log=log,
                     search=search, psl_config=psl_config, state=state, **kw)
        state = res.state if search == "psl" else None
        out.append(res)
    return out
