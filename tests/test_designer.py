import numpy as np
import pytest

from phycore import Designer, design, design_sequence
from phycore.adapters.bench import ProblemPlugin
from phycore.engine.dominance import front_comparison, joint_front

problems = pytest.importorskip("problems")

SPEC = (0.4, 0.5, 0.6)


def _plugin():
    return ProblemPlugin(problems.ToyHX())


def test_nsga2_at_l0_verified_only():
    res = design(_plugin(), SPEC, fidelity=0, n_init=20, rounds=1, batch_size=4,
                 pop=12, generations=3, seed=1)
    assert res.search == "nsga2" and res.fidelity == "L0"
    assert res.front and all(i.feasible for i in res.front)
    assert not any(i.rec.get("corrected_by_surrogate") for i in res.population)
    assert res.evaluations == len(res.population)


def test_nsga2_at_l1_trains_and_gates():
    logs = []
    d = Designer(_plugin(), SPEC, fidelity=1, seed=2, log=lambda *a: logs.append(a))
    res = d.run(n_init=32, rounds=2, batch_size=6, pop=16, generations=4)
    assert res.trained and res.surrogate["n"] >= 24
    assert res.evaluations == 32 + 2 * 6
    assert any(a[0] == "surrogate" for a in logs)
    assert res.cell_gate and "cells" in res.cell_gate
    fc = front_comparison(res.population)
    assert set(fc) >= {"jaccard", "reordered"}
    assert len(joint_front(res.population)) >= 1


def test_psl_runs_and_carries_state():
    torch = pytest.importorskip("torch")
    from phycore.engine.psl import PSLConfig
    cfg = PSLConfig(actor_steps=20, n_candidates=120, n_cheap_init=0, device="cpu",
                    k_ensemble=2, max_inner=0, inner_fixed=0)
    p = _plugin()
    r1 = design(p, SPEC, fidelity=1, n_init=30, rounds=2, batch_size=6, seed=5,
                search="psl", psl_config=cfg)
    assert r1.search == "psl" and r1.state is not None
    assert r1.evaluations == len(r1.population) == 30 + 12
    assert r1.front and all(not i.rec.get("corrected_by_surrogate") for i in r1.front)
    assert r1.n_cheap > r1.evaluations
    r2 = design(p, (0.7, 0.2, 0.5), fidelity=1, n_init=6, rounds=1, batch_size=6,
                seed=6, search="psl", psl_config=cfg, state=r1.state)
    assert r2.state is r1.state
    assert len(r2.state.prev_specs) == 2
    assert r2.evaluations == 12


def test_psl_rejects_foreign_state():
    pytest.importorskip("torch")
    from phycore.engine.psl import PSLConfig, PSLState
    cfg = PSLConfig(actor_steps=1, n_candidates=20, n_cheap_init=0, device="cpu")
    p = _plugin()
    st = PSLState(p, cfg, 0, "cpu")
    st.plugin_id = "other"
    with pytest.raises(ValueError):
        design(p, SPEC, fidelity=1, n_init=4, rounds=1, batch_size=2, search="psl",
               psl_config=cfg, state=st)


def test_design_sequence_threads_state():
    pytest.importorskip("torch")
    from phycore.engine.psl import PSLConfig
    cfg = PSLConfig(actor_steps=10, n_candidates=80, n_cheap_init=0, device="cpu",
                    k_ensemble=2, max_inner=0, inner_fixed=0)
    specs = [(0.3, 0.3, 0.3), (0.6, 0.6, 0.6)]
    out = design_sequence(_plugin(), specs, fidelity=1, search="psl", n_init=26,
                          n_init_next=4, rounds=1, batch_size=4, seed=9, psl_config=cfg)
    assert [r.evaluations for r in out] == [30, 8]
    assert out[1].state is out[0].state
    cold = design_sequence(_plugin(), specs, fidelity=1, search="nsga2", n_init=26,
                           rounds=1, batch_size=4, seed=9, pop=10, generations=2)
    assert all(r.state is None for r in cold)


def test_bad_search_name():
    with pytest.raises(ValueError):
        Designer(_plugin(), SPEC, search="genetic")
