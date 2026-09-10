"""GPU and JIT paths: skipped where the hardware or the compiler is absent."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from phycore import backend as be
from phycore import design
from phycore.adapters.bench import ProblemPlugin

problems = pytest.importorskip("problems")
SPEC = (0.4, 0.5, 0.6)


@pytest.mark.skipif(not be.cuda_available("torch"), reason="sin CUDA")
def test_psl_on_cuda_matches_contract():
    from phycore.engine.psl import PSLConfig
    cfg = PSLConfig(actor_steps=20, n_candidates=120, n_cheap_init=0, device="cuda",
                    k_ensemble=2, max_inner=0, inner_fixed=0)
    res = design(ProblemPlugin(problems.ToyHX()), SPEC, fidelity=1, n_init=30, rounds=1,
                 batch_size=6, seed=5, search="psl", psl_config=cfg)
    assert res.state.dev == "cuda"
    assert next(res.state.actor.parameters()).is_cuda
    assert res.evaluations == 36 and res.front


@pytest.mark.skipif(not be.cuda_available("torch"), reason="sin CUDA")
def test_psl_cuda_state_moves_to_cpu():
    from phycore.engine.psl import PSLConfig
    cfg = PSLConfig(actor_steps=5, n_candidates=60, n_cheap_init=0, device="cuda",
                    k_ensemble=2, max_inner=0, inner_fixed=0)
    p = ProblemPlugin(problems.ToyHX())
    r1 = design(p, SPEC, fidelity=1, n_init=26, rounds=1, batch_size=4, seed=1,
                search="psl", psl_config=cfg)
    cfg2 = PSLConfig(**{**cfg.__dict__, "device": "cpu"})
    r2 = design(p, (0.6, 0.4, 0.5), fidelity=1, n_init=4, rounds=1, batch_size=4, seed=2,
                search="psl", psl_config=cfg2, state=r1.state)
    assert r2.state.dev == "cpu" and not next(r2.state.actor.parameters()).is_cuda


def test_psl_compile_flag_is_guarded():
    pytest.importorskip("torch")
    from phycore.engine.psl import PSLConfig
    cfg = PSLConfig(actor_steps=5, n_candidates=60, n_cheap_init=0, device="cpu",
                    k_ensemble=2, max_inner=0, inner_fixed=0, compile=True)
    res = design(ProblemPlugin(problems.ToyHX()), SPEC, fidelity=1, n_init=26, rounds=1,
                 batch_size=4, seed=3, search="psl", psl_config=cfg)
    assert res.state.jit in ("torch.compile", "none")
    assert res.evaluations == 30
    st = be.compile_status()
    assert (res.state.jit == "torch.compile") == bool(st["ok"]) or st["error"] is not None


def test_jax_jit_backend():
    jax = pytest.importorskip("jax")
    xp, name = be.get_backend("jax")
    assert name == "jax"

    @be.jit(backend="jax")
    def f(x):
        return (xp.sin(x) ** 2).sum()
    assert float(f(xp.ones(4))) == pytest.approx(4 * np.sin(1.0) ** 2)
    assert f.__phycore_jit__ == "jax.jit"
    g = be.grad(lambda x: (x ** 2).sum())
    if g is not None:
        assert np.allclose(np.asarray(g(xp.array([1.0, 2.0]))), [2.0, 4.0])
