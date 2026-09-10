"""Phy-1 in phycore is the family's loop: bit-exact against optimizer_hx."""
import numpy as np
import pytest

problems = pytest.importorskip("problems")
baseline = pytest.importorskip("baseline")

from phycore import Designer                       # noqa: E402
from phycore.adapters.bench import ProblemPlugin   # noqa: E402


@pytest.mark.parametrize("seed", [3, 11])
def test_bit_exact_against_phy_hx_loop(seed):
    prob = problems.ToyHX()
    spec = (0.3, 0.6, 0.4)
    ref = baseline.run_baseline(prob, spec, seed, n_init=30, rounds=2,
                                batch_size=6, pop=16, generations=4)
    d = Designer(ProblemPlugin(prob), spec, fidelity=1, seed=seed)
    res = d.run(n_init=30, rounds=2, batch_size=6, pop=16, generations=4)
    got = res.arrays()
    assert got["n_exp"] == ref["n_exp"]
    np.testing.assert_allclose(got["theta"], ref["theta"], rtol=0, atol=0)
    np.testing.assert_allclose(got["F"], ref["F"], rtol=0, atol=0)
    assert got["ok"].tolist() == ref["ok"].tolist()
    assert res.n_cheap == ref["n_cheap"]
    assert d.gate.get("n") == ref["surrogate_gate"].get("n")
