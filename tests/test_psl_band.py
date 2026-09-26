"""PSL's residual band on the actor's proposals (ENGINE_REVISION 3).

The critic's own split conformal is calibrated on a random split of its
pool, which is not the population it is judged on: the actor's proposals,
chosen BECAUSE the pessimistic prediction looked good. The proposal band
calibrates on fresh truths only (scored before they trained the critic)
and never on the round it measures.
"""
import numpy as np
import pytest

from phycore.uq.conformal import ALPHA, conformal_quantile, group_conformal_quantile


def test_group_quantile_needs_a_populated_group():
    q, per = group_conformal_quantile({"a": np.ones(7), "b": []})
    assert q is None and per == {}
    # the floor is conformal_quantile's own: n_min below eight is not honoured
    q, per = group_conformal_quantile({"a": np.arange(7.0)}, n_min=3)
    assert q is None


def test_group_quantile_is_the_max_of_the_populated_groups():
    rng = np.random.default_rng(0)
    a, b = rng.exponential(1.0, 40), rng.exponential(3.0, 12)
    q, per = group_conformal_quantile({"a": a, "b": b, "c": np.ones(3)})
    assert set(per) == {"a", "b"}
    assert per["a"] == conformal_quantile(a, np.ones(len(a)), ALPHA)
    assert per["b"] == conformal_quantile(b, np.ones(len(b)), ALPHA)
    assert q == max(per.values())


def test_group_max_covers_within_every_group():
    """Two overlapping groups with different score scales: a point of the
    narrow group that is also in the wide one must still be covered at
    1 - alpha, and so must the wide group (Barber et al. 2021)."""
    rng = np.random.default_rng(1)
    hits = {"narrow": [], "wide": []}
    for _ in range(400):
        cal_all = np.abs(rng.normal(0, 1.0, 60))         # every proposal
        cal_spec = np.abs(rng.normal(0, 2.0, 20))        # this spec: harder
        q, _ = group_conformal_quantile({"proposals": cal_all, "spec": cal_spec})
        hits["narrow"].append(abs(rng.normal(0, 1.0)) <= q)
        hits["wide"].append(abs(rng.normal(0, 2.0)) <= q)
    assert np.mean(hits["wide"]) >= 1 - ALPHA - 0.03
    assert np.mean(hits["narrow"]) >= 1 - ALPHA


problems = pytest.importorskip("problems")


def _plugin():
    from phycore.adapters.bench import ProblemPlugin
    return ProblemPlugin(problems.ToyHX())


def _cfg(**kw):
    from phycore.engine.psl import PSLConfig
    return PSLConfig(actor_steps=8, n_candidates=60, n_cheap_init=0, device="cpu",
                     k_ensemble=2, max_inner=0, inner_fixed=0, **kw)


def test_proposal_band_calibrates_on_past_truths_only():
    pytest.importorskip("torch")
    from phycore import design_sequence
    specs = [(0.3, 0.3, 0.3), (0.6, 0.6, 0.6)]
    out = design_sequence(_plugin(), specs, fidelity=1, search="psl", n_init=30,
                          n_init_next=10, rounds=3, batch_size=5, seed=4, psl_config=_cfg())
    st = out[-1].state
    # the second spec's initial truths were scored by the INHERITED critic
    key2 = tuple(np.round(np.asarray(specs[1], float), 6).tolist())
    assert sum(1 for k, kind, _ in st.fresh_scores if kind == "init") >= 8
    assert all(k == key2 for k, kind, _ in st.fresh_scores if kind == "init")
    n_seen = 0
    for res in out:
        for h in res.history:
            band = h["band"]
            if band["source"] == "proposal":
                # calibrated on the proposals scored BEFORE this round, never on its own
                assert band["n"]["proposals"] == n_seen
                assert band["c"] == max(band["q"].values()) >= band["q"]["split"] > 0
            n_seen += h["scored"]
    # first round of the sequence: nothing fresh yet, the critic's own split stays
    assert out[0].history[0]["band"]["source"] == "split"
    assert any(h["band"]["source"] == "proposal" for r in out for h in r.history)
    # the honesty rows and the fresh proposal scores are the same truths
    n_prop = sum(1 for _k, kind, _s in st.fresh_scores if kind == "proposal")
    assert n_prop == n_seen == sum(len(r.honesty) for r in out)
    # the declared band is c * sd_raw: every honesty row's score is err / (sd / c)
    for r in out:
        assert np.all(r.honesty[:, 1] > 0)


def test_split_band_is_the_previous_behaviour():
    pytest.importorskip("torch")
    from phycore import design
    res = design(_plugin(), (0.4, 0.5, 0.6), fidelity=1, n_init=30, rounds=2, batch_size=5,
                 seed=3, search="psl", psl_config=_cfg(band="split"))
    assert all(h["band"]["source"] == "split" for h in res.history)
    assert res.state.fresh_scores == []
