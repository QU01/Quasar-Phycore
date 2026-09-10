import numpy as np
import pytest

from phycore import ContractError, ENGINE_REVISION
from phycore.contracts import (ContractSchema, Mark, check_polygon, count_marks,
                               monotone_z, num, parse_mark, prior_width, strip_marks)
from phycore.product import cache_key, constants_fingerprint, run_meta
from phycore.vvuq import (AMBER, BLOCKED, GREEN, INFO, RED, Campaign, Criterion,
                          Score)


def test_schema_helpers_and_versioning():
    sch = ContractSchema("phyx-test-1", 1, 2, required_top=("rotor",))

    @sch.add
    def _rotor(doc, errs):
        num(doc["rotor"], "r_tip", "rotor", errs, lo=0.0)
        monotone_z(doc["rotor"].get("hub", []), "rotor.hub", errs, n_min=3)
        check_polygon(doc["rotor"].get("section", []), "rotor.section", errs, n_min=4)

    good = sch.stamp({"rotor": {"r_tip": 0.1, "hub": [[0, 1], [1, 1], [2, 1.2]],
                                "section": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]}})
    assert sch.validate(good) == []
    bad = {"schema": "other", "version": "2.0",
           "rotor": {"r_tip": -1, "hub": [[0, 1], [2, 1], [1, 1]],
                     "section": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]]}}
    errs = sch.validate(bad)
    text = "\n".join(errs)
    assert "schema" in text and "major" in text and "< 0" in text
    assert "retrocede" in text and "NO debe repetir" in text
    with pytest.raises(ContractError):
        sch.validate_or_raise(bad)


def test_marks():
    assert parse_mark("K_SHOCK = 0.70 [M] anchored to Rotor 37") is Mark.M
    assert count_marks("[V] [V] [?] [S]")["[V]"] == 2
    assert prior_width(Mark.V) == 0.05 and prior_width("[?]") is None
    assert prior_width("texto [S]") == 0.15 and prior_width(None) is None
    assert strip_marks("eta [V]") == "eta"


def test_campaign_blocks_and_subordinates():
    c = Campaign("test", [
        Criterion("A1", "loss", "<= 5 %", "NASA TP-1730"),
        Criterion("A2", "map", "<= 2 pts", "TN D-8063"),
        Criterion("A6", "efficiency", "<= 1 pt", "TP-1730", subordinate_to=("A2",)),
        Criterion("H1", "energy balance", "< 1e-6", "algebra", informational=True),
    ])

    @c.measure("A1")
    def _a1(crit):
        return Score(crit, RED, measured="7 %")

    @c.measure("A6")
    def _a6(crit):
        return Score(crit, GREEN, measured="0.4 pt")

    @c.measure("H1")
    def _h1(crit):
        return Score(crit, GREEN)

    c.block("A2", ("mapa CT-2 sin leyenda",))
    scores = c.score_all()
    v = {s.criterion.cid: s.verdict for s in scores}
    assert v == {"A1": RED, "A2": BLOCKED, "A6": INFO, "H1": INFO}
    g = c.gate(scores)
    assert not g["passed"]
    assert {b["cid"] for b in g["blocking"]} == {"A1", "A2"}
    assert "Silence is not a pass" in c.report(scores)
    # a criterion without a measurement is blocked, never green
    c2 = Campaign("t2", [Criterion("X", "x", "y", "z")])
    assert c2.score_all()[0].verdict == BLOCKED
    assert Campaign("t3", [Criterion("Y", "y", "t", "a")]).gate(
        [Score(Criterion("Y", "y", "t", "a"), AMBER)])["passed"]


def test_cache_key_and_fingerprint():
    class P:
        plugin_id = "p"
        model_revision = 3

    k1 = cache_key(P(), [0.1, 0.2], {"m": 1.0}, 1, constants="abc")
    k2 = cache_key(P(), [0.1, 0.2], {"m": 1.0}, 1, constants="abd")
    k3 = cache_key(P(), [0.1, 0.2], {"m": 1.0}, 0, constants="abc")
    assert len({k1, k2, k3}) == 3
    P.model_revision = 4
    assert cache_key(P(), [0.1, 0.2], {"m": 1.0}, 1, constants="abc") != k1
    f1 = constants_fingerprint({"K_SHOCK": 0.70, "lower": 1, "TABLE": (1, 2)})
    f2 = constants_fingerprint({"K_SHOCK": 0.71, "TABLE": (1, 2)})
    assert f1 != f2 and f1 == constants_fingerprint({"TABLE": (1, 2), "K_SHOCK": 0.70})
    m = run_meta(P(), {"A": 1.0}, extra={"n_init": 60})
    assert m["engine_revision"] == ENGINE_REVISION and m["n_init"] == 60
    assert "compute" in m and "backend" in m["compute"]
