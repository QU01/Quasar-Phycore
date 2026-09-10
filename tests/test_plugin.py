import numpy as np
import pytest

from phycore import validate_plugin, repair, violation
from phycore.adapters.bench import ProblemPlugin

problems = pytest.importorskip("problems")


def test_toy_plugin_is_valid():
    p = ProblemPlugin(problems.ToyHX())
    spec = (0.5, 0.5, 0.5)
    assert validate_plugin(p, spec) == []
    assert validate_plugin(p, spec, fidelity=1) == []


def test_broken_plugin_is_named():
    class Broken:
        plugin_id = "broken"
        model_revision = 1
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        integer_dims = ()
        objective_names = ("a",)

        def spec_vec(self, spec):
            return np.array([0.5])

        def evaluate(self, theta, spec, fidelity=0):
            return {"theta": theta}

        def objectives(self, rec, spec):
            return np.array([1.0, 2.0])         # wrong length

        def objective_bands(self, rec, spec):
            return np.array([-1.0])             # negative

        def constraints(self, rec, spec):
            return np.array([np.nan])

        def embedding(self, rec):
            return np.zeros(3)

        def residual(self, rec, spec):
            return None

        def apply_residual(self, rec, spec, res):
            return dict(rec)                     # no flag

    errs = validate_plugin(Broken(), None)
    text = "\n".join(errs)
    assert "fidelity_used" in text
    assert "objectives devuelve forma" in text
    assert ">= 0" in text
    assert "no finitos" in text
    assert "corrected_by_surrogate" in text


def test_bad_integer_dims_stop_the_validator():
    class P:
        plugin_id = "p"; model_revision = 1
        bounds = np.array([[0.0, 1.0]]); integer_dims = (5,); objective_names = ("a",)
    errs = validate_plugin(P(), None)
    assert len(errs) == 1 and "integer_dims" in errs[0]


def test_repair_and_violation():
    class P:
        bounds = np.array([[0.0, 1.0], [2.0, 9.0]])
        integer_dims = (1,)
    th = repair(P(), [1.7, 4.4])
    assert th.tolist() == [1.0, 4.0]
    assert violation([-1.0, 0.5, 0.25]) == 0.75
