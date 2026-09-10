import numpy as np
import pytest

from phycore.engine.hypervolume import hypervolume, reference_point


def test_2d_known():
    F = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    ref = np.array([4.0, 4.0])
    # union of rectangles: 3*1 + 2*1 + 1*1 ... exact = 6
    assert hypervolume(F, ref) == pytest.approx(6.0)


def test_dominated_and_outside_points_add_nothing():
    F = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [2.5, 2.5], [5.0, 0.5]])
    assert hypervolume(F, [4.0, 4.0]) == pytest.approx(6.0)
    assert hypervolume(np.zeros((0, 2)), [1, 1]) == 0.0


def test_3d_box():
    # one point at the origin below ref (1,1,1): volume 1
    assert hypervolume(np.zeros((1, 3)), np.ones(3)) == pytest.approx(1.0)
    # two staggered points
    F = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
    # each 0.5*1*... compute: union of [0,1]x[.5,1]x[.5,1] (=.5*.5*.5=.125)
    # and [.5,1]x[0,1]x[.5,1] (.125) minus overlap [.5,1]x[.5,1]x[.5,1] (.125)... careful:
    # box1 = x in[0,1], y in[.5,1], z in[.5,1] -> 1*.5*.5 = .25
    # box2 = x in[.5,1], y in[0,1], z in[.5,1] -> .5*1*.5 = .25
    # overlap = x[.5,1] y[.5,1] z[.5,1] -> .125 ; union = .375
    assert hypervolume(F, np.ones(3)) == pytest.approx(0.375)


def test_matches_pymoo_when_available():
    pymoo = pytest.importorskip("pymoo")
    from pymoo.indicators.hv import HV
    rng = np.random.default_rng(3)
    for d in (2, 3):
        F = rng.random((30, d))
        ref = np.ones(d)
        assert hypervolume(F, ref) == pytest.approx(float(HV(ref_point=ref)(F)), rel=1e-9)


def test_reference_point_union():
    r = reference_point([np.array([[0, 1]]), np.array([[1, 0]])], 2)
    assert np.all(r > 1.0)
    assert reference_point([], 2).tolist() == [1.0, 1.0]
