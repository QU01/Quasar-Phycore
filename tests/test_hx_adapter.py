"""Phy-HX behind the contract, without touching Phy-HX."""
import pytest

from phycore import design, validate_plugin
from phycore.adapters.hx import HXPlugin


@pytest.fixture(scope="module")
def hx():
    try:
        return HXPlugin()
    except Exception as exc:                          # noqa: BLE001
        pytest.skip(f"Phy-HX no importable: {exc!r}")


def test_hx_plugin_validates(hx):
    spec = hx.ds.recuperator_spec()
    assert validate_plugin(hx, spec) == []
    assert validate_plugin(hx, spec, fidelity=1) == []
    assert hx.spec_vec(spec).shape == (4,)


def test_hx_small_l0_design(hx):
    spec = hx.ds.recuperator_spec()
    res = design(hx, spec, fidelity=0, n_init=12, rounds=1, batch_size=4,
                 pop=8, generations=2, seed=4)
    assert res.evaluations == len(res.population) > 0
    assert all(i.cell == ("plate_fin_osf", "dry") for i in res.population)
    assert all(i.label == "plate_fin_osf" for i in res.population)
