"""Smoke tests that do not require crystal trees or ChEMBL."""

from metgeodock import METALS, __version__, dock_ligand
from metgeodock.forcefield import load_params
from metgeodock.maps import load_maps
from metgeodock.metals import METAL_ORDER


def test_version():
    assert __version__


def test_ten_metals():
    assert METAL_ORDER == ("zn", "ca", "mg", "mn", "fe2", "fe3", "cu", "co", "cd", "ni")
    for name in METAL_ORDER:
        assert name in METALS


def test_packaged_params():
    params = load_params()
    assert "pairs" in params
    assert "zn" in params.get("pairs", {}) or params.get("weights")


def test_packaged_maps():
    maps = load_maps()
    assert "rays" in maps or "density" in maps


def test_dock_ligand_is_callable():
    assert callable(dock_ligand)
