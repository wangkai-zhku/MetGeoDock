from metgeodock.geometry import kabsch, rmsd_or
from metgeodock.metals import distance_bin, gp_probability, metal_from_name
from metgeodock.templates import IDEAL
import numpy as np


def test_metal_aliases():
    assert metal_from_name("ZN") == "zn"
    assert metal_from_name("Fe(II)") == "fe2"


def test_distance_bin_gpdock():
    assert distance_bin(1.70) == 0
    assert distance_bin(2.05) == 3
    assert distance_bin(0.1) == 0
    assert distance_bin(9.0) == 16


def test_kabsch_identity():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(8, 3))
    r, t, rmsd = kabsch(pts, pts)
    assert rmsd < 1e-8


def test_rmsd_or_ideal_td():
    metal = np.zeros(3)
    pts = IDEAL["td"] * 2.05
    rmsd, perm = rmsd_or(pts, metal, IDEAL["td"])
    assert rmsd < 1e-6
    assert len(perm) == 4


def test_zn_n_probability_peak():
    # Zn-N histogram peaks at bin 3 (2.00-2.10 A) with 0.40
    p = gp_probability("zn", "N", 2.05)
    assert p > 0.3


def test_ten_metals_have_geometries():
    from metgeodock.metals import METAL_ORDER, METALS

    assert len(METAL_ORDER) == 10
    for name in METAL_ORDER:
        assert METALS[name].geometries
