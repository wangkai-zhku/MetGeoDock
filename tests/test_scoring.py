from metgeodock.metals import expected_coordination, histogram_mode_distance
from metgeodock.scoring import DEFAULT_WEIGHTS, coordination_penalty


def test_default_weights_exclude_vina():
    assert DEFAULT_WEIGHTS["vina"] == 0.0
    assert DEFAULT_WEIGHTS["slef"] == 0.0
    assert DEFAULT_WEIGHTS["ff"] > 0.0


def test_zn_n_mode_distance_near_2():
    d = histogram_mode_distance("zn", "N")
    assert 1.95 <= d <= 2.15


def test_coordination_penalty_in_range_is_zero():
    lo, hi = expected_coordination("zn")
    assert coordination_penalty("zn", lo, 0) == 0.0
    assert coordination_penalty("zn", hi, 0) == 0.0
    assert coordination_penalty("zn", 0, 1) > 0.0
