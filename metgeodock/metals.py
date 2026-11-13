"""Metal-ion parameters ported from GPDOCK (Wang, Brief. Bioinform. 2023).

Distance histograms are 17 bins of 0.1 Angstrom starting at 1.7 Angstrom:
``bin = floor((d - 1.7) * 10)``, clipped to [0, 16].
Ion order matches GPDOCK: Zn, Ca, Mg, Mn, Fe2, Fe3, Cu, Co, Cd, Ni.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

METAL_ORDER: tuple[str, ...] = (
    "zn",
    "ca",
    "mg",
    "mn",
    "fe2",
    "fe3",
    "cu",
    "co",
    "cd",
    "ni",
)
METAL_INDEX: dict[str, int] = {name: i for i, name in enumerate(METAL_ORDER)}

_ALIASES: dict[str, str] = {
    "zn": "zn",
    "zn2": "zn",
    "zinc": "zn",
    "ca": "ca",
    "ca2": "ca",
    "calcium": "ca",
    "c0": "ca",
    "mg": "mg",
    "mg2": "mg",
    "magnesium": "mg",
    "mn": "mn",
    "mn2": "mn",
    "manganese": "mn",
    "fe2": "fe2",
    "fe(ii)": "fe2",
    "fe+2": "fe2",
    "fe3": "fe3",
    "fe(iii)": "fe3",
    "fe+3": "fe3",
    "fe": "fe3",
    "iron": "fe3",
    "cu": "cu",
    "cu2": "cu",
    "copper": "cu",
    "co": "co",
    "co2": "co",
    "cobalt": "co",
    "cd": "cd",
    "cd2": "cd",
    "cadmium": "cd",
    "cad": "cd",
    "ni": "ni",
    "ni2": "ni",
    "nickel": "ni",
}

PDB_RESNAMES: dict[str, tuple[str, ...]] = {
    "zn": ("ZN", "ZN2"),
    "ca": ("CA", "CA2", "C0"),
    "mg": ("MG", "MG2"),
    "mn": ("MN", "MN2", "MN3"),
    "fe2": ("FE2", "FE"),
    "fe3": ("FE", "FE3"),
    "cu": ("CU", "CU1", "CU2"),
    "co": ("CO", "CO2"),
    "cd": ("CD", "CD2", "CAD"),
    "ni": ("NI", "NI2"),
}

# GPDOCK tables (gpml_dock.c). Distances in Angstrom.
DIS_MAX_O = np.array([2.7, 3.2, 3.0, 2.8, 2.8, 2.8, 2.8, 2.8, 3.1, 2.8])
DIS_MAX_N = np.array([2.6, 3.4, 3.2, 2.8, 2.7, 2.8, 2.6, 2.6, 2.8, 2.6])
DIS_MAX_S = np.array([2.8, 3.4, 3.2, 2.9, 2.7, 2.7, 2.8, 2.7, 3.0, 2.7])
DIS_MIN_O = np.array([1.7, 2.0, 1.8, 1.8, 1.8, 1.7, 1.7, 1.7, 1.9, 1.7])
DIS_MIN_N = np.array([1.8, 2.0, 1.8, 1.9, 1.9, 1.8, 1.8, 1.8, 2.0, 1.8])
DIS_MIN_S = np.array([2.0, 2.7, 1.7, 2.1, 1.9, 2.1, 2.0, 2.1, 2.2, 1.9])
DIS_MIN = np.array([1.7, 2.0, 1.7, 1.8, 1.8, 1.7, 1.7, 1.7, 1.9, 1.7])
DIS_MAX = np.array([2.8, 3.3, 3.2, 2.9, 2.8, 2.8, 2.8, 2.8, 3.1, 2.8])

PROB_O = np.array(
    [
        [0.01, 0.05, 0.14, 0.22, 0.18, 0.14, 0.08, 0.07, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.01, 0.03, 0.12, 0.24, 0.23, 0.14, 0.08, 0.05, 0.03, 0.02, 0.02, 0.01, 0.00, 0.00],
        [0.00, 0.03, 0.07, 0.24, 0.24, 0.13, 0.08, 0.06, 0.04, 0.03, 0.03, 0.02, 0.02, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.02, 0.05, 0.13, 0.26, 0.24, 0.11, 0.08, 0.04, 0.03, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.03, 0.07, 0.14, 0.18, 0.20, 0.13, 0.10, 0.07, 0.04, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.01, 0.05, 0.10, 0.19, 0.20, 0.18, 0.10, 0.07, 0.04, 0.03, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.02, 0.06, 0.10, 0.15, 0.14, 0.14, 0.10, 0.10, 0.08, 0.10, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.02, 0.05, 0.10, 0.19, 0.24, 0.18, 0.08, 0.06, 0.03, 0.02, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.01, 0.03, 0.06, 0.15, 0.18, 0.19, 0.16, 0.10, 0.04, 0.02, 0.02, 0.01, 0.00, 0.00, 0.00],
        [0.01, 0.03, 0.07, 0.26, 0.26, 0.16, 0.07, 0.05, 0.03, 0.02, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    ],
    dtype=float,
)
PROB_N = np.array(
    [
        [0.00, 0.02, 0.10, 0.40, 0.26, 0.12, 0.04, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.03, 0.05, 0.06, 0.08, 0.10, 0.07, 0.08, 0.08, 0.05, 0.05, 0.04, 0.07, 0.10, 0.12],
        [0.00, 0.02, 0.03, 0.06, 0.13, 0.18, 0.12, 0.09, 0.07, 0.06, 0.07, 0.04, 0.04, 0.05, 0.03, 0.00, 0.00],
        [0.00, 0.00, 0.02, 0.07, 0.22, 0.34, 0.17, 0.08, 0.04, 0.02, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.03, 0.10, 0.26, 0.29, 0.16, 0.09, 0.04, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.01, 0.04, 0.14, 0.30, 0.28, 0.11, 0.05, 0.03, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.03, 0.15, 0.44, 0.22, 0.09, 0.03, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.02, 0.07, 0.26, 0.35, 0.18, 0.06, 0.03, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.02, 0.07, 0.30, 0.29, 0.16, 0.09, 0.04, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.02, 0.06, 0.18, 0.28, 0.29, 0.09, 0.04, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    ],
    dtype=float,
)
PROB_S = np.array(
    [
        [0.00, 0.00, 0.00, 0.01, 0.06, 0.27, 0.43, 0.14, 0.04, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.06, 0.00, 0.11, 0.11, 0.33, 0.33, 0.06],
        [0.03, 0.00, 0.03, 0.03, 0.00, 0.03, 0.03, 0.06, 0.18, 0.27, 0.12, 0.06, 0.06, 0.00, 0.09, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.02, 0.06, 0.08, 0.12, 0.31, 0.25, 0.10, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.01, 0.04, 0.03, 0.21, 0.36, 0.20, 0.10, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.07, 0.43, 0.31, 0.11, 0.04, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.02, 0.18, 0.30, 0.12, 0.13, 0.13, 0.08, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.14, 0.36, 0.26, 0.11, 0.07, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.06, 0.13, 0.28, 0.29, 0.13, 0.04, 0.02, 0.02, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.02, 0.04, 0.27, 0.30, 0.14, 0.08, 0.10, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    ],
    dtype=float,
)
PROB_HOH = np.array(
    [
        [0.02, 0.05, 0.10, 0.17, 0.19, 0.16, 0.10, 0.09, 0.06, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.01, 0.03, 0.09, 0.22, 0.27, 0.15, 0.08, 0.06, 0.03, 0.02, 0.02, 0.01, 0.00, 0.00],
        [0.00, 0.03, 0.06, 0.28, 0.28, 0.12, 0.07, 0.06, 0.03, 0.02, 0.02, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.02, 0.06, 0.08, 0.12, 0.31, 0.25, 0.10, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.02, 0.04, 0.08, 0.16, 0.22, 0.18, 0.15, 0.08, 0.03, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.02, 0.04, 0.07, 0.15, 0.19, 0.19, 0.13, 0.08, 0.06, 0.04, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.02, 0.06, 0.10, 0.12, 0.13, 0.16, 0.11, 0.11, 0.08, 0.09, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.02, 0.05, 0.09, 0.17, 0.25, 0.18, 0.09, 0.06, 0.05, 0.02, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.02, 0.02, 0.03, 0.06, 0.14, 0.16, 0.19, 0.16, 0.11, 0.04, 0.02, 0.02, 0.01, 0.01, 0.00, 0.00],
        [0.02, 0.04, 0.07, 0.18, 0.25, 0.18, 0.09, 0.06, 0.05, 0.03, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    ],
    dtype=float,
)

CORD_MAX = np.array([6, 8, 7, 7, 7, 7, 6, 7, 8, 7], dtype=int)
CORD_MIN = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=int)

FORMAL_CHARGE = {
    "zn": 2.0,
    "ca": 2.0,
    "mg": 2.0,
    "mn": 2.0,
    "fe2": 2.0,
    "fe3": 3.0,
    "cu": 2.0,
    "co": 2.0,
    "cd": 2.0,
    "ni": 2.0,
}

# Preferred coordination polyhedra for vacant-site matching (GM-DockZn motifs).
PREFERRED_GEOMETRIES: dict[str, tuple[str, ...]] = {
    "zn": ("td", "tbp", "oh"),
    "ca": ("oh", "tbp", "td"),
    "mg": ("oh", "tbp", "td"),
    "mn": ("oh", "tbp", "td"),
    "fe2": ("oh", "tbp", "td"),
    "fe3": ("oh", "tbp", "td"),
    "cu": ("sp", "spy", "td", "oh"),
    "co": ("oh", "td", "tbp"),
    "cd": ("oh", "td", "tbp"),
    "ni": ("sp", "oh", "td", "tbp"),
}


@dataclass(frozen=True)
class MetalParams:
    name: str
    index: int
    charge: float
    dis_min: float
    dis_max: float
    geometries: tuple[str, ...]


METALS: Mapping[str, MetalParams] = {
    name: MetalParams(
        name=name,
        index=i,
        charge=FORMAL_CHARGE[name],
        dis_min=float(DIS_MIN[i]),
        dis_max=float(DIS_MAX[i]),
        geometries=PREFERRED_GEOMETRIES[name],
    )
    for i, name in enumerate(METAL_ORDER)
}


def metal_from_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "")
    if key not in _ALIASES:
        raise KeyError(f"Unknown metal ion: {name!r}")
    return _ALIASES[key]


def distance_bin(distance_angstrom: float) -> int:
    """GPDOCK histogram index for a metal–donor distance."""
    idx = int(np.floor((float(distance_angstrom) - 1.7) * 10.0))
    return int(np.clip(idx, 0, 16))


def gp_probability(metal: str, element: str, distance_angstrom: float) -> float:
    """Look up GPDOCK coordination probability for one metal–donor contact."""
    idx = METAL_INDEX[metal_from_name(metal)]
    el = element.upper()[0]
    table = PROB_O if el == "O" else PROB_N if el == "N" else PROB_S if el == "S" else PROB_O
    p = float(table[idx, distance_bin(distance_angstrom)])
    return max(p, 1.0e-4)


def donor_cutoffs(metal: str) -> dict[str, tuple[float, float]]:
    idx = METAL_INDEX[metal_from_name(metal)]
    return {
        "O": (float(DIS_MIN_O[idx]), float(DIS_MAX_O[idx])),
        "N": (float(DIS_MIN_N[idx]), float(DIS_MAX_N[idx])),
        "S": (float(DIS_MIN_S[idx]), float(DIS_MAX_S[idx])),
    }


def _table_for_element(element: str) -> np.ndarray:
    el = element.upper()[0]
    if el == "N":
        return PROB_N
    if el == "S":
        return PROB_S
    return PROB_O


def histogram_mode_distance(metal: str, element: str) -> float:
    """Distance (Angstrom) at the GPDOCK histogram mode for one metal–element pair."""
    idx = METAL_INDEX[metal_from_name(metal)]
    mode = int(np.argmax(_table_for_element(element)[idx]))
    return 1.7 + (mode + 0.5) * 0.1


def peak_probability(metal: str, element: str) -> float:
    """Maximum GPDOCK bin probability for this metal–element pair."""
    idx = METAL_INDEX[metal_from_name(metal)]
    return float(np.max(_table_for_element(element)[idx]))


def expected_coordination(metal: str) -> tuple[int, int]:
    """Inclusive (min, max) first-shell coordination numbers from GPDOCK."""
    idx = METAL_INDEX[metal_from_name(metal)]
    return int(CORD_MIN[idx]), int(CORD_MAX[idx])
