"""Experimental metal-coordination probability density maps.

Ligand donor directions are accumulated in a metal-centered Fibonacci
spherical histogram, separately for O/N/S, using train-split crystal
complexes only. Querying a pose returns the mean density at each donor
direction times the GPDOCK radial probability.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .coordination import ligand_donors
from .io import Molecule
from .metals import METAL_ORDER, gp_probability, metal_from_name

_ROOT = Path(__file__).resolve().parents[1]
_PKG = Path(__file__).resolve().parent / "data" / "coord_density.npz"
_ROOT_MAP = _ROOT / "processed" / "maps" / "coord_density.npz"
_DEFAULT = _PKG if _PKG.exists() else _ROOT_MAP
N_RAYS = 48


def fibonacci_sphere(n: int = N_RAYS) -> np.ndarray:
    i = np.arange(max(n, 1), dtype=float)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - (2.0 * i + 1.0) / max(n, 1)
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    th = golden * i
    return np.column_stack((r * np.cos(th), r * np.sin(th), z))


@lru_cache(maxsize=1)
def load_maps(path: str | None = None) -> dict:
    p = Path(path) if path else _DEFAULT
    if not p.exists():
        return {
            "rays": fibonacci_sphere(),
            "density": {m: {el: np.ones(N_RAYS) / N_RAYS for el in "ONS"} for m in METAL_ORDER},
            "n_donors": {},
        }
    blob = np.load(p, allow_pickle=True)
    rays = np.asarray(blob["rays"], dtype=float)
    density = {}
    n_donors = {}
    for metal in METAL_ORDER:
        density[metal] = {}
        for el in "ONS":
            key = f"{metal}_{el}"
            density[metal][el] = np.asarray(blob[key], dtype=float) if key in blob.files else np.ones(len(rays)) / len(rays)
        nkey = f"{metal}_n"
        n_donors[metal] = int(blob[nkey]) if nkey in blob.files else 0
    return {"rays": rays, "density": density, "n_donors": n_donors}


def nearest_ray_index(unit: np.ndarray, rays: np.ndarray) -> np.ndarray:
    dots = np.asarray(unit, dtype=float) @ rays.T
    return np.argmax(dots, axis=1)


def map_overlap(
    metal: str,
    ligand: Molecule,
    metal_xyz: np.ndarray,
    maps: dict | None = None,
) -> float:
    """Mean directional density × radial GP probability for coordinating donors."""
    maps = maps or load_maps()
    metal = metal_from_name(metal)
    idx, xyz = ligand_donors(ligand)
    if idx.size == 0:
        return 0.0
    rays = maps["rays"]
    dens = maps["density"].get(metal, {})
    terms = []
    for i, coord in zip(idx, xyz):
        vec = coord - metal_xyz
        dist = float(np.linalg.norm(vec))
        if dist < 1.0e-6 or dist > 3.2:
            continue
        unit = (vec / dist).reshape(1, 3)
        el = str(ligand.elements[int(i)]).upper()[:1]
        ray_i = int(nearest_ray_index(unit, rays)[0])
        ang = float(dens.get(el, np.ones(len(rays)) / len(rays))[ray_i])
        rad = float(gp_probability(metal, el, dist))
        terms.append(ang * rad)
    if not terms:
        return 0.0
    return float(np.mean(terms))
