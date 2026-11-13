"""MetGeoDock: geometry-matching virtual screening for metalloproteins.

Combines GM-DockZn vacant-site geometry with the MetGeo statistical
force field (GPDOCK histograms; no SLEF, no Vina fusion) and CoordNet,
an MLP on metal-warhead probes plus Morgan fingerprints.

Two virtual-screening schemes:

* ``metalgeo`` / ``metgeo`` — dock then rank by MetGeo FF (lower better)
* ``coordnet`` / ``ml`` — neural-net ligand score (higher P(active) better)
"""

from .coordination import CoordinationSphere, detect_coordination, pocket_metal_indices
from .dock import DockResult, dock_ligand
from .io import Molecule, merge_with_waters, read_mol2, read_pdb, strip_waters
from .metals import METAL_INDEX, METALS, metal_from_name
from .scoring import HybridScore, score_pose
from .screen import screen_library

__version__ = "0.1.0"
__all__ = [
    "CoordinationSphere",
    "DockResult",
    "HybridScore",
    "METALS",
    "METAL_INDEX",
    "Molecule",
    "detect_coordination",
    "dock_ligand",
    "merge_with_waters",
    "metal_from_name",
    "pocket_metal_indices",
    "read_mol2",
    "read_pdb",
    "score_pose",
    "screen_library",
    "strip_waters",
]
