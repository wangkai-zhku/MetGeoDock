"""Metal-coordination probes and ligand fingerprints.

Probe SMARTS encode common metalloprotein warheads. Counts plus a compact
Morgan fingerprint give a ligand-level descriptor that does not require a
docked pose. Directional map overlap is added when a pose is available.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .io import Molecule
from .maps import map_overlap
from .metals import METAL_ORDER, metal_from_name, peak_probability

PROBE_SMARTS: dict[str, str] = {
    "hydroxamate": "[#8]=[#6]-[#7]-[#8]",
    "carboxylate": "C(=O)[O,OH,O-]",
    "imidazole": "[n,N]1[c,C][n,N][c,C][c,C]1",
    "thiol": "[#16;X2]",
    "phosphate": "P(=O)([O,OH,O-])",
    "catechol": "c1c([OH,O-])c([OH,O-])ccc1",
    "sulfonamide": "S(=O)(=O)[N,n]",
    "pyridine": "n1ccccc1",
    "phenol": "c[OH]",
    "hydroxyl": "[#8;H1]",
}

PROBE_NAMES: tuple[str, ...] = tuple(PROBE_SMARTS)


def _rdkit_mol(smiles: str | None, ligand: Molecule | None = None):
    from rdkit import Chem

    if smiles:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            return mol
    if ligand is None:
        return None
    # Element-only fallback: no bonds → skip SMARTS, return None
    return None


def probe_counts(smiles: str | None, ligand: Molecule | None = None) -> dict[str, float]:
    mol = _rdkit_mol(smiles, ligand)
    counts = {name: 0.0 for name in PROBE_NAMES}
    if mol is None:
        if ligand is not None:
            els = [str(e).upper()[:1] for e in ligand.elements]
            counts["hydroxyl"] = float(sum(e == "O" for e in els))
            counts["thiol"] = float(sum(e == "S" for e in els))
        return counts
    from rdkit import Chem

    for name, smarts in PROBE_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        counts[name] = float(len(mol.GetSubstructMatches(patt)))
    return counts


def morgan_bits(smiles: str | None, n_bits: int = 128, radius: int = 2) -> np.ndarray:
    vec = np.zeros(n_bits, dtype=float)
    if not smiles:
        return vec
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return vec
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    vec = np.zeros(n_bits, dtype=float)
    from rdkit.DataStructs import ConvertToNumpyArray

    ConvertToNumpyArray(fp, vec)
    return vec


def donor_stats(smiles: str | None, ligand: Molecule | None, metal: str) -> dict[str, float]:
    metal = metal_from_name(metal)
    n_o = n_n = n_s = 0.0
    if ligand is not None:
        for el in ligand.elements:
            e = str(el).upper()[:1]
            if e == "O":
                n_o += 1
            elif e == "N":
                n_n += 1
            elif e == "S":
                n_s += 1
    elif smiles:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            for atom in mol.GetAtoms():
                s = atom.GetSymbol()
                if s == "O":
                    n_o += 1
                elif s == "N":
                    n_n += 1
                elif s == "S":
                    n_s += 1
    prior = (
        n_o * peak_probability(metal, "O")
        + n_n * peak_probability(metal, "N")
        + n_s * peak_probability(metal, "S")
    )
    return {
        "n_o": n_o,
        "n_n": n_n,
        "n_s": n_s,
        "n_donor": n_o + n_n + n_s,
        "coord_prior": float(prior),
        "n_heavy": float(ligand.heavy_mask().sum()) if ligand is not None else n_o + n_n + n_s,
    }


def ligand_feature_vector(
    metal: str,
    smiles: str | None = None,
    ligand: Molecule | None = None,
    metal_xyz: np.ndarray | None = None,
    include_map: bool = True,
) -> dict[str, float]:
    """Fixed-length descriptor for CoordNet (ligand-level + optional pose map)."""
    metal = metal_from_name(metal)
    feats: dict[str, Any] = {}
    feats.update(probe_counts(smiles, ligand))
    feats.update(donor_stats(smiles, ligand, metal))
    for name in METAL_ORDER:
        feats[f"metal_{name}"] = 1.0 if name == metal else 0.0
    overlap = 0.0
    if include_map and ligand is not None and metal_xyz is not None:
        overlap = map_overlap(metal, ligand, metal_xyz)
    feats["map_overlap"] = float(overlap)
    bits = morgan_bits(smiles)
    for i, v in enumerate(bits):
        feats[f"ecfp_{i}"] = float(v)
    return feats


COORDNET_COLUMNS: tuple[str, ...] = tuple(
    list(PROBE_NAMES)
    + ["n_o", "n_n", "n_s", "n_donor", "coord_prior", "n_heavy", "map_overlap"]
    + [f"metal_{m}" for m in METAL_ORDER]
    + [f"ecfp_{i}" for i in range(128)]
)
