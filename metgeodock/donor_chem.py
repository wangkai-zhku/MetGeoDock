"""Metal-specific donor-group chemistry on top of element-level GPDock priors.

Why this exists
---------------
``metals.peak_probability(metal, element)`` only knows O / N / S. A hydroxamate
oxygen (log beta3 ~ 28-30 with Fe3+) and an ester single-bond oxygen (~0) are
treated identically, so a purely geometric score cannot tell a real metal
warhead from an incidental heteroatom.

This module classifies every O/N/S ligand atom into a functional group and maps
(metal, group) -> relative coordination strength in [0, 1]. Strengths are
*relative* preferences (HSAB + Irving-Williams + published log K order), meant
for ranking penalties, not absolute binding free energies.

Everything here is opt-in: nothing changes unless a caller passes
``donor_chem=True`` / ``donor_chem_scale > 0``.
"""

from __future__ import annotations

from typing import Any

METALS = ("zn", "ca", "mg", "mn", "fe2", "fe3", "cu", "co", "cd", "ni")

# group -> (SMARTS, indices of the pattern atoms that act as donors)
GROUP_PATTERNS: dict[str, tuple[str, tuple[int, ...]]] = {
    # Order matters: exclusion rules (amide_N, pyrrole_N) must run BEFORE the
    # aromatic-N rules, otherwise purine-like rings get mislabelled as imidazole.
    "hydroxamate": ("[#7]-[#8]", (0, 1)),
    "catechol": ("c1c([OH,O-])c([OH,O-])ccc1", (1, 3)),
    "carboxylate": ("[#6](=[OX1])[OX2H1,OX1-]", (1, 2)),
    "phosphate": ("[PX4](=[OX1])([OX2H1,OX1-,OX2])[OX2H1,OX1-,OX2]", (1, 2, 3)),
    "amide_N": ("[NX3][#6]=[OX1]", (0,)),
    "pyrrole_N": ("[nX3]", (0,)),
    "thiol": ("[#16;X2;H1]", (0,)),
    "imidazole": ("[nX2]1[cX3][nX3][cX3][cX3]1", (0,)),
    "pyridine": ("[nX2]1[cX3][cX3][cX3][cX3][cX3]1", (0,)),
    "sulfonamide": ("[SX4](=[OX1])(=[OX1])[NX3,nX3]", (2, 3)),
    "carbonyl": ("[#6]=[OX1]", (1,)),
    "hydroxyl": ("[OX2H1]", (0,)),
    "ether_O": ("[OX2]([#6])[#6]", (0,)),
}

# metal -> group -> relative strength (0 = does not coordinate, 1 = strong)
STRENGTH: dict[str, dict[str, float]] = {
    "hydroxamate": {"zn": 1.00, "ca": 0.15, "mg": 0.20, "mn": 0.50, "fe2": 0.85,
                    "fe3": 1.00, "cu": 0.90, "co": 0.80, "cd": 0.85, "ni": 0.80},
    "catechol":    {"zn": 0.60, "ca": 0.35, "mg": 0.40, "mn": 0.60, "fe2": 0.75,
                    "fe3": 1.00, "cu": 0.60, "co": 0.50, "cd": 0.40, "ni": 0.50},
    "carboxylate": {"zn": 0.60, "ca": 0.90, "mg": 0.90, "mn": 0.80, "fe2": 0.70,
                    "fe3": 0.90, "cu": 0.70, "co": 0.60, "cd": 0.60, "ni": 0.60},
    "phosphate":   {"zn": 0.40, "ca": 0.90, "mg": 0.90, "mn": 0.70, "fe2": 0.60,
                    "fe3": 0.90, "cu": 0.40, "co": 0.40, "cd": 0.40, "ni": 0.40},
    "thiol":       {"zn": 0.90, "ca": 0.10, "mg": 0.10, "mn": 0.30, "fe2": 0.70,
                    "fe3": 0.50, "cu": 1.00, "co": 0.80, "cd": 1.00, "ni": 0.90},
    "imidazole":   {"zn": 0.90, "ca": 0.15, "mg": 0.20, "mn": 0.50, "fe2": 0.80,
                    "fe3": 0.40, "cu": 0.90, "co": 0.80, "cd": 0.70, "ni": 0.90},
    "pyridine":    {"zn": 0.80, "ca": 0.15, "mg": 0.20, "mn": 0.45, "fe2": 0.75,
                    "fe3": 0.35, "cu": 0.85, "co": 0.75, "cd": 0.65, "ni": 0.85},
    "amine":       {"zn": 0.60, "ca": 0.30, "mg": 0.35, "mn": 0.40, "fe2": 0.60,
                    "fe3": 0.30, "cu": 0.85, "co": 0.70, "cd": 0.60, "ni": 0.80},
    "sulfonamide": {"zn": 0.50, "ca": 0.50, "mg": 0.50, "mn": 0.50, "fe2": 0.50,
                    "fe3": 0.60, "cu": 0.50, "co": 0.50, "cd": 0.50, "ni": 0.50},
    "carbonyl":    {"zn": 0.45, "ca": 0.40, "mg": 0.45, "mn": 0.45, "fe2": 0.50,
                    "fe3": 0.50, "cu": 0.50, "co": 0.50, "cd": 0.45, "ni": 0.50},
    "hydroxyl":    {"zn": 0.40, "ca": 0.45, "mg": 0.45, "mn": 0.40, "fe2": 0.45,
                    "fe3": 0.55, "cu": 0.40, "co": 0.40, "cd": 0.35, "ni": 0.40},
    "ether_O":     {m: 0.15 for m in METALS},
    "amide_N":     {m: 0.00 for m in METALS},
    "pyrrole_N":   {m: 0.00 for m in METALS},
}

# amine has no dedicated SMARTS above; it is the fallback for aliphatic N
_FALLBACK_N = "amine"
_CACHE: dict[str, Any] = {}


def _patterns():
    key = "patterns"
    if key not in _CACHE:
        from rdkit import Chem
        table = []
        for name, (smarts, donors) in GROUP_PATTERNS.items():
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None:
                table.append((name, patt, donors))
        _CACHE[key] = table
    return _CACHE[key]


def classify_donors(mol) -> dict[int, str]:
    """Map ligand atom index -> functional group for every O/N/S atom.

    First matching pattern wins (specific warheads precede generic ones).
    Aliphatic N with no match falls back to "amine".
    """
    if mol is None:
        return {}
    out: dict[int, str] = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol().upper() not in {"O", "N", "S"}:
            continue
        idx = atom.GetIdx()
        for name, patt, donors in _patterns():
            match = mol.GetSubstructMatch(patt)
            if not match:
                continue
            hit = False
            for m in mol.GetSubstructMatches(patt, uniquify=True):
                for d in donors:
                    if d < len(m) and m[d] == idx:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                out[idx] = name
                break
        else:
            if atom.GetSymbol().upper() == "N":
                out[idx] = _FALLBACK_N
    return out


def donor_strength(group: str, metal: str) -> float:
    metal = str(metal).lower()
    table = STRENGTH.get(group)
    if not table:
        return 0.4
    return float(table.get(metal, 0.4))


def donor_penalty(group: str, metal: str, scale: float = 3.0, floor: float = 0.02) -> float:
    """Penalty added to the coordination energy for a weak donor.

    ``-ln(strength) * scale``: strength 1.0 -> 0, 0.6 -> 1.53, 0.45 -> 2.39,
    0.0 -> large (the donor is dropped upstream instead).
    """
    import math
    s = max(donor_strength(group, metal), floor)
    return float(-math.log(s) * float(scale))


def describe(mol, metal: str) -> list[dict]:
    """Human-readable per-atom donor classification (for tests / debugging)."""
    groups = classify_donors(mol)
    rows = []
    for idx, g in sorted(groups.items()):
        rows.append({"atom": idx, "group": g,
                     "strength": donor_strength(g, metal),
                     "penalty": round(donor_penalty(g, metal), 3)})
    return rows


# ---------------------------------------------------------------- runtime switch
# Kept module-level so the option can reach ff_energy() without threading a new
# argument through every intermediate helper. Default 0.0 == disabled == the
# original geometric scoring, bit for bit.
_ACTIVE: dict[str, float] = {"scale": 0.0}


def set_donor_chem(scale: float | None) -> None:
    """Enable (scale > 0) or disable (0) donor-chemistry penalties."""
    _ACTIVE["scale"] = float(scale or 0.0)


def active_scale() -> float:
    return float(_ACTIVE["scale"])


def is_active() -> bool:
    return _ACTIVE["scale"] > 0.0
