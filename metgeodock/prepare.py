"""Build a docking-ready all-atom receptor from a heavy-atom crystal structure.

MetGeoDock's steric/contact terms assume a full-atom protein: a heavy-atom-only
PDB makes the protein "thin", clashes never fire, and every ligand with any
O/N/S scores ~0 (see results/METHOD_NOTES.md in the HDAC benchmark). This
module is the supported way to add hydrogens, and is exposed as
``metgeodock prepare``.

Pipeline
--------
1. split protein ATOM records from metal/water HETATM records (pdbfixer cannot
   template an ion);
2. pdbfixer: add missing heavy atoms (loop rebuilding is disabled so crystal
   coordinates are preserved) then add hydrogens at the requested pH;
3. drop any hydrogen that lands inside the metal coordination shell - these are
   hydrogens added to a coordinating N/O (e.g. a metal-bound histidine);
4. re-attach the original metal ions and (optionally) waters near the site;
5. report atom counts, detected metal sites and their coordination shell.

Requires ``pdbfixer`` (pulls in OpenMM); ``pip install pdbfixer``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["prepare_receptor", "split_pdb_records"]

_METAL_RESNAMES = {
    "ZN", "ZN2", "CA", "CA2", "C0", "MG", "MG2", "MN", "MN2", "MN3",
    "FE", "FE2", "FE3", "CU", "CU1", "CU2", "CO", "CO2", "CD", "CD2",
    "CAD", "NI", "NI2",
}


def _altloc_ok(line: str) -> bool:
    """Keep only the primary alternate location (' ' or 'A') of each atom.

    Crystal structures frequently carry two conformers; passing both to
    pdbfixer yields duplicate atoms (OpenMM drops them silently and may keep
    the minor conformer).
    """
    alt = line[16] if len(line) > 16 else " "
    return alt in (" ", "A", "1")


def split_pdb_records(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Return ``(protein_atoms, metal_records, water_records)``."""
    prot, metal, water = [], [], []
    for ln in lines:
        if not _altloc_ok(ln):
            continue
        if ln.startswith("ATOM"):
            prot.append(ln)
        elif ln.startswith("HETATM"):
            resn = ln[17:20].strip().upper()
            if resn in _METAL_RESNAMES:
                metal.append(ln)
            elif resn in {"HOH", "WAT", "H2O", "DOD"}:
                water.append(ln)
    return prot, metal, water


def _xyz(line: str) -> np.ndarray:
    return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])


def _element(line: str) -> str:
    return (line[76:78].strip() or line[12:16].strip()[:1]).upper()


def prepare_receptor(
    pdb: str | Path,
    out: str | Path | None = None,
    ph: float = 7.4,
    metal: str | None = None,
    metal_h_cutoff: float = 2.6,
    keep_waters: bool = True,
    water_cutoff: float = 10.0,
    report: str | Path | None = None,
) -> dict[str, Any]:
    """Add hydrogens to a metal-containing receptor and write a docking-ready PDB.

    Parameters
    ----------
    pdb
        Input PDB (heavy atoms, crystal coordinates).
    out
        Output path; defaults to ``<stem>_h.pdb`` next to the input.
    ph
        pH used for protonation-state assignment.
    metal
        Optional metal name hint (e.g. ``"zn"``) used only in the report.
    metal_h_cutoff
        Hydrogens closer than this to a metal ion are removed (they would sit on
        a coordinating atom).
    keep_waters
        Keep crystal waters within ``water_cutoff`` of a metal ion.
    """
    try:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "metgeodock prepare needs pdbfixer/OpenMM for hydrogen placement: "
            "pip install pdbfixer"
        ) from exc

    src = Path(pdb).expanduser()
    if not src.exists():
        raise FileNotFoundError(src)
    dest = Path(out).expanduser() if out else src.with_name(f"{src.stem}_h.pdb")

    all_lines = src.read_text(errors="ignore").splitlines()
    n_altloc_dropped = sum(
        1 for ln in all_lines if ln.startswith(("ATOM", "HETATM")) and not _altloc_ok(ln)
    )
    prot, metals, waters = split_pdb_records(all_lines)
    if not prot:
        raise ValueError(f"no protein ATOM records in {src}")

    tmp = Path(f"/tmp/_prepare_{src.stem}.pdb")
    tmp.write_text("\n".join(prot + ["END"]) + "\n")
    fixer = PDBFixer(filename=str(tmp))
    fixer.findMissingResidues()
    n_missing_res = len(fixer.missingResidues)
    fixer.missingResidues = {}          # keep the crystal geometry: no loop rebuild
    fixer.findMissingAtoms()
    n_missing_atoms = sum(len(v) for v in fixer.missingAtoms.values())
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(float(ph))
    protonated = Path(f"/tmp/_prepare_{src.stem}_h_prot.pdb")
    with protonated.open("w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)

    metal_xyz = [_xyz(m) for m in metals]
    kept, dropped_h = [], 0
    for ln in protonated.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        if _element(ln) == "H" and metal_xyz:
            x = _xyz(ln)
            if min(float(np.linalg.norm(x - m)) for m in metal_xyz) < metal_h_cutoff:
                dropped_h += 1
                continue
        kept.append(ln)

    waters_kept: list[str] = []
    if keep_waters and metal_xyz:
        for w in waters:
            x = _xyz(w)
            if min(float(np.linalg.norm(x - m)) for m in metal_xyz) <= water_cutoff:
                waters_kept.append(w)
    elif keep_waters:
        waters_kept = list(waters)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(kept + metals + waters_kept + ["END"]) + "\n")

    # ---- report -----------------------------------------------------------
    element_counts: dict[str, int] = {}
    for ln in kept:
        element_counts[_element(ln)] = element_counts.get(_element(ln), 0) + 1
    protein_atoms = [
        (ln[17:20].strip(), int(ln[22:26]), ln[12:16].strip(), _element(ln), _xyz(ln))
        for ln in kept
        if ln.startswith("ATOM") and _element(ln) in {"N", "O", "S"}
    ]
    sites = []
    for m in metals:
        z = _xyz(m)
        shell = sorted(
            ((float(np.linalg.norm(x - z)), resn, resi, atom)
             for resn, resi, atom, _el, x in protein_atoms
             if float(np.linalg.norm(x - z)) <= 3.3),
            key=lambda t: t[0],
        )
        sites.append({
            "resname": m[17:20].strip(),
            "chain": m[21],
            "xyz": [round(float(v), 3) for v in z],
            "n_coord": len(shell),
            "coordination": [f"{r}{i}:{a}({d:.2f})" for d, r, i, a in shell],
            "n_water_within_10A": sum(
                1 for w in waters_kept if float(np.linalg.norm(_xyz(w) - z)) <= 10.0
            ),
        })

    info: dict[str, Any] = {
        "input": str(src),
        "output": str(dest),
        "ph": float(ph),
        "metal_hint": metal,
        "n_protein_atoms_in": len(prot),
        "altloc_minor_dropped": n_altloc_dropped,
        "n_atoms_out": len(kept) + len(metals) + len(waters_kept),
        "n_hydrogens": element_counts.get("H", 0),
        "element_counts": dict(sorted(element_counts.items())),
        "missing_residues_not_rebuilt": n_missing_res,
        "missing_atoms_added": n_missing_atoms,
        "hydrogens_removed_near_metal": dropped_h,
        "n_metals": len(metals),
        "n_waters_kept": len(waters_kept),
        "metal_sites": sites,
        "warnings": [],
    }
    if not element_counts.get("H"):
        info["warnings"].append("no hydrogens were added - docking scores will be degenerate")
    if not metals:
        info["warnings"].append("no metal ions found; metgeodock docking needs a metal site")
    for s in sites:
        if s["n_coord"] < 3:
            info["warnings"].append(
                f"metal site at chain {s['chain']} has only {s['n_coord']} coordinating atoms"
            )

    if report:
        Path(report).expanduser().write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return info
