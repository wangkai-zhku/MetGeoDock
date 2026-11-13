#!/usr/bin/env python3
"""Build examples/screening_hdac8/ from the benchmark data.

Copies the open HDAC8 receptor (2V5W chain A) and writes a 20-molecule SDF holding
8 known HDAC8 inhibitors plus 12 decoys, together with a labels.csv for verification.

Usage:  python scripts/data/build_hdac8_example.py
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "examples" / "screening_hdac8"
RECEPTOR = ROOT / "processed" / "hdac_receptors" / "2V5W_A_h.pdb"
ACTIVE_LIB = Path("/home/wangkai/cursor-task/HDAC/docking/libraries/HDAC8_library.sdf")
N_ACT, N_DEC = 8, 12


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RECEPTOR, OUT / "receptor.pdb")

    actives = []
    for mol in Chem.SDMolSupplier(str(ACTIVE_LIB), removeHs=False, sanitize=True):
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        if name.startswith("HDAC8_A"):
            actives.append(mol)
        if len(actives) >= N_ACT:
            break

    # decoys: first N_DEC molecules of the big library (results/biglib/HDAC8.sdf order)
    big = ROOT / "docking" / "biglib" / "HDAC8.sdf"
    decoys = []
    for i, mol in enumerate(Chem.SDMolSupplier(str(big), removeHs=False, sanitize=True)):
        if mol is None:
            continue
        mol.SetProp("_Name", f"decoy_{i:03d}")
        decoys.append(mol)
        if len(decoys) >= N_DEC:
            break

    with Chem.SDWriter(str(OUT / "library.sdf")) as w:
        for mol in actives + decoys:
            w.write(mol)

    with open(OUT / "labels.csv", "w", newline="") as fh:
        rows = csv.writer(fh)
        rows.writerow(["ligand_id", "label"])
        rows.writerows([[m.GetProp("_Name"), 1] for m in actives])
        rows.writerows([[m.GetProp("_Name"), 0] for m in decoys])

    print(f"{len(actives)} actives + {len(decoys)} decoys -> {OUT/'library.sdf'}")


if __name__ == "__main__":
    main()
