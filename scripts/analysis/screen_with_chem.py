"""Screen the fixed common set and record every score term, including the new chemistry terms.

Needed to calibrate the hydrogen-bond / hydrophobic / desolvation weights against real poses:
the earlier runs stored only the total, which cannot be reweighted after the fact.

Usage:
  python scripts/analysis/screen_with_chem.py --target HDAC1 \
      --protein /path/HDAC1.pdb --metal zn
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TERMS = ["total", "coord", "steric", "contact", "geo", "harm", "centroid_pen",
         "map_overlap", "hbond", "hydrophobic", "desolvation", "n_hbond",
         "n_hydrophobic", "n_coord_lig", "n_clash", "min_donor_dist"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--protein", type=Path, required=True)
    ap.add_argument("--metal", default="zn")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/chem_scores")
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    from metgeodock.dock import dock_ligand
    from metgeodock.io import molecule_from_rdkit, read_pdb

    def _alarm(sig, frm):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, _alarm)

    rows = list(csv.DictReader(open(ROOT / f"results/common_set/{a.target}.csv")))
    if a.parts > 1:
        rows = [r for i, r in enumerate(rows) if i % a.parts == a.part]

    protein = read_pdb(a.protein)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    dest = a.out_dir / f"{a.target}_p{a.part}.csv"
    print(f"{a.target} part {a.part}/{a.parts}: {len(rows)} ligands, terms={len(TERMS)}", flush=True)

    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ligand_id", "label"] + TERMS + ["error"])
        w.writeheader()
        for k, r in enumerate(rows, 1):
            rec = {"ligand_id": r["ligand_id"], "label": r["label"]}
            try:
                signal.alarm(120)
                mol = Chem.MolFromSmiles(r["smiles"])
                if mol is None:
                    raise ValueError("parse")
                m = Chem.AddHs(mol)
                AllChem.EmbedMolecule(m, randomSeed=0xF00D)
                m = Chem.RemoveHs(m)
                res = dock_ligand(protein, molecule_from_rdkit(m, r["ligand_id"]),
                                  metal=a.metal, n_spin=6, max_poses=8, include_chem=True)
                signal.alarm(0)
                if res.poses:
                    best = min(res.poses, key=lambda p: p.score.total)
                    for t in TERMS:
                        rec[t] = best.score.terms.get(t, "")
            except Exception as exc:
                signal.alarm(0)
                rec["error"] = type(exc).__name__
            w.writerow(rec); fh.flush()
            if k % 20 == 0:
                print(f"  {a.target}: {k}/{len(rows)}", flush=True)
    print(f"DONE {dest}")


if __name__ == "__main__":
    main()
