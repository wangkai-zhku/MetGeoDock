"""Screen the fixed common set with the literature-constant thermodynamic score.

Records geometry terms (MetGeo), chemistry terms, and thermodynamic terms (buried area,
rim overlap, g_total). No parameter here is fitted: the constants live in
metgeodock/thermo.py with their literature citations.
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

TERMS = ["total", "coord", "geo", "steric", "contact", "map_overlap",
         "hbond", "hydrophobic", "desolvation", "n_hbond", "n_hydrophobic",
         "buried_area", "rim_overlap", "g_buried", "g_hbond", "g_desolv",
         "g_conf", "g_total", "n_rotatable"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--protein", type=Path, required=True)
    ap.add_argument("--metal", default="zn")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/thermo_scores")
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    from metgeodock.dock import dock_ligand
    from metgeodock.io import molecule_from_rdkit, read_pdb
    from metgeodock.chemistry import chemistry_terms
    from metgeodock.thermo import thermodynamic_terms

    def _alarm(sig, frm):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, _alarm)

    rows = list(csv.DictReader(open(ROOT / f"results/common_set/{a.target}.csv")))
    if a.parts > 1:
        rows = [r for i, r in enumerate(rows) if i % a.parts == a.part]

    protein = read_pdb(a.protein)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    dest = a.out_dir / f"{a.target}_p{a.part}.csv"
    print(f"{a.target} part {a.part}/{a.parts}: {len(rows)} ligands", flush=True)

    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ligand_id", "label"] + TERMS + ["error"])
        w.writeheader()
        for k, r in enumerate(rows, 1):
            rec = {"ligand_id": r["ligand_id"], "label": r["label"]}
            try:
                signal.alarm(180)
                mol = Chem.MolFromSmiles(r["smiles"])
                if mol is None:
                    raise ValueError("parse")
                nrot = rdMolDescriptors.CalcNumRotatableBonds(mol)
                m = Chem.AddHs(mol)
                if AllChem.EmbedMolecule(m, randomSeed=0xF00D) != 0:
                    raise ValueError("embed")
                m = Chem.RemoveHs(m)
                res = dock_ligand(protein, molecule_from_rdkit(m, r["ligand_id"]),
                                  metal=a.metal, n_spin=6, max_poses=8)
                signal.alarm(0)
                if res.poses:
                    best = min(res.poses, key=lambda p: p.score.total)
                    for t in ("total", "coord", "geo", "steric", "contact", "map_overlap",
                              "hbond", "hydrophobic", "desolvation", "n_hbond", "n_hydrophobic"):
                        rec[t] = best.score.terms.get(t, "")
                    chem = chemistry_terms(protein, best.ligand, res.sphere.metal_index)
                    th = thermodynamic_terms(protein, best.ligand, res.sphere.metal_index,
                                             n_rotatable=nrot,
                                             n_hbond=int(chem["n_hbond"]),
                                             n_desolv=chem["desolvation"])
                    rec.update(th)
                    rec["n_hbond"] = int(chem["n_hbond"])
                    rec["n_hydrophobic"] = int(chem["n_hydrophobic"])
                    rec["n_rotatable"] = nrot
            except Exception as exc:
                signal.alarm(0)
                rec["error"] = type(exc).__name__
            w.writerow(rec); fh.flush()
            if k % 10 == 0:
                print(f"  {a.target}: {k}/{len(rows)}", flush=True)
    print(f"DONE {dest}")


if __name__ == "__main__":
    main()
