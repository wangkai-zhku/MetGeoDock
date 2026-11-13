"""Score the fixed common set with MetalGeoDock.

For HDAC targets the scores already exist in results/biglib_scores (the big-library build), so
they are looked up. For other targets the common set is docked fresh. Either way the output is
one row per ligand of results/common_set/{target}.csv, so the same molecules carry MetalGeoDock,
Vina and MetalloDock scores.
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--protein", type=Path, default=None)
    ap.add_argument("--metal", default="zn")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    cs = ROOT / f"results/common_set/{a.target}.csv"
    if not cs.exists():
        print(f"missing {cs}"); return
    rows = list(csv.DictReader(open(cs)))

    # ---- existing scores: decoys in the big-library build, actives in the HDAC screen
    existing = {}
    for f in sorted((ROOT / "results/biglib_scores").glob(f"{a.target}_part*.csv")):
        for r in csv.DictReader(open(f)):
            if r.get("score") not in (None, "", "nan"):
                existing[r["ligand_id"]] = float(r["score"])
    hdac_screen = Path("/home/wangkai/cursor-task/HDAC/docking/results") / f"{a.target}.csv"
    if hdac_screen.exists():
        for r in csv.DictReader(open(hdac_screen)):
            if r.get("score") not in (None, "", "nan"):
                existing.setdefault(r["ligand_id"], float(r["score"]))
    covered = [r for r in rows if r["ligand_id"] in existing]

    out = a.out or (ROOT / f"results/common_set/{a.target}_metgeo.csv")
    if len(covered) == len(rows):
        print(f"{a.target}: all {len(rows)} ligands found in existing big-library scores")
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["ligand_id", "label", "score", "error"])
            w.writeheader()
            for r in rows:
                w.writerow({"ligand_id": r["ligand_id"], "label": r["label"],
                            "score": existing[r["ligand_id"]]})
        print(f"WROTE {out}")
        return

    # ---- otherwise dock them
    if a.protein is None:
        print(f"{a.target}: {len(covered)}/{len(rows)} already scored; "
              f"pass --protein to dock the rest"); return
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    from metgeodock.dock import dock_ligand
    from metgeodock.io import molecule_from_rdkit, read_pdb

    def _alarm(sig, frm):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, _alarm)

    protein = read_pdb(a.protein)
    print(f"{a.target}: docking {len(rows)} ligands from the common set", flush=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ligand_id", "label", "score", "error"])
        w.writeheader()
        for k, r in enumerate(rows, 1):
            rec = {"ligand_id": r["ligand_id"], "label": r["label"]}
            try:
                signal.alarm(90)
                mol = Chem.MolFromSmiles(r["smiles"])
                if mol is None:
                    raise ValueError("parse")
                m = Chem.AddHs(mol)
                AllChem.EmbedMolecule(m, randomSeed=0xF00D)
                m = Chem.RemoveHs(m)
                res = dock_ligand(protein, molecule_from_rdkit(m, r["ligand_id"]),
                                  metal=a.metal, n_spin=6, max_poses=8)
                signal.alarm(0)
                rec["score"] = float(min(p.score.total for p in res.poses)) if res.poses else ""
            except Exception as exc:
                signal.alarm(0)
                rec["error"] = type(exc).__name__
            w.writerow(rec)
            if k % 20 == 0:
                print(f"  {a.target}: {k}/{len(rows)}", flush=True)
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
