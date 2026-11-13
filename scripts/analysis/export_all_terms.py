"""Export every internal scoring term, including the ones screen_thermo_ef.py drops.

screen_thermo_ef.py keeps only 19 of the ~30 terms returned by the scorer. The
dropped ones include the raw donor-metal distances (min/mean, measured before
the internal radial refinement) and `harm` (sum of squared distance deviation) -
those are continuous geometric-quality signals, unlike `coord`, which collapses
to a binary "can it coordinate" flag.

Usage:
    python scripts/analysis/export_all_terms.py --target urease --n-act 50 --n-dec 50
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
sys.path.insert(0, str(ROOT))

PDB = {"MMP2": "1QIB", "CAPN1": "2ARY", "CA2": "3CAJ", "METAP2": "1B6A",
       "CDK2": "1HCK", "urease": "1FWJ", "HDAC1": "5ICN", "HDAC2": "3MAX"}
MET = {"MMP2": "zn", "CAPN1": "ca", "CA2": "zn", "METAP2": "co",
       "CDK2": "mg", "urease": "ni", "HDAC1": "zn", "HDAC2": "zn"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--n-act", type=int, default=50)
    ap.add_argument("--n-dec", type=int, default=50)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    from metgeodock.dock import dock_ligand
    from metgeodock.io import molecule_from_rdkit, read_pdb

    rows = list(csv.DictReader(open(ROOT / f"results/crossfamily10/{a.target}_div.csv")))
    acts = [r for r in rows if r["label"].strip() == "1"][: a.n_act]
    decs = [r for r in rows if r["label"].strip() == "0"][: a.n_dec]
    jobs = acts + decs

    prot = read_pdb(str(ROOT / f"processed/receptors10/{PDB[a.target]}_h.pdb"))
    out = Path(a.out or (ROOT / f"results/all_terms_{a.target}.csv"))
    written = []
    for k, r in enumerate(jobs, 1):
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            continue
        m = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(m, randomSeed=0xF00D) != 0:
            continue
        m = Chem.RemoveHs(m)
        rec = {"ligand_id": r["ligand_id"], "label": int(r["label"])}
        try:
            res = dock_ligand(prot, molecule_from_rdkit(m, r["ligand_id"]),
                              metal=MET[a.target], n_spin=6, max_poses=8)
            if not res.poses:
                rec["error"] = "no_pose"
            else:
                best = min(res.poses, key=lambda p: p.score.total)
                for key, val in best.score.terms.items():
                    if isinstance(val, (int, float, np.floating)):
                        rec[key] = float(val)
                rec["n_poses"] = len(res.poses)
        except Exception as exc:
            rec["error"] = type(exc).__name__
        # molecule-level descriptors (needed to test size-controlled effects)
        mm = Chem.MolFromSmiles(r["smiles"])
        rec["n_heavy"] = mm.GetNumHeavyAtoms()
        rec["n_rot"] = rdMolDescriptors.CalcNumRotatableBonds(mm)
        rec["molwt"] = Descriptors.MolWt(mm)
        written.append(rec)
        if k % 20 == 0:
            print(f"  {a.target}: {k}/{len(jobs)}", flush=True)

    if not written:
        print("no rows produced")
        return
    fields = ["ligand_id", "label", "error", "n_heavy", "n_rot", "molwt"]
    for rec in written:
        for key in rec:
            if key not in fields:
                fields.append(key)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(written)
    print(f"WROTE {out}  ({len(written)} 行, {len(fields)} 列)")
    print("列:", ", ".join(f for f in fields if f not in
                          ("ligand_id", "label", "error")))


if __name__ == "__main__":
    main()
