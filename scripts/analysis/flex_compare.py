"""Does ligand flexibility rescue the deep-pocket (MMP) targets?

Hypothesis: a single rigid conformer cannot adapt to a deep pocket, so the sampler
never produces the correct pose and the score is computed on the wrong geometry.

Two flexibility levels are compared on the same ligands:
  rigid      : the single ETKDG conformer used so far
  semi-flex  : metgeodock --semi-flex (rotates donor-adjacent torsions on top seeds)
  ensemble   : N independent ETKDG conformers docked separately, best E kept

Usage:
  flex_compare.py --target MMP9 --pdb 1GKC --metal zn --n-act 30 --n-dec 30
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ef(y, s, frac):
    y = np.asarray(y); s = np.asarray(s, float)
    n, na = len(y), int(y.sum())
    if na == 0:
        return float("nan")
    k = max(1, int(round(frac * n)))
    return float((y[np.argsort(s)[:k]].sum() / k) / (na / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activities", type=Path,
                    default=ROOT / "processed/chembl/curated/all_targets_molecules.csv")
    ap.add_argument("--target", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--metal", default="zn")
    ap.add_argument("--n-act", type=int, default=30)
    ap.add_argument("--n-dec", type=int, default=30)
    ap.add_argument("--n-conf", type=int, default=4, help="conformers for the ensemble mode")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    from metgeodock.dock import dock_ligand
    from metgeodock.io import read_pdb, molecule_from_rdkit
    from metgeodock.prepare import prepare_receptor

    src = Path("/tmp/xtal_train") / f"{a.pdb}.pdb"
    holo = Path("/tmp/xtal_train") / f"{a.pdb}_h.pdb"
    if not holo.exists():
        prepare_receptor(src, out=holo, metal=a.metal)
    protein = read_pdb(holo)

    rows = [r for r in csv.DictReader(open(a.activities)) if r.get("target_name") == a.target]
    act = [r for r in rows if (float(r["pactivity"]) if r["pactivity"] else 0) >= 6][: a.n_act]
    dec = [r for r in rows if (float(r["pactivity"]) if r["pactivity"] else 9) <= 5][: a.n_dec]
    print(f"{a.target}: actives={len(act)} decoys={len(dec)}", flush=True)

    recs = []
    for label, subset in ((1, act), (0, dec)):
        for r in subset:
            smi = r.get("canonical_smiles", "")
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None or not (8 <= mol.GetNumHeavyAtoms() <= 50):
                continue
            recs.append({"label": label, "smiles": smi, "id": r["molecule_chembl_id"]})
    print(f"usable ligands: {len(recs)}", flush=True)

    out = []
    for i, rec in enumerate(recs, 1):
        mol = Chem.MolFromSmiles(rec["smiles"])
        row = {"target": a.target, "ligand": rec["id"], "label": rec["label"]}
        # --- rigid
        try:
            m = Chem.AddHs(mol)
            AllChem.EmbedMolecule(m, randomSeed=0xF00D)
            m = Chem.RemoveHs(m)
            lig = molecule_from_rdkit(m, rec["id"])
            res = dock_ligand(protein, lig, metal=a.metal, n_spin=6, max_poses=8)
            row["E_rigid"] = float(min(p.score.total for p in res.poses)) if res.poses else np.nan
        except Exception:
            row["E_rigid"] = np.nan
        # --- semi-flex
        try:
            m = Chem.AddHs(mol)
            AllChem.EmbedMolecule(m, randomSeed=0xF00D)
            m = Chem.RemoveHs(m)
            lig = molecule_from_rdkit(m, rec["id"])
            res = dock_ligand(protein, lig, metal=a.metal, n_spin=6, max_poses=8, semi_flex=True)
            row["E_semiflex"] = float(min(p.score.total for p in res.poses)) if res.poses else np.nan
        except Exception:
            row["E_semiflex"] = np.nan
        # --- conformer ensemble
        try:
            m = Chem.AddHs(mol)
            ps = AllChem.ETKDGv3(); ps.randomSeed = 0xF00D
            cids = AllChem.EmbedMultipleConfs(m, numConfs=a.n_conf, params=ps)
            best = np.nan
            for cid in cids:
                m2 = Chem.Mol(m, confId=cid)
                m2 = Chem.RemoveHs(m2)
                try:
                    lig = molecule_from_rdkit(m2, rec["id"])
                    res = dock_ligand(protein, lig, metal=a.metal, n_spin=4, max_poses=6)
                    if res.poses:
                        e = float(min(p.score.total for p in res.poses))
                        best = e if not np.isfinite(best) else min(best, e)
                except Exception:
                    continue
            row["E_ensemble"] = best
        except Exception:
            row["E_ensemble"] = np.nan
        out.append(row)
        if i % 10 == 0:
            print(f"  {i}/{len(recs)}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    print(f"\n=== {a.target} ===")
    y = [r["label"] for r in out]
    for key in ("E_rigid", "E_semiflex", "E_ensemble"):
        s = [r[key] for r in out]
        ok = [(yy, ss) for yy, ss in zip(y, s) if np.isfinite(ss)]
        if len(ok) < 20 or len({p[0] for p in ok}) < 2:
            print(f"  {key:12s} insufficient")
            continue
        yy = [p[0] for p in ok]; ss = [p[1] for p in ok]
        print(f"  {key:12s} n={len(ok):3d} AUROC={auroc(yy, ss):.3f} "
              f"EF1%={ef(yy, ss, 0.01):.2f} EF5%={ef(yy, ss, 0.05):.2f}")
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
