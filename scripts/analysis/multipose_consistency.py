"""Does multi-pose consistency separate actives from decoys better than the single best score?

Rationale: a true inhibitor should score well in several binding modes, whereas a decoy
may only reach a good score through one lucky placement. For each ligand we therefore
keep the top-k pose energies and derive:

  E_best      min energy                       (the current score)
  E_top3      mean of the 3 lowest energies
  gap         E_top3 - E_best                  small = many good modes
  spread      std of the top-k energies        small = consistent

Each variant is then evaluated as a library-ranking score on a subsample where the
labels are known (actives from docking/results, decoys from the big library).

Usage:
  python scripts/analysis/multipose_consistency.py --target HDAC1 --n-dec 400 --parts 4 --part 0
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
HDAC = Path("/home/wangkai/cursor-task/HDAC")
sys.path.insert(0, str(ROOT))


def auroc(y, s):
    y = np.asarray(y)
    r = rankdata(np.asarray(s, float), method="average")
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    u = r[y == 1].sum() - n1 * (n1 + 1) / 2
    return float(1.0 - u / (n1 * n0))


def ef(y, s, frac):
    y = np.asarray(y)
    r = rankdata(np.asarray(s, float), method="average")
    n, na = len(y), int(y.sum())
    if na == 0:
        return np.nan
    k = max(1, int(round(frac * n)))
    return float((y[np.argsort(r)[:k]].sum() / k) / (na / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="HDAC1")
    ap.add_argument("--protein", type=Path,
                    default=HDAC / "structures/receptors_h/HDAC1.pdb")
    ap.add_argument("--library", type=Path, default=ROOT / "docking/biglib/HDAC1.sdf")
    ap.add_argument("--metal", default="zn")
    ap.add_argument("--n-act", type=int, default=100)
    ap.add_argument("--n-dec", type=int, default=400)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "results/multipose_hdac1.csv")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    def _alarm(sig, frm):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, _alarm)
    from metgeodock.dock import dock_ligand
    from metgeodock.io import read_pdb, molecule_from_rdkit

    protein = read_pdb(a.protein)

    jobs = []   # (label, smiles, lig_id)
    # actives: prefer SMILES from the curated table so we can re-embed
    # actives: the HDAC dataset carries ligand_id -> smiles directly
    ds = HDAC / "data/processed/hdac_dataset.csv"
    if ds.exists():
        for r in csv.DictReader(open(ds)):
            if r.get("target") == a.target and r.get("label") == "1" and r.get("smiles"):
                jobs.append((1, r["smiles"], r["ligand_id"]))
    jobs = jobs[: a.n_act]

    suppl = Chem.SDMolSupplier(str(a.library), removeHs=False, sanitize=True)
    n_dec = 0
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        jobs.append((0, Chem.MolToSmiles(mol), f"{a.target}_BL{i:05d}"))
        n_dec += 1
        if n_dec >= a.n_dec:
            break

    if a.parts > 1:
        jobs = [j for i, j in enumerate(jobs) if i % a.parts == a.part]
    print(f"{a.target} part {a.part}/{a.parts}: {sum(1 for j in jobs if j[0]==1)} actives + "
          f"{sum(1 for j in jobs if j[0]==0)} decoys, top-{a.topk} poses each", flush=True)

    rows = []
    for k, (label, smi, lid) in enumerate(jobs, 1):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            from rdkit.Chem import AllChem
            signal.alarm(90)          # hard per-ligand cap: some embeddings never converge
            m = Chem.AddHs(mol)
            AllChem.EmbedMolecule(m, randomSeed=0xF00D)
            m = Chem.RemoveHs(m)
            lig = molecule_from_rdkit(m, lid)
            res = dock_ligand(protein, lig, metal=a.metal, n_spin=6, max_poses=10)
            signal.alarm(0)
        except Exception:
            signal.alarm(0)
            continue
        if not res.poses:
            continue
        e = np.sort(np.array([p.score.total for p in res.poses], float))
        top = e[: a.topk]
        rows.append({
            "ligand_id": lid, "label": label, "n_poses": len(e),
            "E_best": float(e[0]),
            "E_top3": float(top.mean()),
            "gap": float(top.mean() - e[0]),
            "spread": float(top.std()),
        })
        if k % 10 == 0:
            print(f"  {k}/{len(jobs)}", flush=True)

    if a.parts > 1:
        a.out = a.out.with_name(a.out.stem + f"_p{a.part}" + a.out.suffix)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("no results"); return
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    y = [r["label"] for r in rows]
    print(f"\n=== {a.target}: {sum(y)} actives / {len(y)-sum(y)} decoys ===")
    print(f"{'score':14s} {'AUROC':>7s} {'EF1%':>6s} {'EF5%':>6s}")
    for name, key, sign in (("E_best", "E_best", 1), ("E_top3", "E_top3", 1),
                            ("E_top3 + gap", "gap", 0), ("spread", "spread", 0)):
        if key == "gap":
            s = [r["E_top3"] + 0.5 * r["gap"] for r in rows]
        elif key == "spread":
            s = [r["spread"] for r in rows]
        else:
            s = [r[key] for r in rows]
        print(f"{name:14s} {auroc(y, s):7.3f} {ef(y, s, 0.01):6.2f} {ef(y, s, 0.05):6.2f}")
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
