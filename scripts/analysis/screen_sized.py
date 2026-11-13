"""Size-matched large-library screen for one target.

Draws decoys from results/decoy_pool_sized.csv by stratified sampling on molecular weight so
that their MW distribution follows the target's actives, then docks both actives and decoys
with the same protocol. This removes the size confound that dominates the 30-vs-30 panel.

Actives come from the curated ChEMBL table for the target; any molecule that belongs to the
target itself is excluded from the decoy draw, so no active can leak into the decoy set.

Usage:
  python scripts/analysis/screen_sized.py --target CA2 --mean-mw 538 --n-dec 1000 --parts 4 --part 0
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RNG_SEED = 20241024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--protein", type=Path, required=True)
    ap.add_argument("--metal", default="zn")
    ap.add_argument("--n-act", type=int, default=30)
    ap.add_argument("--n-dec", type=int, default=1000)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/sized_scores")
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors
    RDLogger.DisableLog("rdApp.*")
    from metgeodock.dock import dock_ligand
    from metgeodock.io import molecule_from_rdkit, read_pdb

    def _alarm(sig, frm):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, _alarm)

    # ---- actives for this target
    table = list(csv.DictReader(open(ROOT / "processed/chembl/curated/all_targets_molecules.csv")))
    acts = []
    for r in table:
        if r.get("target_name") != a.target or not r.get("canonical_smiles"):
            continue
        try:
            pa = float(r["pactivity"])
        except Exception:
            continue
        if pa >= 6:
            acts.append(r)
    acts = acts[: a.n_act]

    # ---- decoy pool, matched on MW to the actives
    pool = [p for p in csv.DictReader(open(ROOT / "results/decoy_pool_sized.csv"))
            if p["target"] != a.target]
    act_mw = []
    for r in acts:
        m = Chem.MolFromSmiles(r["canonical_smiles"])
        if m is not None:
            act_mw.append(Descriptors.MolWt(m))
    if not act_mw:
        print("no actives"); return
    lo, hi = np.percentile(act_mw, [2, 98])
    edges = np.linspace(lo, hi, a.bins + 1)
    rng = np.random.default_rng(RNG_SEED)
    pool_mw = np.array([float(p["mw"]) for p in pool])
    picked = []
    per_bin = max(1, a.n_dec // a.bins)
    for i in range(a.bins):
        in_a = sum(1 for w in act_mw if (w >= edges[i] and (w < edges[i + 1] if i < a.bins - 1 else w <= edges[i + 1])))
        if in_a == 0:
            continue
        idx = np.where((pool_mw >= edges[i]) & (pool_mw < edges[i + 1] if i < a.bins - 1 else pool_mw <= edges[i + 1]))[0]
        if len(idx) == 0:
            continue
        take = rng.choice(idx, size=min(per_bin, len(idx)), replace=False)
        picked.extend(take.tolist())
    decoys = [pool[i] for i in sorted(set(picked))]

    jobs = [(1, r["canonical_smiles"], r["molecule_chembl_id"]) for r in acts]
    jobs += [(0, p["smiles"], f"{a.target}_SZ{i:05d}") for i, p in enumerate(decoys)]
    if a.parts > 1:
        jobs = [j for i, j in enumerate(jobs) if i % a.parts == a.part]

    protein = read_pdb(a.protein)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    dest = a.out_dir / f"{a.target}_p{a.part}.csv"
    fields = ["ligand_id", "label", "score", "n_poses", "error"]
    print(f"{a.target} part {a.part}/{a.parts}: {sum(1 for j in jobs if j[0]==1)} actives + "
          f"{sum(1 for j in jobs if j[0]==0)} decoys (actives MW {lo:.0f}-{hi:.0f})", flush=True)

    t0 = time.time()
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for k, (label, smi, lid) in enumerate(jobs, 1):
            rec = {"ligand_id": lid, "label": label}
            try:
                signal.alarm(90)
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError("parse")
                m = Chem.AddHs(mol)
                AllChem.EmbedMolecule(m, randomSeed=0xF00D)
                m = Chem.RemoveHs(m)
                res = dock_ligand(protein, molecule_from_rdkit(m, lid), metal=a.metal,
                                  n_spin=6, max_poses=8)
                signal.alarm(0)
                rec["score"] = float(min(p.score.total for p in res.poses)) if res.poses else ""
                rec["n_poses"] = len(res.poses)
            except Exception as exc:
                signal.alarm(0)
                rec["error"] = type(exc).__name__
            w.writerow(rec)
            if k % 50 == 0:
                el = time.time() - t0
                print(f"  {a.target}/{a.part}: {k}/{len(jobs)} {el/k:.1f}s/lig "
                      f"eta {(len(jobs)-k)*el/k/3600:.1f}h", flush=True)
    print(f"DONE {dest} ({(time.time()-t0)/3600:.1f}h)", flush=True)


if __name__ == "__main__":
    main()
