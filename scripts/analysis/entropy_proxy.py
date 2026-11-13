"""Conformational-entropy proxy from the docking ensemble.

dG = dH - T dS. Every previous attempt only touched dH. For one target with a
FIXED sampling protocol, the spread of the pose ensemble is a cheap proxy for the
conformational entropy the ligand must give up on binding:

  n_clusters        how many distinct binding modes survive (fewer = more pre-organised)
  mean_pair_rmsd    spread of the ensemble (larger = floppier, worse to bind)
  E_spread          energy range across poses (flat landscape = undiscriminating)
  n_deep            poses within 2 kcal/mol of the best (fewer = more selective)

Sampling settings are identical for every ligand, which is the precondition for
comparing these numbers within a target.
"""
from __future__ import annotations

import argparse, csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HDAC = Path("/home/wangkai/cursor-task/HDAC")
from metgeodock.dock import dock_ligand  # noqa: E402
from metgeodock.io import read_pdb  # noqa: E402
from metgeodock.screen import iter_library  # noqa: E402

CLUSTER_RMSD = 2.0
DEEP_WINDOW = 2.0


def pose_rmsd(a, b):
    n = min(len(a.xyz), len(b.xyz))
    if n == 0:
        return np.inf
    return float(np.sqrt(((a.xyz[:n] - b.xyz[:n]) ** 2).sum(axis=1).mean()))


def stats_for(poses):
    if not poses:
        return None
    energies = np.array([p.score.total for p in poses], float)
    best = energies.min()
    reps = []
    for p in poses:
        d = min((pose_rmsd(p.ligand, q) for q in reps), default=np.inf)
        if d > CLUSTER_RMSD:
            reps.append(p.ligand)
    pair = []
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            pair.append(pose_rmsd(poses[i].ligand, poses[j].ligand))
    return {
        "n_poses": len(poses),
        "n_clusters": len(reps),
        "mean_pair_rmsd": round(float(np.mean(pair)) if pair else 0.0, 3),
        "max_pair_rmsd": round(float(np.max(pair)) if pair else 0.0, 3),
        "E_best": round(float(best), 4),
        "E_spread": round(float(energies.max() - best), 4),
        "E_std": round(float(energies.std()), 4),
        "n_deep": int((energies - best <= DEEP_WINDOW).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--library", type=Path, default=HDAC / "docking/libraries")
    ap.add_argument("--receptors", type=Path, default=HDAC / "structures/receptors_h")
    ap.add_argument("--dataset", type=Path, default=HDAC / "data/processed/hdac_dataset.csv")
    ap.add_argument("--actives-only", action="store_true")
    ap.add_argument("--decoys-only", action="store_true")
    ap.add_argument("--n-spin", type=int, default=8)
    ap.add_argument("--max-poses", type=int, default=15)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    meta = {r["ligand_id"]: r for r in csv.DictReader(open(a.dataset))}
    rows = []
    for t in a.targets:
        lib = a.library / f"{t}_library.sdf"
        rec = a.receptors / f"{t}.pdb"
        if not lib.exists() or not rec.exists():
            print(f"{t}: missing input"); continue
        protein = read_pdb(rec)
        n_done = 0
        for lid, ligand in iter_library(lib):
            m = meta.get(lid)
            if not m: continue
            if a.actives_only and m["label"] != "1": continue
            if a.decoys_only and m["label"] != "0": continue
            try:
                res = dock_ligand(protein, ligand, metal="zn", n_spin=a.n_spin,
                                  max_poses=a.max_poses)
                st = stats_for(res.poses)
            except Exception:
                continue
            if st is None: continue
            rows.append({"ligand_id": lid, "target": t, "label": int(m["label"]),
                         "pchembl": m["pchembl"], **st})
            n_done += 1
        print(f"{t}: {n_done} ligands", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"WROTE {a.out} n={len(rows)}")


if __name__ == "__main__":
    main()
