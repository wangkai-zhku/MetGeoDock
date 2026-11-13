"""Screen a large SDF decoy library against one metalloprotein, in shards.

Writes results/biglib_scores/{target}_part{k}.csv incrementally so a run can be
resumed; the shard is skipped when it already holds the expected row count.

Usage:
  screen_biglib.py --target HDAC3 --protein /home/wangkai/cursor-task/HDAC/structures/receptors_h/HDAC3.pdb \
                   --library docking/biglib/HDAC3.sdf --metal zn --parts 4 --part 0
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--protein", type=Path, required=True)
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--metal", default="zn")
    ap.add_argument("--parts", type=int, default=4, help="total shards")
    ap.add_argument("--part", type=int, default=0, help="which shard this process handles")
    ap.add_argument("--n-spin", type=int, default=6)
    ap.add_argument("--max-poses", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/biglib_scores")
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from metgeodock.dock import dock_ligand
    from metgeodock.io import read_pdb, molecule_from_rdkit

    a.out_dir.mkdir(parents=True, exist_ok=True)
    dest = a.out_dir / f"{a.target}_part{a.part}.csv"
    if dest.exists():
        n_done = sum(1 for _ in open(dest)) - 1
        if n_done > 500:
            print(f"{dest.name}: already has {n_done} rows, skipping", flush=True)
            return

    protein = read_pdb(a.protein)
    suppl = Chem.SDMolSupplier(str(a.library), removeHs=False, sanitize=True)
    mols = [(i, m) for i, m in enumerate(suppl) if m is not None]
    mine = [x for x in mols if x[0] % a.parts == a.part]
    print(f"{a.target} part {a.part}/{a.parts}: {len(mine)} ligands "
          f"(library {len(mols)})", flush=True)

    fields = ["rank", "ligand_id", "score", "hybrid_total", "slef", "gp_nll",
              "geo_rmsd_or", "clash", "dist_harm", "cn_pen", "n_coord_lig",
              "n_coord_pro", "motif", "metal", "pactivity", "pose_path", "error"]
    t0 = time.time()
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for k, (idx, mol) in enumerate(mine, 1):
            rec = {f: "" for f in fields}
            rec["rank"] = k
            rec["ligand_id"] = f"{a.target}_BL{idx:05d}"
            rec["metal"] = a.metal
            try:
                lig = molecule_from_rdkit(mol, rec["ligand_id"])
                res = dock_ligand(protein, lig, metal=a.metal,
                                  n_spin=a.n_spin, max_poses=a.max_poses)
                if res.poses:
                    best = min(res.poses, key=lambda p: p.score.total)
                    rec["score"] = best.score.total
                    for k2, name in (("hybrid_total", "total"), ("slef", "slef"),
                                     ("gp_nll", "gp_nll"), ("geo_rmsd_or", "geo"),
                                     ("clash", "steric"), ("dist_harm", "harm"),
                                     ("cn_pen", "coord"), ("n_coord_lig", "n_coord_lig"),
                                     ("n_coord_pro", "n_coord_pro")):
                        rec[k2] = best.score.terms.get(name, "")
                    rec["motif"] = getattr(best, "motif", "")
            except Exception as exc:
                rec["error"] = type(exc).__name__
            w.writerow(rec)
            if k % 100 == 0:
                el = time.time() - t0
                print(f"  {a.target}/{a.part}: {k}/{len(mine)}  "
                      f"{el/k:.2f}s/lig  eta {(len(mine)-k)*el/k/3600:.1f}h", flush=True)
    print(f"DONE {dest} ({len(mine)} rows, {(time.time()-t0)/3600:.1f}h)", flush=True)


if __name__ == "__main__":
    main()
