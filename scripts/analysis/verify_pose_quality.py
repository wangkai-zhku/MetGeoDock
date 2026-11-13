"""Is a docked pose trustworthy? Redock the co-crystal ligand and measure RMSD.

public10 redocking only proves the sampler works inside a crystal-derived pocket.
This checks the same thing on the very receptors used for the HDAC benchmark, so we
know whether the energies computed on those poses mean anything.
"""
from __future__ import annotations

import csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HDAC = Path("/home/wangkai/cursor-task/HDAC")
from metgeodock.dock import dock_ligand, redock_rmsd  # noqa: E402
from metgeodock.io import load_molecule, read_pdb  # noqa: E402

CASES = [  # target, pdb id, ligand resname
    ("HDAC2", "6WBZ", "TV7"),
    ("HDAC3", "4A69", "I0P"),
    ("HDAC4", "2VQM", "HA3"),
    ("HDAC7", "3C10", "TSN"),
    ("HDAC6", "5EDU", "TSN"),
]


def extract_ligand(pdb: Path, resname: str, dest: Path) -> Path | None:
    lines = []
    for ln in pdb.read_text(errors="ignore").splitlines():
        if ln.startswith("HETATM") and ln[17:20].strip() == resname:
            lines.append(ln)
    if not lines:
        return None
    # keep only the first copy (altloc-free); TSN etc. may appear once per chain
    chains = {}
    for ln in lines:
        chains.setdefault(ln[21], []).append(ln)
    first = chains[sorted(chains)[0]]
    dest.write_text("\n".join(first + ["END"]) + "\n")
    return dest


def main():
    rows = []
    out = ROOT / "results" / "pose_quality.csv"
    for target, pdb_id, resn in CASES:
        src = HDAC / "structures" / f"{pdb_id}.pdb"
        rec = HDAC / "structures" / "receptors_h" / f"{target}.pdb"
        lig_pdb = Path(f"/tmp/xtal_{target}_{resn}.pdb")
        if extract_ligand(src, resn, lig_pdb) is None:
            print(f"{target}: ligand {resn} not found"); continue
        try:
            protein = read_pdb(rec)
            ligand = load_molecule(lig_pdb)
            res = dock_ligand(protein, ligand, metal="zn", n_spin=12, max_poses=20)
            rms = sorted(redock_rmsd(p.ligand, ligand) for p in res.poses)
            if not rms:
                print(f"{target}: no poses"); continue
            row = {"target": target, "pdb_id": pdb_id, "ligand": resn,
                   "n_heavy": int(ligand.heavy_mask().sum()), "n_poses": len(rms),
                   "top1_rmsd": round(rms[0], 2),
                   "top5_rmsd": round(min(rms[:5]), 2),
                   "best_rmsd": round(min(rms), 2),
                   "ok_top1": int(rms[0] <= 2.0),
                   "ok_sampled": int(min(rms) <= 2.0)}
            rows.append(row)
            print(f"{target:8s} {resn} heavy={row['n_heavy']:3d} "
                  f"top1={row['top1_rmsd']:5.2f} top5={row['top5_rmsd']:5.2f} "
                  f"best={row['best_rmsd']:5.2f}  {'PASS' if row['ok_top1'] else 'FAIL'}", flush=True)
        except Exception as exc:
            print(f"{target}: FAIL {exc}")
    if rows:
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        n = len(rows)
        print(f"\ntop-1 success {sum(r['ok_top1'] for r in rows)}/{n} = "
              f"{np.mean([r['ok_top1'] for r in rows]):.2f}")
        print(f"sampled  success {sum(r['ok_sampled'] for r in rows)}/{n} = "
              f"{np.mean([r['ok_sampled'] for r in rows]):.2f}")
        print(f"WROTE {out}")


if __name__ == "__main__":
    main()
