"""Does adding hydrogens (metgeodock prepare) change redocking accuracy?

Picks heavy-atom-only crystal complexes, redocks each with and without the
prepare step, and reports native-pose RMSD / top-1 success side by side.
"""
from __future__ import annotations

import argparse, csv, json, logging, sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from metgeodock.dock import dock_ligand, redock_rmsd  # noqa: E402
from metgeodock.io import read_mol2, read_pdb  # noqa: E402
from metgeodock.prepare import prepare_receptor  # noqa: E402

LOG = logging.getLogger("h-effect")


def element_counts(path):
    c = Counter()
    for ln in Path(path).read_text(errors="ignore").splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            c[(ln[76:78].strip() or ln[12:16].strip()[:1]).upper()] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, default=ROOT / "processed/crystal/inventory.csv")
    ap.add_argument("--per-metal", type=int, default=2)
    ap.add_argument("--n-spin", type=int, default=8)
    ap.add_argument("--max-poses", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=ROOT / "results/hydrogen_effect.csv")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = [r for r in csv.DictReader(open(a.inventory)) if r.get("complete") == "1"]
    # heavy-atom-only complexes only: those are the ones prepare() can help
    picked, by_metal = [], Counter()
    for r in rows:
        if by_metal[r["metal"]] >= a.per_metal:
            continue
        if not Path(r["protein_path"]).exists():
            continue
        if element_counts(r["protein_path"]).get("H", 0) > 0:
            continue
        picked.append(r)
        by_metal[r["metal"]] += 1
    LOG.info("selected %d hydrogen-free complexes across %d metals", len(picked), len(by_metal))

    out_rows = []
    for i, rec in enumerate(picked, 1):
        LOG.info("[%d/%d] %s %s", i, len(picked), rec["metal"], rec["pdb_id"])
        try:
            ligand = read_mol2(rec["ligand_path"])
            res = {}
            for tag in ("raw", "prepared"):
                if tag == "raw":
                    protein = read_pdb(rec["protein_path"])
                    n_h = element_counts(rec["protein_path"]).get("H", 0)
                else:
                    prepped = Path(f"/tmp/hprep_{rec['pdb_id']}.pdb")
                    info = prepare_receptor(rec["protein_path"], out=prepped, metal=rec["metal"])
                    protein = read_pdb(prepped)
                    n_h = info["n_hydrogens"]
                d = dock_ligand(protein, ligand, metal=rec["metal"], n_spin=a.n_spin,
                                max_poses=a.max_poses)
                rmsds = sorted(redock_rmsd(p.ligand, ligand) for p in d.poses)
                top1 = rmsds[0] if rmsds else float("inf")
                best = min(rmsds) if rmsds else float("inf")
                res[tag] = dict(n_h=n_h, top1=top1, best=best,
                                ok_top1=int(top1 <= a.threshold),
                                ok_sampled=int(best <= a.threshold), n_poses=len(rmsds))
            out_rows.append({
                "metal": rec["metal"], "pdb_id": rec["pdb_id"],
                "raw_H": res["raw"]["n_h"], "prepared_H": res["prepared"]["n_h"],
                "raw_top1_rmsd": round(res["raw"]["top1"], 3) if np.isfinite(res["raw"]["top1"]) else "",
                "prep_top1_rmsd": round(res["prepared"]["top1"], 3) if np.isfinite(res["prepared"]["top1"]) else "",
                "raw_best_rmsd": round(res["raw"]["best"], 3) if np.isfinite(res["raw"]["best"]) else "",
                "prep_best_rmsd": round(res["prepared"]["best"], 3) if np.isfinite(res["prepared"]["best"]) else "",
                "raw_ok_top1": res["raw"]["ok_top1"], "prep_ok_top1": res["prepared"]["ok_top1"],
                "raw_ok_sampled": res["raw"]["ok_sampled"], "prep_ok_sampled": res["prepared"]["ok_sampled"],
            })
            print(f"  {rec['metal']:4s} {rec['pdb_id']}  H:{res['raw']['n_h']}->{res['prepared']['n_h']}  "
                  f"top1 {res['raw']['top1']:.2f}->{res['prepared']['top1']:.2f}  "
                  f"ok {res['raw']['ok_top1']}->{res['prepared']['ok_top1']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("  fail %s %s: %s", rec["metal"], rec["pdb_id"], exc)
            out_rows.append({"metal": rec["metal"], "pdb_id": rec["pdb_id"], "error": str(exc)})

    a.out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["metal", "pdb_id", "raw_H", "prepared_H", "raw_top1_rmsd", "prep_top1_rmsd",
            "raw_best_rmsd", "prep_best_rmsd", "raw_ok_top1", "prep_ok_top1",
            "raw_ok_sampled", "prep_ok_sampled", "error"]
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows([{k: r.get(k, "") for k in keys} for r in out_rows])
    ok = [r for r in out_rows if not r.get("error")]
    if ok:
        print(f"\n=== n={len(ok)} ===")
        print(f"top-1 成功率   raw={np.mean([r['raw_ok_top1'] for r in ok]):.3f}  "
              f"prepared={np.mean([r['prep_ok_top1'] for r in ok]):.3f}")
        print(f"采样成功率     raw={np.mean([r['raw_ok_sampled'] for r in ok]):.3f}  "
              f"prepared={np.mean([r['prep_ok_sampled'] for r in ok]):.3f}")
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
