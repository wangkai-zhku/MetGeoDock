"""Does `metgeodock prepare` change the published Public-10 redocking numbers?

All ten MetalloDock public complexes ship as heavy-atom-only PDB files, so the
published top-1/top-3 were obtained with a degenerate coordination term.
Redock every complex with the raw receptor and with the hydrogenated one.
"""
from __future__ import annotations

import argparse, csv, sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from metgeodock.dock import dock_ligand, redock_rmsd  # noqa: E402
from metgeodock.io import load_molecule, read_pdb  # noqa: E402
from metgeodock.metals import metal_from_name  # noqa: E402
from metgeodock.prepare import prepare_receptor  # noqa: E402

METAL_FROM_RESNAME = {"ZN": "zn", "ZN2": "zn", "CA": "ca", "CA2": "ca", "C0": "ca",
                      "MG": "mg", "MG2": "mg", "MN": "mn", "MN2": "mn", "MN3": "mn",
                      "FE": "fe3", "FE2": "fe2", "FE3": "fe3", "CU": "cu", "CU1": "cu",
                      "CU2": "cu", "CO": "co", "CO2": "co", "CD": "cd", "CD2": "cd",
                      "NI": "ni", "NI2": "ni"}


def detect_metal(pdb: Path) -> str | None:
    for ln in pdb.read_text(errors="ignore").splitlines():
        if ln.startswith("HETATM"):
            r = ln[17:20].strip().upper()
            if r in METAL_FROM_RESNAME:
                return METAL_FROM_RESNAME[r]
    return None


def count_h(pdb: Path) -> int:
    c = 0
    for ln in pdb.read_text(errors="ignore").splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            e = ln[76:78].strip() or ln[12:16].strip()[:1]
            if e.upper() == "H":
                c += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=ROOT / "processed/metallodock_ref/dataset/metallo_complex")
    ap.add_argument("--n-spin", type=int, default=12)
    ap.add_argument("--max-poses", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=ROOT / "results/public10_hydrogen.csv")
    a = ap.parse_args()

    rows = []
    ids = sorted(d.name for d in a.root.iterdir() if d.is_dir())
    for i, cid in enumerate(ids, 1):
        d = a.root / cid
        # match the published protocol: pocket-trimmed receptor, not the full entry
        pro = d / f"{cid}_pro_pocket.pdb"
        if not pro.exists():
            pro = d / f"{cid}.pdb"
        lig = d / f"{cid}_ligand.pdb"
        if not lig.exists():
            lig = d / f"{cid}_ligand.sdf"
        if pro is None or not lig.exists():
            print(f"[{i}/{len(ids)}] {cid}: missing input", flush=True)
            rows.append({"pdb_id": cid, "error": "missing input"})
            continue
        metal = detect_metal(pro)
        try:
            ligand = load_molecule(lig)
            out = {"pdb_id": cid, "metal": metal, "raw_H": count_h(pro)}
            for tag in ("raw", "prepared"):
                if tag == "raw":
                    protein = read_pdb(pro)
                else:
                    prepped = Path(f"/tmp/p10_{cid}_h.pdb")
                    info = prepare_receptor(pro, out=prepped, metal=metal)
                    protein = read_pdb(prepped)
                    out["prepared_H"] = info["n_hydrogens"]
                res = dock_ligand(protein, ligand, metal=metal, n_spin=a.n_spin,
                                  max_poses=a.max_poses)
                rms = sorted(redock_rmsd(p.ligand, ligand) for p in res.poses)
                out[f"{tag}_top1"] = round(rms[0], 3) if rms else ""
                out[f"{tag}_top3"] = round(min(rms[:3]), 3) if len(rms) >= 3 else (
                    round(rms[-1], 3) if rms else "")
                out[f"{tag}_ok_top1"] = int(rms[0] <= a.threshold) if rms else 0
                out[f"{tag}_ok_top3"] = int(min(rms[:3]) <= a.threshold) if rms else 0
            print(f"[{i}/{len(ids)}] {cid} {metal} H:{out['raw_H']}->{out.get('prepared_H')} "
                  f"top1 {out['raw_top1']}->{out['prepared_top1']} "
                  f"ok1 {out['raw_ok_top1']}->{out['prepared_ok_top1']}", flush=True)
            rows.append(out)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(ids)}] {cid}: FAIL {exc}", flush=True)
            rows.append({"pdb_id": cid, "metal": metal, "error": str(exc)[:120]})

    keys = ["pdb_id", "metal", "raw_H", "prepared_H", "raw_top1", "prepared_top1",
            "raw_top3", "prepared_top3", "raw_ok_top1", "prepared_ok_top1",
            "raw_ok_top3", "prepared_ok_top3", "error"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows([{k: r.get(k, "") for k in keys} for r in rows])
    ok = [r for r in rows if not r.get("error")]
    if ok:
        print(f"\n=== Public-10 (n={len(ok)}) ===")
        print(f"top-1 成功率 raw={np.mean([r['raw_ok_top1'] for r in ok]):.3f}  "
              f"prepared={np.mean([r['prepared_ok_top1'] for r in ok]):.3f}")
        print(f"top-3 成功率 raw={np.mean([r['raw_ok_top3'] for r in ok]):.3f}  "
              f"prepared={np.mean([r['prepared_ok_top3'] for r in ok]):.3f}")
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
