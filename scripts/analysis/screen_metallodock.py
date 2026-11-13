"""Run MetalloDock on the fixed common set so it can be compared with MetalGeoDock and Vina.

Each ligand is docked in its own subprocess because the MetalloDock entry point is a script
that reads one ligand name at a time. Scores land in results/metallodock_scores/{target}.csv.

Usage:
  python scripts/analysis/screen_metallodock.py --target HDAC1 --protein <pdb> --limit 20
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MDL = ROOT / "processed/metallodock_ref"
# Persisted copy: the upstream default pointed at /tmp/mdl/, which does not survive a reboot.
MODEL = MDL / "trained_models/docking_metallodock_time.pkl"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from _embed import embed_3d  # noqa: E402


def make_pocket(protein: Path, out: Path, radius: float = 18.0, metal: str | None = None):
    """以**催化金属**为中心裁 18 Å 口袋。

    旧实现取文件里第一个金属，对多金属结构会选错位点：MMP9 (2 Zn + 3 Ca) 的首个金属
    与 detect_coordination 选中的催化 Zn 相距 13.25 Å，等于把口袋开在另一个位点上。
    MetalGeoDock 与 Vina 两条线都用 detect_coordination 定位，这里必须保持一致。
    """
    from metgeodock.dock import detect_coordination
    from metgeodock.io import read_pdb
    lines = protein.read_text(errors="ignore").splitlines()
    try:
        sph = detect_coordination(read_pdb(str(protein)), metal=metal)
        met = np.array(sph.metal_xyz, dtype=float)
    except Exception:
        met = None
    if met is None:
        for ln in lines:
            if ln.startswith(("HETATM", "ATOM")) and ln[17:20].strip().upper() in {
                    "ZN", "FE", "FE2", "FE3", "MN", "MN2", "MG", "CA", "CO", "CU", "NI", "CD"}:
                met = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
                break
    if met is None:
        raise SystemExit("no metal in receptor")
    keep = []
    for ln in lines:
        if ln.startswith(("ATOM", "HETATM")):
            try:
                xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            except Exception:
                continue
            if np.linalg.norm(xyz - met) <= radius:
                keep.append(ln)
    out.write_text("\n".join(keep) + "\nEND\n")
    return len(keep), met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--protein", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--coord-idx", type=int, default=0)
    ap.add_argument("--metal", default=None,
                    help="金属元素（zn/fe3/mn/ni/co/mg），用于与另两条线选同一催化位点")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/metallodock_scores")
    ap.add_argument("--set-suffix", default="", help="use common_set/{target}{suffix}.csv")
    ap.add_argument("--common-set", type=Path, default=None,
                    help="explicit ligand CSV (ligand_id,label,smiles); overrides common_set/")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors
    RDLogger.DisableLog("rdApp.*")

    cs = a.common_set or (ROOT / f"results/common_set/{a.target}{a.set_suffix}.csv")
    rows = list(csv.DictReader(open(cs)))
    print(f"{a.target}: ligand set {cs}", flush=True)
    if a.limit:
        rows = rows[: a.limit]
    if a.parts > 1:
        rows = [r for i, r in enumerate(rows) if i % a.parts == a.part]

    a.out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="mdl_"))
    pocket = work / "receptor_pocket.pdb"
    n_atoms, met = make_pocket(a.protein, pocket, metal=a.metal)
    print(f"{a.target}: pocket {n_atoms} atoms, metal at {met.round(2)}", flush=True)

    lig_dir = work / "ligands"
    lig_dir.mkdir()
    dest = a.out_dir / f"{a.target}{a.set_suffix}" / f"p{a.part}.csv" if a.parts > 1 \
        else a.out_dir / f"{a.target}{a.set_suffix}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ligand_id", "label", "score", "error"])
        w.writeheader()
        for k, r in enumerate(rows, 1):
            rec = {"ligand_id": r["ligand_id"], "label": r["label"]}
            try:
                mol = Chem.MolFromSmiles(r["smiles"])
                if mol is None:
                    raise ValueError("parse")
                m = embed_3d(mol)
                if m is None:
                    raise ValueError("embed")
                sdf = lig_dir / f"{r['ligand_id']}.sdf"
                sw = Chem.SDWriter(str(sdf)); sw.write(m); sw.close()

                out_dir = work / "out" / r["ligand_id"]
                cmd = [sys.executable, "utils/ligand_docking_custom.py",
                       "--ligand_path", str(lig_dir),
                       "--ligand_name", r["ligand_id"],
                       "--pocket_file", str(pocket),
                       "--coordination_idx", str(a.coord_idx),
                       "--voc_file", "dataset/chembl_fragments_voc_dict.pkl",
                       "--model_file", str(MODEL),
                       "--out_dir", str(out_dir),
                       "--docking", "True", "--scoring", "True"]
                subprocess.run(cmd, cwd=str(MDL), capture_output=True, timeout=900)
                sj = out_dir / "0" / "scores.jsonl"
                if sj.exists():
                    last = sj.read_text().strip().splitlines()[-1]
                    rec["score"] = float(json.loads(last)["score"][0])
                else:
                    rec["error"] = "no_score"
            except Exception as exc:
                rec["error"] = type(exc).__name__
            w.writerow(rec); fh.flush()
            if k % 10 == 0:
                print(f"  {a.target}: {k}/{len(rows)}", flush=True)
    print(f"WROTE {dest}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
