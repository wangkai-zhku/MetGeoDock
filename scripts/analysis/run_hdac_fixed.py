"""用重建后的 HDAC 受体重跑 MetalGeoDock 完整打分，与旧受体的结果对比。

受体来自 scripts/analysis/prepare_hdac_fixed.py：
  HDAC1_fix  5ICN + MTA1 + IP6          HDAC2_fix  3MAX 单拷贝
  HDAC3_fix  4A69 + SMRT DAD + InsP4    HDAC6_fix  5G0J (CD1+CD2 串联)
  HDAC8_2V5W / HDAC8_2V5X  两个晶体构象（L1 lid open/closed 对照）

配体集与 bench22 完全一致（results/bench22/common/<T>.csv，100 actives + 50 decoys），
所以新结果可以直接和 metalgeo_full_<T>.csv 比 AUC。

用法：
    python scripts/analysis/run_hdac_fixed.py --workers 12
    python scripts/analysis/run_hdac_fixed.py --workers 12 --only HDAC3_fix,HDAC6_fix
断点续跑：已有 CSV 里存在的配体会被跳过。
"""
from __future__ import annotations

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from _embed import embed_3d  # noqa: E402

OUT = ROOT / "results/bench22"
COMMON = OUT / "common"
FIXED = ROOT / "processed/hdac_fixed"

# 受体名 -> (pdb, metal, 对应的 common set 名称)
TARGETS = {
    "HDAC1_fix":   ("HDAC1_fix_h.pdb", "zn", "HDAC1"),
    "HDAC2_fix":   ("HDAC2_fix_h.pdb", "zn", "HDAC2"),
    "HDAC3_fix":   ("HDAC3_fix_h.pdb", "zn", "HDAC3"),
    "HDAC6_fix":   ("HDAC6_fix_h.pdb", "zn", "HDAC6"),
    "HDAC8_2V5W":  ("HDAC8_2V5W_h.pdb", "zn", "HDAC8"),
    "HDAC8_2V5X":  ("HDAC8_2V5X_h.pdb", "zn", "HDAC8"),
}
_REC = {}


def _get_rec(tid):
    if tid not in _REC:
        from metgeodock.io import read_pdb
        pdb, metal, _ = TARGETS[tid]
        _REC[tid] = (read_pdb(str(FIXED / pdb)), metal)
    return _REC[tid]


def dock_one(args):
    tid, lig_id, smiles, label = args
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from metgeodock.dock import dock_ligand
    from metgeodock.io import molecule_from_rdkit
    prot, metal = _get_rec(tid)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return tid, lig_id, label, None
    m = embed_3d(mol)
    if m is None:
        return tid, lig_id, label, None
    try:
        res = dock_ligand(prot, molecule_from_rdkit(m, lig_id), metal=metal,
                          n_spin=8, max_poses=8, include_chem=True)
    except Exception as exc:
        sys.stderr.write(f"[dock_one] {tid} {lig_id}: {type(exc).__name__}: {exc}\n")
        return tid, lig_id, label, None
    if not res.poses:
        return tid, lig_id, label, None
    best = min(res.poses, key=lambda p: p.score.total)
    return tid, lig_id, label, float(best.score.terms.get("total", 0.0))


def load_set(setname: str):
    p = COMMON / f"{setname}.csv"
    return [(r["ligand_id"], r["smiles"], int(r["label"])) for r in csv.DictReader(open(p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    todo = [t.strip() for t in a.only.split(",") if t.strip()] or list(TARGETS)

    for tid in todo:
        pdb, metal, setname = TARGETS[tid]
        dst = OUT / f"hdacfix_{tid}.csv"
        rows = load_set(setname)
        done = list(csv.DictReader(open(dst))) if dst.exists() and dst.stat().st_size else []
        have = {r["ligand_id"] for r in done}
        jobs = [(tid, l, s, lab) for l, s, lab in rows if l not in have]
        if not jobs:
            print(f"{tid}: 已完成 ({len(done)} 条)，跳过", flush=True)
            continue
        print(f"{tid} ({pdb}, {metal}): 目标 {len(rows)}，已有 {len(done)}，待补 {len(jobs)}", flush=True)
        tmp = dst.with_suffix(".csv.tmp")
        n_new = 0
        with open(tmp, "w", newline="") as f, Pool(a.workers) as pool:
            w = csv.writer(f)
            w.writerow(["ligand_id", "label", "score"])
            for r in done:
                w.writerow([r["ligand_id"], r["label"], r["score"]])
            f.flush()
            BATCH = a.workers * 4
            for i in range(0, len(jobs), BATCH):
                for _, lig, lab, sc in pool.map(dock_one, jobs[i:i + BATCH], chunksize=1):
                    if sc is None:
                        continue
                    w.writerow([lig, lab, f"{sc:.6e}"])
                    n_new += 1
                f.flush()
                print(f"  {tid}: 新增 {n_new}/{len(jobs)}", flush=True)
        tmp.replace(dst)
        print(f"  {tid}: 写出 {len(done) + n_new} 行 -> {dst.name}", flush=True)
    print("全部完成")


if __name__ == "__main__":
    main()
