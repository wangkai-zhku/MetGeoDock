"""22 靶点 top-1 vs top-3 富集对比（完整保存每个配体的全部位姿能量）。

背景：项目里 multipose_consistency.py 提出"真抑制剂应在多个结合模式下都得分好，
decoy 往往只靠一次侥幸放置拿高分"，因此用 top-3 位姿均值（E_top3）或一致性
（gap / spread）代替单一最优分（E_best）可能提升富集。该实验此前只跑了一行日志
就中断（logs/multipose_hdac1.log），从未产出结果。

本脚本对 22 靶点 × 1:10 配体集重跑，并把每个配体的**全部位姿能量**都写出来，
便于事后任意比较聚合方式（E_best / E_top3 / E_top5 / gap / spread / 中位数...）。

对接参数与 results/bench22_1to10 完全一致（n_spin=8, max_poses=8,
include_chem=True, metal 按靶点），因此 E_best 应当能复现 metalgeo_full_<T>.csv
的 score —— 脚本会在每靶点结束时做这个一致性自检。

输出：
  results/bench22_multipose/1to10/<T>.csv
    列：ligand_id,label,n_poses,E_best,E_top3,E_top5,gap,spread,energies
    energies = 所有位姿的 total，分号分隔、升序（负值越小越好）

用法：
    python scripts/analysis/run_multipose_bench22.py --workers 36
    python scripts/analysis/run_multipose_bench22.py --workers 36 --targets CA2,HDAC1
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

BENCH = ROOT / "results/bench22_1to10"
COMMON = BENCH / "common"
OUT = ROOT / "results/bench22_multipose/1to10"
TARGETS = {
    "CA2": ("processed/receptors10/3CAJ_h.pdb", "zn"),
    "CA12": ("processed/receptors22/CA12_h.pdb", "zn"),
    "HDAC1": ("processed/receptors10/5ICN_h.pdb", "zn"),
    "HDAC2": ("processed/receptors10/3MAX_h.pdb", "zn"),
    "HDAC3": ("processed/hdac_receptors/4A69_h.pdb", "zn"),
    "HDAC6": ("processed/hdac_receptors/5EDU_CD2_h.pdb", "zn"),
    "HDAC8": ("processed/hdac_receptors/2V5W_A_h.pdb", "zn"),
    "MMP1": ("processed/receptors22/MMP1_cat_h.pdb", "zn"),
    "MMP2": ("processed/receptors10/1QIB_h.pdb", "zn"),
    "MMP3": ("processed/receptors22/MMP3_h.pdb", "zn"),
    "MMP9": ("processed/receptors22/MMP9_cat_h.pdb", "zn"),
    "MMP13": ("processed/receptors22/MMP13_h.pdb", "zn"),
    "ALOX5": ("processed/receptors10/3O8Y_h.pdb", "fe2"),
    "ARG1": ("processed/receptors22/ARG1_h.pdb", "mn"),
    "ARG2": ("processed/receptors10/1PQ3_h.pdb", "mn"),
    "urease": ("processed/receptors10/1FWJ_h.pdb", "ni"),
    "METAP2": ("processed/receptors10/1B6A_h.pdb", "co"),
    "CYP1A2": ("processed/receptors22/CYP1A2_h.pdb", "fe3"),
    "CYP2E1": ("processed/receptors22/CYP2E1_h.pdb", "fe3"),
    "CYP19A1": ("processed/receptors22/CYP19A1_h.pdb", "fe3"),
    "EGLN2": ("processed/receptors22/EGLN2_h.pdb", "fe3"),
    "HIV1_integrase": ("processed/receptors22/HIV1_cat_h.pdb", "mg"),
}

def acquire_lock(name: str):
    """单实例锁：同名任务已在运行时直接退出，避免两个程序跑同一个任务浪费算力。"""
    import fcntl
    lock = ROOT / "logs" / f".lock_{name}"
    lock.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"[{name}] 已有实例在运行（锁 {lock}），退出以避免重复计算。", flush=True)
        raise SystemExit(1)
    return fh

_REC = {}


def _get_rec(tid):
    if tid not in _REC:
        from metgeodock.io import read_pdb
        rel, metal = TARGETS[tid]
        _REC[tid] = (read_pdb(str(ROOT / rel)), metal)
    return _REC[tid]


def dock_one(args):
    tid, lid, smiles, label = args
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from metgeodock.dock import dock_ligand
    from metgeodock.io import molecule_from_rdkit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return tid, lid, label, None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) > 1:                      # 脱盐，与 bench22_1to10 的修正保持一致
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    m = embed_3d(mol)
    if m is None:
        return tid, lid, label, None
    prot, metal = _get_rec(tid)
    try:
        res = dock_ligand(prot, molecule_from_rdkit(m, lid), metal=metal,
                          n_spin=8, max_poses=8, include_chem=True)
    except Exception as exc:
        sys.stderr.write(f"[{tid} {lid}] {type(exc).__name__}: {exc}\n")
        return tid, lid, label, None
    if not res.poses:
        return tid, lid, label, None
    energies = sorted(float(p.score.terms.get("total", 0.0)) for p in res.poses)
    return tid, lid, label, energies


def load_set(tid):
    rows = []
    for r in csv.DictReader(open(COMMON / f"{tid}.csv")):
        rows.append((tid, r["ligand_id"], r["smiles"], int(r["label"])))
    return rows


def stats(en):
    import numpy as np
    a = np.asarray(en, dtype=float)
    best = float(a[0])
    top3 = float(a[:3].mean()) if len(a) >= 3 else float(a.mean())
    top5 = float(a[:5].mean()) if len(a) >= 5 else float(a.mean())
    return best, top3, top5, top3 - best, float(a.std())


def main():
    _lock = acquire_lock("multipose")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=36)
    ap.add_argument("--targets", default="")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    todo = list(TARGETS) if a.all or not a.targets else [t.strip() for t in a.targets.split(",") if t.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    for tid in todo:
        dst = OUT / f"{tid}.csv"
        rows = load_set(tid)
        done = list(csv.DictReader(open(dst))) if dst.exists() and dst.stat().st_size else []
        have = {r["ligand_id"] for r in done if r.get("E_best")}
        jobs = [j for j in rows if j[1] not in have]
        if not jobs:
            print(f"{tid}: 已完成 ({len(done)})，跳过", flush=True)
            continue
        print(f"\n{tid}: 目标 {len(rows)}，待算 {len(jobs)}（workers={a.workers}）", flush=True)
        tmp = dst.with_suffix(".csv.tmp")
        n_new = 0
        with open(tmp, "w", newline="") as fh, Pool(a.workers) as pool:
            w = csv.writer(fh)
            w.writerow(["ligand_id", "label", "n_poses", "E_best", "E_top3",
                        "E_top5", "gap", "spread", "energies"])
            for r in done:                      # 保留已有
                w.writerow([r["ligand_id"], r["label"], r["n_poses"], r["E_best"],
                            r["E_top3"], r["E_top5"], r["gap"], r["spread"], r["energies"]])
            fh.flush()
            BATCH = a.workers * 2
            for i in range(0, len(jobs), BATCH):
                for _, lid, lab, en in pool.map(dock_one, jobs[i:i + BATCH], chunksize=1):
                    if en is None:
                        continue
                    best, t3, t5, gap, sp = stats(en)
                    w.writerow([lid, lab, len(en), f"{best:.6e}", f"{t3:.6e}",
                                f"{t5:.6e}", f"{gap:.6e}", f"{sp:.6e}",
                                ";".join(f"{x:.6e}" for x in en)])
                    n_new += 1
                fh.flush()
                print(f"  {tid}: {n_new}/{len(jobs)}", flush=True)
        tmp.replace(dst)
        # 自检：E_best 是否与 bench22_1to10 的 metalgeo_full 一致
        ref = BENCH / f"metalgeo_full_{tid}.csv"
        n_same = n_tot = 0
        if ref.exists():
            old = {r["ligand_id"]: float(r["score"]) for r in csv.DictReader(open(ref)) if r["score"]}
            for r in csv.DictReader(open(dst)):
                if r["ligand_id"] in old and r.get("E_best"):
                    n_tot += 1
                    if abs(float(r["E_best"]) - old[r["ligand_id"]]) < 1e-5:
                        n_same += 1
        print(f"[{tid}] 写出 {n_new} 行；E_best 与 metalgeo_full 一致 {n_same}/{n_tot}", flush=True)

    print("\n全部完成")


if __name__ == "__main__":
    main()
