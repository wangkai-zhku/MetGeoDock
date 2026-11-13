"""agri_vs 并行重跑（保持与 benchmark_agri_vs.py 完全相同的对接参数）。

原脚本是纯串行循环（504 个配体 × 约 100 s ≈ 14 小时），这里只把内层循环并行化，
其余一律照抄，保证结果可与 results/agri_vs.* 直接对比：

    dock_ligand(protein, lig, metal=rec.metal, n_spin=6, max_poses=6,
                keep_water=False, multi_metal=True, max_pocket_metals=2,
                coord_water=2)

三个靶点的金属：
    hppd  1 × Fe2+          （单金属）
    gs    4 × Mn2+          （multi_metal 会联合使用最多 2 个）
    igpd  2 × Mn2+

输出：results/agri_vs_v2.csv（target,ligand_id,label,score,metal,error）
      支持断点续跑（按 target+ligand_id 跳过）。

用法：
    python scripts/analysis/benchmark_agri_vs_parallel.py --workers 12
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


def _protein(tid, pocket_path):
    if tid not in _REC:
        from metgeodock.io import read_pdb
        _REC[tid] = read_pdb(str(pocket_path))
    return _REC[tid]


def dock_one(args):
    tid, pocket_path, ligand_id, smiles, label, metal = args
    from metgeodock.dock import dock_ligand
    # 直接复用 benchmark_agri_vs.py 的配体构象生成（ETKDGv3 + UFF 40 步，seed=0），
    # 保证与 results/agri_vs.* 的旧结果逐位可比
    from benchmark_agri_vs import smiles_to_ligand
    protein = _protein(tid, pocket_path)
    try:
        lig = smiles_to_ligand(str(smiles), str(ligand_id))
        if lig is None:
            return tid, ligand_id, label, None, "", "embed_fail"
        result = dock_ligand(
            protein, lig, metal=str(metal),
            n_spin=6, max_poses=6, keep_water=False,
            multi_metal=True, max_pocket_metals=2, coord_water=2,
        )
    except Exception as exc:  # noqa: BLE001
        return tid, ligand_id, label, None, "", type(exc).__name__
    if not result.poses:
        return tid, ligand_id, label, None, "", "no_pose"
    best = min(result.poses, key=lambda p: p.score.total)
    return tid, ligand_id, label, float(best.score.total), str(result.sphere.metal_name), ""


def main():
    _lock = acquire_lock("agri_vs")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--lib-dir", type=Path, default=ROOT / "processed/agri_vs")
    ap.add_argument("--targets", default="hppd,gs,igpd")
    ap.add_argument("--out-csv", type=Path, default=ROOT / "results/agri_vs_v2.csv")
    a = ap.parse_args()

    import pandas as pd

    jobs = []
    done = set()
    if a.out_csv.exists():
        for r in csv.DictReader(open(a.out_csv)):
            if r.get("score"):
                done.add((r["target"], r["ligand_id"]))
        print(f"续跑：已有 {len(done)} 条有效记录", flush=True)

    for tid in [t.strip() for t in a.targets.split(",") if t.strip()]:
        pocket = a.lib_dir / tid / "protein_pocket.pdb"
        libp = a.lib_dir / tid / "library.csv"
        if not pocket.exists() or not libp.exists():
            print(f"跳过 {tid}（缺文件）")
            continue
        lib = pd.read_csv(libp)
        n = 0
        for rec in lib.itertuples(index=False):
            if (tid, str(rec.ligand_id)) in done:
                continue
            jobs.append((tid, str(pocket), rec.ligand_id, rec.smiles, int(rec.label), rec.metal))
            n += 1
        print(f"{tid}: 待算 {n} / 总 {len(lib)}", flush=True)

    if not jobs:
        print("没有待算配体")
        return

    new, write_header = [], not a.out_csv.exists()
    with open(a.out_csv, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["target", "ligand_id", "label", "score", "metal", "error"])
        with Pool(a.workers) as pool:
            for i, (tid, lid, lab, sc, met, err) in enumerate(
                    pool.imap_unordered(dock_one, jobs, chunksize=1), 1):
                w.writerow([tid, lid, lab, "" if sc is None else f"{sc:.6e}", met, err])
                if i % 20 == 0:
                    fh.flush()
                    print(f"  {i}/{len(jobs)}", flush=True)
    print(f"写出 {a.out_csv}")


if __name__ == "__main__":
    main()
