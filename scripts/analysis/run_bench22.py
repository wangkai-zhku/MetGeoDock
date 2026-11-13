"""22 靶点 benchmark：MetalGeoDock 对接打分。

用法：python run_bench22.py metalgeo [--targets CA2,HDAC1] [--workers 16]
输出：results/bench22/metalgeo_<TARGET>.csv  (ligand_id,label,score)
断点续跑：已有且行数完整的 CSV 会跳过。
"""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/bench22"
OUT.mkdir(parents=True, exist_ok=True)
SETS = ROOT / "processed/ligand_sets22"
_DEC_CAP = 50          # main() 里按 --dec-cap 覆盖（1:10 集用 1000）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _embed import embed_3d  # noqa: E402

# 靶点 -> (受体路径, 金属, 最多取多少 decoy)
TARGETS = {
    "CA2":            ("processed/receptors10/3CAJ_h.pdb", "zn", 50),
    "CA12":           ("processed/receptors22/CA12_h.pdb", "zn", 50),
    "HDAC1":          ("processed/receptors10/5ICN_h.pdb", "zn", 50),
    "HDAC2":          ("processed/receptors10/3MAX_h.pdb", "zn", 50),
    "HDAC3":          ("processed/hdac_receptors/4A69_h.pdb", "zn", 50),
    "HDAC6":          ("processed/hdac_receptors/5EDU_CD2_h.pdb", "zn", 50),
    "HDAC8":          ("processed/hdac_receptors/2V5W_A_h.pdb", "zn", 50),
    "MMP1":           ("processed/receptors22/MMP1_cat_h.pdb", "zn", 50),
    "MMP2":           ("processed/receptors10/1QIB_h.pdb", "zn", 50),
    "MMP3":           ("processed/receptors22/MMP3_h.pdb", "zn", 50),
    "MMP9":           ("processed/receptors22/MMP9_cat_h.pdb", "zn", 50),
    "MMP13":          ("processed/receptors22/MMP13_h.pdb", "zn", 50),
    "ALOX5":          ("processed/receptors10/3O8Y_h.pdb", "fe2", 50),
    "ARG1":           ("processed/receptors22/ARG1_h.pdb", "mn", 50),
    "ARG2":           ("processed/receptors10/1PQ3_h.pdb", "mn", 50),
    "urease":         ("processed/receptors10/1FWJ_h.pdb", "ni", 50),
    "METAP2":         ("processed/receptors10/1B6A_h.pdb", "co", 50),
    "CYP1A2":         ("processed/receptors22/CYP1A2_h.pdb", "fe3", 50),
    "CYP2E1":         ("processed/receptors22/CYP2E1_h.pdb", "fe3", 50),
    "CYP19A1":        ("processed/receptors22/CYP19A1_h.pdb", "fe3", 50),
    "EGLN2":          ("processed/receptors22/EGLN2_h.pdb", "fe3", 50),
    "HIV1_integrase": ("processed/receptors22/HIV1_cat_h.pdb", "mg", 50),
}
_REC = {}
# main() 里按 --score-mode 设定：full = 含化学项完整打分，base = 旧版仅几何/力场打分
_INCLUDE_CHEM = True


def _get_rec(tid):
    """按需加载受体（每个 worker 只缓存自己用到的，避免 22 个受体被 fork 到每个进程）"""
    if tid not in _REC:
        from metgeodock.io import read_pdb
        p, m, _ = TARGETS[tid]
        _REC[tid] = (read_pdb(str(ROOT / p)), m)
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
                          n_spin=8, max_poses=8, include_chem=_INCLUDE_CHEM)
    except Exception as exc:
        import traceback
        sys.stderr.write(f"[dock_one] {tid} {lig_id}: {type(exc).__name__}: {exc}\n")
        traceback.print_exc(file=sys.stderr)
        return tid, lig_id, label, None
    if not res.poses:
        return tid, lig_id, label, None
    best = min(res.poses, key=lambda p: p.score.total)
    return tid, lig_id, label, float(best.score.terms.get("total", 0.0))


def load_set(tid: str):
    rows = []
    for fn, lab, cap in ((f"{tid}_actives.csv", 1, 100), (f"{tid}_decoys.csv", 0, _DEC_CAP)):
        p = SETS / fn
        if not p.exists():
            continue
        for i, r in enumerate(csv.DictReader(open(p))):
            if i >= cap:
                break
            rows.append((tid, r["ligand_id"], r["smiles"], lab))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["metalgeo"])
    ap.add_argument("--targets", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--prefix", default="metalgeo")
    ap.add_argument("--sets-dir", default="", help="配体集目录，默认 processed/ligand_sets22")
    ap.add_argument("--out-dir", default="", help="结果目录，默认 results/bench22")
    ap.add_argument("--dec-cap", type=int, default=50, help="每靶点最多用多少个 decoy（1:10 集填 1000）")
    ap.add_argument("--score-mode", choices=["full", "base"], default="full",
                    help="full = 含化学项完整打分（对应 metalgeo_full_*）；"
                         "base = 旧版仅几何/力场打分（对应 metalgeo_* / base_*，可复现且快 5 倍）")
    a = ap.parse_args()
    global _INCLUDE_CHEM, SETS, OUT, _DEC_CAP
    _INCLUDE_CHEM = a.score_mode == "full"
    _DEC_CAP = a.dec_cap
    if a.sets_dir:
        SETS = Path(a.sets_dir)
    if a.out_dir:
        OUT = Path(a.out_dir)
        OUT.mkdir(parents=True, exist_ok=True)
    todo = [t.strip() for t in a.targets.split(",") if t.strip()] or list(TARGETS)
    print(f"score-mode={a.score_mode}（include_chem={_INCLUDE_CHEM}），workers={a.workers}", flush=True)

    for tid in todo:
        dst = OUT / f"{a.prefix}_{tid}.csv"
        jobs = load_set(tid)
        done_rows = list(csv.DictReader(open(dst))) if dst.exists() and dst.stat().st_size else []
        have = {r["ligand_id"] for r in done_rows}
        todo_jobs = [j for j in jobs if j[1] not in have]
        if not todo_jobs:
            print(f"{tid}: 已完成 ({len(done_rows)} 条)，跳过", flush=True)
            continue
        print(f"{tid}: 目标 {len(jobs)}，已有 {len(done_rows)}，待补 {len(todo_jobs)}", flush=True)
        tmp = dst.with_suffix(".csv.tmp")
        n_new = 0
        with open(tmp, "w", newline="") as f, Pool(a.workers) as pool:
            w = csv.writer(f)
            w.writerow(["ligand_id", "label", "score"])
            for r in done_rows:                      # 保留已有记录，只追加缺的配体
                w.writerow([r["ligand_id"], r["label"], r["score"]])
            f.flush()
            BATCH = a.workers * 4
            for i in range(0, len(todo_jobs), BATCH):
                chunk = todo_jobs[i:i + BATCH]
                for t_, lig, lab, sc in pool.map(dock_one, chunk, chunksize=1):
                    if sc is None:
                        continue
                    w.writerow([lig, lab, f"{sc:.6e}"])
                    n_new += 1
                f.flush()
                print(f"  {tid}: 新增 {n_new}/{len(todo_jobs)}", flush=True)
        tmp.replace(dst)
        print(f"  {tid}: 补齐 {n_new} 条，共 {len(done_rows) + n_new} 行 -> {dst.name}", flush=True)
    print("全部完成")


if __name__ == "__main__":
    main()
