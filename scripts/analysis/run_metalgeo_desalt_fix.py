"""方案 A：只重跑配体集里的多组分（含盐）分子，用脱盐版替换 MetalGeoDock 的分数。

依据：审计实验（audit_metalgeo_desalt.py，48 个配体）显示
  * 单组分配体：脱盐前后分数 **Δ ≡ 0**（完全不变）；
  * 多组分配体：盐片段在口袋外被硬塞进去，产生巨大 steric 惩罚（HDAC1 里可达 165），
    脱盐后回到正常量级，个别配体 Δ 达 −96.6；整体 Spearman = 0.900。
所以只需重跑多组分分子（约 1535 个），等价于全量脱盐，但省 4 天。

流程（每靶点）：
  1. 从 common set 找出多组分的 ligand_id；
  2. 只对这些配体做脱盐 → embed → dock_ligand(include_chem=True)；
  3. 把新分数替换进 results/bench22_1to10/metalgeo_full_<T>.csv（保持原行序）；
  4. 备份原文件到 *.predeep_salt.bak 一次。

用法：
    python scripts/analysis/run_metalgeo_desalt_fix.py --workers 12
    python scripts/analysis/run_metalgeo_desalt_fix.py --workers 12 --targets CA2,HDAC1
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
_REC = {}


def _get_rec(tid):
    if tid not in _REC:
        from metgeodock.io import read_pdb
        p, metal = TARGETS[tid]
        _REC[tid] = (read_pdb(str(ROOT / p)), metal)
    return _REC[tid]


def dock_desalted(args):
    tid, lid, smiles = args
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from metgeodock.dock import dock_ligand
    from metgeodock.io import molecule_from_rdkit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return lid, None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    m = embed_3d(mol)
    if m is None:
        return lid, None
    prot, metal = _get_rec(tid)
    try:
        res = dock_ligand(prot, molecule_from_rdkit(m, lid), metal=metal,
                          n_spin=8, max_poses=8, include_chem=True)
        if not res.poses:
            return lid, None
        best = min(res.poses, key=lambda p: p.score.total)
        return lid, float(best.score.terms.get("total", 0.0))
    except Exception as exc:
        sys.stderr.write(f"[{tid} {lid}] {type(exc).__name__}: {exc}\n")
        return lid, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--targets", default="")
    a = ap.parse_args()
    todo = [t.strip() for t in a.targets.split(",") if t.strip()] or list(TARGETS)

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    grand = 0
    for tid in todo:
        dst = BENCH / f"metalgeo_full_{tid}.csv"
        if not dst.exists():
            print(f"{tid}: 缺结果文件，跳过", flush=True)
            continue
        rows = list(csv.DictReader(open(dst)))
        done = {r["ligand_id"]: r for r in rows}
        smi = {r["ligand_id"]: r["smiles"] for r in csv.DictReader(open(COMMON / f"{tid}.csv"))}

        jobs = []
        for lid in done:
            s = smi.get(lid)
            if not s:
                continue
            mol = Chem.MolFromSmiles(s)
            if mol is not None and len(Chem.GetMolFrags(mol)) > 1:
                jobs.append((tid, lid, s))
        if not jobs:
            print(f"{tid}: 无多组分配体，跳过", flush=True)
            continue
        print(f"{tid}: 重跑 {len(jobs)} 个多组分配体", flush=True)

        new = {}
        with Pool(a.workers) as pool:
            for i, (lid, sc) in enumerate(pool.imap_unordered(dock_desalted, jobs, chunksize=1), 1):
                if sc is not None:
                    new[lid] = sc
                if i % 20 == 0:
                    print(f"  {tid}: {i}/{len(jobs)}", flush=True)

        bak = dst.with_suffix(".csv.presalt.bak")
        if not bak.exists():
            bak.write_bytes(dst.read_bytes())
        tmp = dst.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["ligand_id", "label", "score"])
            for r in rows:
                sc = new.get(r["ligand_id"])
                w.writerow([r["ligand_id"], r["label"],
                            f"{sc:.6e}" if sc is not None else r["score"]])
        tmp.replace(dst)
        print(f"[{tid}] 替换 {len(new)}/{len(jobs)} 个脱盐分数 -> {dst.name}", flush=True)
        grand += len(new)

    print(f"\n完成，共替换 {grand} 个配体的分数")


if __name__ == "__main__":
    main()
