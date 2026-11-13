"""审计：多组分（含盐）配体对 MetalGeoDock 打分/排序的影响。

对照同一批配体的两种输入：
  A) 原始 SMILES（含盐，现状做法）
  B) 只保留最大片段（脱盐）
比较分数差异与排名一致性。MetalGeoDock 的 molecule_from_rdkit 不做脱盐，
盐片段会作为配体的一部分参与配位/接触/化学项打分。
"""
from __future__ import annotations
import csv, sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/analysis"))
from _embed import embed_3d

CASES = [("CA2", "processed/receptors10/3CAJ_h.pdb", "zn", 12),
         ("HDAC1", "processed/receptors10/5ICN_h.pdb", "zn", 10),
         ("CYP1A2", "processed/receptors22/CYP1A2_h.pdb", "fe3", 8)]
_REC = {}


def _rec(tid, rel, metal):
    if tid not in _REC:
        from metgeodock.io import read_pdb
        _REC[tid] = (read_pdb(str(ROOT / rel)), metal)
    return _REC[tid]


def one(args):
    tid, rel, metal, lid, smi, desalt = args
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from metgeodock.dock import dock_ligand
    from metgeodock.io import molecule_from_rdkit
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return tid, lid, desalt, None
    if desalt:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    m = embed_3d(mol)
    if m is None:
        return tid, lid, desalt, None
    prot, mt = _rec(tid, rel, metal)
    try:
        res = dock_ligand(prot, molecule_from_rdkit(m, lid), metal=mt,
                          n_spin=8, max_poses=8, include_chem=True)
        if not res.poses:
            return tid, lid, desalt, None
        best = min(res.poses, key=lambda p: p.score.total)
        return tid, lid, desalt, float(best.score.terms.get("total", 0.0))
    except Exception:
        return tid, lid, desalt, None


def main():
    jobs = []
    for tid, rel, metal, n_multi in CASES:
        rows = list(csv.DictReader(open(ROOT / "processed/ligand_sets22_1to10" / f"{tid}_decoys.csv")))
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        multi, single = [], []
        for r in rows:
            mol = Chem.MolFromSmiles(r["smiles"])
            if mol is None:
                continue
            (multi if len(Chem.GetMolFrags(mol)) > 1 else single).append(r)
            if len(multi) >= n_multi and len(single) >= 6:
                break
        for r in multi[:n_multi] + single[:6]:
            for desalt in (False, True):
                jobs.append((tid, rel, metal, r["ligand_id"], r["smiles"], desalt))
    print(f"对照任务 {len(jobs)} 个（每配体跑 含盐/脱盐 两次）", flush=True)
    out = {}
    with Pool(12) as pool:
        for tid, lid, desalt, sc in pool.map(one, jobs, chunksize=1):
            out.setdefault((tid, lid), {})["desalt" if desalt else "raw"] = sc
    rows = [(k[0], k[1], v.get("raw"), v.get("desalt")) for k, v in out.items()]
    import numpy as np
    both = [(a, b) for _, _, a, b in rows if a is not None and b is not None]
    print(f"\n{'靶点':8s}{'配体':16s}{'含盐':>10s}{'脱盐':>10s}{'Δ':>9s}")
    for t, l, a, b in sorted(rows):
        if a is None or b is None:
            print(f"{t:8s}{l:16s}{'失败':>10s}{'失败':>10s}")
            continue
        print(f"{t:8s}{l:16s}{a:>10.3f}{b:>10.3f}{b-a:>+9.3f}")
    if both:
        x = np.array([a for a, _ in both]); y = np.array([b for _, b in both])
        from scipy.stats import spearmanr
        print(f"\n共同 {len(both)} 个：Spearman(含盐 vs 脱盐) = {spearmanr(x, y).statistic:.3f}")
        print(f"平均 |Δ| = {np.abs(y-x).mean():.3f}，最大 |Δ| = {np.abs(y-x).max():.3f}")


if __name__ == "__main__":
    main()
