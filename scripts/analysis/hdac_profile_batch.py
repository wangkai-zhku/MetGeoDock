"""Batch interaction-profile run for the HDAC selectivity test (50 compounds x 6 subtypes).

Fixes over the 5-compound pilot:
  * salt_bridge now uses Gasteiger partial charges (formal charges were all zero),
  * evaluation normalises WITHIN each subtype before comparing across subtypes -
    the pilot showed HDAC6 scores high for everyone simply because its pocket is
    more open, which is a baseline artefact, not selectivity,
  * covariance whitening is estimated on the full matrix instead of 30 rows.

Runs in parallel; each worker keeps the six receptors resident.
"""
from __future__ import annotations
import argparse, csv, json, sys, itertools
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
sys.path.insert(0, str(ROOT))
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
from metgeodock.dock import dock_ligand
from metgeodock.io import molecule_from_rdkit, read_pdb
from metgeodock.interactions import interaction_profile, INTERACTION_TERMS

PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC8": "1T64"}
_REC = {}


def init_worker():
    global _REC
    for n, p in PDB.items():
        _REC[n] = read_pdb(str(ROOT / f"processed/hdac_receptors/{p}_h.pdb"))


def task(args):
    mid, smiles, name = args
    prot = _REC[name]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return mid, name, None
    m = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(m, randomSeed=0xF00D) != 0:
        return mid, name, None
    m = Chem.RemoveHs(m)
    try:
        res = dock_ligand(prot, molecule_from_rdkit(m, f"{mid}_{name}"),
                          metal="zn", n_spin=6, max_poses=8)
    except Exception:
        return mid, name, None
    if not res.poses:
        return mid, name, None
    best = min(res.poses, key=lambda p: p.score.total)
    pose = np.asarray(best.ligand.xyz, float)
    mm = Chem.Mol(m)
    if mm.GetNumAtoms() == len(pose):
        conf = mm.GetConformer()
        for i in range(mm.GetNumAtoms()):
            conf.SetAtomPosition(i, pose[i].tolist())
        prof = interaction_profile(prot, best.ligand, rd_mol=mm)
    else:
        prof = interaction_profile(prot, best.ligand, rd_mol=None)
    return mid, name, {k: float(prof[k]) for k in INTERACTION_TERMS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="results/hdac_profile_50.json")
    a = ap.parse_args()

    meta = {r["molecule_id"]: r for r in csv.DictReader(
        open(ROOT / "processed/hdac_selectivity_set.csv"))}
    # balanced sample over target subtypes
    by_t = {}
    for mid, r in meta.items():
        by_t.setdefault(r["target_subtype"], []).append(mid)
    want = {"HDAC8": int(a.n * 0.5), "HDAC6": int(a.n * 0.3), "HDAC1": int(a.n * 0.12),
            "HDAC2": 2, "HDAC4": 2}
    picks = []
    for t, k in want.items():
        picks += by_t.get(t, [])[:k]
    picks = picks[:a.n]
    print(f"样本 {len(picks)} 个，目标亚型分布: "
          f"{ {t: sum(1 for p in picks if meta[p]['target_subtype'] == t) for t in want} }")

    jobs = [(mid, meta[mid]["smiles"], name) for mid in picks for name in PDB]
    print(f"总任务 {len(jobs)}（{len(picks)} 化合物 × 6 亚型），并行 {a.workers}", flush=True)
    out = {}
    with Pool(a.workers, initializer=init_worker) as pool:
        for i, (mid, name, prof) in enumerate(pool.imap_unordered(task, jobs, chunksize=2), 1):
            if prof:
                out.setdefault(mid, {})[name] = prof
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}  已完成化合物 {len(out)}", flush=True)

    complete = {m: v for m, v in out.items() if len(v) == 6}
    print(f"\n完成: {len(complete)} 个化合物在 6 个亚型上都有谱")

    # ---- build matrix
    ids = list(complete)
    M = np.array([[complete[m][s][t] for t in INTERACTION_TERMS]
                  for m in ids for s in PDB], float)
    print(f"谱矩阵 {M.shape}")
    print("\n各项的跨样本统计（方差接近 0 的项无信息）:")
    for j, t in enumerate(INTERACTION_TERMS):
        v = M[:, j]
        print(f"  {t:<16} 均值={v.mean():7.3f} 标准差={v.std():7.3f} 零值占比={np.mean(v==0):.0%}")

    # ---- subtype-wise z-normalisation (the fix)
    Z = M.reshape(len(ids), len(PDB), len(INTERACTION_TERMS)).copy()
    for s_i in range(len(PDB)):
        block = Z[:, s_i, :]
        mu = block.mean(0)
        sd = block.std(0)
        sd[sd < 1e-9] = 1.0
        Z[:, s_i, :] = (block - mu) / sd
    Zf = Z.reshape(-1, len(INTERACTION_TERMS))

    # ---- whitening on the normalised matrix, dropping near-constant terms
    keep = [j for j in range(len(INTERACTION_TERMS)) if M[:, j].std() > 1e-6]
    print(f"\n保留 {len(keep)} 项参与组合: {[INTERACTION_TERMS[j] for j in keep]}")
    Zw = Zf[:, keep]
    S = np.cov(Zw, rowvar=False)
    S = np.atleast_2d(S) + np.eye(len(keep)) * 1e-6 * max(np.trace(S), 1)
    w, V = np.linalg.eigh(S)
    W = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
    Zw = (Zw - Zw.mean(0)) @ W

    # every term is "more contacts = better", so sum with a positive sign
    score = Zw.sum(axis=1)
    hit = 0
    rows = []
    print("\n" + "=" * 100)
    print("亚型内归一化 + 白化组合（越大越好）")
    print("=" * 100)
    print(f"{'化合物':<16}{'目标':<8}{'排名':>6}  {'分数 (HDAC1/2/3/4/6/8)'}")
    for i, m in enumerate(ids):
        sub = meta[m]["target_subtype"]
        sc = {s: score[i * len(PDB) + s_i] for s_i, s in enumerate(PDB)}
        order = sorted(sc, key=lambda k: -sc[k])
        rank = order.index(sub) + 1 if sub in order else -1
        hit += (rank == 1)
        rows.append({"molecule": m, "target": sub, "rank": rank,
                     "scores": {k: round(float(v), 3) for k, v in sc.items()}})
        if i < 25:
            print(f"{m:<16}{sub:<8}{rank:>6}  " +
                  " ".join(f"{sc[s]:+6.2f}" for s in PDB))
    n = len(ids)
    print(f"\n目标亚型排名第1: {hit}/{n} = {hit/n:.1%}   (随机 {1/6:.1%})")
    top2 = sum(1 for r in rows if r["rank"] <= 2)
    top3 = sum(1 for r in rows if r["rank"] <= 3)
    print(f"前2名: {top2}/{n} = {top2/n:.1%} (随机 {2/6:.1%})   前3名: {top3}/{n} = {top3/n:.1%} (随机 {3/6:.1%})")

    json.dump({"scores": rows, "profiles": complete}, open(ROOT / a.out, "w"), indent=2)
    print(f"\n写出 {a.out}")


if __name__ == "__main__":
    main()
