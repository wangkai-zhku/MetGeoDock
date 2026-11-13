"""Does a covariance-combined graded interaction profile rank the true subtype first?

Pipeline per (compound, subtype):
   dock -> pose -> graded interaction profile (7 terms, all smooth functions)
Then:
   * stack every (compound, subtype) profile into a matrix,
   * whiten with its covariance (no activity labels - the weights come from the
     descriptor covariance structure, which is what makes this not a fitted model),
   * combine with equal weight after whitening,
   * ask whether the compound's own subtype scores highest.
"""
from __future__ import annotations
import argparse, csv, json, sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1, help="compounds per target subtype")
    ap.add_argument("--out", default="results/hdac_interaction_profile.json")
    a = ap.parse_args()

    meta = {r["molecule_id"]: r for r in csv.DictReader(
        open(ROOT / "processed/hdac_selectivity_set.csv"))}
    picks, seen = [], {}
    for mid, r in meta.items():
        t = r["target_subtype"]
        if seen.get(t, 0) >= a.limit:
            continue
        picks.append(mid); seen[t] = seen.get(t, 0) + 1
    print(f"样本 {len(picks)}: {[(m, meta[m]['target_subtype']) for m in picks]}\n")

    REC = {n: read_pdb(str(ROOT / f"processed/hdac_receptors/{p}_h.pdb"))
           for n, p in PDB.items()}

    rows, mat = [], []
    for mid in picks:
        sub = meta[mid]["target_subtype"]
        mol = Chem.MolFromSmiles(meta[mid]["smiles"])
        if mol is None:
            continue
        m = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(m, randomSeed=0xF00D) != 0:
            continue
        m = Chem.RemoveHs(m)
        rec = {"molecule": mid, "target": sub, "profiles": {}}
        print(f"{mid} (目标 {sub})")
        print(f"   {'亚型':<8}" + "".join(f"{t[:9]:>11}" for t in INTERACTION_TERMS))
        for name, prot in REC.items():
            try:
                res = dock_ligand(prot, molecule_from_rdkit(m, f"{mid}_{name}"),
                                  metal="zn", n_spin=6, max_poses=8)
            except Exception:
                continue
            if not res.poses:
                continue
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
            rec["profiles"][name] = {k: round(float(prof[k]), 3) for k in INTERACTION_TERMS}
            mat.append([prof[k] for k in INTERACTION_TERMS])
            print(f"   {name:<8}" + "".join(f"{prof[t]:>11.2f}" for t in INTERACTION_TERMS))
        rows.append(rec)

    M = np.array(mat, float)
    print(f"\n谱矩阵: {M.shape}")
    if M.shape[0] >= 8:
        mu = M.mean(0)
        S = np.cov(M, rowvar=False)
        S = np.atleast_2d(S) + np.eye(len(INTERACTION_TERMS)) * 1e-6 * max(np.trace(S), 1)
        w, V = np.linalg.eigh(S)
        W = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
        Zw = (M - mu) @ W
        print("白化后各项的方差（应≈1）:", np.round(Zw.var(0), 2))
        print("\n各项对第一主成分的载荷（绝对值为权重）:")
        pc1 = V[:, -1]
        for t, l in sorted(zip(INTERACTION_TERMS, pc1), key=lambda kv: -abs(kv[1])):
            print(f"  {t:<16}{l:+.3f}")
    else:
        Zw = M

    # combine: whitened, equal weight, using direction that rewards MORE favourable contacts
    print("\n" + "=" * 92)
    print("组合打分（白化后等权求和，越大越好）")
    print("=" * 92)
    hit = 0
    idx = 0
    for rec in rows:
        sub = rec["target"]
        n = len(rec["profiles"])
        z = Zw[idx:idx + n].sum(axis=1)
        idx += n
        order = sorted(rec["profiles"], key=lambda k: -z[list(rec["profiles"]).index(k)])
        rank = order.index(sub) + 1 if sub in order else -1
        ok = rank == 1
        hit += ok
        print(f"  {rec['molecule']:<16} 目标={sub:<7} 排名={rank}/{n}  {'✓' if ok else '✗'}   "
              f"分数: " + ", ".join(f"{k}={v:+.2f}" for k, v in
                                    zip(rec["profiles"], np.round(z, 2))))
    print(f"\n目标亚型排名第1: {hit}/{len(rows)}")

    json.dump(rows, open(ROOT / a.out, "w"), indent=2)
    print(f"写出 {a.out}")


if __name__ == "__main__":
    main()
