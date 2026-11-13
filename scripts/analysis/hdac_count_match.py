"""Count-vs-count matching between ligand functional groups and contacted residues.

Previous attempt compared *fractions* (ligand aromatic atoms 60% vs pocket aromatic
residues 9-18%), which is a scale mismatch. Here both sides are COUNTS over the
same four chemical classes, but the pocket side counts only residues the pose
actually touches (<=4.5 A), not the whole 8-16 A shell -- so the comparison is
between "what the ligand offers" and "what this pocket actually presents to it".

match = cosine between the two 4-vectors, plus a demand-satisfaction variant
       mean_i min(need_i, supply_i) / need_i  (saturating, reported for contrast)
"""
from __future__ import annotations
import csv, json, sys
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

PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "5ZOP_both", "HDAC6": "5EDU_CD2", "HDAC8": "2V5W_A"}
S = list(PDB)
CL = {"HDAC1": "I", "HDAC2": "I", "HDAC3": "I", "HDAC8": "I", "HDAC4": "IIa", "HDAC6": "IIb"}
ARO = {"PHE", "TYR", "TRP", "HIS"}
POL = {"SER", "THR", "ASN", "GLN", "TYR", "CYS"}
CHG = {"ASP", "GLU", "LYS", "ARG"}
HYD = {"LEU", "ILE", "VAL", "MET", "ALA", "PRO", "GLY"}
_REC = {}


def init_worker():
    for name, pid in PDB.items():
        p = read_pdb(str(ROOT / f"processed/hdac_receptors/{pid}_h.pdb"))
        mi = None
        for i in range(len(p)):
            if str(p.resnames[i]).strip().upper().startswith("ZN"):
                mi = i; break
        _REC[name] = (p, mi)


def lig_groups(mol):
    """官能团计数：芳香环、氢键供体、氢键受体、疏水碳（非芳香、非极性）"""
    m = Chem.RemoveHs(mol)
    ri = m.GetRingInfo()
    ar_rings = sum(1 for r in ri.AtomRings()
                   if all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r))
    hbd = sum(1 for a in m.GetAtoms() if a.GetTotalNumHs() > 0 and a.GetAtomicNum() in (7, 8))
    hba = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() in (7, 8))
    hyd = sum(1 for a in m.GetAtoms()
              if a.GetAtomicNum() == 6 and not a.GetIsAromatic()
              and all(n.GetAtomicNum() == 6 or n.GetAtomicNum() == 1 for n in a.GetNeighbors()))
    return np.array([ar_rings, max(hbd, 1), max(hba, 1), max(hyd, 1)], float)


def pocket_supply(prot, lig_xyz):
    per = {}
    for i in range(len(prot)):
        if str(prot.elements[i]).strip().upper() == "H":
            continue
        k = int(prot.resseqs[i])
        per.setdefault(k, {"rn": str(prot.resnames[i]).strip().upper(), "xyz": []})
        per[k]["xyz"].append(np.asarray(prot.xyz[i], float))
    c = {"aro": 0, "pol": 0, "chg": 0, "hyd": 0}
    for v in per.values():
        arr = np.array(v["xyz"])
        d = np.sqrt(((lig_xyz[:, None, :] - arr[None, :, :]) ** 2).sum(-1)).min()
        if d > 4.5:
            continue
        rn = v["rn"]
        if rn in ARO: c["aro"] += 1
        if rn in POL: c["pol"] += 1
        if rn in CHG: c["chg"] += 1
        if rn in HYD: c["hyd"] += 1
    # 供给向量：芳香/极性(供体+受体)/带电/疏水
    return np.array([c["aro"], max(c["pol"], 1), max(c["chg"], 1), max(c["hyd"], 1)], float)


def task(args):
    mid, smiles, name = args
    prot, mi = _REC[name]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return mid, name, None
    m = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(m, randomSeed=0xF00D) != 0:
        return mid, name, None
    m = Chem.RemoveHs(m)
    need = lig_groups(m)
    try:
        res = dock_ligand(prot, molecule_from_rdkit(m, f"{mid}_{name}"),
                          metal="zn", n_spin=18, max_poses=40, metal_index=mi)
    except Exception:
        return mid, name, None
    if not res.poses:
        return mid, name, None
    cos_list, sat_list = [], []
    for pose in res.poses[:10]:
        xyz = np.asarray(pose.ligand.xyz, float)
        lm = pose.ligand.heavy_mask()
        xyz = xyz[lm] if lm.any() else xyz
        sup = pocket_supply(prot, xyz)
        cs = float(need @ sup / (np.linalg.norm(need) * np.linalg.norm(sup)))
        sat = float(np.mean([min(a, b) / a for a, b in zip(need, sup)]))
        cos_list.append(cs); sat_list.append(sat)
    return mid, name, {"need": need.tolist(), "cos_max": max(cos_list),
                       "cos_mean": float(np.mean(cos_list)), "sat_mean": float(np.mean(sat_list)),
                       "n_pose": len(res.poses)}


def main():
    meta = {r["molecule_id"]: r for r in csv.DictReader(
        open(ROOT / "processed/hdac_cap_ligands.csv"))}
    picks, seen = [], {}
    for m, r in meta.items():
        cl = CL[r["target_subtype"]]
        if seen.get(cl, 0) >= 4:
            continue
        picks.append(m); seen[cl] = seen.get(cl, 0) + 1
    jobs = [(m, meta[m]["smiles"], s) for m in picks for s in S]
    print(f"{len(picks)} 配体 x {len(S)} 亚型 = {len(jobs)} 任务", flush=True)
    out = {}
    with Pool(16, initializer=init_worker) as pool:
        for i, (mid, name, r) in enumerate(pool.imap_unordered(task, jobs, chunksize=1), 1):
            if r:
                out.setdefault(mid, {})[name] = r
            if i % 12 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    rows = [{"molecule": m, "target": meta[m]["target_subtype"],
             "class": CL[meta[m]["target_subtype"]], "data": out.get(m, {})} for m in picks]
    json.dump(rows, open(ROOT / "results/count_match.json", "w"), indent=2)

    print("\n计数匹配（cosine，越大越匹配）")
    print(f"{'化合物':<16}{'目标':<6}" + "".join(f"{s:>9}" for s in S) + "  判定")
    hits = {"cos_max": 0, "cos_mean": 0, "sat_mean": 0}
    n = 0
    for r in rows:
        if not r["data"]:
            continue
        n += 1
        for key, hk in (("cos_max", "cos_max"), ("cos_mean", "cos_mean"), ("sat_mean", "sat_mean")):
            pass
        best = {k: max(S, key=lambda s: r["data"][s][k]) for k in hits}
        for k in hits:
            hits[k] += (best[k] == r["target"])
        print(f"{r['molecule']:<16}{r['target']:<6}"
              + "".join(f"{r['data'][s]['cos_max']:>9.3f}" for s in S) + f"  {best['cos_max']}")
    print(f"\n准确率（n={n}）")
    for k, v in hits.items():
        print(f"  {k:<10}{v}/{n} = {v/max(n,1):.1%}")
    print(f"  随机      16.7%")


if __name__ == "__main__":
    main()
