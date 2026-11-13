"""Absolute physicochemical descriptors for docked poses.

The `total` score saturates at exactly 0.00 for five subtypes at once (steric=0,
coord~1e-25), so it cannot rank anything. The seven graded interaction terms in
metgeodock.interactions are Gaussian functions of distance -- continuous and
non-saturating -- so they can be compared as raw numbers.

For every (ligand, subtype) we keep the lowest-total pose and record:
  total, steric, geo, contact      (from the scoring function)
  hbond, hydrophobic, pi_pi, cation_pi, salt_bridge, halogen, polar_contact

Physiologically relevant receptors:
  HDAC1 5ICN, HDAC2 3MAX, HDAC3 4A69(+SMRT DAD+InsP4),
  HDAC4 5ZOP (HDAC4 + SMRT peptide), HDAC6 5EDU_CD2 (MBP removed), HDAC8 2V5W (open lid)
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
from metgeodock.interactions import interaction_profile, INTERACTION_TERMS

PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "5ZOP_both", "HDAC6": "5EDU_CD2", "HDAC8": "2V5W_A"}
CLASS = {"HDAC1": "I", "HDAC2": "I", "HDAC3": "I", "HDAC8": "I",
         "HDAC4": "IIa", "HDAC6": "IIb"}
_REC = {}


def init_worker():
    for name, pid in PDB.items():
        p = read_pdb(str(ROOT / f"processed/hdac_receptors/{pid}_h.pdb"))
        mi = None
        for i in range(len(p)):
            if str(p.resnames[i]).strip().upper().startswith("ZN"):
                mi = i; break
        _REC[name] = (p, mi)


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
    rd = molecule_from_rdkit(m, f"{mid}_{name}")
    try:
        res = dock_ligand(prot, rd, metal="zn", n_spin=18, max_poses=40, metal_index=mi)
    except Exception:
        return mid, name, None
    if not res.poses:
        return mid, name, None
    bp = min(res.poses, key=lambda p: p.score.total)
    t = {k: float(v) for k, v in bp.score.terms.items()
         if k in ("total", "steric", "coord", "geo", "contact", "map_overlap")}
    # 最佳 pose（total 最低）与最好接触 pose 的相互作用谱都算
    prof = interaction_profile(prot, bp.ligand, mi, rd_mol=m)
    # 再找 top-5 pose 的平均（降低单 pose 抖动）
    tops = sorted(res.poses, key=lambda p: p.score.total)[:5]
    profs = [interaction_profile(prot, p.ligand, mi, rd_mol=m) for p in tops]
    avg = {k: float(np.mean([q[k] for q in profs])) for k in INTERACTION_TERMS}
    return mid, name, {"terms": t, "profile": {k: float(v) for k, v in prof.items()},
                       "profile_top5_avg": avg, "n_pose": len(res.poses)}


def main():
    meta = {r["molecule_id"]: r for r in csv.DictReader(
        open(ROOT / "processed/hdac_cap_ligands.csv"))}
    picks, seen = [], {}
    for m, r in meta.items():
        cl = CLASS[r["target_subtype"]]
        if seen.get(cl, 0) >= 4:
            continue
        picks.append(m); seen[cl] = seen.get(cl, 0) + 1
    jobs = [(m, meta[m]["smiles"], s) for m in picks for s in PDB]
    print(f"{len(picks)} 配体 x {len(PDB)} 亚型 = {len(jobs)} 任务", flush=True)
    out = {}
    with Pool(16, initializer=init_worker) as pool:
        for i, (mid, name, r) in enumerate(pool.imap_unordered(task, jobs, chunksize=1), 1):
            if r:
                out.setdefault(mid, {})[name] = r
            if i % 12 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    rows = [{"molecule": m, "target": meta[m]["target_subtype"],
             "class": CLASS[meta[m]["target_subtype"]], "data": out.get(m, {})}
            for m in picks]
    json.dump(rows, open(ROOT / "results/chem_descriptors.json", "w"), indent=2)
    print(f"写出 results/chem_descriptors.json（{len(rows)} 配体）")

    # 逐项检查：目标亚型的该项是否系统性不同
    S = list(PDB)
    print("\n各亚型 7 项相互作用的平均（top5 pose）")
    print(f"{'亚型':<8}" + "".join(f"{k[:9]:>10}" for k in INTERACTION_TERMS))
    for s in S:
        vals = [r["data"][s]["profile_top5_avg"][k] for r in rows if s in r["data"]]
        if not vals:
            continue
        print(f"{s:<8}" + "".join(f"{np.mean([r['data'][s]['profile_top5_avg'][k] for r in rows if s in r['data']]):>10.3f}"
                                  for k in INTERACTION_TERMS))


if __name__ == "__main__":
    main()
