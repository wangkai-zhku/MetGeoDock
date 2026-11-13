"""Open-conformer HDAC8 + pose-ensemble scoring.

The two things never tried together:
  1. HDAC8 uses 2V5W (open L1 lid) instead of 1T64 (closed). L1 swings 6.2 A between
     them, so HDAC8-selective ligands were being penalised by a shut gate.
  2. Scoring uses the whole pose ensemble, not the lowest-total pose. Pose RMSD
     between two near-identical receptors is 8-15 A, so a single pose is not a
     reproducible input.

score(subtype) = fraction of returned poses whose ligand reaches a residue that is
class-enriched in that subtype's cap region (8-16 A shell).
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
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

# HDAC4 -> 6FYZ chain B (most complete HDAC4; 2VQM lacks part of the Loop2 cap rim)
# HDAC6 -> 5EDU CD2 only  (the deposited entry is an MBP fusion; 114-478 is MBP)
# HDAC8 -> 2V5W (open L1 lid; 1T64 has the lid shut)
PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "6FYZ_B", "HDAC6": "5EDU_CD2", "HDAC8": "2V5W_A"}
CLASS = {"HDAC1": "I", "HDAC2": "I", "HDAC3": "I", "HDAC8": "I",
         "HDAC4": "IIa", "HDAC6": "IIb"}
R_IN, R_OUT, CONTACT = 8.0, 16.0, 4.5
W = json.load(open(ROOT / "results/hdac_class_cap_enrichment.json"))
_REC = {}


def load_cap(pid):
    prot = read_pdb(str(ROOT / f"processed/hdac_receptors/{pid}_h.pdb"))
    metal = None
    for i in range(len(prot)):
        if str(prot.resnames[i]).strip().upper().startswith("ZN"):
            metal = np.asarray(prot.xyz[i], float); break
    per = {}
    for i in range(len(prot)):
        if str(prot.elements[i]).strip().upper() == "H":
            continue
        k = str(prot.resseqs[i])
        per.setdefault(k, {"rn": str(prot.resnames[i]).strip().upper(), "xyz": []})
        per[k]["xyz"].append(np.asarray(prot.xyz[i], float))
    cap = {}
    for k, v in per.items():
        d = float(np.linalg.norm(np.array(v["xyz"]) - metal, axis=1).min())
        if R_IN <= d <= R_OUT:
            cap[k] = v
    return prot, metal, cap


def init_worker():
    for name, pid in PDB.items():
        _REC[name] = load_cap(pid)


def task(args):
    mid, smiles, name = args
    prot, metal, cap = _REC[name]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return mid, name, None
    m = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(m, randomSeed=0xF00D) != 0:
        return mid, name, None
    m = Chem.RemoveHs(m)
    try:
        res = dock_ligand(prot, molecule_from_rdkit(m, f"{mid}_{name}"),
                          metal="zn", n_spin=18, max_poses=40)
    except Exception:
        return mid, name, None
    if not res.poses:
        return mid, name, None
    wmap = W[CLASS[name]]
    n_ok, n_pose, best, contacts = 0, len(res.poses), 0.0, 0
    for pose in res.poses:
        lig = np.asarray(pose.ligand.xyz, float)
        lm = pose.ligand.heavy_mask()
        lig = lig[lm] if lm.any() else lig
        hit, sc = 0, 0.0
        for k, v in cap.items():
            arr = np.array(v["xyz"])
            d = np.sqrt(((lig[:, None, :] - arr[None, :, :]) ** 2).sum(-1)).min()
            if d < CONTACT:
                w = wmap.get(v["rn"], 0.0)
                if w > 0.3:
                    hit += 1
                sc += max(w, 0.0)
        if hit:
            n_ok += 1
        if sc > best:
            best, contacts = sc, hit
    # 同时保存原始打分数值（不二值化），供绝对比较
    allp = sorted(res.poses, key=lambda p: p.score.total)
    bp = allp[0]
    terms = {k: float(v) for k, v in bp.score.terms.items()
             if k in ("total", "steric", "coord", "geo", "cpen", "contact", "map", "overlap")}
    return mid, name, {"rate": n_ok / max(n_pose, 1), "n_pose": n_pose,
                       "best": best, "n_enriched": contacts,
                       "score": terms,
                       "n_poses_total": len(res.poses)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/hdac_open_ensemble.json")
    ap.add_argument("--hdac4", default="2VQM")
    ap.add_argument("--hdac8", default="1T64")
    a = ap.parse_args()
    PDB["HDAC4"] = a.hdac4
    PDB["HDAC8"] = a.hdac8

    meta = {r["molecule_id"]: r for r in csv.DictReader(
        open(ROOT / "processed/hdac_cap_ligands.csv"))}
    picks, seen = [], defaultdict(int)
    for mid, r in meta.items():
        cl = CLASS[r["target_subtype"]]
        if seen[cl] >= a.per_class:
            continue
        picks.append(mid); seen[cl] += 1
    print(f"样本 {len(picks)}（HDAC8 用 2V5W 开放构象）: "
          f"{[(m, CLASS[meta[m]['target_subtype']]) for m in picks]}")
    print(f"任务 {len(picks) * 6}，并行 {a.workers}", flush=True)

    jobs = [(mid, meta[mid]["smiles"], name) for mid in picks for name in PDB]
    out = {}
    with Pool(a.workers, initializer=init_worker) as pool:
        for i, (mid, name, r) in enumerate(pool.imap_unordered(task, jobs, chunksize=1), 1):
            if r:
                out.setdefault(mid, {})[name] = r
            if i % 12 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    print("\n" + "=" * 92)
    print("充分采样 (n_spin=18, max_poses=40) + pose 系综：每类平均的 pose 成功率")
    print("=" * 92)
    print(f"{'化合物':<16}{'目标':<6}{'排名':>5}  " + "".join(f"{s:>8}" for s in PDB) + "   类平均 I/IIa/IIb")
    hit = 0
    rows = []
    for mid in picks:
        sub = meta[mid]["target_subtype"]; cl = CLASS[sub]
        r = out.get(mid, {})
        if not r:
            continue
        byc = defaultdict(list)
        for s, v in r.items():
            byc[CLASS[s]].append(v["rate"])
        cs = {k: float(np.mean(v)) for k, v in byc.items()}
        order = sorted(cs, key=lambda k: -cs[k])
        rank = order.index(cl) + 1
        hit += (rank == 1)
        rows.append({"molecule": mid, "target": sub, "class": cl, "rank": rank,
                     "rates": {s: round(v["rate"], 3) for s, v in r.items()},
                     "scores": {s: {k: float(x) for k, x in v.get("score", {}).items()}
                                for s, v in r.items()},
                     "class_scores": {k: round(v, 3) for k, v in cs.items()}})
        print(f"{mid:<16}{cl:<6}{rank:>5}  " +
              "".join(f"{r[s]['rate']:>8.2f}" if s in r else f"{'-':>8}" for s in PDB) +
              "   " + "/".join(f"{cs.get(k, 0):.2f}" for k in ("I", "IIa", "IIb")))
    n = len(rows)
    from scipy.stats import binomtest
    p = binomtest(hit, n, 1 / 3, alternative="greater").pvalue if n else 1.0
    print(f"\n类层次命中: {hit}/{n} = {hit/max(n,1):.1%}   (随机 33.3%, p={p:.3f})")
    by_cls = defaultdict(lambda: [0, 0])
    for rr in rows:
        by_cls[rr["class"]][1] += 1
        by_cls[rr["class"]][0] += (rr["rank"] == 1)
    for k, (a_, b_) in sorted(by_cls.items()):
        print(f"  class {k:<4}{a_}/{b_}")
    json.dump(rows, open(ROOT / a.out, "w"), indent=2)
    print(f"写出 {a.out}")


if __name__ == "__main__":
    main()
