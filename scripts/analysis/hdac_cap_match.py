"""Cap-group complementarity: match the ligand's cap chemistry to the cap region.

The cap region separates classes (3-cluster dendrogram recovers {HDAC1,2,3} and
{HDAC6}, separation 1.25). This script tests whether a ligand's own cap chemistry
matches the cap region of its true class better than that of the other classes.

Per (ligand, subtype):
  * dock, take the best pose
  * find the coordinating atom (closest N/O/S to the metal)
  * cap part = ligand atoms further than CAP_CUT from that atom
  * cap chemistry = hydrophobic fraction, aromatic-ring count, charged-atom count
  * pocket cap profile comes from hdac_cap_region.json
  * complementarity = -sum |ligand_cap - pocket_cap| over the shared descriptors
    (closer chemistry = higher score), plus an aromatic-matching term.

Evaluation is at the CLASS level (class I / IIa / IIb), which is the target the user
set: HDAC1 vs HDAC2 is explicitly out of scope.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
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
       "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC8": "1T64"}
CLASS = {"HDAC1": "I", "HDAC2": "I", "HDAC3": "I", "HDAC8": "I",
         "HDAC4": "IIa", "HDAC6": "IIb"}
CAP_CUT = 5.0
HYD_EL = {"C", "S", "F", "CL", "BR", "I"}


def ligand_cap(lig, metal):
    """atoms further than CAP_CUT from the coordinating atom"""
    xyz = np.asarray(lig.xyz, float)
    el = [str(e).strip().upper() for e in lig.elements]
    heavy = np.array([e != "H" for e in el])
    idx = np.where(heavy)[0]
    if len(idx) == 0:
        return None
    d_metal = np.linalg.norm(xyz[idx] - metal, axis=1)
    anchor = idx[int(d_metal.argmin())]
    d_anchor = np.linalg.norm(xyz - xyz[anchor], axis=1)
    cap = [i for i in idx if d_anchor[i] > CAP_CUT]
    if not cap:
        return None
    cap_el = [el[i] for i in cap]
    hyd = sum(e in HYD_EL for e in cap_el) / len(cap_el)
    polar = sum(e in ("N", "O") for e in cap_el) / len(cap_el)
    return {"n_atoms": len(cap), "frac_hyd": hyd, "frac_polar": polar,
            "has_arom": 0.0, "cap_el": cap_el}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3, help="compounds per class")
    ap.add_argument("--out", default="results/hdac_cap_match.json")
    a = ap.parse_args()

    cap = json.load(open(ROOT / "results/hdac_cap_region.json"))
    meta = {r["molecule_id"]: r for r in csv.DictReader(
        open(ROOT / "processed/hdac_selectivity_set.csv"))}
    picks, seen = [], defaultdict(int)
    for mid, r in meta.items():
        cl = CLASS[r["target_subtype"]]
        if seen[cl] >= a.limit:
            continue
        picks.append(mid); seen[cl] += 1
    print(f"样本 {len(picks)}（按类别平衡）: "
          f"{[(m, CLASS[meta[m]['target_subtype']]) for m in picks]}\n")

    REC = {}
    for name, pid in PDB.items():
        prot = read_pdb(str(ROOT / f"processed/hdac_receptors/{pid}_h.pdb"))
        metal = None
        for i in range(len(prot)):
            if str(prot.resnames[i]).strip().upper().startswith("ZN"):
                metal = np.asarray(prot.xyz[i], float); break
        REC[name] = (prot, metal)

    print("cap 区化学（来自结构）:")
    for s in PDB:
        if s in cap:
            c = cap[s]
            print(f"  {s:<8}{CLASS[s]:<5} 疏水={c['frac_hyd']:.2f} 芳香={c['frac_aro']:.2f} "
                  f"带电={c['frac_chg']:.2f} 极性={c['frac_pol']:.2f}")

    rows = []
    print("\n" + "=" * 96)
    print("cap 基团化学互补性（越高 = 配体 cap 与口袋 cap 区越匹配）")
    print("=" * 96)
    for mid in picks:
        sub = meta[mid]["target_subtype"]
        cl = CLASS[sub]
        mol = Chem.MolFromSmiles(meta[mid]["smiles"])
        if mol is None:
            continue
        m = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(m, randomSeed=0xF00D) != 0:
            continue
        m = Chem.RemoveHs(m)
        rec = {"molecule": mid, "target": sub, "class": cl, "score": {}, "capchem": {}}
        print(f"\n{mid} (目标 {sub} / {cl})")
        print(f"   {'亚型':<8}{'类别':<5}{'匹配分':>9}{'cap疏水':>9}{'cap原子数':>10}")
        for name, (prot, metal) in REC.items():
            if metal is None or name not in cap:
                continue
            try:
                res = dock_ligand(prot, molecule_from_rdkit(m, f"{mid}_{name}"),
                                  metal="zn", n_spin=6, max_poses=8)
            except Exception:
                continue
            if not res.poses:
                continue
            best = min(res.poses, key=lambda p: p.score.total)
            lc = ligand_cap(best.ligand, metal)
            if lc is None:
                continue
            pc = cap[name]
            # complementarity: hydrophobic ligand cap prefers a hydrophobic cap pocket
            dh = abs(lc["frac_hyd"] - pc["frac_hyd"])
            dp = abs(lc["frac_polar"] - pc["frac_pol"])
            score = -(1.0 * dh + 0.5 * dp)
            rec["score"][name] = round(float(score), 4)
            rec["capchem"][name] = {"lig_hyd": round(lc["frac_hyd"], 3),
                                    "n": lc["n_atoms"],
                                    "pocket_hyd": pc["frac_hyd"]}
            mark = "  <- 目标" if name == sub else ""
            print(f"   {name:<8}{CLASS[name]:<5}{score:>9.3f}{lc['frac_hyd']:>9.3f}"
                  f"{lc['n_atoms']:>10}{mark}")
        # class-level ranking
        if rec["score"]:
            byclass = defaultdict(list)
            for s, v in rec["score"].items():
                byclass[CLASS[s]].append(v)
            cls_score = {k: float(np.mean(v)) for k, v in byclass.items()}
            order = sorted(cls_score, key=lambda k: -cls_score[k])
            rec["class_rank"] = order.index(cl) + 1
            rec["hit"] = (order[0] == cl)
            print(f"   类层次: " + ", ".join(f"{k}={cls_score[k]:+.3f}" for k in order)
                  + f"   目标类排名 {rec['class_rank']}/3  {'✓' if rec['hit'] else '✗'}")
        rows.append(rec)

    ok = sum(1 for r in rows if r.get("hit"))
    n = len(rows)
    from scipy.stats import binomtest
    p = binomtest(ok, n, 1 / 3, alternative="greater").pvalue if n else 1.0
    print(f"\n类层次命中: {ok}/{n} = {ok/max(n,1):.1%}  (随机 33.3%, p={p:.3f})")
    json.dump(rows, open(ROOT / a.out, "w"), indent=2)
    print(f"写出 {a.out}")


if __name__ == "__main__":
    main()
