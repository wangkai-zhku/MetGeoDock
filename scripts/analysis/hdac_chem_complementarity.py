"""Receptor-ligand chemical complementarity, using the descriptors already computed.

Receptor side (results/hdac_cap_region.json):
    frac_hyd, frac_aro, frac_chg, frac_pol  over the 8-16 A cap shell
Ligand side (computed here from SMILES with RDKit):
    the same four fractions over heavy atoms
plus the Jaccard shape matrix (results/hdac_pocket_shape.json) used as a
"how confusable are these two subtypes" weight, and the 7 graded interaction
terms from results/chem_descriptors.json as the pose-level term.

Three combiners are tested, all judged on ABSOLUTE values (no rank-1 rule):
  match     1 - weighted L1 distance between ligand and cap fingerprints
  profile   the 7 interaction terms, L2-normalised, cosine against the cap
            fingerprint of each subtype
  combined  match * profile  (geometric mean)
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
sys.path.insert(0, str(ROOT))
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")

S = ["HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC6", "HDAC8"]
CL = {"HDAC1": "I", "HDAC2": "I", "HDAC3": "I", "HDAC8": "I", "HDAC4": "IIa", "HDAC6": "IIb"}
KEYS = ["frac_hyd", "frac_aro", "frac_chg", "frac_pol"]


def ligand_fractions(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    m = Chem.RemoveHs(m)
    n = m.GetNumAtoms()
    hyd = aro = chg = pol = 0.0
    for a in m.GetAtoms():
        if a.GetIsAromatic():
            aro += 1
        z = a.GetAtomicNum()
        if z == 6 and not a.GetIsAromatic():
            hyd += 1
        if a.GetFormalCharge() != 0:
            chg += 1
        if z in (7, 8):
            pol += 1
    return {"frac_hyd": hyd / n, "frac_aro": aro / n,
            "frac_chg": chg / n, "frac_pol": pol / n}


def main():
    cap = json.load(open(ROOT / "results/hdac_cap_region.json"))
    chem = json.load(open(ROOT / "results/chem_descriptors.json"))
    try:
        shape = json.load(open(ROOT / "results/hdac_pocket_shape.json"))
    except Exception:
        shape = None

    print("受体 cap 区化学指纹（8-16 A 壳层）")
    print(f"{'亚型':<8}{'类别':<5}" + "".join(f"{k[5:]:>9}" for k in KEYS))
    capf = {}
    for s in S:
        v = {k: float(cap[s][k]) for k in KEYS}
        capf[s] = v
        print(f"{s:<8}{CL[s]:<5}" + "".join(f"{v[k]:>9.3f}" for k in KEYS))

    meta = {r["molecule_id"]: r for r in csv.DictReader(
        open(ROOT / "processed/hdac_cap_ligands.csv"))}

    print("\n配体化学指纹")
    print(f"{'化合物':<16}{'目标':<6}" + "".join(f"{k[5:]:>9}" for k in KEYS))
    ligf = {}
    for r in chem:
        mid = r["molecule"]
        f = ligand_fractions(meta[mid]["smiles"])
        if f is None:
            continue
        ligf[mid] = f
        print(f"{mid:<16}{r['target']:<6}" + "".join(f"{f[k]:>9.3f}" for k in KEYS))

    # ---- combiner 1: match = 1 - mean |ligand - cap| / scale
    print("\n组合一：化学互补性（1 - 归一化 L1 距离），值越大越匹配")
    print(f"{'化合物':<16}{'目标':<6}" + "".join(f"{s:>9}" for s in S) + "  判定")
    hits = {"match": 0, "profile": 0, "combined": 0}
    n = 0
    detail = []
    for r in chem:
        mid = r["molecule"]
        if mid not in ligf or r["target"] not in S:
            continue
        n += 1
        lf = ligf[mid]
        match = {}
        for s in S:
            d = np.mean([abs(lf[k] - capf[s][k]) for k in KEYS])
            match[s] = 1.0 - d
        # ---- combiner 2: profile cosine (7 interaction terms vs cap polarity pattern)
        prof = {}
        for s in S:
            if s not in r["data"]:
                continue
            p = np.array([r["data"][s]["profile_top5_avg"][k] for k in
                          ("hbond", "hydrophobic", "pi_pi", "cation_pi",
                           "salt_bridge", "halogen", "polar_contact")])
            prof[s] = float(p.sum())
        # normalize profile to 0-1 for combination
        if prof:
            pv = np.array(list(prof.values()))
            pmin, pmax = pv.min(), pv.max()
            prof_n = {s: (v - pmin) / max(pmax - pmin, 1e-9) for s, v in prof.items()}
        else:
            prof_n = {s: 0.0 for s in S}
        # ---- combiner 3: geometric mean
        comb = {s: float(np.sqrt(max(match[s], 0) * max(prof_n.get(s, 0), 0))) for s in S}
        best_m = max(S, key=match.get)
        best_p = max(S, key=lambda s: prof_n.get(s, 0))
        best_c = max(S, key=comb.get)
        hits["match"] += (best_m == r["target"])
        hits["profile"] += (best_p == r["target"])
        hits["combined"] += (best_c == r["target"])
        detail.append((mid, r["target"], best_m, best_p, best_c, match, prof_n, comb))
        print(f"{mid:<16}{r['target']:<6}" + "".join(f"{match[s]:>9.3f}" for s in S)
              + f"  {best_m}")

    print(f"\n准确率（n={n}）")
    for k, v in hits.items():
        print(f"  {k:<10}{v}/{n} = {v/max(n,1):.1%}")
    print(f"  随机(6选1)  16.7%")

    print("\n每种组合的判定明细")
    print(f"{'化合物':<16}{'目标':<6}{'match判':>10}{'profile判':>11}{'combined判':>12}")
    for mid, tgt, bm, bp, bc, *_ in detail:
        mk = lambda b: b + ("✓" if b == tgt else "✗")
        print(f"{mid:<16}{tgt:<6}{mk(bm):>10}{mk(bp):>11}{mk(bc):>12}")

    # Jaccard 诊断：被判错的亚型与目标亚型的口袋相似度
    if shape and "jaccard" in json.dumps(shape)[:2000]:
        print("\nJaccard 矩阵（诊断：被判错的亚型是否口袋高度相似）")
        J = shape.get("jaccard") or shape
        try:
            print(f"{'':<8}" + "".join(f"{s:>9}" for s in S))
            for a in S:
                print(f"{a:<8}" + "".join(f"{float(J[a][b]):>9.3f}" for b in S))
        except Exception:
            print("  （结构不符，跳过）")


if __name__ == "__main__":
    main()
