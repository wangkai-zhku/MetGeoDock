"""Specificity score: contact the VARIABLE positions.

Baseline idea (user's formulation): the free pocket has a fixed pattern; a ligand
only perturbs what it touches; the positions that differ between subtypes carry the
selectivity information. So the quantity to score is

    spec(ligand, subtype) = sum over pocket residues in contact with the ligand
                            of  variability(that position across the six subtypes)

variability = normalised Shannon entropy of the residue type at that aligned position
(0 = identical in all six subtypes, 1 = all different).

A ligand that only touches the conserved catalytic machinery scores ~0 everywhere;
one that reaches into the subtype-variable rim scores high, and should score highest
on its own subtype.
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
A3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
      "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
      "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
CUT = 4.5


def variability():
    aln = json.load(open(ROOT / "results/hdac_sequence_alignment.json"))
    types = defaultdict(list)
    seqs = {}
    for name, pid in PDB.items():
        prot = read_pdb(str(ROOT / f"processed/hdac_receptors/{pid}.pdb"))
        seen = {}
        for i in range(len(prot)):
            rn = str(prot.resnames[i]).strip().upper()
            if rn in A3:
                seen.setdefault(str(prot.resseqs[i]), rn)
        seqs[name] = {k: A3[v] for k, v in seen.items()}
    ref = "HDAC1"
    for name in PDB:
        if name == ref:
            mp = {k: k for k in seqs[ref]}
        else:
            raw = aln["pairs"].get(name, {}).get("mapping", {})
            mp = {k[1:]: v[1:] for k, v in raw.items()}
        for rk in seqs[ref]:
            tk = mp.get(rk)
            if tk and tk in seqs[name]:
                types[rk].append(seqs[name][tk])
    var = {}
    for rk, ts in types.items():
        if len(ts) < 3:
            var[rk] = 0.0
            continue
        c = defaultdict(int)
        for t in ts:
            c[t] += 1
        n = sum(c.values())
        H = -sum((x / n) * np.log(x / n) for x in c.values())
        var[rk] = float(H / np.log(len(PDB)))
    return var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--cut", type=float, default=CUT)
    ap.add_argument("--out", default="results/hdac_variable_contact.json")
    a = ap.parse_args()

    VAR = variability()
    nz = [v for v in VAR.values() if v > 0]
    print(f"位置可变性: {len(nz)}/{len(VAR)} 个位置可变 (均值 {np.mean(nz):.3f})\n")

    aln = json.load(open(ROOT / "results/hdac_sequence_alignment.json"))
    REC = {}
    for name, pid in PDB.items():
        prot = read_pdb(str(ROOT / f"processed/hdac_receptors/{pid}_h.pdb"))
        by = {}
        for i in range(len(prot)):
            k = str(prot.resseqs[i])
            by.setdefault(k, {"rn": str(prot.resnames[i]).strip().upper(), "xyz": []})
            el = str(prot.elements[i]).strip().upper()
            if el != "H":
                by[k]["xyz"].append(np.asarray(prot.xyz[i], float))
        if name == "HDAC1":
            to_ref = {k: k for k in by}
        else:
            raw = aln["pairs"].get(name, {}).get("mapping", {})
            inv = {v[1:]: k[1:] for k, v in raw.items()}
            to_ref = {k: inv.get(k, k) for k in by}
        REC[name] = {"prot": prot, "by": by, "to_ref": to_ref,
                     "refvar": [VAR.get(to_ref.get(k, k), 0.0) for k in by]}
        nv = sum(1 for v in REC[name]["refvar"] if v > 0)
        print(f"{name}: 残基 {len(by)}, 其中可变位置 {nv}")

    meta = {r["molecule_id"]: r for r in csv.DictReader(
        open(ROOT / "processed/hdac_selectivity_set.csv"))}
    picks, seen = [], defaultdict(int)
    for mid, r in meta.items():
        t = r["target_subtype"]
        if seen[t] >= a.limit:
            continue
        picks.append(mid); seen[t] += 1
    print(f"\n样本 {len(picks)}\n")

    rows = []
    print("=" * 96)
    print(f"可变位置接触分（配体 {a.cut} A 内接触残基的可变性之和）")
    print("=" * 96)
    for mid in picks:
        sub = meta[mid]["target_subtype"]
        mol = Chem.MolFromSmiles(meta[mid]["smiles"])
        if mol is None:
            continue
        m = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(m, randomSeed=0xF00D) != 0:
            continue
        m = Chem.RemoveHs(m)
        rec = {"molecule": mid, "target": sub, "score": {}, "nvar": {}, "ncon": {}}
        print(f"\n{mid} (目标 {sub})")
        print(f"   {'亚型':<8}{'可变接触分':>11}{'可变残基':>9}{'总接触':>8}")
        for name, R in REC.items():
            try:
                res = dock_ligand(R["prot"], molecule_from_rdkit(m, f"{mid}_{name}"),
                                  metal="zn", n_spin=6, max_poses=8)
            except Exception:
                continue
            if not res.poses:
                continue
            best = min(res.poses, key=lambda p: p.score.total)
            lig = np.asarray(best.ligand.xyz, float)
            lm = best.ligand.heavy_mask()
            lig = lig[lm] if lm.any() else lig
            score, nvar, ncon = 0.0, 0, 0
            for k, v in R["by"].items():
                arr = np.array(v["xyz"])
                d = np.sqrt(((lig[:, None, :] - arr[None, :, :]) ** 2).sum(-1)).min()
                if d < a.cut:
                    ncon += 1
                    w = VAR.get(R["to_ref"].get(k, k), 0.0)
                    if w > 0:
                        nvar += 1
                        score += w
            rec["score"][name] = round(score, 3)
            rec["nvar"][name] = nvar
            rec["ncon"][name] = ncon
            mark = "  <- 目标" if name == sub else ""
            print(f"   {name:<8}{score:>11.2f}{nvar:>9}{ncon:>8}{mark}")
        if rec["score"]:
            order = sorted(rec["score"], key=lambda k: -rec["score"][k])
            rec["hit"] = (order[0] == sub)
            rec["rank"] = order.index(sub) + 1
            print(f"   最高 = {order[0]}  目标排名 {rec['rank']}/6  {'✓' if rec['hit'] else '✗'}")
        rows.append(rec)

    ok = sum(1 for r in rows if r.get("hit"))
    print(f"\n目标亚型打分最高: {ok}/{len(rows)}")
    from scipy.stats import binomtest
    if rows:
        p = binomtest(ok, len(rows), 1 / 6, alternative="greater").pvalue
        print(f"二项检验 p = {p:.3f}  (随机 16.7%)")
    json.dump(rows, open(ROOT / a.out, "w"), indent=2)
    print(f"写出 {a.out}")


if __name__ == "__main__":
    main()
