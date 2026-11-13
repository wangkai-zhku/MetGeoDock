"""Class-specific cap contact score.

The cap region separates classes structurally (3-cluster dendrogram recovers
{HDAC1,2,3} and {HDAC6}). The earlier "chemical similarity" match failed because
ligand caps are uniformly more hydrophobic (0.62-0.91) than any pocket cap region
(0.47-0.64), so the score just picked the most hydrophobic pocket.

This version asks a different question: does the ligand's cap touch the residue
types that are enriched in its own class's cap region?

    score(ligand, subtype) = sum over cap-region residues in contact with the
                             ligand cap of  w[class(subtype)][residue type]

with w = log2 enrichment of that residue type in the class cap region versus the
other classes (from hdac_class_cap_enrichment.json). A class I selective ligand
should make contacts to LYS/CYS/ILE (class I-enriched) and therefore score highest
when evaluated against the class I weights.
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
R_IN, R_OUT = 8.0, 16.0
CONTACT = 4.5


def cap_residues(pid):
    """residue key -> (type, heavy atom coords) for the cap region"""
    prot = read_pdb(str(ROOT / f"processed/hdac_receptors/{pid}_h.pdb"))
    metal = None
    for i in range(len(prot)):
        if str(prot.resnames[i]).strip().upper().startswith("ZN"):
            metal = np.asarray(prot.xyz[i], float); break
    if metal is None:
        return {}, None
    per = {}
    for i in range(len(prot)):
        el = str(prot.elements[i]).strip().upper()
        if el == "H":
            continue
        k = str(prot.resseqs[i])
        per.setdefault(k, {"rn": str(prot.resnames[i]).strip().upper(), "xyz": []})
        per[k]["xyz"].append(np.asarray(prot.xyz[i], float))
    cap = {}
    for k, v in per.items():
        arr = np.array(v["xyz"])
        d = float(np.linalg.norm(arr - metal, axis=1).min())
        if R_IN <= d <= R_OUT:
            cap[k] = v
    return cap, metal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3, help="compounds per class")
    ap.add_argument("--out", default="results/hdac_cap_class_contact.json")
    a = ap.parse_args()

    W = json.load(open(ROOT / "results/hdac_class_cap_enrichment.json"))
    meta = {r["molecule_id"]: r for r in csv.DictReader(
        open(ROOT / "processed/hdac_cap_ligands.csv"))}
    picks, seen = [], defaultdict(int)
    for mid, r in meta.items():
        cl = CLASS[r["target_subtype"]]
        if seen[cl] >= a.limit:
            continue
        picks.append(mid); seen[cl] += 1
    print(f"样本 {len(picks)}: {[(m, CLASS[meta[m]['target_subtype']]) for m in picks]}\n")

    CAP = {}
    for name, pid in PDB.items():
        cap, metal = cap_residues(pid)
        CAP[name] = (cap, metal)
        n = sum(1 for v in cap.values() if v["rn"] in W.get(CLASS[name], {}))
        print(f"{name:<8}{CLASS[name]:<5} cap 残基 {len(cap)}")

    rows = []
    print("\n" + "=" * 96)
    print("类特异 cap 接触分（接触残基按该类富集度加权）")
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
        rec = {"molecule": mid, "target": sub, "class": cl, "score": {}, "hits": {}}
        print(f"\n{mid} (目标 {sub} / {cl})")
        print(f"   {'亚型':<8}{'类别':<5}{'特异接触分':>11}{'接触cap残基':>12}{'富集残基':>9}")
        for name, (cap, metal) in CAP.items():
            if metal is None or not cap:
                continue
            prot = read_pdb(str(ROOT / f"processed/hdac_receptors/{PDB[name]}_h.pdb"))
            try:
                res = dock_ligand(prot, molecule_from_rdkit(m, f"{mid}_{name}"),
                                  metal="zn", n_spin=6, max_poses=8)
            except Exception:
                continue
            if not res.poses:
                continue
            wmap = W[CLASS[name]]
            # FLEXIBILITY as a RATE: what fraction of the returned poses manage to
            # reach class-enriched residues at all? A selective ligand should find
            # many such poses in its own subtype and few in the others.
            n_ok, n_pose = 0, len(res.poses)
            best_sc, ncon, nrich = 0.0, 0, 0
            for pose in res.poses:
                lig = np.asarray(pose.ligand.xyz, float)
                lm = pose.ligand.heavy_mask()
                lig = lig[lm] if lm.any() else lig
                s_p, c_p, r_p = 0.0, 0, 0
                for k, v in cap.items():
                    arr = np.array(v["xyz"])
                    d = np.sqrt(((lig[:, None, :] - arr[None, :, :]) ** 2).sum(-1)).min()
                    if d < CONTACT:
                        c_p += 1
                        w = wmap.get(v["rn"], 0.0)
                        if w > 0.3:
                            r_p += 1
                        s_p += max(w, 0.0)
                if r_p > 0:
                    n_ok += 1
                if s_p > best_sc:
                    best_sc, ncon, nrich = s_p, c_p, r_p
            score = n_ok / max(n_pose, 1)
            rec["score"][name] = round(float(score), 3)
            rec["hits"][name] = nrich
            mark = "  <- 目标" if name == sub else ""
            print(f"   {name:<8}{CLASS[name]:<5}{score:>11.2f}{ncon:>12}{nrich:>9}{mark}")
        if rec["score"]:
            byc = defaultdict(list)
            for s, v in rec["score"].items():
                byc[CLASS[s]].append(v)
            cs = {k: float(np.mean(v)) for k, v in byc.items()}
            order = sorted(cs, key=lambda k: -cs[k])
            rec["class_rank"] = order.index(cl) + 1
            rec["hit"] = (order[0] == cl)
            print(f"   类层次: " + ", ".join(f"{k}={cs[k]:.2f}" for k in order)
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
