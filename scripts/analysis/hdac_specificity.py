"""Subtype-specific contact score: does the ligand exploit the residues that make
this subtype different?

The current score, total = steric + 0.35*coord, measures whether a ligand fits at
all. It has no term that could distinguish "fits HDAC8" from "fits HDAC6", which is
why subtype ranking came out at chance. This adds that missing term.

Definition
----------
1. Per subtype, take all residues with any atom in the 6-14 A shell (= the rim/foot
   pocket, where the earlier analysis showed 80-95% of the inter-subtype difference
   lives).
2. Compare each subtype's residue-TYPE composition with the mean over the other five
   subtypes -> an enrichment weight per residue type (log ratio).
3. For a docked pose, count ligand heavy atoms within 4.5 A of each residue, weight
   by that residue's type enrichment, and sum:

       spec = sum_over_residues  log2( f_this / f_others ) * contacts  /  n_lig_heavy

   A ligand that parks itself against the residues this subtype has more of than the
   others scores high; one that binds the conserved core scores ~0 for every subtype.

Only steric geometry is used, no activity labels, no crystal waters.
"""
from __future__ import annotations
import argparse, csv, glob, json, sys
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
SHELL = (6.0, 14.0)
CONTACT = 4.5


def load_residues(pid):
    """residue -> (type, [heavy atom coords])"""
    res = defaultdict(lambda: {"type": None, "xyz": []})
    metal = None
    for ln in (ROOT / f"processed/hdac_receptors/{pid}_h.pdb").read_text(errors="ignore").splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        rn = ln[17:20].strip().upper()
        el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        if ln.startswith("ATOM") and el != "H":
            key = f"{rn}{ln[21]}{ln[22:26].strip()}"
            res[key]["type"] = rn
            res[key]["xyz"].append(xyz)
        elif rn.startswith("ZN"):
            metal = xyz
    return res, metal


def enrichment(table):
    """per subtype, log2 enrichment of each residue type in the pocket shell"""
    types = sorted({t for v in table.values() for t in v})
    mat = {}
    for s, comp in table.items():
        tot = sum(comp.values()) or 1
        mat[s] = {t: comp.get(t, 0) / tot for t in types}
    out = {}
    for s in table:
        others = [x for x in table if x != s]
        w = {}
        for t in types:
            fa = max(mat[s][t], 1e-4)
            fo = max(np.mean([mat[o][t] for o in others]), 1e-4)
            w[t] = float(np.log2(fa / fo))
        out[s] = w
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2, help="compounds per target subtype")
    ap.add_argument("--out", default="results/hdac_specificity.json")
    a = ap.parse_args()

    meta = {r["molecule_id"]: r for r in csv.DictReader(
        open(ROOT / "processed/hdac_selectivity_set.csv"))}
    picks, seen = [], defaultdict(int)
    for mid, r in meta.items():
        t = r["target_subtype"]
        if seen[t] >= a.limit:
            continue
        picks.append(mid); seen[t] += 1
    print(f"样本 {len(picks)} 个: {[(m, meta[m]['target_subtype']) for m in picks]}\n")

    RES, METAL, SHELLRES, W = {}, {}, {}, {}
    comp = {}
    for name, pid in PDB.items():
        res, metal = load_residues(pid)
        RES[name], METAL[name] = res, metal
        in_shell = {k: v for k, v in res.items()
                    if metal is not None and
                    min(float(np.linalg.norm(x - metal)) for x in v["xyz"]) <= SHELL[1]
                    and min(float(np.linalg.norm(x - metal)) for x in v["xyz"]) >= SHELL[0]}
        SHELLRES[name] = in_shell
        comp[name] = defaultdict(int)
        for v in in_shell.values():
            comp[name][v["type"]] += 1
        print(f"{name}: 壳层残基 {len(in_shell)}, 类型 {len(comp[name])}")
    W = enrichment(comp)
    print("\n各亚型的特征残基类型（log2 富集 > 0.5）:")
    for s in PDB:
        top = sorted(W[s].items(), key=lambda kv: -kv[1])[:4]
        print(f"  {s:<8}" + ", ".join(f"{t}({w:+.2f})" for t, w in top))

    rows = []
    print("\n" + "=" * 98)
    print("特异性接触分 spec（越大 = 越依赖该亚型的特征残基）")
    print("=" * 98)
    for mid in picks:
        sub = meta[mid]["target_subtype"]
        mol = Chem.MolFromSmiles(meta[mid]["smiles"])
        if mol is None:
            continue
        m = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(m, randomSeed=0xF00D) != 0:
            continue
        m = Chem.RemoveHs(m)
        row = {"molecule": mid, "target": sub, "spec": {}, "steric": {}}
        print(f"\n{mid} (目标 {sub})")
        print(f"   {'亚型':<8}{'spec':>9}{'steric':>9}{'接触残基数':>11}")
        for name in PDB:
            if METAL[name] is None:
                continue
            prot = read_pdb(str(ROOT / f"processed/hdac_receptors/{PDB[name]}_h.pdb"))
            try:
                r = dock_ligand(prot, molecule_from_rdkit(m, f"{mid}_{name}"),
                                metal="zn", n_spin=6, max_poses=8)
            except Exception:
                continue
            if not r.poses:
                continue
            best = min(r.poses, key=lambda p: p.score.total)
            lig = np.asarray(best.ligand.xyz, float)
            lm = best.ligand.heavy_mask()
            lig = lig[lm] if lm.any() else lig
            spec, nres = 0.0, 0
            for key, v in SHELLRES[name].items():
                arr = np.array(v["xyz"])
                d = np.sqrt(((lig[:, None, :] - arr[None, :, :]) ** 2).sum(-1))
                contacts = int((d.min(axis=1) < CONTACT).sum())
                if contacts:
                    nres += 1
                    spec += W[name].get(v["type"], 0.0) * contacts
            spec /= max(len(lig), 1)
            row["spec"][name] = round(float(spec), 4)
            row["steric"][name] = round(float(best.score.terms.get("steric", np.nan)), 2)
            mark = "  ← 目标" if name == sub else ""
            print(f"   {name:<8}{spec:>9.3f}{row['steric'][name]:>9.2f}{nres:>11}{mark}")
        if row["spec"]:
            best_sub = max(row["spec"], key=lambda k: row["spec"][k])
            row["hit"] = (best_sub == sub)
            print(f"   spec 最高 = {best_sub}   {'✓ 正确' if row['hit'] else '✗'}")
        rows.append(row)

    ok = sum(1 for r in rows if r.get("hit"))
    print(f"\n目标亚型 spec 最高: {ok}/{len(rows)}")
    json.dump(rows, open(ROOT / a.out, "w"), indent=2)
    print(f"写出 {a.out}")


if __name__ == "__main__":
    main()
