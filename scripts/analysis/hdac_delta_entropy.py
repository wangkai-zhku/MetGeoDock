"""Ligand-induced rotamer entropy loss: the pocket descriptor meets the pose.

Static rotamer entropy only sees the crystal structure, so it cannot rank ligands.
But once a ligand is docked, its atoms squeeze the side chains exactly like the
metal ion and the rest of the protein do. The entropy the pocket loses on binding,

    dS = S(pocket + ligand) - S(pocket alone)

depends on how well the ligand's shape fits: a complementary pose displaces little,
a clashing pose locks many side chains. That gives a per-(ligand, subtype) number
instead of a per-protein number, which is what selectivity actually needs.

Only steric (LJ) terms are used - metal steric included, electrostatics excluded
(measured earlier to be harmful here). Crystal waters are never used; the docking
already supplies predicted coordinating waters via coord_water.
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

spec_path = ROOT / "scripts/analysis/pocket_rotamer_metal.py"
import importlib.util
spec = importlib.util.spec_from_file_location("pm", spec_path)
pm = importlib.util.module_from_spec(spec)
sys.modules["pm"] = pm
spec.loader.exec_module(pm)

PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC8": "1T64"}
SHELL = (8.0, 14.0)
RT = 0.593
EPS_OFF = 1e6          # electrostatics off
PROBE_FREE = 0.0


def pocket_entropy(protein_xyz, protein_el, by_res, dmin, metal, extra_xyz, extra_el, rt,
                   all_keys=None):
    """sum of chi1 entropies over pocket residues, with extra atoms in the environment"""
    pocket = [k for k, d in dmin.items() if SHELL[0] < d <= SHELL[1]]
    all_keys = all_keys if all_keys is not None else []
    S, rows = 0.0, {}
    for key in pocket:
        v = by_res[key]
        # environment = every protein atom not in this residue, plus the ligand
        mask = np.array([k != key for k in all_keys], bool)
        if len(extra_xyz):
            ex = np.vstack([protein_xyz[mask], extra_xyz])
            exel = [e for e, m in zip(protein_el, mask) if m] + list(extra_el)
        else:
            ex = protein_xyz[mask]
            exel = [e for e, m in zip(protein_el, mask) if m]
        r = pm.residue_entropy(v["rn"], v["atoms"], ex, exel, metal, rt, EPS_OFF)
        if r is None:
            continue
        S += r["S"]
        rows[key] = r["S"]
    return float(S), rows, len(pocket)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--out", default="results/hdac_delta_S.json")
    a = ap.parse_args()

    meta = {r["molecule_id"]: r for r in csv.DictReader(
        open(ROOT / "processed/hdac_selectivity_set.csv"))}
    # pick compounds spread over subtypes
    picks, seen = [], set()
    for mid, r in meta.items():
        t = r["target_subtype"]
        if t in seen:
            continue
        picks.append(mid); seen.add(t)
        if len(picks) >= a.limit:
            break
    print(f"样本: {[(m, meta[m]['target_subtype']) for m in picks]}\n")

    # pre-parse receptors
    REC = {}
    for name, pid in PDB.items():
        prot = read_pdb(str(ROOT / f"processed/hdac_receptors/{pid}_h.pdb"))
        by = {}
        for i in range(len(prot)):
            key = f"{str(prot.resnames[i]).strip()}{prot.resseqs[i]}"
            by.setdefault(key, {"rn": str(prot.resnames[i]).strip(),
                                "atoms": [], "keys": []})
        # rebuild with atom-level keys for environment exclusion
        keys = [f"{str(prot.resnames[i]).strip()}{prot.resseqs[i]}" for i in range(len(prot))]
        for i in range(len(prot)):
            by[keys[i]]["atoms"].append((str(prot.resnames[i]).strip(), keys[i],
                                         str(prot.names[i]).strip(), prot.xyz[i],
                                         str(prot.elements[i]).strip().upper()))
            by[keys[i]]["keys"].append(keys[i])
        metal = None
        for i in range(len(prot)):
            if str(prot.resnames[i]).strip().upper().startswith("ZN"):
                metal = prot.xyz[i]; break
        dmin = {}
        for key, v in by.items():
            if metal is not None:
                dmin[key] = min(float(np.linalg.norm(x[3] - metal)) for x in v["atoms"])
        REC[name] = {"prot": prot, "by": by, "keys": keys, "metal": metal, "dmin": dmin,
                     "xyz": np.asarray(prot.xyz, float),
                     "el": [str(e).strip().upper() for e in prot.elements]}
        print(f"受体 {name}: {len(prot)} 原子, 金属={'有' if metal is not None else '无'}, "
              f"口袋残基 {sum(1 for d in dmin.values() if SHELL[0] < d <= SHELL[1])}")

    results = []
    print("\n" + "=" * 96)
    print("配体诱导的 rotamer 熵损失 dS（越小 = 越契合）")
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
        row = {"molecule": mid, "target": sub, "dS": {}}
        print(f"\n{mid}  (已知目标 = {sub})")
        print(f"   {'亚型':<8}{'S_free':>9}{'S_bound':>9}{'dS':>9}{'steric':>9}{'total':>11}")
        for name in PDB:
            R = REC[name]
            if R["metal"] is None:
                continue
            try:
                res = dock_ligand(R["prot"], molecule_from_rdkit(m, f"{mid}_{name}"),
                                  metal="zn", n_spin=6, max_poses=8)
            except Exception as exc:
                print(f"   {name:<8} 对接异常 {type(exc).__name__}")
                continue
            if not res.poses:
                continue
            best = min(res.poses, key=lambda p: p.score.total)
            lig_xyz = np.asarray(best.ligand.xyz, float)
            lig_el = [str(e).strip().upper() for e in best.ligand.elements]
            S_free, _, nres = pocket_entropy(R["xyz"], R["el"], R["by"], R["dmin"],
                                            R["metal"], np.zeros((0, 3)), [], RT, R["keys"])
            S_bound, _, _ = pocket_entropy(R["xyz"], R["el"], R["by"], R["dmin"],
                                           R["metal"], lig_xyz, lig_el, RT, R["keys"])
            dS = S_bound - S_free
            t = best.score.terms
            row["dS"][name] = {"S_free": round(S_free, 3), "S_bound": round(S_bound, 3),
                               "dS": round(dS, 3), "steric": round(float(t.get("steric", np.nan)), 2),
                               "total": float(t.get("total", np.nan)), "n_res": nres}
            mark = "  ← 目标" if name == sub else ""
            print(f"   {name:<8}{S_free:>9.3f}{S_bound:>9.3f}{dS:>+9.3f}"
                  f"{t.get('steric', float('nan')):>9.2f}{t.get('total', float('nan')):>11.3e}{mark}")
        # does the target subtype have the smallest dS?
        if row["dS"]:
            best_sub = min(row["dS"], key=lambda k: row["dS"][k]["dS"])
            row["target_is_min_dS"] = (best_sub == sub)
            row["rank_of_target"] = sorted(row["dS"], key=lambda k: row["dS"][k]["dS"]).index(sub) + 1
            print(f"   dS 最小的亚型 = {best_sub}   目标排名 = {row['rank_of_target']}/6"
                  f"   {'✓ 正确' if row['target_is_min_dS'] else '✗'}")
        results.append(row)

    json.dump(results, open(ROOT / a.out, "w"), indent=2)
    ok = sum(1 for r in results if r.get("target_is_min_dS"))
    print(f"\n目标亚型 dS 最小: {ok}/{len(results)}")
    print(f"写出 {a.out}")


if __name__ == "__main__":
    main()
