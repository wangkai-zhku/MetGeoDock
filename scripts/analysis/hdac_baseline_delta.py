"""Baseline-plus-change score: does the ligand perturb the VARIABLE positions?

Idea (from the user)
--------------------
The free pocket has a baseline rotamer pattern. A ligand perturbs only the residues
it actually touches; the rest stay put. So:

    score(ligand, subtype) = sum over pocket residues
                             |dS_residue|  x  variability(position)

where
    dS_residue  = S(rotamer | ligand present) - S(rotamer | free)   [negative = locked]
    variability = positional entropy across the six subtypes (0 = identical residue
                  in all six, high = the position differs between subtypes)

Why this avoids the earlier failure: the *difference region* used before depended on
which crystal structure you used as reference (noise = 63% of signal). Here nothing
depends on a reference structure - dS comes from the pose, and variability comes from
the alignment, both stable quantities.

A selective ligand should perturb the positions that actually differ between
subtypes; a non-selective one perturbs only conserved positions and scores low.
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, sys
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

spec = importlib.util.spec_from_file_location("pm", ROOT / "scripts/analysis/pocket_rotamer_metal.py")
pm = importlib.util.module_from_spec(spec)
sys.modules["pm"] = pm
spec.loader.exec_module(pm)

PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC8": "1T64"}
SHELL = (4.0, 12.0)
RT, EPS = 0.593, 1e6        # electrostatics off
A3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
      "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
      "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


# ---------------------------------------------------------------- receptor loading
def load_receptor(pid):
    prot = read_pdb(str(ROOT / f"processed/hdac_receptors/{pid}_h.pdb"))
    by = defaultdict(lambda: {"rn": None, "atoms": []})
    keys = []
    for i in range(len(prot)):
        rn = str(prot.resnames[i]).strip().upper()
        an = str(prot.names[i]).strip().upper()
        key = str(prot.resseqs[i])
        keys.append(key)
        by[key]["rn"] = rn
        by[key]["atoms"].append((rn, key, an, np.asarray(prot.xyz[i], float),
                                 str(prot.elements[i]).strip().upper()))
    metal = None
    for i in range(len(prot)):
        if str(prot.resnames[i]).strip().upper().startswith("ZN"):
            metal = np.asarray(prot.xyz[i], float); break
    xyz = np.asarray(prot.xyz, float)
    el = [str(e).strip().upper() for e in prot.elements]
    return prot, by, keys, xyz, el, metal


def pocket_entropies(by, keys, xyz, el, metal, extra_xyz=None, extra_el=None):
    """per-residue chi1 entropy for pocket residues, with optional ligand in env"""
    d = np.linalg.norm(xyz - metal, axis=1)
    pocket = []
    seen = set()
    for i, k in enumerate(keys):
        if k in seen:
            continue
        if SHELL[0] < d[i] <= SHELL[1]:
            pocket.append(k); seen.add(k)
    extra_xyz = np.zeros((0, 3)) if extra_xyz is None else extra_xyz
    extra_el = [] if extra_el is None else extra_el
    out = {}
    for key in pocket:
        v = by[key]
        mask = np.array([k != key for k in keys], bool)
        ex = np.vstack([xyz[mask], extra_xyz]) if len(extra_xyz) else xyz[mask]
        exel = [e for e, m in zip(el, mask) if m] + list(extra_el)
        r = pm.residue_entropy(v["rn"], v["atoms"], ex, exel, metal, RT, EPS)
        if r is not None:
            out[key] = r["S"]
    return out


# ---------------------------------------------------------------- variability weights
def ln_chain(prot, i):
    """chain id for atom i (Molecule may not expose chains; fall back to 'A')"""
    try:
        return str(prot.chains[i])
    except Exception:
        return "A"


def positional_variability():
    """entropy of the residue type at each aligned position across the six subtypes"""
    aln = json.load(open(ROOT / "results/hdac_sequence_alignment.json"))
    # gather per-position residue letters: ref position -> types
    seqs = {}
    for name, pid in PDB.items():
        prot = read_pdb(str(ROOT / f"processed/hdac_receptors/{pid}.pdb"))
        seen, order = {}, []
        ch = None
        for i in range(len(prot)):
            if str(prot.resnames[i]).strip().upper().startswith("ZN"):
                ch = str(prot.chains[i]) if hasattr(prot, "chains") else None
        for i in range(len(prot)):
            rn = str(prot.resnames[i]).strip().upper()
            if rn not in A3:
                continue
            k = str(prot.resseqs[i])              # numbering only; receptors are single-chain
            if k not in seen:
                seen[k] = rn
                order.append(k)
        seqs[name] = {k: A3[v] for k, v in seen.items()}

    ref = "HDAC1"
    var = {}
    # every ref position gets the multiset of residue types across subtypes
    types = defaultdict(list)
    for name in PDB:
        if name == ref:
            mapping = {k: k for k in seqs[ref]}
        else:
            mp_raw = aln["pairs"].get(name, {}).get("mapping", {})
            mapping = {k[1:]: v[1:] for k, v in mp_raw.items()}   # strip chain letter
        for rk in seqs[ref]:
            tk = mapping.get(rk)
            if tk and tk in seqs[name]:
                types[rk].append(seqs[name][tk])
    for rk, ts in types.items():
        if len(ts) < 3:
            var[rk] = 0.0
            continue
        cnt = defaultdict(int)
        for t in ts:
            cnt[t] += 1
        n = sum(cnt.values())
        H = -sum((c / n) * np.log(c / n) for c in cnt.values())
        var[rk] = float(H / np.log(len(PDB)))     # normalise to 0..1
    return var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2, help="compounds per target subtype")
    ap.add_argument("--min-dS", type=float, default=0.002, help="perturbation threshold")
    ap.add_argument("--out", default="results/hdac_baseline_delta.json")
    a = ap.parse_args()

    meta = {r["molecule_id"]: r for r in csv.DictReader(
        open(ROOT / "processed/hdac_selectivity_set.csv"))}
    picks, seen = [], defaultdict(int)
    for mid, r in meta.items():
        t = r["target_subtype"]
        if seen[t] >= a.limit:
            continue
        picks.append(mid); seen[t] += 1
    print(f"样本 {len(picks)}: {[(m, meta[m]['target_subtype']) for m in picks]}\n")

    VAR = positional_variability()
    nz = [v for v in VAR.values() if v > 0]
    if nz:
        print(f"位置可变性: {len(nz)} 个位置 > 0，均值 {np.mean(nz):.3f}，最大 {max(nz):.3f}\n")
    else:
        print("警告: 可变性全部为 0（mapping 未匹配），退化为不加权\n")

    REC = {}
    for name, pid in PDB.items():
        prot, by, keys, xyz, el, metal = load_receptor(pid)
        free = pocket_entropies(by, keys, xyz, el, metal)
        # map receptor residue keys onto reference positions (for variability lookup)
        if name == "HDAC1":
            to_ref = {k: k for k in free}
        else:
            aln = json.load(open(ROOT / "results/hdac_sequence_alignment.json"))
            mp = aln["pairs"].get(name, {}).get("mapping", {})
            inv = {v: k for k, v in mp.items()}
            to_ref = {k: inv.get(k, k) for k in free}
        REC[name] = dict(prot=prot, by=by, keys=keys, xyz=xyz, el=el, metal=metal,
                         free=free, to_ref=to_ref)
        print(f"{name}: 口袋残基 {len(free)}, 平均基线熵 {np.mean(list(free.values())):.3f}")

    print("\n" + "=" * 100)
    print("基线 + 变化量打分：sum |dS_residue| x variability(position)")
    print("=" * 100)
    rows = []
    for mid in picks:
        sub = meta[mid]["target_subtype"]
        mol = Chem.MolFromSmiles(meta[mid]["smiles"])
        if mol is None:
            continue
        m = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(m, randomSeed=0xF00D) != 0:
            continue
        m = Chem.RemoveHs(m)
        rec = {"molecule": mid, "target": sub, "score": {}, "n_perturbed": {}}
        print(f"\n{mid} (目标 {sub})")
        print(f"   {'亚型':<8}{'变值打分':>10}{'受影响残基':>11}{'加权残基':>10}")
        for name, R in REC.items():
            try:
                res = dock_ligand(R["prot"], molecule_from_rdkit(m, f"{mid}_{name}"),
                                  metal="zn", n_spin=6, max_poses=8)
            except Exception:
                continue
            if not res.poses:
                continue
            best = min(res.poses, key=lambda p: p.score.total)
            lig_xyz = np.asarray(best.ligand.xyz, float)
            lig_el = [str(e).strip().upper() for e in best.ligand.elements]
            bound = pocket_entropies(R["by"], R["keys"], R["xyz"], R["el"], R["metal"],
                                     lig_xyz, lig_el)
            score = 0.0
            nper = 0
            nw = 0
            for k, s0 in R["free"].items():
                if k not in bound:
                    continue
                dS = bound[k] - s0
                if abs(dS) < a.min_dS:
                    continue
                nper += 1
                w = VAR.get(R["to_ref"].get(k, k), 0.0)
                if w > 0:
                    nw += 1
                score += abs(dS) * (0.2 + w)          # every perturbation counts, variable more
            rec["score"][name] = round(float(score), 3)
            rec["n_perturbed"][name] = nper
            mark = "  <- 目标" if name == sub else ""
            print(f"   {name:<8}{score:>10.3f}{nper:>11}{nw:>10}{mark}")
        if rec["score"]:
            best_sub = max(rec["score"], key=lambda k: rec["score"][k])
            rec["hit"] = (best_sub == sub)
            order = sorted(rec["score"], key=lambda k: -rec["score"][k])
            print(f"   最高 = {best_sub}  排名 {order.index(sub)+1}/6  "
                  f"{'✓' if rec['hit'] else '✗'}")
        rows.append(rec)

    ok = sum(1 for r in rows if r.get("hit"))
    print(f"\n目标亚型打分最高: {ok}/{len(rows)}")
    json.dump(rows, open(ROOT / a.out, "w"), indent=2)
    print(f"写出 {a.out}")


if __name__ == "__main__":
    main()
