"""Structure-restricted rotamer entropy of the binding pocket.

Idea
----
A generic rotamer entropy depends only on the amino-acid type, so two isoforms
with the same pocket sequence get the same number. What matters physically is the
entropy the side chain actually has *in this pocket*: a residue packed against a
neighbouring backbone may only tolerate one of its three chi1 rotamers, while the
same residue type in an open environment keeps all three.

Procedure, per pocket residue:
  1. take the crystallographic chi1 (N-CA-CB-CG) and the side-chain atoms beyond CB,
  2. rebuild the three canonical chi1 states (g+ 60, t 180, g- -60) by rigid
     rotation about the CA->CB axis (all other chi held at their crystal values),
  3. test each state for steric clash against everything else in the structure
     (own residue and the metal excluded),
  4. keep only clash-free states, renormalise their Dunbrack-style probabilities,
     and take  S = -sum p ln p.

A residue with no clash-free state contributes S = 0 (fully locked). So two
structurally different pockets made of identical residues can and do differ.

Usage:
    python scripts/analysis/pocket_rotamer_entropy.py
    python scripts/analysis/pocket_rotamer_entropy.py --shell 8 14 --probe 2.5
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC8": "1T64"}

# side-chain atoms beyond CB, PDB standard names
SIDECHAIN = {
    "ARG": ["CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["CG", "OD1", "ND2"], "ASP": ["CG", "OD1", "OD2"], "CYS": ["SG"],
    "GLN": ["CG", "CD", "OE1", "NE2"], "GLU": ["CG", "CD", "OE1", "OE2"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"], "ILE": ["CG1", "CG2", "CD1"],
    "LEU": ["CG", "CD1", "CD2"], "LYS": ["CG", "CD", "CE", "NZ"],
    "MET": ["CG", "SD", "CE"], "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": [], "SER": ["OG"], "THR": ["OG1", "CG2"],
    "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["CG1", "CG2"], "ALA": [], "GLY": [],
}
# chi1 rotamer probabilities, g+ / t / g-  (Dunbrack-style summary)
CHI1P = {
    "ARG": (0.34, 0.34, 0.32), "ASN": (0.34, 0.36, 0.30), "ASP": (0.36, 0.34, 0.30),
    "CYS": (0.28, 0.36, 0.36), "GLN": (0.35, 0.34, 0.31), "GLU": (0.38, 0.32, 0.30),
    "HIS": (0.31, 0.39, 0.30), "ILE": (0.33, 0.42, 0.25), "LEU": (0.36, 0.40, 0.24),
    "LYS": (0.35, 0.34, 0.31), "MET": (0.30, 0.36, 0.34),
    "PHE": (0.31, 0.40, 0.29), "SER": (0.44, 0.31, 0.25), "THR": (0.51, 0.31, 0.18),
    "TRP": (0.32, 0.38, 0.30), "TYR": (0.30, 0.40, 0.30), "VAL": (0.06, 0.70, 0.24),
}
CHI1_STATES = (60.0, 180.0, -60.0)      # g+, t, g-  for L-amino acids
METALS = {"ZN"}


def read_structure(pid):
    atoms = []          # (resname, reskey, atomname, xyz)
    metal = []
    for ln in (ROOT / f"processed/hdac_receptors/{pid}_h.pdb").read_text(errors="ignore").splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        rn = ln[17:20].strip().upper()
        an = ln[12:16].strip()
        el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        if ln.startswith("ATOM") and el != "H":
            atoms.append((rn, ln[21] + ln[22:26].strip(), an, xyz))
        elif rn in METALS:
            metal.append(xyz)
    return atoms, metal[0]


def chi1_angle(n, ca, cb, cg):
    b1, b2, b3 = ca - n, cb - ca, cg - cb
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    m = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2); y = np.dot(m, n2)
    return np.degrees(np.arctan2(y, x))


def rotate_about(pts, origin, axis, deg):
    a = axis / np.linalg.norm(axis)
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    v = pts - origin
    return (v * c + np.cross(a, v) * s + np.outer(v @ a, a) * (1 - c)) + origin


def residue_entropy(resname, resatoms, others, probe, metal_xyz):
    """S = -sum p ln p over sterically allowed chi1 states."""
    if resname in ("ALA", "GLY", "PRO"):
        return None                                   # no chi1: not informative
    sc = SIDECHAIN.get(resname, [])
    if not sc:
        return None
    names = {a[2]: a[3] for a in resatoms}
    need = ["N", "CA", "CB", "CG"]
    if not all(k in names for k in need):
        return None
    if not all(k in names for k in sc):
        return None
    origin = names["CA"]
    axis = names["CB"] - names["CA"]
    moving = np.array([names[k] for k in sc])
    cur = chi1_angle(names["N"], names["CA"], names["CB"], names["CG"])

    probs = np.array(CHI1P.get(resname, (1 / 3, 1 / 3, 1 / 3)), float)
    probs /= probs.sum()
    own = {a[2] for a in resatoms}
    # environment: every heavy atom not in this residue
    env = np.array([a[3] for a in others])
    if len(env) == 0:
        return None
    d_metal = np.linalg.norm(moving - metal_xyz, axis=1).min()

    allowed, detail = [], []
    for i, target in enumerate(CHI1_STATES):
        rot = rotate_about(moving, origin, axis, target - cur)
        d = np.sqrt(((rot[:, None, :] - env[None, :, :]) ** 2).sum(-1))
        clash = float(d.min())
        ok = clash > probe
        allowed.append(ok)
        detail.append(round(clash, 2))
    allowed = np.array(allowed)
    if not allowed.any():
        return {"S": 0.0, "n_allowed": 0, "min_dists": detail,
                "raw_chi1": round(float(cur), 1), "metal_clash": round(float(d_metal), 2)}
    p = probs[allowed] / probs[allowed].sum()
    S = float(-(p * np.log(p)).sum())
    return {"S": S, "n_allowed": int(allowed.sum()), "min_dists": detail,
            "raw_chi1": round(float(cur), 1), "metal_clash": round(float(d_metal), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", nargs=2, type=float, default=[8.0, 14.0])
    ap.add_argument("--probe", type=float, default=2.5, help="clash threshold (A)")
    ap.add_argument("--out", default="results/hdac_rotamer_entropy.json")
    a = ap.parse_args()

    out = {}
    print(f"口袋壳层 {a.shell[0]}-{a.shell[1]} A，碰撞阈值 {a.probe} A\n")
    print(f"{'亚型':<8}{'可用残基':>9}{'结构受限总熵':>13}{'每残基':>8}{'锁定残基%':>10}{'单rotamer%':>11}")
    for name, pid in PDB.items():
        atoms, metal = read_structure(pid)
        by_res = {}
        for rn, key, an, xyz in atoms:
            by_res.setdefault(key, {"rn": rn, "atoms": []})["atoms"].append((rn, key, an, xyz))
        # residue minimum distance to metal
        info = {}
        for key, v in by_res.items():
            d = min(np.linalg.norm(xyz - metal) for _, _, _, xyz in v["atoms"])
            info[key] = d
        pocket = [k for k, d in info.items() if a.shell[0] < d <= a.shell[1]]
        all_atoms = [(rn, key, an, xyz) for rn, key, an, xyz in atoms]

        rows = {}
        for key in pocket:
            v = by_res[key]
            others = [x for x in all_atoms if x[1] != key]
            r = residue_entropy(v["rn"], v["atoms"], others, a.probe, metal)
            if r is not None:
                rows[key] = {"rn": v["rn"], **r}
        if not rows:
            continue
        S = np.array([r["S"] for r in rows.values()])
        locked = float(np.mean([r["n_allowed"] == 0 for r in rows.values()]))
        single = float(np.mean([r["n_allowed"] <= 1 for r in rows.values()]))
        out[name] = {"n_residues": len(rows), "total_S": round(float(S.sum()), 3),
                     "mean_S": round(float(S.mean()), 4),
                     "frac_locked": round(locked, 3), "frac_single": round(single, 3),
                     "per_residue": {k: {"rn": v["rn"], "S": round(v["S"], 3),
                                         "n": v["n_allowed"], "d": v["min_dists"]}
                                     for k, v in rows.items()}}
        print(f"{name:<8}{len(rows):>9}{S.sum():>13.2f}{S.mean():>8.3f}"
              f"{locked:>9.0%}{single:>10.0%}")

    json.dump(out, open(ROOT / a.out, "w"), indent=2)
    print(f"\n写出 {a.out}")
    print("\n配对比较（结构受限熵；若序列相同而结构不同，这里应出现差异）")
    keys = [k for k in out]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a1, a2 = out[keys[i]]["total_S"], out[keys[j]]["total_S"]
            print(f"  {keys[i]:<8} vs {keys[j]:<8} {a1:7.2f} vs {a2:7.2f}  "
                  f"Δ={a1 - a2:+7.2f}")


if __name__ == "__main__":
    main()
