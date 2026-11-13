"""Rotamer entropy with the metal ion included: steric + electrostatic + coordination lock.

Previous version ignored the metal entirely (it was read into a separate variable
and never entered the environment atom list). But in a metalloprotein the metal is
not a spectator:

  1. steric   - the ion occupies space; a rotamer that would pass through it is
                forbidden. Handled by adding ZN to the LJ environment.
  2. charge   - Zn2+ at ~2 A from a carboxylate oxygen is a huge electrostatic term.
                Handled by a screened Coulomb term with per-element partial charges.
  3. lock     - any side-chain atom within coordination range of the metal cannot
                rotate at all; those residues contribute S ~ 0 by construction
                instead of by accident of the clash test.

Physics
-------
    E_i = E_LJ(i) + E_coul(i)
    E_coul = 332.0 * q_metal * q_atom / (eps * r)      [kcal/mol, r in A]
    w_i = p_i exp(-(E_i - min E)/RT);  S = -sum w~ ln w~

Usage:
    python scripts/analysis/pocket_rotamer_metal.py --eps 4
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
CORPUS = Path("/home/wangkai/cursor-task/metalloprotein/ORI_PDB_LIGAND")
PDB_DEFAULT = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
               "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC8": "1T64"}

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
CHI1P = {
    "ARG": (0.34, 0.34, 0.32), "ASN": (0.34, 0.36, 0.30), "ASP": (0.36, 0.34, 0.30),
    "CYS": (0.28, 0.36, 0.36), "GLN": (0.35, 0.34, 0.31), "GLU": (0.38, 0.32, 0.30),
    "HIS": (0.31, 0.39, 0.30), "ILE": (0.33, 0.42, 0.25), "LEU": (0.36, 0.40, 0.24),
    "LYS": (0.35, 0.34, 0.31), "MET": (0.30, 0.36, 0.34),
    "PHE": (0.31, 0.40, 0.29), "SER": (0.44, 0.31, 0.25), "THR": (0.51, 0.31, 0.18),
    "TRP": (0.32, 0.38, 0.30), "TYR": (0.30, 0.40, 0.30), "VAL": (0.06, 0.70, 0.24),
}
CHI1_STATES = (60.0, 180.0, -60.0)
VDW = {"C": (1.70, 0.1094), "N": (1.55, 0.0945), "O": (1.52, 0.0945),
       "S": (1.80, 0.1460), "H": (1.20, 0.0157), "ZN": (1.39, 0.0500)}
# crude partial charges for the electrostatic term
Q = {"O": -0.50, "N": -0.40, "S": -0.20, "C": 0.10, "H": 0.10, "ZN": 2.0}
COORD_CUT = 2.8       # side-chain atom within this of the metal => coordinated


def read_any(path):
    atoms, metal = [], []
    for ln in Path(path).read_text(errors="ignore").splitlines():
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
            atoms.append((rn, ln[21] + ln[22:26].strip(), an, xyz, el))
        elif rn == "ZN":
            metal.append(xyz)
    return atoms, metal


def chi1(n, ca, cb, cg):
    b1, b2, b3 = ca - n, cb - ca, cg - cb
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    m = np.cross(n1, b2 / np.linalg.norm(b2))
    return np.degrees(np.arctan2(np.dot(m, n2), np.dot(n1, n2)))


def rot(pts, origin, axis, deg):
    a = axis / np.linalg.norm(axis)
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    v = pts - origin
    return (v * c + np.cross(a, v) * s + np.outer(v @ a, a) * (1 - c)) + origin


def energy(xyz, els, env_xyz, env_el, metal, eps, cut_lj=8.0, cut_el=20.0):
    """steric (LJ, metal included) + screened Coulomb to metal + all polar env atoms"""
    E = 0.0
    if len(env_xyz):
        env_el = np.asarray(env_el)
        d = np.sqrt(((xyz[:, None, :] - env_xyz[None, :, :]) ** 2).sum(-1))
        for i in range(len(xyz)):
            sel = d[i] < cut_lj
            if sel.any():
                si, ei = VDW.get(els[i], VDW["C"])
                sj = np.array([VDW.get(e, VDW["C"])[0] for e in env_el[sel]])
                ej = np.array([VDW.get(e, VDW["C"])[1] for e in env_el[sel]])
                sig = (si + sj) / 2.0
                epsij = np.sqrt(ei * ej)
                r6 = (sig / d[i][sel]) ** 6
                E += float(np.sum(4 * epsij * (r6 ** 2 - r6)))
            # electrostatics: only polar/charged atoms of the environment
            selq = (d[i] < cut_el) & np.isin(env_el, ["O", "N", "S"])
            if selq.any():
                qi = Q.get(els[i], 0.0)
                if qi:
                    qj = np.array([Q.get(e, 0.0) for e in env_el[selq]])
                    E += float(np.sum(332.0 * qi * qj / (eps * d[i][selq])))
    if metal is not None:
        dm = np.linalg.norm(xyz - metal, axis=1)
        # metal steric
        for i, r in enumerate(dm):
            if r < cut_lj:
                si, ei = VDW.get(els[i], VDW["C"])
                sm, em = VDW["ZN"]
                sig = (si + sm) / 2.0
                epsij = np.sqrt(ei * em)
                r6 = (sig / max(r, 1e-3)) ** 6
                E += float(4 * epsij * (r6 ** 2 - r6))
        # metal charge
        qi = np.array([Q.get(e, 0.0) for e in els])
        sel = (dm < cut_el) & (qi != 0.0)
        if sel.any():
            E += float(np.sum(332.0 * Q["ZN"] * qi[sel] / (eps * dm[sel])))
    return E


def residue_entropy(resname, resatoms, env_xyz, env_el, metal, rt, eps):
    if resname in ("ALA", "GLY", "PRO"):
        return None
    sc = SIDECHAIN.get(resname, [])
    if not sc:
        return None
    names = {a[2]: a[3] for a in resatoms}
    els = {a[2]: a[4] for a in resatoms}
    if not all(k in names for k in ["N", "CA", "CB", "CG"]) or not all(k in names for k in sc):
        return None
    origin, axis = names["CA"], names["CB"] - names["CA"]
    moving = np.array([names[k] for k in sc])
    m_el = [els.get(k, "C") for k in sc]
    cur = chi1(names["N"], names["CA"], names["CB"], names["CG"])

    # coordinated? then the side chain is locked by the metal
    if metal is not None and np.linalg.norm(moving - metal, axis=1).min() < COORD_CUT:
        return {"S": 0.0, "coordinated": True, "energies": [], "weights": []}

    p = np.array(CHI1P.get(resname, (1 / 3, 1 / 3, 1 / 3)), float)
    p /= p.sum()
    Es = []
    for tgt in CHI1_STATES:
        r = rot(moving, origin, axis, tgt - cur)
        Es.append(energy(r, m_el, env_xyz, env_el, metal, eps))
    Es = np.array(Es)
    w = p * np.exp(-(Es - Es.min()) / rt)
    w /= w.sum()
    S = float(-(w * np.log(np.maximum(w, 1e-12))).sum())
    return {"S": S, "coordinated": False, "energies": [round(float(x), 2) for x in Es],
            "weights": [round(float(x), 3) for x in w]}


def find(pid):
    hits = sorted(CORPUS.glob(f"*/zn_{pid}_pro_*.pdb")) or sorted(CORPUS.glob(f"*/*_{pid}_pro_*.pdb"))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=4.0, help="dielectric for Coulomb")
    ap.add_argument("--rt", type=float, default=0.593)
    ap.add_argument("--shell", nargs=2, type=float, default=[8.0, 14.0])
    ap.add_argument("--use-corpus", action="store_true", help="read from the 16k corpus")
    a = ap.parse_args()

    print(f"含金属（位阻+静电+配位锁定）的 rotamer 熵   eps={a.eps} RT={a.rt}  "
          f"壳层={a.shell[0]}-{a.shell[1]}\n")
    print(f"{'亚型':<8}{'PDB':<7}{'残基':>6}{'配位锁定':>9}{'总熵':>8}{'每残基':>8}")
    out = {}
    for name, pid in PDB_DEFAULT.items():
        path = find(pid) if a.use_corpus else (ROOT / f"processed/hdac_receptors/{pid}.pdb")
        if path is None or not Path(path).exists():
            print(f"{name:<8}{pid:<7}  文件缺失")
            continue
        atoms, metals = read_any(path)
        if not metals:
            print(f"{name:<8}{pid:<7}  无金属")
            continue
        metal = metals[0]
        by = {}
        for rn, key, an, xyz, el in atoms:
            by.setdefault(key, {"rn": rn, "atoms": []})["atoms"].append((rn, key, an, xyz, el))
        dmin = {k: min(np.linalg.norm(x[3] - metal) for x in v["atoms"]) for k, v in by.items()}
        pocket = [k for k, d in dmin.items() if a.shell[0] < d <= a.shell[1]]
        all_xyz = np.array([x[3] for x in atoms])
        all_el = [x[4] for x in atoms]
        vals, ncoord = [], 0
        for key in pocket:
            v = by[key]
            keep = np.array([x[1] != key for x in atoms])
            r = residue_entropy(v["rn"], v["atoms"], all_xyz[keep],
                                [e for e, kk in zip(all_el, keep) if kk], metal, a.rt, a.eps)
            if r is not None:
                vals.append(r["S"])
                ncoord += int(r["coordinated"])
        if not vals:
            continue
        out[name] = {"n": len(vals), "n_coordinated": ncoord,
                     "total": round(float(np.sum(vals)), 3),
                     "mean": round(float(np.mean(vals)), 4)}
        print(f"{name:<8}{pid:<7}{len(vals):>6}{ncoord:>9}{np.sum(vals):>8.2f}{np.mean(vals):>8.4f}")

    print("\n配对差（每残基）")
    ks = list(out)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            d = out[ks[i]]["mean"] - out[ks[j]]["mean"]
            print(f"  {ks[i]:<8} vs {ks[j]:<8} {d:+.4f}")

    json.dump(out, open(ROOT / f"results/hdac_rotamer_metal_eps{a.eps}.json", "w"), indent=2)
    print(f"\n写出 results/hdac_rotamer_metal_eps{a.eps}.json")


if __name__ == "__main__":
    main()
