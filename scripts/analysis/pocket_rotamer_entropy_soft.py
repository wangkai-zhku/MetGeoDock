"""Structure-restricted rotamer entropy, continuous (Boltzmann) version.

The hard-clash version is unstable: a rotamer either clashes or it does not, so the
entropy jumps discontinuously and the HDAC1/HDAC2 difference flips sign with the
threshold (2.2 A -> -2.76, 2.5 A -> +0.40, 2.8 A -> -0.69). That is an artefact of
the threshold, not biology.

Fix: replace the binary test by a Boltzmann weight over steric energy, which is a
continuous function of the atomic environment:

    E_i    = sum_j 4 eps_ij [ (sigma_ij/d_ij)^12 - (sigma_ij/d_ij)^6 ]   (truncated, shifted)
    w_i    = p_i * exp(-E_i / RT)          p_i = Dunbrack chi1 probability
    w~     = w / sum w                     (restricted partition function)
    S      = -sum w~ ln w~

Now a rotamer that sits slightly too close still contributes a little, and nothing
snaps. If the ordering survives as RT and the truncation change, the signal is real.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
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
# vdW sigma (A) and epsilon (kcal/mol), united-atom-ish
VDW = {"C": (1.70, 0.1094), "N": (1.55, 0.0945), "O": (1.52, 0.0945),
       "S": (1.80, 0.1460), "H": (1.20, 0.0157), "ZN": (1.39, 0.0500),
       "F": (1.47, 0.0610), "CL": (1.75, 0.1500), "BR": (1.85, 0.1700)}
METALS = {"ZN"}


def read_structure(pid):
    atoms, metal = [], []
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
            atoms.append((rn, ln[21] + ln[22:26].strip(), an, xyz, el))
        elif rn in METALS:
            metal.append(xyz)
    return atoms, metal[0]


def chi1_angle(n, ca, cb, cg):
    b1, b2, b3 = ca - n, cb - ca, cg - cb
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    m = np.cross(n1, b2 / np.linalg.norm(b2))
    return np.degrees(np.arctan2(np.dot(m, n2), np.dot(n1, n2)))


def rotate_about(pts, origin, axis, deg):
    a = axis / np.linalg.norm(axis)
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    v = pts - origin
    return (v * c + np.cross(a, v) * s + np.outer(v @ a, a) * (1 - c)) + origin


def lj_energy(xyz, elements, env_xyz, env_el, cutoff=8.0):
    """truncated, shifted Lennard-Jones; returns total steric energy in kcal/mol"""
    if len(env_xyz) == 0:
        return 0.0
    env_el = np.asarray(env_el)
    d = np.sqrt(((xyz[:, None, :] - env_xyz[None, :, :]) ** 2).sum(-1))
    mask = d < cutoff
    if not mask.any():
        return 0.0
    E = 0.0
    for i in range(len(xyz)):
        sel = mask[i]
        if not sel.any():
            continue
        s_i, e_i = VDW.get(elements[i], VDW["C"])
        dij = d[i][sel]
        ei = env_el[sel]
        s_j = np.array([VDW.get(e, VDW["C"])[0] for e in ei])
        e_j = np.array([VDW.get(e, VDW["C"])[1] for e in ei])
        sig = (s_i + s_j) / 2.0
        eps = np.sqrt(e_i * e_j)
        r6 = (sig / dij) ** 6
        E += float(np.sum(4 * eps * (r6 ** 2 - r6)))
    return E


def residue_entropy(resname, resatoms, env_xyz, env_el, metal_xyz, rt):
    """continuous Boltzmann-weighted chi1 entropy"""
    if resname in ("ALA", "GLY", "PRO"):
        return None
    sc = SIDECHAIN.get(resname, [])
    if not sc:
        return None
    names = {a[2]: a[3] for a in resatoms if len(a) > 3}
    el_of = {a[2]: (a[4] if len(a) > 4 else "C") for a in resatoms if len(a) > 3}
    if not all(k in names for k in ["N", "CA", "CB", "CG"]):
        return None
    if not all(k in names for k in sc):
        return None
    origin, axis = names["CA"], names["CB"] - names["CA"]
    moving = np.array([names[k] for k in sc])
    m_el = [el_of.get(k, "C") for k in sc]
    cur = chi1_angle(names["N"], names["CA"], names["CB"], names["CG"])
    p = np.array(CHI1P.get(resname, (1 / 3, 1 / 3, 1 / 3)), float)
    p /= p.sum()

    energies, rots = [], []
    for target in CHI1_STATES:
        rot = rotate_about(moving, origin, axis, target - cur)
        E = lj_energy(rot, m_el, env_xyz, env_el)
        energies.append(E)
        rots.append(rot)
    energies = np.array(energies)
    Emin = energies.min()
    w = p * np.exp(-(energies - Emin) / rt)
    w /= w.sum()
    S = float(-(w * np.log(np.maximum(w, 1e-12))).sum())
    return {"S": S, "energies": [round(float(e), 2) for e in energies],
            "weights": [round(float(x), 3) for x in w], "rt": rt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", nargs=2, type=float, default=[8.0, 14.0])
    ap.add_argument("--rt", type=float, default=0.593, help="kcal/mol (298 K)")
    ap.add_argument("--cutoff", type=float, default=8.0)
    a = ap.parse_args()

    print(f"连续（Boltzmann）rotamer 熵  壳层 {a.shell[0]}-{a.shell[1]} A  RT={a.rt}  截断={a.cutoff} A\n")
    print(f"{'亚型':<8}{'残基':>6}{'总熵':>8}{'每残基':>8}{'高熵残基%':>10}")
    out, detail = {}, {}
    for name, pid in PDB.items():
        atoms, metal = read_structure(pid)
        by_res = {}
        for rn, key, an, xyz, el in atoms:
            by_res.setdefault(key, {"rn": rn, "atoms": []})["atoms"].append((rn, key, an, xyz, el))
        dmin = {k: min(np.linalg.norm(x[3] - metal) for x in v["atoms"]) for k, v in by_res.items()}
        pocket = [k for k, d in dmin.items() if a.shell[0] < d <= a.shell[1]]
        all_xyz = np.array([x[3] for x in atoms])
        all_el = [x[4] for x in atoms]
        rows = {}
        for key in pocket:
            v = by_res[key]
            keep = np.array([x[1] != key for x in atoms])
            r = residue_entropy(v["rn"], v["atoms"], all_xyz[keep], [e for e, k in zip(all_el, keep) if k],
                                metal, a.rt)
            if r is not None:
                rows[key] = {"rn": v["rn"], **r}
        if not rows:
            continue
        S = np.array([r["S"] for r in rows.values()])
        out[name] = {"n": len(rows), "total_S": round(float(S.sum()), 3),
                     "mean_S": round(float(S.mean()), 4),
                     "frac_high": round(float((S > 0.8).mean()), 3)}
        detail[name] = rows
        print(f"{name:<8}{len(rows):>6}{S.sum():>8.2f}{S.mean():>8.3f}{out[name]['frac_high']:>9.0%}")

    print("\n配对差值（连续版）")
    for i, k1 in enumerate(PDB):
        for k2 in list(PDB)[i + 1:]:
            d = out[k1]["total_S"] - out[k2]["total_S"]
            print(f"  {k1:<8} vs {k2:<8} {out[k1]['total_S']:6.2f} vs {out[k2]['total_S']:6.2f}  Δ={d:+6.2f}")

    json.dump({"summary": out,
               "per_residue": {k: {kk: {"rn": vv["rn"], "S": round(vv["S"], 3)}
                                   for kk, vv in v.items()} for k, v in detail.items()}},
              open(ROOT / "results/hdac_rotamer_entropy_soft.json", "w"), indent=2)
    print("\n写出 results/hdac_rotamer_entropy_soft.json")


if __name__ == "__main__":
    main()
