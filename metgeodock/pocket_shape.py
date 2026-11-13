"""Pocket classification from geometry alone, used to decide the sign of the conformational
entropy term.

The idea: whether ligand flexibility helps or hurts should follow from the pocket's shape, not
from its activity data. A narrow elongated channel can only be filled by a molecule that can
bend along it, so flexibility pays. A shallow open bowl is filled by a compact rigid molecule,
so flexibility costs entropy for nothing.

Measured quantities (all from the receptor structure, no ligands involved):
  cavity_volume   A^3 of free space within 12 A of the metal (grid, 0.8 A)
  elongation      major/minor axis ratio of that cavity (PCA) - channels are elongated
  mouth_frac      fraction of cavity surface that is open to solvent (a wide mouth = shallow)
  max_reach       longest straight line from the metal through free space (depth proxy)

Classification:
  channel  elongation >= 2.0 and mouth_frac <= 0.45   -> flexibility helps
  bowl     max_reach < 11 A or mouth_frac > 0.60       -> flexibility hurts
  mixed    otherwise                                  -> term is off
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

METALS = {"ZN", "FE", "FE2", "FE3", "MN", "MN2", "MG", "CA", "CO", "CU", "NI", "CD"}
GRID = 0.8
RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "F": 1.47, "CL": 1.75,
         "BR": 1.85, "I": 1.98, "P": 1.80}


def load_receptor(pdb: str | Path):
    met = None
    prot = []
    for ln in Path(pdb).read_text(errors="ignore").splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        rn = ln[17:20].strip().upper()
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except Exception:
            continue
        if rn in METALS:
            met = xyz if met is None else met
        elif ln.startswith("ATOM") and el != "H":
            prot.append((el, xyz))
    return met, prot


def cavity(met, prot, radius=12.0, grid=GRID, probe=1.4):
    """Free-space grid around the metal, with basic shape descriptors."""
    P = np.array([p[1] for p in prot])
    E = [p[0] for p in prot]
    near = np.linalg.norm(P - met, axis=1) <= radius + max(RADII.values())
    P, E = P[near], [e for e, k in zip(E, near) if k]
    if len(P) == 0:
        return None
    rad = np.array([RADII.get(e, 1.7) for e in E]) + probe

    n = int(2 * radius / grid) + 1
    ax = np.linspace(-radius, radius, n)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]) + met
    keep = np.linalg.norm(pts - met, axis=1) <= radius
    pts = pts[keep]

    free = np.ones(len(pts), bool)
    for i in range(0, len(P), 500):          # chunked distance test
        d = np.linalg.norm(pts[:, None, :] - P[None, i:i + 500, :], axis=2)
        free &= ~(d < rad[None, i:i + 500]).any(axis=1)
    cav = pts[free]
    if len(cav) < 20:
        return None

    vol = len(cav) * grid ** 3
    c = cav - cav.mean(axis=0)
    cov = np.cov(c.T)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    elong = float(np.sqrt(ev[0] / max(ev[-1], 1e-6)))

    # mouth: free grid points on the sphere boundary are reachable from solvent
    r = np.linalg.norm(cav - met, axis=1)
    shell = cav[r >= radius * 0.92]
    mouth = float(len(shell) / max(1, len(cav[r >= radius * 0.5]))) if len(cav) else 0.0

    # depth: longest ray that stays in free space
    i = np.arange(600) + 0.5
    phi = np.arccos(1 - 2 * i / 600)
    th = np.pi * (1 + 5 ** 0.5) * i
    D = np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])
    steps = np.arange(1.0, radius, grid)
    pts_ray = met[None, None, :] + D[:, None, :] * steps[None, :, None]
    flat = pts_ray.reshape(-1, 3)
    blocked = np.zeros(len(flat), bool)
    for k in range(0, len(P), 500):
        dd = np.linalg.norm(flat[:, None, :] - P[None, k:k + 500, :], axis=2)
        blocked |= (dd < rad[None, k:k + 500]).any(axis=1)
    blocked = blocked.reshape(len(D), len(steps))
    reach = np.where(blocked, 0.0, steps[None, :]).max(axis=1)
    max_reach = float(reach.max())

    return dict(cavity_volume=float(vol), elongation=elong, mouth_frac=float(mouth),
                max_reach=max_reach, n_free=int(len(cav)))


def classify(desc):
    if desc is None:
        return "unknown", 0
    if desc["elongation"] >= 2.0 and desc["mouth_frac"] <= 0.45:
        return "channel", -1          # flexibility helps -> entropy term reversed
    if desc["max_reach"] < 11.0 or desc["mouth_frac"] > 0.60:
        return "bowl", +1             # flexibility hurts -> entropy term as physics says
    return "mixed", 0


def pocket_profile(pdb):
    met, prot = load_receptor(pdb)
    if met is None or not prot:
        return None
    d = cavity(met, prot)
    kind, sign = classify(d)
    out = {"pocket_type": kind, "g_conf_sign": sign}
    if d:
        out.update(d)
    return out


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        r = pocket_profile(p)
        print(Path(p).name, r)
