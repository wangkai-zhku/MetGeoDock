"""Foot-pocket geometry per HDAC subtype (validation against published numbers).

Literature anchors this script is designed to test:
  * class I HDAC active site = ~11 A channel + adjacent ~14 A internal (foot) cavity
  * HDAC1 cavity pocket area 327.7 A^2 vs HDAC2 409.2 A^2, entrance 12.9 vs 23.3 A^2
  * HDAC3's foot pocket is narrowed by Tyr107 (replacing Ser113/Ser114 of HDAC1/2),
    which pushes Leu133 inward and blocks bulky groups

So the expected ordering is HDAC2 > HDAC1 > HDAC3 for foot-pocket volume. If the
descriptor cannot reproduce that, it is not measuring the right thing.

Channel direction is defined as (metal - protein centroid), i.e. outward from the
protein body. Foot pocket = open cavity in the 8-14 A shell that lies OUTSIDE the
channel cone; channel = the same shell inside the cone.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC8": "1T64"}
MET = {"ZN"}
PROBE, STEP = 1.4, 0.8
R_IN, R_OUT = 7.0, 14.0        # the internal cavity sits in this shell
CONE_DEG = 55.0


def load(pid):
    prot, met = [], []
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
            prot.append(xyz)
        elif rn in MET:
            met.append(xyz)
    return np.array(prot), met[0]


def analyse(protein, metal):
    centroid = protein.mean(0)
    axis = metal - centroid
    axis /= np.linalg.norm(axis)
    n = int(2 * R_OUT / STEP) + 1
    ax = np.linspace(-R_OUT, R_OUT, n)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    rel = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    r = np.linalg.norm(rel, axis=1)
    shell = (r >= R_IN) & (r <= R_OUT)
    rel = rel[shell]
    r = r[shell]
    cosang = rel @ axis / np.maximum(r, 1e-9)
    cone = cosang > np.cos(np.deg2rad(CONE_DEG))
    pts = rel + metal
    dmin = np.empty(len(pts))
    for i in range(0, len(pts), 4000):
        blk = pts[i:i + 4000]
        dmin[i:i + 4000] = np.sqrt(((blk[:, None, :] - protein[None, :, :]) ** 2).sum(-1)).min(1)
    open_vox = (dmin > PROBE) * STEP ** 3
    return {
        "axis": axis,
        "foot_volume": float(open_vox[~cone].sum()),
        "channel_volume": float(open_vox[cone].sum()),
        "total_volume": float(open_vox.sum()),
        "foot_voxels": int((open_vox[~cone] > 0).sum()),
        "channel_voxels": int((open_vox[cone] > 0).sum()),
        "n_open": int((open_vox > 0).sum()),
    }


for name, pid in PDB.items():
    prot, m = load(pid)
    r = analyse(prot, m)
    print(f"{name:<8}{pid:<7} foot={r['foot_volume']:>8.0f} A^3  channel={r['channel_volume']:>8.0f} A^3  "
          f"foot/channel={r['foot_volume']/max(r['channel_volume'],1):.2f}")

print("\n文献预期顺序（class I）: HDAC2 > HDAC1 > HDAC3 的 foot pocket")
res = {n: analyse(*load(p)) for n, p in PDB.items()}
for a, b in (("HDAC1", "HDAC2"), ("HDAC1", "HDAC3"), ("HDAC2", "HDAC3")):
    va, vb = res[a]["foot_volume"], res[b]["foot_volume"]
    print(f"  {a} vs {b}: {va:.0f} vs {vb:.0f} -> {a if va > vb else b} 更大 "
          f"(比值 {max(va,vb)/max(min(va,vb),1):.2f})")

json.dump({k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
               for kk, vv in v.items()} for k, v in res.items()},
          open(ROOT / "results/hdac_foot_pocket.json", "w"), indent=2)
print(f"\n写出 results/hdac_foot_pocket.json")
