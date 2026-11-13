"""Pocket shape fingerprints for the HDAC subtypes.

Residue-type counting cannot separate HDAC1/2/3 (their 12 A residue sets are
identical). Selectivity between close homologues has to come from the *shape* of
the cavity, not its composition. Here each pocket is rasterised into a 1 A
occupancy grid centred on the metal, and pockets are compared by:

  * shape difference volume  - grid cells where one pocket is open and the other
                               is blocked (the actual "difference region")
  * Jaccard similarity of the open-cavity voxel sets
  * cavity volume and radial profile

The difference regions are what a selective inhibitor would have to exploit, so
they are written out as PDB-style marker files for inspection.
"""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC8": "1T64"}
MET = {"ZN", "CA", "MG", "MN", "CO", "NI", "FE"}
RMAX, STEP = 13.0, 1.0
PROBE = 1.4          # a voxel is "open" if no heavy atom within this radius


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
        elif rn in MET and rn.startswith("ZN"):
            met.append(xyz)
    return np.array(prot), met[0]


def grid(protein, metal):
    n = int(2 * RMAX / STEP) + 1
    ax = np.linspace(-RMAX, RMAX, n)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    rel = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])   # same for every subtype
    pts = rel + metal
    r = np.linalg.norm(rel, axis=1)
    # nearest protein atom distance (chunked to bound memory)
    dmin = np.empty(len(pts))
    for i in range(0, len(pts), 4000):
        blk = pts[i:i + 4000]
        d = np.sqrt(((blk[:, None, :] - protein[None, :, :]) ** 2).sum(-1))
        dmin[i:i + 4000] = d.min(1)
    return rel, r, dmin


def main():
    grids = {}
    for name, pid in PDB.items():
        prot, m = load(pid)
        pts, r, dmin = grid(prot, m)
        open_mask = dmin > PROBE
        grids[name] = dict(metal=m, pts=pts, r=r, open=open_mask, dmin=dmin)
        vol = open_mask.sum() * STEP ** 3
        shell = [(r >= lo) & (r < lo + 2) for lo in (3, 5, 7, 9, 11)]
        prof = [float((open_mask & s).sum() * STEP ** 3) for s in shell]
        print(f"{name:<8}{pid:<7} 空腔体积={vol:>7.0f} A^3   径向剖面(3-11A)={['%.0f' % x for x in prof]}")

    print("\n亚型间空腔形状比较（Jaccard 相似度，1.0 = 完全相同）")
    names = list(grids)
    print(f"{'':<8}" + "".join(f"{n:>9}" for n in names))
    jac = {}
    for a in names:
        line = f"{a:<8}"
        for b in names:
            oa, ob = grids[a]["open"], grids[b]["open"]
            j = (oa & ob).sum() / max((oa | ob).sum(), 1)
            jac[(a, b)] = j
            line += f"{j:>9.3f}"
        print(line)

    # difference regions between the closest homologues
    print("\n最相似的一对与差异体积")
    pairs = sorted(((jac[(a, b)], a, b) for a, b in itertools.combinations(names, 2)), reverse=True)
    for j, a, b in pairs[:4]:
        oa, ob = grids[a]["open"], grids[b]["open"]
        diff = (oa ^ ob).sum() * STEP ** 3
        print(f"  {a} vs {b}: Jaccard={j:.3f}  差异体积={diff:.0f} A^3")

    out = ROOT / "results/hdac_pocket_shape.json"
    json.dump({n: {"cavity_volume": float(g["open"].sum() * STEP ** 3),
                   "n_open_voxels": int(g["open"].sum()),
                   "metal": g["metal"].tolist()} for n, g in grids.items()},
              open(out, "w"), indent=2)
    print(f"\n写出 {out}")

    # write the difference region of the most similar pair as markers
    j, a, b = pairs[0]
    oa, ob = grids[a]["open"], grids[b]["open"]
    sel = oa ^ ob
    dest = ROOT / f"results/pocket_diff_{a}_{b}.pdb"
    with dest.open("w") as fh:
        pts_abs = grids[a]["pts"] + grids[a]["metal"][None, :]
        for i, p in enumerate(pts_abs[sel]):
            kind = "OPEN" if oa[sel][i] else "SHUT"
            tag = "O" if kind == "OPEN" else "S"
            fh.write(f"HETATM{ i+1:5d}  {tag}   DIF X   1    "
                     f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00          {tag:>2}\n")
    print(f"写出差异区域标记 {dest}（{int(sel.sum())} 个体素）")


if __name__ == "__main__":
    main()
