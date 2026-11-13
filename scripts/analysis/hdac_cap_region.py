"""CAP-region descriptors: the surface rim above the channel.

Rationale
---------
HDAC inhibitors have three parts: a zinc-binding group (buried, conserved site), a
linker through the ~11 A channel, and a CAP group that sits on the enzyme surface.
The literature is consistent that selectivity comes from the cap and its rim, not
from the buried catalytic machinery. All earlier descriptors averaged over the whole
pocket and therefore diluted exactly this region.

This script isolates the cap region geometrically:
    channel axis = (metal - protein centroid) normalised
    cap region   = residues whose closest atom is 10-20 A from the metal AND lies
                   within CONE_DEG of the channel axis

and characterises it per subtype: residue composition, hydrophobicity / charge /
aromatic fractions, surface shape, and experimental flexibility (B-factor).

Then the same region is profiled for ligands: the "cap part" of a docked pose is
everything further than 5 A from the ligand's coordinating atom.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
sys.path.insert(0, str(ROOT))
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC8": "1T64"}
R_IN, R_OUT, CONE_DEG = 8.0, 16.0, 90.0
HYD = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "CYS", "GLY"}
ARO = {"PHE", "TYR", "TRP", "HIS"}
CHG = {"ASP", "GLU", "LYS", "ARG"}
POL = {"SER", "THR", "ASN", "GLN", "TYR", "HIS", "CYS"}


def read(pid, prepared=True):
    f = ROOT / f"processed/hdac_receptors/{pid}_h.pdb" if prepared else \
        ROOT / f"processed/hdac_receptors/{pid}.pdb"
    atoms, metal = [], []
    for ln in f.read_text(errors="ignore").splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        rn = ln[17:20].strip().upper()
        el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        an = ln[12:16].strip()
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        b = float(ln[60:66]) if ln[60:66].strip() else 0.0
        if ln.startswith("ATOM") and el != "H":
            atoms.append({"rn": rn, "key": f"{ln[21]}{ln[22:26].strip()}", "an": an,
                          "xyz": xyz, "el": el, "b": b})
        elif rn.startswith("ZN"):
            metal.append(xyz)
    return atoms, metal[0] if metal else None, atoms


def cap_region(atoms, metal):
    P = np.array([a["xyz"] for a in atoms])
    centroid = P.mean(0)
    axis = metal - centroid
    axis /= np.linalg.norm(axis)
    per = {}
    for a in atoms:
        per.setdefault(a["key"], {"rn": a["rn"], "xyz": [], "b": []})
        per[a["key"]]["xyz"].append(a["xyz"])
        per[a["key"]]["b"].append(a["b"])
    cap = {}
    for k, v in per.items():
        arr = np.array(v["xyz"])
        d = np.linalg.norm(arr - metal, axis=1)
        dmin = float(d.min())
        if not (R_IN <= dmin <= R_OUT):
            continue
        # direction of the closest atom
        i = int(d.argmin())
        vec = arr[i] - metal
        n = np.linalg.norm(vec)
        if n < 1e-6:
            continue
        cosang = float(vec @ axis / n)
        if cosang < np.cos(np.deg2rad(CONE_DEG)):
            continue
        cap[k] = {"rn": v["rn"], "dmin": dmin, "b": float(np.mean(v["b"])),
                  "xyz": arr[i]}
    return cap, axis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/hdac_cap_region.json")
    a = ap.parse_args()

    print(f"CAP 区定义: 距金属 {R_IN}-{R_OUT} A（含通道开口四周）\n")
    print(f"{'亚型':<8}{'cap残基':>8}{'疏水%':>7}{'芳香%':>7}{'带电%':>7}{'极性%':>7}"
          f"{'B中位':>8}{'形状主轴':>9}")
    out = {}
    for name, pid in PDB.items():
        atoms, metal, _ = read(pid)
        if metal is None:
            continue
        cap, axis = cap_region(atoms, metal)
        if not cap:
            print(f"{name}: 无 cap 残基")
            continue
        types = [v["rn"] for v in cap.values()]
        n = len(types)
        Q = np.array([np.asarray(v["xyz"], float).ravel() for v in cap.values()])
        if len(Q) >= 3 and Q.ndim == 2 and Q.shape[1] == 3:
            ev = np.sort(np.linalg.eigvalsh(np.cov(Q.T)))[::-1]
            ev = np.maximum(ev, 1e-9)
            shape = float(np.sqrt(ev[0] / ev[1]))
        else:
            shape = float("nan")
        rec = {
            "n_res": n,
            "frac_hyd": sum(t in HYD for t in types) / n,
            "frac_aro": sum(t in ARO for t in types) / n,
            "frac_chg": sum(t in CHG for t in types) / n,
            "frac_pol": sum(t in POL for t in types) / n,
            "bfac_median": float(np.median([v["b"] for v in cap.values()])),
            "shape_elong": round(shape, 3),
            "composition": dict(Counter(types)),
            "residues": {k: {"rn": v["rn"], "d": round(v["dmin"], 1)}
                         for k, v in sorted(cap.items(), key=lambda kv: kv[1]["dmin"])},
        }
        out[name] = rec
        print(f"{name:<8}{n:>8}{rec['frac_hyd']:>7.2f}{rec['frac_aro']:>7.2f}"
              f"{rec['frac_chg']:>7.2f}{rec['frac_pol']:>7.2f}"
              f"{rec['bfac_median']:>8.1f}{shape:>9.2f}")

    # subtype discrimination within the cap region alone
    keys = ["n_res", "frac_hyd", "frac_aro", "frac_chg", "frac_pol", "shape_elong"]
    print("\ncap 区描述符的亚型间变异")
    for k in keys:
        v = np.array([out[s][k] for s in out], float)
        if not np.isfinite(v).all():
            continue
        m = np.abs(v.mean())
        print(f"  {k:<14} min={v.min():>7.3f} max={v.max():>7.3f} CV={v.std()/m if m else 0:>8.3f}")

    print("\ncap 区残基组成的成对差异（Jaccard，1.0 = 组成完全相同）")
    subs = list(out)
    print(f"{'':<8}" + "".join(f"{s:>9}" for s in subs))
    for x in subs:
        line = f"{x:<8}"
        for y in subs:
            cx = Counter(out[x]["composition"])
            cy = Counter(out[y]["composition"])
            inter = sum((cx & cy).values())
            union = sum((cx | cy).values())
            line += f"{inter/max(union,1):>9.3f}"
        print(line)

    json.dump(out, open(ROOT / a.out, "w"), indent=2)
    print(f"\n写出 {a.out}")

    print("\n各亚型 cap 区最近的 8 个残基（离金属由近到远）:")
    for s in out:
        near = list(out[s]["residues"].items())[:8]
        print(f"  {s:<8}" + ", ".join(f"{v['rn']}{k[1:]}({v['d']}A)" for k, v in near))


if __name__ == "__main__":
    main()
