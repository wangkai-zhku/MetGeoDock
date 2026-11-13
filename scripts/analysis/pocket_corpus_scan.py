"""Pocket descriptors over the whole metalloprotein corpus (16k+ crystal structures).

Why the full corpus: developing a pocket descriptor on six HDAC subtypes alone is
a six-point fit. The reference distribution has to come from every metalloprotein
we have, so that "how unusual is this pocket" is measured against a real
population rather than against its own family.

Lightweight by design - no voxel grid, no ray casting over hundreds of
directions - so ~17k structures finish in minutes:

  shell        first coordination shell (<=3 A): residue types, count (cn)
  layers       heavy-atom and residue counts in 3-6, 6-9, 9-12, 12-15 A shells
  chemistry    hydrophobic / polar / charged / aromatic fractions, per layer
  shape        PCA axis lengths and axis ratios of the 12 A pocket atoms
  openness     fraction of 48 rays from the metal that stay clear to 12 A
  fingerprint  sorted residue-type multiset (for cross-structure comparison)

Input: /home/wangkai/cursor-task/metalloprotein/ORI_PDB_LIGAND/{MM,SM,SW}/*_pro_*.pdb
Output: results/pocket_corpus_descriptors.csv

Usage:
    python scripts/analysis/pocket_corpus_scan.py --workers 16
    python scripts/analysis/pocket_corpus_scan.py --limit 500     # smoke test
"""
from __future__ import annotations
import argparse, csv, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

CORPUS = Path("/home/wangkai/cursor-task/metalloprotein/ORI_PDB_LIGAND")
ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")

METAL_NAMES = {"ZN", "ZN2", "C0", "CA", "CA2", "MG", "MG2", "MN", "MN2", "MN3",
               "FE", "FE2", "FE3", "CU", "CU1", "CU2", "CO", "CO2", "CD", "CD2",
               "CAD", "NI", "NI2"}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "GLY", "CYS"}
AROMATIC = {"PHE", "TYR", "TRP", "HIS"}
CHARGED = {"ASP", "GLU", "LYS", "ARG"}
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "HIS", "CYS"}
DONOR = {"N", "O", "S"}

FIELDS = ["file", "family", "metal", "pdb_id", "lig_id", "n_atoms", "n_residues",
          "cn", "shell_residues", "n_atoms_3_6", "n_atoms_6_9", "n_atoms_9_12", "n_atoms_12_15",
          "hyd_3_9", "pol_3_9", "chg_3_9", "aro_3_9",
          "hyd_9_12", "pol_9_12", "chg_9_12", "aro_9_12",
          "hyd_12_15", "pol_12_15", "chg_12_15", "aro_12_15",
          "pca_a1", "pca_a2", "pca_a3", "elong", "flat", "openness", "fp"]


def describe(path: str) -> dict | None:
    p = Path(path)
    parts = p.stem.split("_")            # e.g. ca_1A8A_pro_5
    metal = parts[0]
    pdb_id = parts[1] if len(parts) > 1 else "?"
    lig_id = parts[-1]
    prot, met = [], []
    try:
        text = p.read_text(errors="ignore")
    except OSError:
        return None
    for ln in text.splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        rn = ln[17:20].strip().upper()
        el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        try:
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        except ValueError:
            continue
        if ln.startswith("ATOM") and el != "H":
            prot.append((rn, ln[22:26].strip(), xyz))
        elif rn in METAL_NAMES:
            met.append(xyz)
    if not prot or not met:
        return None
    P = np.array([x[2] for x in prot], float)
    m = np.array(met[0], float)
    d = np.linalg.norm(P - m, axis=1)

    rec = {"file": str(p), "family": p.parent.name, "metal": metal,
           "pdb_id": pdb_id, "lig_id": lig_id,
           "n_atoms": len(prot), "n_residues": len({(x[0], x[1]) for x in prot})}

    # shell
    shell = [(prot[i][0], prot[i][1], d[i]) for i in range(len(prot)) if d[i] <= 3.0]
    rec["cn"] = len(shell)
    rec["shell_residues"] = ",".join(sorted({f"{r}{i}" for r, i, _ in shell}))

    # layers
    for lo, hi, name in ((3, 6, "3_6"), (6, 9, "6_9"), (9, 12, "9_12"), (12, 15, "12_15")):
        sel = (d > lo) & (d <= hi)
        rec[f"n_atoms_{name}"] = int(sel.sum())
    # chemistry per layer
    for lo, hi, name in ((3, 9, "3_9"), (9, 12, "9_12"), (12, 15, "12_15")):
        types = [prot[i][0] for i in range(len(prot)) if lo < d[i] <= hi]
        n = max(len(types), 1)
        rec[f"hyd_{name}"] = round(sum(t in HYDROPHOBIC for t in types) / n, 4)
        rec[f"pol_{name}"] = round(sum(t in POLAR for t in types) / n, 4)
        rec[f"chg_{name}"] = round(sum(t in CHARGED for t in types) / n, 4)
        rec[f"aro_{name}"] = round(sum(t in AROMATIC for t in types) / n, 4)

    # shape from 12 A atoms
    Q = P[d <= 12.0] - m
    if len(Q) >= 10:
        ev = np.sort(np.linalg.eigvalsh(np.cov(Q.T)))[::-1]
        ev = np.maximum(ev, 1e-9)
        a1, a2, a3 = np.sqrt(ev)
        rec.update(pca_a1=round(float(a1), 3), pca_a2=round(float(a2), 3),
                   pca_a3=round(float(a3), 3),
                   elong=round(float(a1 / a2), 3), flat=round(float(a2 / a3), 3))
    else:
        rec.update(pca_a1=np.nan, pca_a2=np.nan, pca_a3=np.nan,
                   elong=np.nan, flat=np.nan)

    # openness: 48 deterministic directions
    phi = np.linspace(0, np.pi, 8, endpoint=False)
    th = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    dirs = np.array([[np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)]
                     for a in phi for b in th])
    cheap = P[d <= 14.0]
    rs = np.arange(1.6, 12.0, 1.0)
    free = 0
    for v in dirs:
        pts = m + v[None, :] * rs[:, None]
        dm = np.sqrt(((pts[:, None, :] - cheap[None, :, :]) ** 2).sum(-1)).min()
        if dm > 1.9:
            free += 1
    rec["openness"] = round(free / len(dirs), 4)
    rec["fp"] = "|".join(sorted({prot[i][0] for i in range(len(prot)) if d[i] <= 10.0}))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/pocket_corpus_descriptors.csv"))
    a = ap.parse_args()

    files = sorted(str(p) for p in CORPUS.glob("*/*_pro_*.pdb"))
    if a.limit:
        files = files[: a.limit]
    print(f"待处理结构: {len(files)}  并行 {a.workers}", flush=True)

    rows = []
    with Pool(a.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(describe, files, chunksize=50), 1):
            if rec:
                rows.append(rec)
            if i % 1000 == 0:
                print(f"  {i}/{len(files)}  ({len(rows)} 成功)", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWROTE {out}  ({len(rows)} 行)")
    import collections
    print("按金属:", dict(collections.Counter(r["metal"] for r in rows).most_common(12)))


if __name__ == "__main__":
    main()
