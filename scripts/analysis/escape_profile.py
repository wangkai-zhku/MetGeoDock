"""Escape-path energy profile: can a bound pose actually stay bound?

A static score answers "is this pose geometrically plausible"; it cannot answer
"will this molecule stay". Residence time is governed by the barrier along the
escape path (Bell: koff ~ exp(-dE/kT)), so for every docked pose we translate the
ligand along the pocket-opening axis in 0.5 A steps and evaluate the
Lennard-Jones contact energy with the receptor.
"""
from __future__ import annotations

import argparse, csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HDAC = Path("/home/wangkai/cursor-task/HDAC")
SIGMA = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
         "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98}
EPS = {"C": 0.15, "N": 0.20, "O": 0.20, "S": 0.25, "P": 0.25,
       "F": 0.12, "CL": 0.25, "BR": 0.30, "I": 0.35}
SHIFTS = np.arange(0.0, 8.5, 0.5)
CUTOFF = 12.0


def read_atoms(path: Path, mode: str):
    xyz, el = [], []
    for ln in path.read_text(errors="ignore").splitlines():
        if ln.startswith("ATOM"):
            pass
        elif ln.startswith("HETATM"):
            r = ln[17:20].strip().upper()
            if mode == "protein" and r != "ZN":
                continue
        else:
            continue
        e = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        if e in ("H", "ZN") or e not in SIGMA:
            continue
        xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        el.append(e)
    return np.asarray(xyz, float).reshape(-1, 3), np.asarray(el)


def lj_energy(lxyz, lel, rxyz, rel):
    if len(lxyz) == 0 or len(rxyz) == 0:
        return 0.0, 0
    d = np.linalg.norm(lxyz[:, None, :] - rxyz[None, :, :], axis=2)
    sig = np.array([SIGMA[e] for e in lel])[:, None]
    rsig = np.array([SIGMA[e] for e in rel])[None, :]
    sig_ij = 0.5 * (sig + rsig)
    eps_ij = np.sqrt(np.array([EPS[e] for e in lel])[:, None] * np.array([EPS[e] for e in rel])[None, :])
    mask = d < CUTOFF
    sr = np.where(mask, sig_ij / np.maximum(d, 1e-3), 10.0)
    e = eps_ij * (sr ** 12 - 2.0 * sr ** 6)
    e = np.where(mask, e, 0.0)
    return float(e.sum()), int((d < 4.6).sum())


def profile(rec: Path, lig: Path):
    rxyz, rel = read_atoms(rec, "protein")
    lxyz, lel = read_atoms(lig, "ligand")
    if len(lxyz) == 0 or len(rxyz) == 0:
        return None
    zn = None
    for ln in rec.read_text(errors="ignore").splitlines():
        if ln.startswith("HETATM") and ln[17:20].strip().upper() == "ZN":
            zn = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            break
    if zn is None:
        zn = rxyz.mean(0)
    axis = zn - rxyz.mean(0)
    n = np.linalg.norm(axis)
    axis = axis / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
    energies, ncont = [], []
    for s in SHIFTS:
        e, nc = lj_energy(lxyz + axis * s, lel, rxyz, rel)
        energies.append(e); ncont.append(nc)
    E = np.array(energies); e0 = E[0]
    return {
        "E0": e0, "E_min": float(E.min()),
        "dE_barrier": float(E.max() - e0),
        "dE_escape": float(E[-1] - e0),
        "slope_2A": float((E[4] - e0) / 2.0),
        "slope_4A": float((E[8] - e0) / 4.0),
        "n_contact_0": ncont[0],
        "E_curve": "|".join(f"{v:.3f}" for v in E),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=Path, default=HDAC / "docking/poses_sd")
    ap.add_argument("--receptors", type=Path, default=HDAC / "structures/receptors_h")
    ap.add_argument("--dataset", type=Path, default=HDAC / "data/processed/hdac_dataset.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "hdac_escape.csv")
    a = ap.parse_args()
    meta = {r["ligand_id"]: r for r in csv.DictReader(open(a.dataset))}
    pfiles = sorted(a.poses.rglob("*_best.pdb"))
    print(f"poses: {len(pfiles)}", flush=True)
    rows = []
    for i, pf in enumerate(pfiles, 1):
        lid = pf.stem.replace("_best", "")
        m = meta.get(lid)
        if not m: continue
        rec = a.receptors / f"{m['target']}.pdb"
        if not rec.exists(): continue
        try:
            pr = profile(rec, pf)
        except Exception:
            continue
        if pr is None: continue
        rows.append({"ligand_id": lid, "target": m["target"], "label": int(m["label"]),
                     "pchembl": m["pchembl"], "decoy_type": m["decoy_type"], **pr})
        if i % 200 == 0: print(f"  {i}/{len(pfiles)}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"WROTE {a.out}  n={len(rows)}")


if __name__ == "__main__":
    main()
