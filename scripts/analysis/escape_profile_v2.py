"""Escape-path energy with electrostatics, hydrogen bonds and formal charges.

v1 used plain Lennard-Jones, i.e. basically molecular size - no signal. v2 adds the
physics that actually distinguishes warheads at the same geometry:

  * Gasteiger partial charges on the ligand (carries protonation state: a
    carboxylate/hydroxamate anion has sum(q) ~ -1)
  * Coulomb term with a distance-dependent dielectric (eps = 4r), including the
    Zn(2+) - ligand attraction (electronic proxy for zinc-binding strength)
  * Directional hydrogen-bond bonus for N/O...N/O pairs in 2.6-3.5 A
  * Lennard-Jones contact as before

All terms are O(N^2) analytic sums: no MM-GBSA, no FEP, same cost class as the
existing score. The escape profile (translate along the pocket-opening axis)
still converts a static pose into a kinetic-like descriptor.
"""
from __future__ import annotations

import argparse, csv
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]
HDAC = Path("/home/wangkai/cursor-task/HDAC")

SIGMA = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47, "CL": 1.75}
EPS = {"C": 0.15, "N": 0.20, "O": 0.20, "S": 0.25, "P": 0.25, "F": 0.12, "CL": 0.25}
SHIFTS = np.arange(0.0, 8.5, 0.5)
CUT_LJ, CUT_COUL = 12.0, 20.0
HB_LO, HB_HI, HB_E = 2.6, 3.5, -1.5
COUL_K = 332.0
ZN_Q = 2.0
# formal charges on ionised side chains (heavy atoms only)
FORMAL = {"ASP:OD1": -0.5, "ASP:OD2": -0.5, "GLU:OE1": -0.5, "GLU:OE2": -0.5,
          "LYS:NZ": 1.0, "ARG:NE": 0.33, "ARG:NH1": 0.33, "ARG:NH2": 0.33}


def gasteiger(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    m = Chem.AddHs(m)
    try:
        AllChem.ComputeGasteigerCharges(m)
    except Exception:
        return None
    q = np.array([float(a.GetProp("_GasteigerCharge")) for a in m.GetAtoms() if a.GetSymbol() != "H"])
    return np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)


def read_receptor(path: Path):
    xyz, el, q = [], [], []
    zn = None
    for ln in path.read_text(errors="ignore").splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        rn = ln[17:20].strip().upper()
        an = ln[12:16].strip().upper()
        e = (ln[76:78].strip() or an[:1]).upper()
        pos = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        if rn == "ZN":
            zn = np.array(pos); continue
        if e in ("H",) or e not in SIGMA:
            continue
        xyz.append(pos); el.append(e)
        q.append(FORMAL.get(f"{rn}:{an}", 0.0))
    return np.asarray(xyz, float).reshape(-1, 3), np.asarray(el), np.asarray(q, float), zn


def read_ligand(path: Path):
    xyz, el = [], []
    for ln in path.read_text(errors="ignore").splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        e = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        if e == "H" or e not in SIGMA:
            continue
        xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        el.append(e)
    return np.asarray(xyz, float).reshape(-1, 3), np.asarray(el)


def energy(lxyz, lel, lq, rxyz, rel, rq, zn):
    """LJ + Coulomb(dielectric 4r) + H-bond bonus + Zn attraction."""
    if len(lxyz) == 0 or len(rxyz) == 0:
        return 0.0
    d = np.linalg.norm(lxyz[:, None, :] - rxyz[None, :, :], axis=2)
    sig = 0.5 * (np.array([SIGMA[e] for e in lel])[:, None] + np.array([SIGMA[e] for e in rel])[None, :])
    eps = np.sqrt(np.array([EPS[e] for e in lel])[:, None] * np.array([EPS[e] for e in rel])[None, :])
    m_lj = d < CUT_LJ
    sr = np.where(m_lj, sig / np.maximum(d, 1e-3), 10.0)
    e_lj = np.where(m_lj, eps * (sr ** 12 - 2.0 * sr ** 6), 0.0).sum()

    m_c = d < CUT_COUL
    eps_r = 4.0 * np.maximum(d, 1.0)
    e_coul = np.where(m_c, COUL_K * (lq[:, None] * rq[None, :]) / eps_r, 0.0).sum()

    lpol = np.isin(lel, ["N", "O", "S"])
    rpol = np.isin(rel, ["N", "O", "S"])
    m_hb = (d >= HB_LO) & (d <= HB_HI) & lpol[:, None] & rpol[None, :]
    e_hb = HB_E * m_hb.sum()

    e_zn = 0.0
    if zn is not None:
        dz = np.linalg.norm(lxyz - zn[None, :], axis=1)
        mz = dz < 4.5
        e_zn = float(np.where(mz, COUL_K * lq * ZN_Q / (4.0 * np.maximum(dz, 1.0)), 0.0).sum())
    return float(e_lj + e_coul + e_hb + e_zn)


def profile(rec: Path, lig: Path, lq):
    rxyz, rel, rq, zn = read_receptor(rec)
    lxyz, lel = read_ligand(lig)
    if len(lxyz) == 0 or len(rxyz) == 0 or lq is None or len(lq) != len(lxyz):
        return None
    ref = zn if zn is not None else rxyz.mean(0)
    axis = ref - rxyz.mean(0)
    n = np.linalg.norm(axis)
    axis = axis / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
    E = np.array([energy(lxyz + axis * s, lel, lq, rxyz, rel, rq, zn) for s in SHIFTS])
    e0 = E[0]
    return {"E0": e0, "E_min": float(E.min()),
            "dE_barrier": float(E.max() - e0), "dE_escape": float(E[-1] - e0),
            "slope_2A": float((E[4] - e0) / 2.0), "slope_4A": float((E[8] - e0) / 4.0),
            "E_curve": "|".join(f"{v:.2f}" for v in E)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=Path, default=HDAC / "docking/poses_sd")
    ap.add_argument("--receptors", type=Path, default=HDAC / "structures/receptors_h")
    ap.add_argument("--dataset", type=Path, default=HDAC / "data/processed/hdac_dataset.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "hdac_escape_v2.csv")
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
        lq = gasteiger(m["smiles"])
        try:
            pr = profile(rec, pf, lq)
        except Exception:
            continue
        if pr is None: continue
        rows.append({"ligand_id": lid, "target": m["target"], "label": int(m["label"]),
                     "pchembl": m["pchembl"], "decoy_type": m["decoy_type"], **pr})
        if i % 200 == 0: print(f"  {i}/{len(pfiles)}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"WROTE {a.out}  n={len(rows)}")


if __name__ == "__main__":
    main()
