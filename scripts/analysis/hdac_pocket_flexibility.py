"""Pocket flexibility / conformational entropy from a Gaussian network model.

Why not volume: pocket volume is an ensemble property. A crystal structure is one
snapshot, and the literature is explicit that these sites are dynamic (MMP Omega
loops are far more flexible in MD than in any single structure; HDAC2 selectivity
involves slow-binding with residence times of hours). An absolute volume measured
off one conformer therefore has no stable physical meaning across structures.

What is more defensible: the *shape of the fluctuation field* around the pocket,
which GNM estimates from a single structure and which is far less sensitive to
crystal-packing accidents than a raw volume.

Model
-----
Gaussian network model on Calpha atoms, cutoff Rc = 10 A:
    Gamma_ij = -1 if i,j within Rc else 0;  Gamma_ii = -sum_j Gamma_ij
    <dR_i^2> = (3 kT / gamma) * [Gamma^+]_ii        (Kirchhoff pseudo-inverse)
The first eigenvector of Gamma (slowest mode) gives the direction of the largest
collective motion; projecting pocket residues onto it says whether the pocket
breathes as a unit or is structurally rigid.

Outputs per subtype:
  msf   mean square fluctuation of pocket residues (8-14 A), normalised by the
        whole-protein mean so different structures are comparable
  slow  participation of pocket residues in the slowest mode
  rigid fraction of pocket residues with normalised MSF below the protein median
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC8": "1T64"}
RC = 10.0


def ca_and_metal(pid):
    ca, res, met = [], [], []
    for ln in (ROOT / f"processed/hdac_receptors/{pid}_h.pdb").read_text(errors="ignore").splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        an = ln[12:16].strip()
        rn = ln[17:20].strip().upper()
        el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        if ln.startswith("ATOM") and an == "CA":
            ca.append(xyz); res.append((rn, ln[22:26].strip()))
        elif ln.startswith("HETATM") and rn == "ZN":
            met.append(xyz)
    return np.array(ca), res, met[0]


def gnm(ca, rc=RC):
    n = len(ca)
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=2)
    A = (d < rc).astype(float)
    np.fill_diagonal(A, 0)
    G = -A
    np.fill_diagonal(G, A.sum(1))
    vals, vecs = np.linalg.eigh(G)
    # remove the zero mode
    keep = vals > 1e-8
    inv = np.zeros(n)
    inv[keep] = 1.0 / vals[keep]
    msf = (vecs ** 2) @ inv                      # [Gamma^+]_ii, up to 3kT/gamma
    slow = vecs[:, np.argmax(vals)]              # slowest non-trivial mode
    return msf, np.abs(slow), vals


def main():
    out = {}
    for name, pid in PDB.items():
        ca, res, metal = ca_and_metal(pid)
        if len(ca) < 50:
            print(f"{name}: too few CA atoms ({len(ca)})")
            continue
        msf, slow, vals = gnm(ca)
        d = np.linalg.norm(ca - metal, axis=1)
        msf_n = msf / msf.mean()
        pocket = (d > 8.0) & (d <= 14.0)
        core = d <= 6.0
        whole_med = float(np.median(msf_n))
        rec = {
            "pdb": pid, "n_ca": len(ca),
            "msf_pocket": round(float(msf_n[pocket].mean()), 4),
            "msf_core": round(float(msf_n[core].mean()), 4) if core.any() else None,
            "msf_ratio_pocket_core": round(float(msf_n[pocket].mean() /
                                                 max(msf_n[core].mean(), 1e-9)), 3) if core.any() else None,
            "slow_pocket": round(float(slow[pocket].mean()), 4),
            "rigid_frac_pocket": round(float((msf_n[pocket] < whole_med).mean()), 3),
            "eig_ratio_1_2": round(float(vals[-1] / max(vals[-2], 1e-9)), 3),
        }
        out[name] = rec
        print(f"{name:<8}{pid:<7} CA={len(ca):<4} MSF_pocket={rec['msf_pocket']:.3f} "
              f"MSF_core={rec['msf_core']:.3f} 比值={rec['msf_ratio_pocket_core']:.2f} "
              f"慢模参与={rec['slow_pocket']:.3f} 刚性比例={rec['rigid_frac_pocket']:.2f}")

    print("\n口袋柔性排序（越低 = 越刚；按 MSF_pocket）")
    for k, v in sorted(out.items(), key=lambda kv: kv[1]["msf_pocket"]):
        print(f"  {k:<8}{v['msf_pocket']:.3f}")

    print("\n关键对照：文献说 HDAC1/2 的 foot pocket 可塑、HDAC3 被 Tyr107 收窄")
    print("  若 HDAC3 的 pocket/core 比值最低，说明其口袋相对更刚——与文献一致")
    for k, v in out.items():
        print(f"  {k:<8}pocket/core = {v['msf_ratio_pocket_core']}")

    json.dump(out, open(ROOT / "results/hdac_pocket_flexibility.json", "w"), indent=2)
    print("\n写出 results/hdac_pocket_flexibility.json")


if __name__ == "__main__":
    main()
