"""Fine-grained pocket descriptors for the six HDAC subtypes.

Rationale: a selective inhibitor sees the SAME ligand in all six receptors, so
any ligand-intrinsic descriptor (MW, nRot, chelate count) is a constant across
subtypes and cannot possibly explain selectivity. Only ligand-pocket interaction
terms can. This script characterises the pockets themselves so we can later test
which pocket property tracks the observed selectivity.

Descriptors are computed from the metal outward:
  shell      first coordination shell composition (residue types, count)
  size       protein heavy atoms / pocket volume proxy within 8, 10, 12 A
  shape      PCA axis ratios of pocket atoms (elongation, flatness)
  chemistry  hydrophobic / polar / charged / aromatic residue fractions
  opening    fraction of directions from the metal that are unobstructed
  mouth      radius of the largest free channel (ray casting)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC8": "1T64"}
METAL = {"ZN", "ZN2", "CA", "CA2", "MG", "MG2", "MN", "FE", "FE2", "FE3",
         "CU", "CO", "NI", "CD"}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "CYS", "GLY"}
AROMATIC = {"PHE", "TYR", "TRP", "HIS"}
CHARGED = {"ASP", "GLU", "LYS", "ARG"}
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "HIS", "CYS"}


def read_pocket(path, metal_name="ZN"):
    prot, met = [], []
    for ln in Path(path).read_text(errors="ignore").splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        rn = ln[17:20].strip().upper()
        el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        if ln.startswith("ATOM") and el != "H":
            prot.append((rn, int(ln[22:26]) if ln[22:26].strip().lstrip('-').isdigit() else 0, xyz))
        elif rn in METAL and rn.startswith(metal_name):
            met.append(xyz)
    return prot, met


def ray_free_fraction(metal, atoms, n_dir=200, rmax=12.0, step=0.5):
    """Fraction of directions from the metal that stay clear out to rmax."""
    rng = np.random.default_rng(7)
    dirs = rng.normal(size=(n_dir, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rs = np.arange(1.5, rmax, step)
    free = 0
    for d in dirs:
        pts = metal + d[None, :] * rs[:, None]
        dmin = np.min(np.linalg.norm(pts[:, None, :] - atoms[None, :, :], axis=2))
        if dmin > 1.9:          # nothing within 1.9 A along the whole ray
            free += 1
    return free / n_dir


def main():
    out = {}
    for name, pid in PDB.items():
        prot, met = read_pocket(ROOT / f"processed/hdac_receptors/{pid}_h.pdb")
        if not met:
            print(f"{name}: no metal, skipped")
            continue
        m = met[0]
        P = np.array([p[2] for p in prot])
        d = np.linalg.norm(P - m, axis=1)
        rec = {"pdb": pid, "metal": "ZN",
               "n_protein_atoms": len(prot),
               "n_metals_in_pdb": len(met)}
        # --- shell and residue composition
        for r in (3.0, 6.0, 8.0, 10.0, 12.0):
            rec[f"n_atoms_{int(r)}A"] = int((d <= r).sum())
        near = [(prot[i][0], d[i]) for i in range(len(prot)) if d[i] <= 8.0]
        res8 = {prot[i][1] for i in range(len(prot)) if d[i] <= 8.0}
        rec["n_residues_8A"] = len(res8)
        shell = sorted({prot[i][0] for i in range(len(prot)) if d[i] <= 3.0})
        rec["shell_residues"] = ",".join(shell)
        rec["shell_n"] = len(shell)
        types = [p[0] for p in near]
        n = max(len(types), 1)
        rec["frac_hydrophobic"] = sum(t in HYDROPHOBIC for t in types) / n
        rec["frac_polar"] = sum(t in POLAR for t in types) / n
        rec["frac_charged"] = sum(t in CHARGED for t in types) / n
        rec["frac_aromatic"] = sum(t in AROMATIC for t in types) / n
        rec["n_his_8A"] = sum(t == "HIS" for t in types)
        rec["n_asp_glu_8A"] = sum(t in ("ASP", "GLU") for t in types)
        rec["n_cys_8A"] = sum(t == "CYS" for t in types)
        rec["n_phe_tyr_trp_8A"] = sum(t in ("PHE", "TYR", "TRP") for t in types)
        # --- shape from pocket atoms
        Q = P[d <= 12.0] - m
        if len(Q) >= 10:
            ev = np.sort(np.linalg.eigvalsh(np.cov(Q.T)))[::-1]
            ev = np.maximum(ev, 1e-9)
            rec["pca_axis1"] = float(np.sqrt(ev[0]))
            rec["pca_axis2"] = float(np.sqrt(ev[1]))
            rec["pca_axis3"] = float(np.sqrt(ev[2]))
            rec["elongation"] = float(np.sqrt(ev[0] / ev[1]))
            rec["flatness"] = float(np.sqrt(ev[1] / ev[2]))
        # --- opening / mouth
        rec["ray_free_frac"] = ray_free_fraction(m, P[d <= 20.0])
        rec["radius_of_gyration_12A"] = float(np.linalg.norm(Q, axis=1).mean()) if len(Q) else float("nan")
        out[name] = rec
        print(f"{name:<8}{pid:<7} 原子8A={rec['n_atoms_8A']:<5} 残基8A={rec['n_residues_8A']:<4} "
              f"壳层={rec['shell_n']:<2} 疏水={rec['frac_hydrophobic']:.2f} "
              f"开口={rec['ray_free_frac']:.2f} 细长={rec.get('elongation', float('nan')):.2f}")

    dest = ROOT / "results/hdac_pocket_descriptors.json"
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\n写出 {dest}")

    # subtype separation summary
    keys = ["n_atoms_8A", "n_residues_8A", "shell_n", "frac_hydrophobic", "frac_polar",
            "frac_charged", "n_his_8A", "n_asp_glu_8A", "n_cys_8A", "n_phe_tyr_trp_8A",
            "elongation", "flatness", "ray_free_frac"]
    print("\n各描述的亚型间变异（CV），越大说明该描述越能区分亚型")
    for k in keys:
        v = np.array([out[s][k] for s in out if k in out[s]], float)
        m = np.abs(v.mean())
        print(f"  {k:<20}{'min=%.3g  max=%.3g  CV=%.3f' % (v.min(), v.max(), v.std()/m if m > 0 else 0)}")


if __name__ == "__main__":
    main()
