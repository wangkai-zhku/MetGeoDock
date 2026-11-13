"""Pocket descriptors for metalloenzymes, used to group targets for per-class calibration.

Features per receptor (all measured around the catalytic metal):
  cn              number of protein N/O/S within 3.3 A  (coordination number)
  n_6/n_8/n_10    protein heavy atoms within 6/8/10 A  (burial / pocket size)
  frac_8          fraction of the whole protein inside 8 A (openness)
  rg_pocket       radius of gyration of atoms within 10 A (pocket compactness)
  n_his/n_asp/n_cys/n_glu   donor residue composition of the first shell
  lig_mw_mean     mean MW of the target's known active ligands (from ChEMBL)

These let us group targets into pocket classes without using any activity value in
the descriptor itself (lig_mw_mean only summarises the chemotype, not potency).
"""
from __future__ import annotations

import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
HDAC = Path("/home/wangkai/cursor-task/HDAC")
XTAL = Path("/tmp/xtal_train")
METALS = {"ZN", "ZN2", "FE", "FE3", "FE2", "CA", "CA2", "C0", "MG", "MG2", "MN", "MN2", "MN3",
          "CO", "CO2", "CU", "CU1", "CU2", "NI", "NI2", "CD", "CD2"}


def descriptors(path: Path):
    met, prot = [], []
    for ln in path.read_text(errors="ignore").splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        resn = ln[17:20].strip().upper()
        atom = ln[12:16].strip().upper()
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        if resn in METALS:
            met.append(xyz)
        elif ln.startswith("ATOM") and el != "H":
            prot.append((resn, atom, el, xyz))
    if not met or not prot:
        return None
    z = met[0]
    P = np.asarray([p[3] for p in prot], float)
    d = np.linalg.norm(P - z, axis=1)

    shell = [p for p, dd in zip(prot, d) if dd <= 3.3 and p[2] in ("N", "O", "S")]
    res_of_shell = [p[0] for p in shell]
    pocket = P[d <= 10.0]
    rg = float(np.sqrt(((pocket - pocket.mean(0)) ** 2).sum(1).mean())) if len(pocket) else np.nan
    return {
        "cn": len(shell),
        "n_his": res_of_shell.count("HIS"),
        "n_asp": res_of_shell.count("ASP") + res_of_shell.count("GLU"),
        "n_cys": res_of_shell.count("CYS"),
        "n_6A": int((d <= 6).sum()), "n_8A": int((d <= 8).sum()), "n_10A": int((d <= 10).sum()),
        "frac_8A": round(float((d <= 8).mean()), 4),
        "rg_pocket": round(rg, 2),
        "n_metal": len(met),
    }


def main():
    acts = defaultdict(list)
    pdb_by_target = {}
    for r in csv.DictReader(open(ROOT / "processed/chembl/curated/all_targets_molecules.csv")):
        if r.get("pdb") and r["pdb"] not in ("", "None"):
            pdb_by_target.setdefault(r["target_name"], r["pdb"])
        if r.get("metal") != "fe3" and r.get("target_name", "").upper().startswith("HDAC"):
            continue
        try:
            pa = float(r["pactivity"])
        except Exception:
            continue
        if pa >= 6 and r.get("canonical_smiles"):
            acts[r["target_name"]].append(r["canonical_smiles"])

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")

    def mean_mw(smiles_list, k=40):
        vals = []
        for s in smiles_list[:k]:
            m = Chem.MolFromSmiles(s)
            if m:
                vals.append(Descriptors.MolWt(m))
        return round(float(np.mean(vals)), 1) if vals else np.nan

    rows = []
    # cross-family receptors (post-prep)
    for f in sorted(glob.glob(str(XTAL / "*_h.pdb"))):
        pdb = Path(f).stem.replace("_h", "")
        target = next((t for t, p in pdb_by_target.items() if p == pdb), None)
        if not target:
            continue
        d = descriptors(Path(f))
        if not d:
            continue
        d.update({"target": target, "pdb": pdb, "family": "cross-family",
                  "lig_mw_mean": mean_mw(acts.get(target, []))})
        rows.append(d)
    # HDAC receptors
    for t in ("HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC6", "HDAC7", "HDAC8"):
        f = HDAC / "structures/receptors_h" / f"{t}.pdb"
        if not f.exists():
            continue
        d = descriptors(f)
        if not d:
            continue
        d.update({"target": t, "pdb": {"HDAC1": "4BKX", "HDAC2": "6WBZ", "HDAC3": "4A69",
                                       "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC7": "3C10",
                                       "HDAC8": "5DC8"}[t], "family": "HDAC",
                  "lig_mw_mean": mean_mw(acts.get(t, []))})
        rows.append(d)

    keys = ["target", "family", "pdb", "cn", "n_his", "n_asp", "n_cys", "n_6A", "n_8A",
            "n_10A", "frac_8A", "rg_pocket", "n_metal", "lig_mw_mean"]
    dest = ROOT / "results/pocket_descriptors.csv"
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"{'target':14s} {'family':12s} {'cn':>3s} {'His':>4s} {'Asp':>4s} {'Cys':>4s} "
          f"{'n8A':>5s} {'rg':>6s} {'ligMW':>7s}")
    for r in sorted(rows, key=lambda x: (x["family"], x["target"])):
        print(f"{r['target']:14s} {r['family']:12s} {r['cn']:3d} {r['n_his']:4d} {r['n_asp']:4d} "
              f"{r['n_cys']:4d} {r['n_8A']:5d} {r['rg_pocket']:6.1f} {r['lig_mw_mean']:7.1f}")
    print(f"\nWROTE {dest}  n={len(rows)}")


if __name__ == "__main__":
    main()
