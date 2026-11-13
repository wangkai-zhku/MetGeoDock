"""Interaction-fingerprint features from docked poses.

MetGeo's terms describe whether a pose is geometrically plausible; they carry no
affinity information, which is why library ranking sits at AUROC ~0.5 even for the
trained hybrid. This module extracts the standard protein-ligand interaction
fingerprint instead: which residue types the ligand touches, and how.

Encoding: residue class (20 amino acids + hetero) x interaction type
(hydrogen bond / hydrophobic / polar) counts, plus summary descriptors.
"""
from __future__ import annotations

import argparse, csv
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AA = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
      "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
AA_SET = set(AA)
INTERACTIONS = ("hbond", "hydrophobic", "polar")
HB_DIST, HYD_DIST, POL_DIST = 3.5, 4.5, 4.2


def read_pdb_atoms(path: Path, mode: str = "protein"):
    """mode='protein' keeps ATOM (+Zn); mode='ligand' keeps every atom.

    Docked poses are HETATM/LIG records, so a protein-only filter yields zero
    ligand atoms and the fingerprint silently comes out empty.
    """
    xyz, el, resn, prot = [], [], [], []
    for ln in path.read_text(errors="ignore").splitlines():
        if ln.startswith("ATOM"):
            is_p = True
        elif ln.startswith("HETATM"):
            r = ln[17:20].strip().upper()
            if mode == "ligand":
                is_p = False
            elif r == "ZN":
                is_p = False
            else:
                continue
        else:
            continue
        e = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        if e == "H":
            continue
        xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        el.append(e); resn.append(ln[17:20].strip().upper()); prot.append(is_p)
    return (np.asarray(xyz, float).reshape(-1, 3), np.asarray(el),
            np.asarray(resn), np.asarray(prot, bool))


def features_for_pose(rec_path: Path, lig_path: Path, cutoff: float = 15.0) -> dict:
    rxyz, rel, rresn, rprot = read_pdb_atoms(rec_path)
    lxyz, lel, _, _ = read_pdb_atoms(lig_path, mode="ligand")
    if lxyz.size == 0 or rxyz.size == 0:
        return {}
    zn_mask = (rel == "ZN") & (~rprot)
    zn_xyz = rxyz[zn_mask][0] if zn_mask.any() else None
    if zn_xyz is not None:
        keep = np.linalg.norm(rxyz - zn_xyz, axis=1) <= cutoff
        rxyz, rel, rresn = rxyz[keep], rel[keep], rresn[keep]

    d = np.linalg.norm(lxyz[:, None, :] - rxyz[None, :, :], axis=2)
    lpol = np.isin(lel, ["N", "O", "S"])
    rpol = np.isin(rel, ["N", "O", "S"])
    lc = lel == "C"; rc = rel == "C"
    is_hb = (d <= HB_DIST) & lpol[:, None] & rpol[None, :]
    is_hy = (d <= HYD_DIST) & lc[:, None] & rc[None, :] & ~is_hb
    is_po = (d <= POL_DIST) & (lpol[:, None] ^ rpol[None, :]) & ~is_hb & ~is_hy

    counts = Counter()
    for name, mask in (("hbond", is_hb), ("hydrophobic", is_hy), ("polar", is_po)):
        if not mask.any():
            continue
        idx = np.argwhere(mask)
        for j in idx[:, 1]:
            counts[(rresn[j], name)] += 1

    feat = {}
    for aa in AA:
        for it in INTERACTIONS:
            feat[f"ifp_{aa}_{it}"] = float(counts.get((aa, it), 0))
    for it in INTERACTIONS:
        feat[f"ifp_HET_{it}"] = float(sum(c for (r, i2), c in counts.items() if i2 == it and r not in AA_SET))
    feat["ifp_total_hbond"] = float(sum(c for (r, i2), c in counts.items() if i2 == "hbond"))
    feat["ifp_total_hydrophobic"] = float(sum(c for (r, i2), c in counts.items() if i2 == "hydrophobic"))
    feat["ifp_n_residues_contacted"] = float(len({r for (r, _) in counts}))
    feat["ifp_HIS_any"] = float(sum(c for (r, _), c in counts.items() if r == "HIS"))
    feat["ifp_ASP_any"] = float(sum(c for (r, _), c in counts.items() if r == "ASP"))
    if zn_xyz is not None and len(lxyz):
        dz = np.linalg.norm(lxyz - zn_xyz, axis=1)
        feat["ifp_min_dist_to_zn"] = float(dz.min())
        feat["ifp_n_atoms_within_4A_zn"] = float((dz <= 4.0).sum())
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=Path, default=Path("/home/wangkai/cursor-task/HDAC/docking/poses_sd"))
    ap.add_argument("--receptors", type=Path, default=Path("/home/wangkai/cursor-task/HDAC/structures/receptors_h"))
    ap.add_argument("--dataset", type=Path, default=Path("/home/wangkai/cursor-task/HDAC/data/processed/hdac_dataset.csv"))
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "hdac_ifp.csv")
    a = ap.parse_args()
    meta = {r["ligand_id"]: r for r in csv.DictReader(open(a.dataset))}
    rows = []
    pose_files = sorted(a.poses.rglob("*_best.pdb"))
    print(f"found {len(pose_files)} pose files", flush=True)
    for i, pf in enumerate(pose_files, 1):
        lid = pf.stem.replace("_best", "")
        m = meta.get(lid)
        if m is None:
            continue
        rec = a.receptors / f"{m['target']}.pdb"
        if not rec.exists():
            continue
        try:
            feat = features_for_pose(rec, pf)
        except Exception as exc:
            print(f"  fail {lid}: {exc}"); continue
        if feat:
            rows.append({"ligand_id": lid, "target": m["target"], "label": int(m["label"]),
                         "pchembl": m["pchembl"], "decoy_type": m["decoy_type"], **feat})
        if i % 200 == 0:
            print(f"  {i}/{len(pose_files)}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"WROTE {a.out}  n={len(rows)}  n_features={max(0,len(keys)-5)}")


if __name__ == "__main__":
    main()
