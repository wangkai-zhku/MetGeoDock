"""Do the poses actually differ between closed (1T64) and open (2V5W) HDAC8?

The earlier gating test only compared scores, which was the wrong check: identical
steric can hide completely different poses. Here the poses themselves are compared:

  pose RMSD          same ligand, same atom order -> direct coordinate RMSD
  span               end-to-end distance (is the ligand stretched out?)
  far / centroid     distance from the metal (does it reach into the cap region?)
  cap contacts       residues within 4.5 A of the ligand that lie 8-16 A from Zn
  pocket volume use  number of receptor residues within 4.5 A

If the open conformer lets the ligand extend further or touch more cap residues,
the gating matters even when steric is zero for both.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
sys.path.insert(0, str(ROOT))
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
from metgeodock.dock import dock_ligand
from metgeodock.io import molecule_from_rdkit, read_pdb

CONF = {"1T64_closed": ROOT / "processed/hdac_receptors/1T64_h.pdb",
        "2V5W_open": ROOT / "processed/hdac_receptors/2V5W_A_h.pdb"}


def pocket_info(prot):
    metal = None
    for i in range(len(prot)):
        if str(prot.resnames[i]).strip().upper().startswith("ZN"):
            metal = np.asarray(prot.xyz[i], float); break
    per = {}
    for i in range(len(prot)):
        el = str(prot.elements[i]).strip().upper()
        if el == "H":
            continue
        k = str(prot.resseqs[i])
        per.setdefault(k, []).append(np.asarray(prot.xyz[i], float))
    cap = {k: np.array(v) for k, v in per.items()
           if 8.0 <= float(np.linalg.norm(np.array(v) - metal, axis=1).min()) <= 16.0}
    allres = {k: np.array(v) for k, v in per.items()}
    return metal, cap, allres


def main():
    meta = {r["molecule_id"]: r for r in csv.DictReader(
        open(ROOT / "processed/hdac_cap_ligands.csv"))}
    picks = [m for m, r in meta.items() if r["target_subtype"] == "HDAC8"][:4]
    P = {k: read_pdb(str(v)) for k, v in CONF.items()}
    INFO = {k: pocket_info(v) for k, v in P.items()}

    print(f"HDAC8 选择性抑制剂 {len(picks)} 个：pose 层面的比较", flush=True)
    for mid in picks:
        m = Chem.AddHs(Chem.MolFromSmiles(meta[mid]["smiles"]))
        if AllChem.EmbedMolecule(m, randomSeed=0xF00D) != 0:
            continue
        m = Chem.RemoveHs(m)
        poses = {}
        print(f"\n{mid}", flush=True)
        print(f"  {'构象':<14}{'steric':>8}{'端到端':>8}{'远端':>7}{'质心':>7}"
              f"{'cap接触':>8}{'总接触':>7}", flush=True)
        for name, p in P.items():
            try:
                res = dock_ligand(p, molecule_from_rdkit(m, f"{mid}_{name}"),
                                  metal="zn", n_spin=6, max_poses=8)
            except Exception:
                continue
            if not res.poses:
                continue
            best = min(res.poses, key=lambda x: x.score.total)
            xyz = np.asarray(best.ligand.xyz, float)
            lm = best.ligand.heavy_mask()
            xyz = xyz[lm] if lm.any() else xyz
            metal, cap, allres = INFO[name]
            d = np.linalg.norm(xyz - metal, axis=1)
            span = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
            ncap = sum(1 for arr in cap.values()
                       if np.sqrt(((xyz[:, None, :] - arr[None, :, :]) ** 2).sum(-1)).min() < 4.5)
            nall = sum(1 for arr in allres.values()
                       if np.sqrt(((xyz[:, None, :] - arr[None, :, :]) ** 2).sum(-1)).min() < 4.5)
            poses[name] = xyz
            print(f"  {name:<14}{best.score.terms.get('steric', 0):>8.2f}{span:>8.1f}"
                  f"{d.max():>7.1f}{np.linalg.norm(xyz.mean(0) - metal):>7.1f}"
                  f"{ncap:>8}{nall:>7}", flush=True)
        if len(poses) == 2:
            ks = list(poses)
            a, b = poses[ks[0]], poses[ks[1]]
            if a.shape == b.shape:
                rmsd = float(np.sqrt(((a - b) ** 2).sum(1).mean()))
                ca, cb = a.mean(0), b.mean(0)
                print(f"  -> pose RMSD = {rmsd:.1f} A   "
                      f"质心位移 = {float(np.linalg.norm(ca - cb)):.1f} A", flush=True)


if __name__ == "__main__":
    main()
