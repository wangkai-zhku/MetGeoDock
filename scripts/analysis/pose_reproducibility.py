"""Is the pose sampling reproducible and sufficient?

Two checks on the same ligand/receptor pair:
  1. determinism  - run the identical call twice; poses must be identical
  2. sampling     - n_spin=6/max_poses=8 (what every earlier test used) versus
                    n_spin=18/max_poses=40 (library default). If the best score
                    or the pose itself moves, all earlier numbers were sampled
                    too thinly to be stable.
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

REC = ROOT / "processed/hdac_receptors/3MAX_h.pdb"   # HDAC2, stable class I pocket


def heavy_xyz(pose):
    xyz = np.asarray(pose.ligand.xyz, float)
    lm = pose.ligand.heavy_mask()
    return xyz[lm] if lm.any() else xyz


def main():
    meta = {r["molecule_id"]: r for r in csv.DictReader(
        open(ROOT / "processed/hdac_cap_ligands.csv"))}
    mid = [m for m, r in meta.items() if r["target_subtype"] == "HDAC8"][0]
    mol = Chem.AddHs(Chem.MolFromSmiles(meta[mid]["smiles"]))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    mol = Chem.RemoveHs(mol)
    prot = read_pdb(str(REC))
    print(f"配体 {mid}  受体 HDAC2 (3MAX)  重原子 {mol.GetNumAtoms()}", flush=True)

    def run(n_spin, max_poses):
        res = dock_ligand(prot, molecule_from_rdkit(mol, mid), metal="zn",
                          n_spin=n_spin, max_poses=max_poses)
        best = min(res.poses, key=lambda p: p.score.total)
        return best, len(res.poses)

    print("\n[1] 确定性：同参数跑两次", flush=True)
    a, na = run(6, 8)
    b, nb = run(6, 8)
    xa, xb = heavy_xyz(a), heavy_xyz(b)
    same_shape = xa.shape == xb.shape
    rmsd = float(np.sqrt(((xa - xb) ** 2).sum(1).mean())) if same_shape else float("nan")
    print(f"  n_pose: {na} vs {nb}", flush=True)
    print(f"  total : {a.score.terms.get('total', 0):.6e} vs {b.score.terms.get('total', 0):.6e}", flush=True)
    print(f"  pose RMSD = {rmsd:.4f} A   -> {'确定性可复现' if rmsd < 1e-6 else '存在随机性！'}", flush=True)

    print("\n[2] 采样充分性：n_spin=6/8poses  vs  18/40poses", flush=True)
    c, nc = run(18, 40)
    xc = heavy_xyz(c)
    r6 = float(np.sqrt(((xa - xc) ** 2).sum(1).mean())) if xa.shape == xc.shape else float("nan")
    print(f"  稀疏采样 n_pose={na}  total={a.score.terms.get('total', 0):.6e}  "
          f"steric={a.score.terms.get('steric', 0):.2f}", flush=True)
    print(f"  充分采样 n_pose={nc}  total={c.score.terms.get('total', 0):.6e}  "
          f"steric={c.score.terms.get('steric', 0):.2f}", flush=True)
    print(f"  两个最优 pose 的 RMSD = {r6:.2f} A", flush=True)

    # how many of the rich poses exist only in the fuller run
    print("\n[3] 充分采样下，接触富集残基的 pose 比例", flush=True)
    metal = None
    for i in range(len(prot)):
        if str(prot.resnames[i]).strip().upper().startswith("ZN"):
            metal = np.asarray(prot.xyz[i], float); break
    per = {}
    for i in range(len(prot)):
        if str(prot.elements[i]).strip().upper() == "H":
            continue
        k = str(prot.resseqs[i])
        per.setdefault(k, {"rn": str(prot.resnames[i]).strip().upper(), "xyz": []})
        per[k]["xyz"].append(np.asarray(prot.xyz[i], float))
    cap = {k: np.array(v["xyz"]) for k, v in per.items()
           if 8.0 <= float(np.linalg.norm(np.array(v["xyz"]) - metal, axis=1).min()) <= 16.0}
    for label, n_spin, mp in (("稀疏 6/8", 6, 8), ("充分 18/40", 18, 40)):
        res = dock_ligand(prot, molecule_from_rdkit(mol, f"{mid}_{label}"),
                          metal="zn", n_spin=n_spin, max_poses=mp)
        ncon = []
        for p in res.poses:
            xyz = heavy_xyz(p)
            k = sum(1 for arr in cap.values()
                    if np.sqrt(((xyz[:, None, :] - arr[None, :, :]) ** 2).sum(-1)).min() < 4.5)
            ncon.append(k)
        print(f"  {label:<10} poses={len(res.poses):>2}  cap 接触数 "
              f"min={min(ncon)} 中位={int(np.median(ncon))} max={max(ncon)}", flush=True)


if __name__ == "__main__":
    main()
