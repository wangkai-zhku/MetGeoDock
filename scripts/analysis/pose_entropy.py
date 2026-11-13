"""Information-theoretic pose entropy (ITPE).

Boltzmann weights p_i = exp(-E_i/kT)/Z give the Gibbs entropy
S = -k_B sum p_i ln p_i = k_B * H, so H (nats) IS the entropy in units of k_B.

Two estimators are reported:
  H_cont : one state per docked pose, Boltzmann weighted
  H_disc : poses clustered by RMSD first, weights summed per cluster (more
           robust to pose count)

Binding costs entropy: dG = dH - T(S_bound - S_free). We therefore also estimate
H_free from a free-ligand conformer ensemble using the same estimator and report
    dS_kcal = kT * (H_disc - H_free)
in kcal/mol.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HDAC = Path("/home/wangkai/cursor-task/HDAC")
from metgeodock.dock import dock_ligand  # noqa: E402
from metgeodock.io import read_pdb  # noqa: E402
from metgeodock.screen import iter_library  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

RDLogger.DisableLog("rdApp.*")
KB_KCAL = 0.001987204
T_DEFAULT = 300.0
CLUSTER_RMSD = 2.0


def boltzmann_entropy(energies, T=T_DEFAULT):
    E = np.asarray(energies, float)
    if len(E) == 0:
        return np.nan, 0
    E = E - E.min()
    w = np.exp(-E / (KB_KCAL * T))
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        return np.nan, len(E)
    p = w / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()), len(E)


def cluster_entropy(poses, T=T_DEFAULT):
    if not poses:
        return np.nan, 0, 0
    E = np.array([p.score.total for p in poses], float)
    E = E - E.min()
    w = np.exp(-E / (KB_KCAL * T))
    reps, assign = [], []
    for p in poses:
        idx = None
        for k, r in enumerate(reps):
            a, b = p.ligand, r
            n = min(len(a.xyz), len(b.xyz))
            if n and np.sqrt(((a.xyz[:n] - b.xyz[:n]) ** 2).sum(axis=1).mean()) <= CLUSTER_RMSD:
                idx = k
                break
        if idx is None:
            reps.append(p.ligand)
            idx = len(reps) - 1
        assign.append(idx)
    pc = np.zeros(len(reps))
    for i, c in enumerate(assign):
        pc[c] += w[i]
    pc = pc / pc.sum()
    pc = pc[pc > 0]
    return float(-(pc * np.log(pc)).sum()), len(reps), len(poses)


def free_entropy(smiles, n_conf=20, T=T_DEFAULT, seed=0xF00D):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=ps)
    if not cids:
        return np.nan
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
        energies = np.array([r[1] for r in res], float)
    except Exception:
        energies = np.zeros(len(cids))
    entropy, _ = boltzmann_entropy(energies, T)
    return entropy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--library", type=Path, default=HDAC / "docking/libraries")
    ap.add_argument("--receptors", type=Path, default=HDAC / "structures/receptors_h")
    ap.add_argument("--dataset", type=Path, default=HDAC / "data/processed/hdac_dataset.csv")
    ap.add_argument("--actives-only", action="store_true")
    ap.add_argument("--n-spin", type=int, default=8)
    ap.add_argument("--max-poses", type=int, default=15)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    meta = {r["ligand_id"]: r for r in csv.DictReader(open(a.dataset))}
    rows = []
    for t in a.targets:
        lib = a.library / f"{t}_library.sdf"
        rec = a.receptors / f"{t}.pdb"
        if not lib.exists() or not rec.exists():
            print(f"{t}: missing input", flush=True)
            continue
        protein = read_pdb(rec)
        n = 0
        for lid, ligand in iter_library(lib):
            m = meta.get(lid)
            if not m:
                continue
            if a.actives_only and m["label"] != "1":
                continue
            try:
                res = dock_ligand(protein, ligand, metal="zn", n_spin=a.n_spin,
                                  max_poses=a.max_poses)
                if not res.poses:
                    continue
                E = [p.score.total for p in res.poses]
                H_cont, _ = boltzmann_entropy(E)
                H_disc, k_cl, n_tot = cluster_entropy(res.poses)
                H_free = free_entropy(m["smiles"])
            except Exception:
                continue
            dS = KB_KCAL * T_DEFAULT * ((H_disc if np.isfinite(H_disc) else 0.0)
                                        - (H_free if np.isfinite(H_free) else 0.0))
            rows.append({
                "ligand_id": lid, "target": t, "label": int(m["label"]),
                "pchembl": m["pchembl"], "E_best": round(float(min(E)), 4),
                "H_cont": round(H_cont, 4) if np.isfinite(H_cont) else "",
                "H_disc": round(H_disc, 4) if np.isfinite(H_disc) else "",
                "n_clusters": k_cl, "n_poses": n_tot,
                "H_free": round(H_free, 4) if np.isfinite(H_free) else "",
                "dS_kcal": round(dS, 4),
            })
            n += 1
        print(f"{t}: {n} ligands", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"WROTE {a.out} n={len(rows)}")


if __name__ == "__main__":
    main()
