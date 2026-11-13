"""Transfer test: do the alpha/beta fitted on HDAC (Zn) work on Fe/Mn targets?

The agrochemical retrospective set (agri_vs) has 168 molecules against HPPD (Fe),
GS (Mn) and IGPD (Mn) - different metals, different protein family, different
ligands. Nothing here was used to fit the entropy weights, so this is a genuine
out-of-domain test of whether the three-term score generalises.

For each (ligand, target):
  E          : MetGeo score, already in results/agri_vs.csv
  H_free     : free-ligand conformational entropy (ligand only, no docking)
  n_clusters : distinct binding modes from a fresh docking run
Score = z(E) + alpha*z(H_free) - beta*z(n_clusters), z-scored inside each target.
"""
from __future__ import annotations

import argparse
import csv
import math
import signal
from pathlib import Path

import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
KB_KCAL = 0.001987204
T_DEFAULT = 300.0
N_CONF = 10
CLUSTER_RMSD = 2.0


class Timeout(Exception):
    pass


def _alarm(sig, frm):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)


def boltzmann_entropy(energies, T=T_DEFAULT):
    E = np.asarray(energies, float)
    if len(E) == 0:
        return np.nan
    E = E - E.min()
    w = np.exp(-E / (KB_KCAL * T))
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        return np.nan
    p = w / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def h_free(smiles, n_conf=N_CONF, seed=0xF00D):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    try:
        signal.alarm(4)
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=ps)
        signal.alarm(0)
    except Exception:
        signal.alarm(0)
        return np.nan
    if not cids:
        return np.nan
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=100)
        return boltzmann_entropy([r[1] for r in res])
    except Exception:
        return np.nan


def n_clusters_for(protein, ligand_mol, metal=None):
    import sys
    sys.path.insert(0, str(ROOT))
    from metgeodock.dock import dock_ligand
    from metgeodock.io import molecule_from_rdkit
    lig = molecule_from_rdkit(ligand_mol, "lig")
    res = dock_ligand(protein, lig, metal=(metal or None), n_spin=8, max_poses=15)
    poses = res.poses
    if not poses:
        return np.nan, np.nan
    reps = []
    for p in poses:
        best = np.inf
        for r in reps:
            n = min(len(p.ligand.xyz), len(r.xyz))
            if n:
                d = np.sqrt(((p.ligand.xyz[:n] - r.xyz[:n]) ** 2).sum(axis=1).mean())
                best = min(best, d)
        if best > CLUSTER_RMSD:
            reps.append(p.ligand)
    return len(reps), min(p.score.total for p in poses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=ROOT / "results/agri_vs.csv")
    ap.add_argument("--library", type=Path, default=ROOT / "examples/agri_vs_168/library_all.csv")
    ap.add_argument("--pockets", type=Path, default=ROOT / "processed/agri_vs")
    ap.add_argument("--out", type=Path, default=ROOT / "results/transfer_fe_mn.csv")
    ap.add_argument("--target", default=None, help="restrict to one target (for parallelism)")
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    import sys
    sys.path.insert(0, str(ROOT))
    from metgeodock.io import read_pdb

    smi = {r["ligand_id"]: r["smiles"] for r in csv.DictReader(open(a.library))}
    rows = [r for r in csv.DictReader(open(a.scores)) if r.get("error", "") in ("", "None")]
    if a.target:
        rows = [r for r in rows if r["target"] == a.target]
        print(f"restricted to {a.target}: {len(rows)} rows", flush=True)

    hf_cache = {}
    for r in rows:
        lid = r["ligand_id"]
        if lid not in hf_cache:
            hf_cache[lid] = h_free(smi.get(lid, ""))
    print(f"H_free done for {len(hf_cache)} ligands", flush=True)

    print("docking for n_clusters ...", flush=True)
    prot_cache, nc_cache = {}, {}
    out = []
    for i, r in enumerate(rows):
        t, lid = r["target"], r["ligand_id"]
        try:
            E = float(r["score"])
        except Exception:
            continue
        if t not in prot_cache:
            prot_cache[t] = read_pdb(a.pockets / t / "protein_pocket.pdb")
        key = (t, lid)
        if key not in nc_cache:
            mol = Chem.MolFromSmiles(smi.get(lid, ""))
            if mol is None:
                nc_cache[key] = (np.nan, np.nan)
            else:
                try:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
                    mol = Chem.RemoveHs(mol)
                    nc_cache[key] = n_clusters_for(prot_cache[t], mol, r.get("metal"))
                except Exception as exc:
                    if (i + 1) <= 3:
                        print(f"  dock fail {lid}: {type(exc).__name__} {str(exc)[:80]}", flush=True)
                    nc_cache[key] = (np.nan, np.nan)
        nc, e_dock = nc_cache[key]
        out.append({"target": t, "ligand_id": lid, "label": int(r["label"]),
                    "pchembl": r["pchembl"], "E": E,
                    "H_free": hf_cache.get(lid, np.nan), "n_clusters": nc})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"WROTE {a.out} n={len(out)}")


if __name__ == "__main__":
    main()
