"""Fit the three score coefficients from crystal structures only (no activity data).

Model
-----
    score = a * E_metalgeo + b * H_free - c * n_clusters      (a, b, c >= 0)

The three quantities are not commensurate (energy-like, entropy-like, a count), so
each gets its own coefficient and they are z-scored within a receptor before use.

Training signal
---------------
For each crystal complex P we build a small library: P's own co-crystal ligand
(label 1) plus N ligands taken from OTHER crystal complexes (label 0). The only
supervision is "which ligand is the one actually observed in the crystal", which
is a structural fact. No IC50/Ki/pChEMBL value is read anywhere in this script.

Fitting
-------
Pairwise logistic regression on feature differences (native vs each decoy), which
yields a, b and c together with a grouped cross-validated estimate of how often the
native ligand ranks first.

Usage:
  structure_triplet.py --metal zn --n-complex 50 --n-decoy 6 --out part_zn.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import signal
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
KB_KCAL = 0.001987204
T_DEFAULT = 300.0
N_CONF = 10
HDAC_RECEPTORS = Path("/home/wangkai/cursor-task/HDAC/structures/receptors_h")


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


def h_free_of(mol, cache, key):
    if key in cache:
        return cache[key]
    from rdkit import Chem
    from rdkit.Chem import AllChem
    val = np.nan
    try:
        m = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 0xF00D
        signal.alarm(4)
        cids = AllChem.EmbedMultipleConfs(m, numConfs=N_CONF, params=ps)
        signal.alarm(0)
        if cids:
            res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=100)
            val = boltzmann_entropy([r[1] for r in res])
    except Exception:
        signal.alarm(0)
    cache[key] = val
    return val


def ligand_mol(path: Path):
    from rdkit import Chem
    for kw in (dict(removeHs=True, sanitize=True), dict(removeHs=False, sanitize=False)):
        try:
            m = Chem.MolFromMol2File(str(path), **kw)
            if m is not None and m.GetNumAtoms() > 0:
                return m
        except Exception:
            continue
    return None


def cluster_count(poses, cutoff=2.0):
    reps = []
    for p in poses:
        best = np.inf
        for r in reps:
            n = min(len(p.ligand.xyz), len(r.xyz))
            if n:
                best = min(best, float(np.sqrt(((p.ligand.xyz[:n] - r.xyz[:n]) ** 2).sum(axis=1).mean())))
        if best > cutoff:
            reps.append(p.ligand)
    return len(reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, default=ROOT / "processed/crystal/inventory.csv")
    ap.add_argument("--pool", type=Path, default=ROOT / "results/decoy_pool.json")
    ap.add_argument("--metal", required=True)
    ap.add_argument("--n-complex", type=int, default=50)
    ap.add_argument("--n-decoy", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from metgeodock.dock import dock_ligand
    from metgeodock.io import read_pdb, molecule_from_rdkit
    from metgeodock.prepare import prepare_receptor

    rng = random.Random(a.seed)
    inv = [r for r in csv.DictReader(open(a.inventory))
           if r.get("complete") == "1" and r["metal"] == a.metal]
    rng.shuffle(inv)
    inv = inv[: a.n_complex]
    pool = json.loads(a.pool.read_text()) if a.pool.exists() else []
    pool_mols = []
    rng.shuffle(pool)
    for lp in pool:
        m = ligand_mol(ROOT / lp)
        if m is not None:
            pool_mols.append((lp, m))
        if len(pool_mols) >= max(40, a.n_decoy * 6):
            break

    work = Path("/tmp/triplet_rec")
    work.mkdir(exist_ok=True)
    hf_cache = {}
    groups = []
    for i, rec in enumerate(inv, 1):
        src_pro = ROOT / rec["protein_path"]
        holo = work / f"{rec['pdb_id']}_{rec['rec_id']}.pdb"
        if not holo.exists():
            try:
                prepare_receptor(src_pro, out=holo, metal=rec["metal"])
            except Exception:
                continue
        try:
            protein = read_pdb(holo)
        except Exception:
            continue

        nat_mol = ligand_mol(ROOT / rec["ligand_path"])
        if nat_mol is None:
            continue
        cand = [(rec["ligand_path"], nat_mol, 1)]
        used = {rec["ligand_path"]}
        for lp, m in pool_mols:
            if lp in used:
                continue
            cand.append((lp, m, 0))
            if len(cand) > a.n_decoy:
                break

        rows = []
        for lp, mol, is_nat in cand:
            try:
                ligand = molecule_from_rdkit(mol, lp)
                res = dock_ligand(protein, ligand, metal=rec["metal"], n_spin=8, max_poses=10)
                if not res.poses:
                    continue
                rows.append({
                    "E": float(min(p.score.total for p in res.poses)),
                    "H_free": float(h_free_of(mol, hf_cache, lp)),
                    "n_clusters": float(cluster_count(res.poses)),
                    "is_native": int(is_nat),
                })
            except Exception:
                continue
        if len(rows) >= 4 and sum(r["is_native"] for r in rows) == 1:
            groups.append({"pdb_id": rec["pdb_id"], "metal": a.metal, "ligands": rows})
            print(f"  [{i}/{len(inv)}] {rec['pdb_id']}: {len(rows)} ligands", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(groups))
    print(f"WROTE {a.out} groups={len(groups)}")


if __name__ == "__main__":
    main()
