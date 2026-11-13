"""Free-ligand conformational entropy for a whole SDF library.

H_free = Shannon entropy of the Boltzmann-weighted conformer ensemble, the same
estimator used for the bound side. Only depends on the ligand, not on docking, so
it is cheap and can be precomputed once per library.
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
KB_KCAL = 0.001987204
T_DEFAULT = 300.0
N_CONF = 10


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


def h_free(smiles: str, n_conf=N_CONF, seed=0xF00D):
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
        energies = np.array([r[1] for r in res], float)
    except Exception:
        return np.nan
    return boltzmann_entropy(energies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    supp = Chem.SDMolSupplier(str(a.sdf), removeHs=True, sanitize=True)
    rows, n = [], 0
    for m in supp:
        if m is None:
            continue
        name = m.GetProp("_Name") if m.HasProp("_Name") else f"mol{n}"
        try:
            hf = h_free(Chem.MolToSmiles(m))
        except Exception:
            hf = np.nan
        rows.append({"ligand_id": name, "H_free": round(float(hf), 4) if np.isfinite(hf) else ""})
        n += 1
        if n % 1000 == 0:
            print(f"  {n}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ligand_id", "H_free"])
        w.writeheader()
        w.writerows(rows)
    print(f"WROTE {a.out} n={len(rows)}")


if __name__ == "__main__":
    main()
