"""Compute H_free and n_clusters for every ligand of the cross-family benchmark.

Both quantities were only available for HDAC so far. They are needed to fit the
per-pocket-class coefficients b (entropy) and c (conformational spread).

n_clusters : RMSD clustering (1.5 A) of the docking poses returned for one ligand
H_free     : conformer free energy (ETKDG ensemble + MMFF), same definition as
             scripts/analysis/free_entropy_lib.py

Writes results/entropy_big/{metal}_{target}.csv with one row per ligand.
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CLUSTER_RMSD = 1.5


def _alarm(sig, frm):
    raise TimeoutError()


signal.signal(signal.SIGALRM, _alarm)


def rmsd(a, b):
    """RMSD with optimal superposition (Kabsch)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(0); b = b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt((((a @ rot) - b) ** 2).sum(1).mean()))


def cluster_count(conformers, cutoff=CLUSTER_RMSD):
    """Number of distinct binding modes (RMSD clustering).

    Degenerate cases return at least 1 whenever poses exist, so a numerical
    failure in the superposition can never masquerade as "zero clusters".
    """
    confs = []
    for c in conformers:
        try:
            a = np.asarray(c, float)
            if a.ndim == 2 and a.shape[1] == 3 and np.isfinite(a).all():
                confs.append(a)
        except Exception:
            continue
    if not confs:
        return 0
    reps = [confs[0]]
    for conf in confs[1:]:
        try:
            if all(rmsd(conf, r) > cutoff for r in reps):
                reps.append(conf)
        except Exception:
            continue
    return max(1, len(reps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activities", type=Path,
                    default=ROOT / "processed/chembl/curated/all_targets_molecules.csv")
    ap.add_argument("--pdb-dir", type=Path, default=Path("/tmp/xtal_train"))
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/entropy_big")
    ap.add_argument("--n-act", type=int, default=30)
    ap.add_argument("--n-dec", type=int, default=30)
    ap.add_argument("--n-conf", type=int, default=8)
    ap.add_argument("--max-poses", type=int, default=10)
    ap.add_argument("--metal", default=None)
    ap.add_argument("--target", default=None, help="only process this target")
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    from metgeodock.dock import dock_ligand
    from metgeodock.io import read_pdb, molecule_from_rdkit
    from metgeodock.prepare import prepare_receptor

    # target -> (metal, pdb)
    pdb_of = {}
    metal_of = {}
    for r in csv.DictReader(open(a.activities)):
        if r.get("pdb") and r["pdb"] not in ("", "None"):
            pdb_of.setdefault(r["target_name"], r["pdb"])
            metal_of.setdefault(r["target_name"], r.get("metal", "zn"))

    rows = list(csv.DictReader(open(a.activities)))
    by_t = defaultdict(list)
    for r in rows:
        n = r.get("target_name", "")
        if not n or "HDAC" in n.upper():
            continue
        by_t[n].append(r)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, lst in sorted(by_t.items()):
        pdb, metal = pdb_of.get(name), metal_of.get(name, "zn")
        if not pdb:
            continue
        if a.metal and metal != a.metal:
            continue
        if a.target and name != a.target:
            continue
        src = a.pdb_dir / f"{pdb}_h.pdb"
        if not src.exists():
            src = a.pdb_dir / f"{pdb}.pdb"
            if not src.exists():
                continue
            holo = a.pdb_dir / f"{pdb}_h.pdb"
            try:
                prepare_receptor(src, out=holo, metal=metal)
                src = holo
            except Exception as exc:
                print(f"  {name}: prep failed {str(exc)[:40]}", flush=True)
                continue
        try:
            protein = read_pdb(src)
        except Exception as exc:
            print(f"  {name}: read failed {str(exc)[:40]}", flush=True)
            continue

        act = [r for r in lst if (float(r["pactivity"]) if r["pactivity"] else 0) >= 6][: a.n_act]
        dec = [r for r in lst if (float(r["pactivity"]) if r["pactivity"] else 9) <= 5][: a.n_dec]
        out = []
        for label, subset in ((1, act), (0, dec)):
            for r in subset:
                smi = r.get("canonical_smiles", "")
                mol = Chem.MolFromSmiles(smi) if smi else None
                if mol is None or not (8 <= mol.GetNumHeavyAtoms() <= 50):
                    continue
                lid = r["molecule_chembl_id"]
                # --- conformer free energy (with hard timeout)
                h_free = np.nan
                try:
                    signal.alarm(4)
                    m = Chem.AddHs(mol)
                    ps = AllChem.ETKDGv3(); ps.randomSeed = 0xF00D
                    cids = AllChem.EmbedMultipleConfs(m, numConfs=a.n_conf, params=ps)
                    ens = []
                    for cid in cids:
                        try:
                            mp = AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s")
                            if mp is None:
                                continue
                            ff = AllChem.MMFFGetMoleculeForceField(m, mp, confId=cid)
                            ens.append(ff.CalcEnergy())
                        except Exception:
                            continue
                    if ens:
                        e = np.array(ens, float)
                        p = np.exp(-(e - e.min()) / 0.593)
                        p /= p.sum()
                        h_free = float(-(p * np.log(p + 1e-300)).sum() * 0.593)
                    signal.alarm(0)
                except (TimeoutError, Exception):
                    signal.alarm(0)
                # --- docking: clusters over returned poses
                nclust, nposes, ebest = 0, 0, np.nan
                try:
                    signal.alarm(90)
                    m = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(m, randomSeed=0xF00D)
                    m = Chem.RemoveHs(m)
                    lig = molecule_from_rdkit(m, lid)
                    res = dock_ligand(protein, lig, metal=metal, n_spin=6,
                                      max_poses=a.max_poses)
                    nposes = len(res.poses)
                    if res.poses:
                        ebest = float(min(p.score.total for p in res.poses))
                        nclust = cluster_count([np.asarray(p.ligand.xyz, float) for p in res.poses])
                    signal.alarm(0)
                except (TimeoutError, Exception):
                    signal.alarm(0)
                out.append({"target": name, "metal": metal, "ligand_id": lid,
                            "label": label, "h_free": h_free, "n_clusters": nclust,
                            "n_poses": nposes, "E_best": ebest})
        dest = a.out_dir / f"{metal}_{name}.csv"
        _ok = False
        if dest.exists():
            _rows = list(csv.DictReader(open(dest)))
            _ok = len(_rows) > 20 and sum(1 for r in _rows if r.get("E_best", "nan") not in ("nan", "", "None")) > 10
        if _ok:
            print(f"  {name:14s} skip (done)", flush=True)
            continue
        with dest.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["target", "metal", "ligand_id", "label",
                                               "h_free", "n_clusters", "n_poses", "E_best"])
            w.writeheader(); w.writerows(out)
        print(f"  {name:14s} {metal:4s} n={len(out):3d} -> {dest.name}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
