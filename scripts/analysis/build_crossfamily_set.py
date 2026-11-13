"""Cross-family training data: metal enzymes EXCLUDING the HDAC family.

Purpose: fit b and c (with a = 1) for
    score = E_metalgeo + b*H_free - c*n_clusters
on a broad set of metalloenzymes, then validate on HDAC without ever touching HDAC
activity during training.

For each selected target:
  actives : pactivity >= 6
  decoys  : pactivity <= 5 (real weak binders of the SAME target, not random molecules)
  cap     : N_ACT actives + N_DEC decoys
Features per ligand: E (docked), H_free (free-ligand entropy), n_clusters (binding modes).
"""
from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
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


def h_free_mol(mol, cache, key):
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
    ap.add_argument("--activities", type=Path,
                    default=ROOT / "processed/chembl/curated/all_targets_molecules.csv")
    ap.add_argument("--pdb-dir", type=Path, default=Path("/tmp/xtal_train"), help="where downloaded PDBs live")
    ap.add_argument("--n-act", type=int, default=40)
    ap.add_argument("--n-dec", type=int, default=40)
    ap.add_argument("--per-metal", type=int, default=2, help="targets per metal")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only-metal", default=None, help="restrict to one metal (parallel mode)")
    ap.add_argument("--valid-list", type=Path, default=None,
                    help="file with pdb,metal lines of receptors that truly contain the ion")
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    valid = set()
    if a.valid_list and a.valid_list.exists():
        for ln in a.valid_list.read_text().splitlines():
            if ln.strip():
                p_, m_ = ln.split(",")
                valid.add((p_.strip(), m_.strip()))
    rows = list(csv.DictReader(open(a.activities)))
    by_target = defaultdict(list)
    meta = {}
    for r in rows:
        name = r.get("target_name", "")
        if "HDAC" in name.upper() or "SIRT" in name.upper():
            continue                                   # HDAC stays held out
        if not r.get("pdb") or r["pdb"] in ("", "None"):
            continue
        if valid and (r["pdb"], r["metal"]) not in valid:
            continue
        by_target[r["target_chembl_id"]].append(r)
        meta[r["target_chembl_id"]] = {"name": name, "metal": r["metal"], "pdb": r["pdb"]}

    by_metal = defaultdict(list)
    for tid, lst in by_target.items():
        if len(lst) < 20:
            continue
        by_metal[meta[tid]["metal"]].append((tid, len(lst)))
    chosen = []
    for m, lst in by_metal.items():
        if a.only_metal and m != a.only_metal:
            continue
        lst.sort(key=lambda x: -x[1])
        chosen += [(tid, m) for tid, _ in lst[: a.per_metal]]
    print(f"training targets: {len(chosen)} across {len(by_metal)} metals", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for tid, metal in chosen:
        lst = by_target[tid]
        name, pdb = meta[tid]["name"], meta[tid]["pdb"]
        act = [r for r in lst if float(r["pactivity"] or 0) >= 6][: a.n_act]
        dec = [r for r in lst if float(r["pactivity"] or 9) <= 5][: a.n_dec]
        if len(act) < 10 or len(dec) < 10:
            print(f"  {name}: skipped (act={len(act)}, dec={len(dec)})", flush=True)
            continue

        pro = a.pdb_dir / f"{pdb}.pdb"
        if not pro.exists():
            print(f"  {name}: missing {pro}", flush=True)
            continue
        try:
            from metgeodock.prepare import prepare_receptor
            from metgeodock.io import read_pdb
            holo = a.pdb_dir / f"{pdb}_h.pdb"
            if not holo.exists():
                prepare_receptor(pro, out=holo, metal=metal)
            protein = read_pdb(holo)
        except Exception as exc:
            print(f"  {name}: receptor prep failed {type(exc).__name__}: {str(exc)[:60]}", flush=True)
            continue

        from metgeodock.dock import dock_ligand
        from metgeodock.io import molecule_from_rdkit
        cache, ok = {}, 0
        for label, subset in ((1, act), (0, dec)):
            for r in subset:
                smi = r.get("canonical_smiles", "")
                mol = Chem.MolFromSmiles(smi) if smi else None
                if mol is None:
                    continue
                if not (8 <= mol.GetNumHeavyAtoms() <= 50):
                    continue
                try:
                    m3 = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(m3, randomSeed=0xF00D)
                    m3 = Chem.RemoveHs(m3)
                    ligand = molecule_from_rdkit(m3, r["molecule_chembl_id"])
                    res = dock_ligand(protein, ligand, metal=metal, n_spin=6, max_poses=8)
                    if not res.poses:
                        continue
                    out_rows.append({
                        "target_id": tid, "target_name": name, "metal": metal,
                        "ligand_id": r["molecule_chembl_id"], "label": label,
                        "pactivity": float(r["pactivity"]),
                        "E": float(min(p.score.total for p in res.poses)),
                        "H_free": float(h_free_mol(m3, cache, r["molecule_chembl_id"])),
                        "n_clusters": float(cluster_count(res.poses)),
                    })
                    ok += 1
                except Exception:
                    continue
        print(f"  {metal:4s} {name:12s} ({pdb}): {ok} ligands", flush=True)

    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["target_id", "target_name", "metal", "ligand_id",
                                           "label", "pactivity", "E", "H_free", "n_clusters"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"WROTE {a.out} rows={len(out_rows)}")


if __name__ == "__main__":
    main()
