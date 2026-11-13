"""Structure-only calibration of the entropy weights (no activity data at all).

For each selected crystal complex P (protein + its co-crystal ligand L_native):
  * build a small screening library: L_native + N decoy ligands taken from OTHER
    crystal complexes (all are metal-binding ligands, so chemistry is comparable)
  * dock every ligand into P -> E (MetGeo), and per ligand the conformational
    features H_free (ligand only) and n_clusters (distinct binding modes in P)
  * label: 1 for the ligand that is actually in the crystal, 0 for the rest

alpha/beta are then fitted so that the native ligand ranks first, i.e. the
training signal is "can this score recover the experimentally observed ligand",
which is purely structural and independent of any IC50/Ki data.
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


def h_free_from_mol(mol, n_conf=N_CONF, seed=0xF00D):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    if mol is None:
        return np.nan
    m = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    try:
        signal.alarm(4)
        cids = AllChem.EmbedMultipleConfs(m, numConfs=n_conf, params=ps)
        signal.alarm(0)
    except Exception:
        signal.alarm(0)
        return np.nan
    if not cids:
        return np.nan
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=100)
        return boltzmann_entropy([r[1] for r in res])
    except Exception:
        return np.nan


def ligand_mol(mol2_path: Path):
    from rdkit import Chem
    try:
        m = Chem.MolFromMol2File(str(mol2_path), removeHs=True, sanitize=True)
    except Exception:
        m = None
    if m is None:
        try:
            m = Chem.MolFromMol2File(str(mol2_path), removeHs=False, sanitize=False)
        except Exception:
            m = None
    return m


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


def run_one(rec, decoy_mols, pref_dir: Path, work: Path):
    """Dock native + decoys into one receptor; return per-ligand features."""
    from metgeodock.dock import dock_ligand, redock_rmsd
    from metgeodock.io import read_pdb, molecule_from_rdkit
    from metgeodock.prepare import prepare_receptor

    metal = rec["metal"]
    src_pro = ROOT / rec["protein_path"]
    holo = work / f"{rec['pdb_id']}_{rec['rec_id']}_h.pdb"
    if not holo.exists():
        prepare_receptor(src_pro, out=holo, metal=metal)
    protein = read_pdb(holo)

    # native ligand
    nat_mol = ligand_mol(ROOT / rec["ligand_path"])
    if nat_mol is None:
        return None
    rows = []
    native_lig = None
    for k, mol in enumerate([nat_mol] + decoy_mols):
        is_native = k == 0
        try:
            ligand = molecule_from_rdkit(mol, f"lig{k}")
            res = dock_ligand(protein, ligand, metal=metal, n_spin=8, max_poses=12)
        except Exception:
            continue
        if not res.poses:
            continue
        E = min(p.score.total for p in res.poses)
        nc = cluster_count(res.poses)
        hf = h_free_from_mol(mol)
        blocked = 0
        if is_native:
            try:
                blocked = int(min(redock_rmsd(p.ligand, ligand) for p in res.poses) <= 2.0)
            except Exception:
                blocked = 0
            native_lig = ligand
        rows.append({"pdb_id": rec["pdb_id"], "metal": metal, "ligand": f"lig{k}",
                     "is_native": int(is_native), "E": E, "n_clusters": nc,
                     "H_free": hf, "native_recoverable": blocked})
    return rows if len(rows) >= 5 else None


def zsc(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-9)


def evaluate(groups, alpha, beta):
    """Mean normalised rank of the native ligand (lower is better)."""
    ranks = []
    for rows in groups:
        keep = [r for r in rows if np.isfinite(r["E"]) and np.isfinite(r["H_free"]) and np.isfinite(r["n_clusters"])]
        if len(keep) < 5 or sum(r["is_native"] for r in keep) != 1:
            continue
        E = zsc([r["E"] for r in keep])
        H = zsc([r["H_free"] for r in keep])
        N = zsc([r["n_clusters"] for r in keep])
        s = E + alpha * H - beta * N
        order = np.argsort(s)
        pos = [i for i, r in enumerate(keep) if r["is_native"] == 1][0]
        rank = int(np.where(order == pos)[0][0])
        ranks.append(rank / (len(keep) - 1))
    if not ranks:
        return float("nan"), 0
    return float(np.mean(ranks)), len(ranks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, default=ROOT / "processed/crystal/inventory.csv")
    ap.add_argument("--per-metal", type=int, default=10)
    ap.add_argument("--n-decoy", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "results/structure_calibration.json")
    ap.add_argument("--only-metal", default=None, help="process only this metal (parallel mode)")
    ap.add_argument("--dump-groups", type=Path, default=None, help="write per-group features here")
    ap.add_argument("--decoy-pool", type=Path, default=None, help="json list of ligand_path for cross-decoy pool")
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    rng = random.Random(a.seed)
    inv = [r for r in csv.DictReader(open(a.inventory)) if r.get("complete") == "1"]
    by_metal = {}
    for r in inv:
        by_metal.setdefault(r["metal"], []).append(r)
    picked = []
    for m, lst in by_metal.items():
        rng.shuffle(lst)
        picked += lst[: a.per_metal]
    if a.only_metal:
        picked = [r for r in picked if r["metal"] == a.only_metal]
    print(f"selected {len(picked)} complexes across {len(by_metal)} metals", flush=True)

    all_mols = {}
    for r in picked:
        m = ligand_mol(ROOT / r["ligand_path"])
        if m is not None:
            all_mols[r["pdb_id"]] = m
    print(f"parsed {len(all_mols)} ligands", flush=True)

    work = Path("/tmp/structure_calib")
    work.mkdir(exist_ok=True)
    groups = []
    for i, r in enumerate(picked, 1):
        pid = r["pdb_id"]
        if pid not in all_mols:
            continue
        if a.decoy_pool and a.decoy_pool.exists():
            pool_paths = [x for x in json.loads(a.decoy_pool.read_text()) if x != r["ligand_path"]]
            rng.shuffle(pool_paths)
            decoys = []
            for lp in pool_paths[: a.n_decoy * 2]:
                m = ligand_mol(ROOT / lp)
                if m is not None:
                    decoys.append(m)
                if len(decoys) >= a.n_decoy:
                    break
        else:
            pool = [m for k, m in all_mols.items() if k != pid]
            rng.shuffle(pool)
            decoys = pool[: a.n_decoy]
        try:
            rows = run_one(r, decoys, work.parent, work)
        except Exception as exc:
            print(f"  [{i}/{len(picked)}] {pid} FAIL {type(exc).__name__}: {str(exc)[:60]}", flush=True)
            continue
        if rows:
            groups.append(rows)
            nat = [x for x in rows if x["is_native"] == 1][0]
            print(f"  [{i}/{len(picked)}] {r['metal']:4s} {pid}: {len(rows)} ligands, "
                  f"native E={nat['E']:.3g} ncl={nat['n_clusters']:.0f} "
                  f"recoverable={nat['native_recoverable']}", flush=True)

    print(f"\nusable groups: {len(groups)}", flush=True)
    if a.dump_groups:
        a.dump_groups.parent.mkdir(parents=True, exist_ok=True)
        a.dump_groups.write_text(json.dumps(groups))
        print(f"WROTE groups -> {a.dump_groups}")
        return
    base, n0 = evaluate(groups, 0.0, 0.0)
    print(f"baseline (E only): mean normalised native rank = {base:.3f} (n={n0})")

    grid = np.arange(0.0, 6.01, 0.5)
    best = None
    for al in grid:
        for be in grid:
            v, n = evaluate(groups, al, be)
            if np.isfinite(v) and (best is None or v < best[0]):
                best = (v, al, be, n)
    v, al, be, n = best
    print(f"best: alpha={al:.1f}, beta={be:.1f} -> mean rank {v:.3f} (n={n})")
    print(f"improvement: {base:.3f} -> {v:.3f}  ({100*(base-v)/base:+.1f}% relative)")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "n_groups": len(groups), "n_evaluated": n,
        "baseline_mean_rank_E_only": round(base, 4),
        "calibrated": {"alpha": float(al), "beta": float(be),
                       "mean_rank": round(v, 4),
                       "relative_improvement_pct": round(100 * (base - v) / base, 2)},
        "note": "labels are crystal ligands only; no activity data used",
        "groups": groups,
    }, indent=2))
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
