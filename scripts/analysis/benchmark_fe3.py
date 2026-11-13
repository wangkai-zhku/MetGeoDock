"""Fe(III) benchmark: CYP450 heme enzymes with the iron extracted from HEM.

Same protocol as the other metals (30 actives pactivity>=6, 30 decoys pactivity<=5,
zinc-family E scoring unchanged), so the result is directly comparable to the
Zn/Ca/Mg/Mn/Fe2/Co/Cu/Ni numbers.
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


class Timeout(Exception):
    pass


def _alarm(sig, frm):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ef(y, s, frac):
    y = np.asarray(y); s = np.asarray(s, float)
    n, na = len(y), int(y.sum())
    if na == 0:
        return float("nan")
    k = max(1, int(round(frac * n)))
    return float((y[np.argsort(s)[:k]].sum() / k) / (na / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activities", type=Path,
                    default=ROOT / "processed/chembl/curated/all_targets_molecules.csv")
    ap.add_argument("--pdb-dir", type=Path, default=Path("/tmp/xtal_train"))
    ap.add_argument("--n-act", type=int, default=30)
    ap.add_argument("--n-dec", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    from metgeodock.dock import dock_ligand
    from metgeodock.io import read_pdb, molecule_from_rdkit
    from metgeodock.prepare import prepare_receptor

    rows = list(csv.DictReader(open(a.activities)))
    by_t = defaultdict(list)
    pdb_of = {}
    for r in rows:
        if r.get("metal") != "fe3":
            continue
        f = a.pdb_dir / f"{r['pdb']}_fe3.pdb"
        if not f.exists():
            continue
        by_t[r["target_name"]].append(r)
        pdb_of[r["target_name"]] = r["pdb"]
    print(f"Fe3 targets with extracted heme iron: {len(by_t)}", flush=True)

    out_rows = []
    for name, lst in sorted(by_t.items()):
        act = [r for r in lst if (float(r["pactivity"]) if r["pactivity"] else 0) >= 6][: a.n_act]
        dec = [r for r in lst if (float(r["pactivity"]) if r["pactivity"] else 9) <= 5][: a.n_dec]
        if len(act) < 10 or len(dec) < 10:
            print(f"  {name}: skipped (act={len(act)}, dec={len(dec)})", flush=True)
            continue
        pdb = pdb_of[name]
        src = a.pdb_dir / f"{pdb}_fe3.pdb"
        holo = a.pdb_dir / f"{pdb}_fe3_h.pdb"
        if not holo.exists():
            try:
                prepare_receptor(src, out=holo, metal="fe3")
            except Exception as exc:
                print(f"  {name}: prep failed {str(exc)[:50]}", flush=True)
                continue
        protein = read_pdb(holo)
        ok = 0
        for label, subset in ((1, act), (0, dec)):
            for r in subset:
                smi = r.get("canonical_smiles", "")
                mol = Chem.MolFromSmiles(smi) if smi else None
                if mol is None or not (8 <= mol.GetNumHeavyAtoms() <= 50):
                    continue
                try:
                    m3 = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(m3, randomSeed=0xF00D)
                    m3 = Chem.RemoveHs(m3)
                    lig = molecule_from_rdkit(m3, r["molecule_chembl_id"])
                    res = dock_ligand(protein, lig, metal="fe3", n_spin=6, max_poses=8)
                    if not res.poses:
                        continue
                    out_rows.append({"target": name, "metal": "fe3",
                                     "ligand": r["molecule_chembl_id"], "label": label,
                                     "pactivity": float(r["pactivity"]),
                                     "E": float(min(p.score.total for p in res.poses))})
                    ok += 1
                except Exception:
                    continue
        print(f"  {name:10s} ({pdb}): {ok} ligands", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["target", "metal", "ligand", "label", "pactivity", "E"])
        w.writeheader(); w.writerows(out_rows)

    print(f"\n=== Fe3 per-target ===")
    per = defaultdict(list)
    for r in out_rows:
        per[r["target"]].append(r)
    aucs = []
    for t, rs in sorted(per.items()):
        y = [r["label"] for r in rs]; s = [r["E"] for r in rs]
        if len(set(y)) < 2 or len(rs) < 15:
            continue
        au = auroc(y, s); aucs.append(au)
        print(f"  {t:10s} n={len(rs):3d} act={sum(y):3d} AUROC={au:.3f} "
              f"EF1%={ef(y, s, 0.01):.2f} EF5%={ef(y, s, 0.05):.2f}")
    if aucs:
        print(f"\n  Fe3 mean AUROC = {np.mean(aucs):.3f}  (n={len(aucs)} targets)")
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
