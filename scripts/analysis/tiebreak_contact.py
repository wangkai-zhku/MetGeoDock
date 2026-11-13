"""Tie-break for saturated scores: rank by contact count inside the saturated band.

Diagnosis: for deep-pocket targets every ligand reaches the same floor (E ~ 1e-31)
because a hydroxamate oxygen can always be placed at the MetGeo mode distance.
The ordering inside that band is then pure float noise.

Fix: keep E as the primary key, but when E is below SAT_FLOOR (i.e. saturated), break
ties by the number of protein-ligand contacts of the best pose - a quantity that is
NOT saturated and physically means "how much of the pocket is filled".

Only the saturated band is affected, so normal (non-saturated) parts of the ranking
are untouched.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SAT_FLOOR = 1e-18


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
    ap.add_argument("--target", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--metal", default="zn")
    ap.add_argument("--n-act", type=int, default=25)
    ap.add_argument("--n-dec", type=int, default=25)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    from metgeodock.dock import dock_ligand
    from metgeodock.io import read_pdb, molecule_from_rdkit
    from metgeodock.prepare import prepare_receptor

    holo = Path("/tmp/xtal_train") / f"{a.pdb}_h.pdb"
    if not holo.exists():
        prepare_receptor(Path("/tmp/xtal_train") / f"{a.pdb}.pdb", out=holo, metal=a.metal)
    protein = read_pdb(holo)

    rows = [r for r in csv.DictReader(open(a.activities)) if r.get("target_name") == a.target]
    act = [r for r in rows if (float(r["pactivity"]) if r["pactivity"] else 0) >= 6][: a.n_act]
    dec = [r for r in rows if (float(r["pactivity"]) if r["pactivity"] else 9) <= 5][: a.n_dec]
    recs = []
    for label, subset in ((1, act), (0, dec)):
        for r in subset:
            smi = r.get("canonical_smiles", "")
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None or not (8 <= mol.GetNumHeavyAtoms() <= 50):
                continue
            recs.append({"label": label, "smiles": smi, "id": r["molecule_chembl_id"]})
    print(f"{a.target}: {len(recs)} ligands (act {sum(r['label'] for r in recs)})", flush=True)

    out = []
    for i, rec in enumerate(recs, 1):
        mol = Chem.MolFromSmiles(rec["smiles"])
        try:
            m = Chem.AddHs(mol)
            AllChem.EmbedMolecule(m, randomSeed=0xF00D)
            m = Chem.RemoveHs(m)
            lig = molecule_from_rdkit(m, rec["id"])
            res = dock_ligand(protein, lig, metal=a.metal, n_spin=6, max_poses=8)
            if not res.poses:
                continue
            best = min(res.poses, key=lambda p: p.score.total)
            out.append({"target": a.target, "ligand": rec["id"], "label": rec["label"],
                        "E": float(best.score.total),
                        "contact": float(best.score.terms.get("contact", 0.0)),
                        "steric": float(best.score.terms.get("steric", 0.0)),
                        "n_coord_lig": float(best.score.terms.get("n_coord_lig", 0.0))})
        except Exception:
            continue
        if i % 10 == 0:
            print(f"  {i}/{len(recs)}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    y = [r["label"] for r in out]
    E = np.array([r["E"] for r in out], float)
    C = np.array([r["contact"] for r in out], float)
    sat = E < SAT_FLOOR
    print(f"\n=== {a.target}: n={len(out)}  saturated={int(sat.sum())} ({100*sat.mean():.0f}%) ===")

    print(f"  E only                AUROC={auroc(y, E):.3f}  EF1%={ef(y, E, 0.01):.2f}")
    print(f"  E, tie-break -contact AUROC={auroc(y, E - 0.0*C):.3f} (same primary)")

    # tiered ranking: saturated band ordered by -contact, rest by E
    order = sorted(range(len(out)), key=lambda i: (0, -C[i]) if sat[i] else (1, E[i]))
    rank_pos = {idx: k for k, idx in enumerate(order)}
    y_sorted = [y[i] for i in order]
    print(f"  tiered (sat by -contact)  AUROC={auroc(y_sorted, list(range(len(y_sorted)))):.3f}  "
          f"EF1%={ef(y_sorted, list(range(len(y_sorted))), 0.01):.2f}")

    # continuous blend inside the saturated band only
    for lam in (0.0, 0.05, 0.1, 0.2, 0.5):
        cz = (C - C.mean()) / (C.std() + 1e-9)
        s = np.where(sat, -lam * cz, 0.0) + (E - E.min()) / (E.max() - E.min() + 1e-30)
        print(f"  blend lambda={lam:<4}        AUROC={auroc(y, s):.3f}  EF1%={ef(y, s, 0.01):.2f}")
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
