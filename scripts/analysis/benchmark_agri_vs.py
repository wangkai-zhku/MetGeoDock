#!/usr/bin/env python3
"""Virtual screen agricultural metalloproteins with MetGeoDock.

Actives are experimental ChEMBL/PubChem pesticides for that target; decoys
are other-MoA pesticides plus weak ChEMBL rows. Success is enrichment of
actives at the top of the MetGeo ranking (no wet-lab). Does not mutate
all-agricultural-dock.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.dock import dock_ligand  # noqa: E402
from metgeodock.io import Molecule, read_pdb  # noqa: E402
from metgeodock.nnscore import coordnet_score_ligand  # noqa: E402

LOG = logging.getLogger("benchmark_agri_vs")


def smiles_to_ligand(smiles: str, name: str, seed: int = 0) -> Molecule | None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed) % 10000
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    AllChem.UFFOptimizeMolecule(mol, maxIters=40)
    conf = mol.GetConformer()
    xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)
    els = np.array([a.GetSymbol() for a in mol.GetAtoms()])
    dummy = np.array(["LIG"] * len(els))
    zeros = np.zeros(len(els))
    seqs = np.zeros(len(els), dtype=int)
    return Molecule(
        name=name,
        xyz=xyz,
        elements=els,
        names=els,
        resnames=dummy,
        resseqs=seqs,
        charges=zeros,
        atom_types=els,
    )


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-s)
    ys = y[order]
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    tpr = np.concatenate([[0.0], tps / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fps / n_neg, [1.0]])
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trap(tpr, fpr))


def enrichment(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    y = np.asarray(y, dtype=int)
    order = np.argsort(-np.asarray(s, dtype=float))
    k = max(1, int(np.ceil(frac * len(y))))
    hit = float(y[order][:k].mean())
    base = float(y.mean())
    return float(hit / base) if base > 0 else float("nan")


def topk_hits(y: np.ndarray, s: np.ndarray, k: int) -> dict:
    y = np.asarray(y, dtype=int)
    order = np.argsort(-np.asarray(s, dtype=float))
    kk = min(int(k), len(y))
    hits = int(y[order][:kk].sum())
    return {"k": kk, "n_actives_in_top": hits, "recall": float(hits / y.sum()) if y.sum() else None}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lib-dir", type=Path, default=Path("processed/agri_vs"))
    p.add_argument("--n-spin", type=int, default=6)
    p.add_argument("--max-poses", type=int, default=6)
    p.add_argument("--coord-water", dest="coord_water", type=int, default=2)
    p.add_argument("--out-csv", type=Path, default=Path("results/agri_vs.csv"))
    p.add_argument("--out-json", type=Path, default=Path("results/agri_vs.json"))
    p.add_argument("--log", type=Path, default=Path("logs/benchmark_agri_vs.log"))
    p.add_argument("--targets", default="hppd,gs,igpd")
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    wanted = [t.strip() for t in str(args.targets).split(",") if t.strip()]
    rows = []
    done = set()
    if args.out_csv.exists():
        prev = pd.read_csv(args.out_csv)
        rows = prev.to_dict(orient="records")
        for rec in rows:
            done.add((str(rec.get("target")), str(rec.get("ligand_id"))))
        LOG.info("resume already_done=%s", len(done))
    by_target = {}
    for tid in wanted:
        pocket = args.lib_dir / tid / "protein_pocket.pdb"
        lib_path = args.lib_dir / tid / "library.csv"
        if not pocket.exists() or not lib_path.exists():
            LOG.warning("skip %s missing files", tid)
            continue
        protein = read_pdb(pocket)
        lib = pd.read_csv(lib_path)
        LOG.info("%s n_lib=%s n_act=%s", tid, len(lib), int((lib["label"] == 1).sum()))
        for rec in lib.itertuples(index=False):
            key = (tid, str(rec.ligand_id))
            if key in done:
                continue
            try:
                lig = smiles_to_ligand(str(rec.smiles), str(rec.ligand_id))
                if lig is None:
                    rows.append({"target": tid, "ligand_id": rec.ligand_id, "label": rec.label, "error": "embed_fail"})
                    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
                    continue
                result = dock_ligand(
                    protein,
                    lig,
                    metal=str(rec.metal),
                    n_spin=args.n_spin,
                    max_poses=args.max_poses,
                    keep_water=False,
                    multi_metal=True,
                    max_pocket_metals=2,
                    coord_water=int(args.coord_water),
                )
            except Exception as exc:  # noqa: BLE001
                rows.append({"target": tid, "ligand_id": rec.ligand_id, "label": rec.label, "error": str(exc)})
                pd.DataFrame(rows).to_csv(args.out_csv, index=False)
                continue
            if not result.poses:
                rows.append({"target": tid, "ligand_id": rec.ligand_id, "label": rec.label, "error": "no_pose"})
                pd.DataFrame(rows).to_csv(args.out_csv, index=False)
                continue
            best = result.best
            p_act = coordnet_score_ligand(result.sphere.metal_name, smiles=str(rec.smiles), ligand=best.ligand, metal_xyz=result.sphere.metal_xyz)
            t = best.score.terms
            rec_out = {
                "target": tid,
                "ligand_id": rec.ligand_id,
                "name": getattr(rec, "name", rec.ligand_id),
                "source": getattr(rec, "source", ""),
                "label": int(rec.label),
                "pchembl": getattr(rec, "pchembl", float("nan")),
                "score": float(best.score.total),
                "metalgeo_higher": float(-best.score.total),
                "coordnet": float(p_act) if np.isfinite(p_act) else float("nan"),
                "n_coord_lig": int(best.score.n_coord_lig),
                "n_coord_water": float(t.get("n_coord_water", t.get("n_water", 0.0))),
                "steric": float(t.get("steric", best.score.clash)),
                "contact": float(t.get("contact", 0.0)),
                "centroid_pen": float(t.get("centroid_pen", 0.0)),
                "map_overlap": float(t.get("map_overlap", 0.0)),
                "n_clash": float(t.get("n_clash", 0.0)),
                "min_donor_dist": float(t.get("min_donor_dist", 9.0)),
                "motif": best.motif,
                "metal": result.sphere.metal_name,
                "error": "",
            }
            LOG.info("%s %s label=%s score=%.3f", tid, rec.ligand_id, rec.label, rec_out["score"])
            rows.append(rec_out)
            args.out_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        def _ok(rec: dict) -> bool:
            if str(rec.get("target")) != tid:
                return False
            err = rec.get("error")
            if err is None or (isinstance(err, float) and pd.isna(err)):
                return True
            return str(err).strip() in ("", "nan")

        sub = pd.DataFrame([r for r in rows if _ok(r)])
        if sub.empty:
            by_target[tid] = {"n": 0}
            continue
        y = sub["label"].to_numpy(dtype=int)
        s = sub["metalgeo_higher"].to_numpy(dtype=float)
        sc = sub["coordnet"].to_numpy(dtype=float)
        ranked = sub.sort_values("score", kind="mergesort")
        top_names = ranked["name"].astype(str).head(10).tolist()
        top_lab = ranked["label"].astype(int).head(10).tolist()
        named = sub[sub["source"].astype(str).str.contains("named", na=False)].copy()
        named_metrics = None
        if len(named) and named["label"].nunique() > 1:
            yn = named["label"].to_numpy(dtype=int)
            sn = named["metalgeo_higher"].to_numpy(dtype=float)
            named_ranked = named.sort_values("score", kind="mergesort")
            named_metrics = {
                "n": int(len(named)),
                "n_actives": int(yn.sum()),
                "metalgeo_auroc": auroc(yn, sn),
                "metalgeo_top5": topk_hits(yn, sn, 5),
                "metalgeo_top10": topk_hits(yn, sn, 10),
                "top10_names": named_ranked["name"].astype(str).head(10).tolist(),
                "top10_labels": named_ranked["label"].astype(int).head(10).tolist(),
            }
        nw = sub["n_coord_water"] if "n_coord_water" in sub.columns else None
        by_target[tid] = {
            "n": int(len(sub)),
            "n_actives": int(y.sum()),
            "n_decoys": int((1 - y).sum()),
            "frac_best_pose_with_coord_water": float((pd.to_numeric(nw, errors="coerce").fillna(0) > 0).mean()) if nw is not None else None,
            "metalgeo_auroc": auroc(y, s),
            "metalgeo_ef_10pct": enrichment(y, s, 0.10),
            "metalgeo_ef_20pct": enrichment(y, s, 0.20),
            "metalgeo_top5": topk_hits(y, s, 5),
            "metalgeo_top10": topk_hits(y, s, 10),
            "coordnet_auroc": auroc(y, sc) if np.isfinite(sc).any() else None,
            "top10_names": top_names,
            "top10_labels": top_lab,
            "named_pesticides": named_metrics,
        }
    frame = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "by_target": by_target,
        "n_spin": args.n_spin,
        "max_poses": args.max_poses,
        "coord_water": args.coord_water,
        "keep_water": False,
        "csv": str(args.out_csv),
        "note": "Actives are experimental pesticides/ChEMBL pChEMBL>=6; decoys are other-MoA pesticides. Ranking uses MetGeo (lower energy better).",
    }
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("wrote %s", args.out_json)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
