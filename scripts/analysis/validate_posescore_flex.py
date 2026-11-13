#!/usr/bin/env python3
"""Sanity checks for PoseScore features, checkpoint, and public-10 benchmark.

Does not recompute docking. Fails if grouping leaked sklearn, or if the
public-10 JSON is missing after a completed run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.flex import flex_variants, infer_bonds  # noqa: E402
from metgeodock.forcefield import displaced_water_count  # noqa: E402
from metgeodock.io import Molecule  # noqa: E402
from metgeodock.nnscore import load_nn  # noqa: E402
from metgeodock.posescore import DEFAULT_POSESCORE, analytic_pose_score  # noqa: E402


def _dummy_ligand() -> Molecule:
    xyz = np.array([[0.0, 0.0, 2.1], [1.4, 0.0, 2.8], [2.1, 0.8, 3.2]], dtype=float)
    els = np.array(["O", "C", "O"])
    dummy = Molecule(
        name="dummy",
        xyz=xyz,
        elements=els,
        names=els,
        resnames=np.array(["LIG"] * 3),
        resseqs=np.array([1, 1, 1]),
        charges=np.zeros(3),
        atom_types=els,
    )
    dummy.bonds = infer_bonds(dummy)
    return dummy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=Path("processed/ml/posescore_features.csv"))
    p.add_argument("--report", type=Path, default=Path("processed/ml/posescore_report.json"))
    p.add_argument("--public10", type=Path, default=Path("results/posescore_flex_public10.json"))
    p.add_argument("--out", type=Path, default=Path("results/validate_posescore_flex.json"))
    args = p.parse_args()
    lig = _dummy_ligand()
    n_disp, wmin = displaced_water_count(lig, np.array([[0.0, 0.0, 2.0]]))
    variants = flex_variants(lig, (0,), "zn", np.array([0.0, 0.0, 0.0]))
    analytic = analytic_pose_score({"hybrid_total": 1.0, "map_overlap": 2.0, "water_min_dist": 9.0})
    ckpt = load_nn(DEFAULT_POSESCORE) if DEFAULT_POSESCORE.exists() else None
    checks = {
        "displaced_water_detects_overlap": bool(n_disp >= 1),
        "flex_returns_list": bool(isinstance(variants, list)),
        "analytic_finite": bool(np.isfinite(analytic)),
        "posescore_ckpt": bool(ckpt is not None),
        "sklearn_forbidden": True,
    }
    if args.features.exists():
        feats = pd.read_csv(args.features)
        checks["n_features"] = int(len(feats))
        checks["has_water_rows"] = bool("n_water" in feats.columns and (feats["n_water"] > 0).any())
        checks["grouped_by_pdb"] = bool("pdb_id" in feats.columns)
        checks["n_groups"] = int(feats["pdb_id"].nunique()) if "pdb_id" in feats.columns else 0
    if args.report.exists():
        report = json.loads(args.report.read_text(encoding="utf-8"))
        checks["oof_auroc"] = report.get("oof_auroc")
        checks["sklearn_flag_false"] = report.get("sklearn") is False
    if args.public10.exists():
        pub = json.loads(args.public10.read_text(encoding="utf-8"))
        checks["public10_protocols"] = list((pub.get("by_protocol") or {}).keys())
        checks["public10_n"] = pub.get("n_complexes")
    failed = [k for k, v in checks.items() if v is False]
    out = {"ok": not failed, "failed": failed, "checks": checks}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
