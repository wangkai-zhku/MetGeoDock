#!/usr/bin/env python3
"""Validate that both MetGeoDock VS schemes beat random and donor baselines.

Reads results/two_vs_summary.json and processed/ml/coordnet_report.json.
Does not invent numbers. Writes results/two_vs_validation.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--two-vs", type=Path, default=Path("results/two_vs_summary.json"))
    p.add_argument("--coordnet", type=Path, default=Path("processed/ml/coordnet_report.json"))
    p.add_argument("--out", type=Path, default=Path("results/two_vs_validation.json"))
    args = p.parse_args()
    two = json.loads(args.two_vs.read_text(encoding="utf-8")) if args.two_vs.exists() else {}
    cv = json.loads(args.coordnet.read_text(encoding="utf-8")) if args.coordnet.exists() else {}
    struct = two.get("structure_based") or {}
    models = cv.get("models") or two.get("ligand_based_cv") or {}
    mlp = (models.get("coordnet_mlp") or {}).get("oof_auroc")
    et = (models.get("extratrees") or {}).get("oof_auroc")
    donor = (models.get("donor_logreg") or {}).get("oof_auroc")
    metalgeo = struct.get("metalgeo_mean_auroc")
    checks = {
        "scheme_a_metalgeo_beats_random": (
            bool(float(metalgeo) > 0.5) if metalgeo is not None else None
        ),
        "scheme_a_metalgeo_beats_donor": struct.get("metalgeo_beats_donor"),
        "scheme_b_mlp_beats_random": bool(float(mlp) > 0.5) if mlp is not None else None,
        "scheme_b_et_beats_random": bool(float(et) > 0.5) if et is not None else None,
        "scheme_b_mlp_beats_donor": (
            bool(float(mlp) > float(donor)) if mlp is not None and donor is not None else None
        ),
        "scheme_b_et_beats_donor": (
            bool(float(et) > float(donor)) if et is not None and donor is not None else None
        ),
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "both_schemes_superior": bool(
            checks["scheme_a_metalgeo_beats_random"]
            and (checks["scheme_b_mlp_beats_random"] or checks["scheme_b_et_beats_random"])
        ),
        "values": {
            "metalgeo_mean_auroc": metalgeo,
            "coordnet_mlp_oof_auroc": mlp,
            "extratrees_oof_auroc": et,
            "donor_logreg_oof_auroc": donor,
            "random_auroc": 0.5,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
