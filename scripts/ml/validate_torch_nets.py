#!/usr/bin/env python3
"""Sanity-check PyTorch CoordNet/PoseNet reports vs stored sklearn baselines.

Does not import scikit-learn. Reads processed/ml/torch_report.json and the
previous sklearn numbers in processed/ml/coordnet_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.nnscore import load_nn, predict_active_proba  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--torch-report", type=Path, default=Path("processed/ml/torch_report.json"))
    p.add_argument("--sklearn-report", type=Path, default=Path("processed/ml/coordnet_report.json"))
    p.add_argument("--posenet-report", type=Path, default=Path("processed/ml/posenet_report.json"))
    p.add_argument("--out", type=Path, default=Path("results/torch_vs_sklearn.json"))
    args = p.parse_args()
    report = json.loads(args.torch_report.read_text(encoding="utf-8"))
    sk = json.loads(args.sklearn_report.read_text(encoding="utf-8")) if args.sklearn_report.exists() else {}
    pose_sk = json.loads(args.posenet_report.read_text(encoding="utf-8")) if args.posenet_report.exists() else {}
    sk_mlp = float((sk.get("models") or {}).get("coordnet_mlp", {}).get("oof_auroc") or 0.6261199091219717)
    sk_et = float((sk.get("models") or {}).get("extratrees", {}).get("oof_auroc") or 0.6771910270487415)
    sk_donor = float((sk.get("models") or {}).get("donor_logreg", {}).get("oof_auroc") or 0.6073398271481381)
    torch_auc = float(report["coordnet"]["oof_auroc"])
    donor_auc = float(report["donor_logistic_torch"]["oof_auroc"])
    pose_auc = float((report.get("posenet") or {}).get("oof_auroc") or float("nan"))
    bundle = load_nn("processed/ml/coordnet.pt", kind="coordnet")
    assert bundle is not None, "coordnet.pt missing"
    dummy = {c: 0.0 for c in bundle.get("columns", [])}
    dummy["n_donor"] = 3.0
    proba = predict_active_proba(bundle, dummy)
    assert 0.0 <= proba <= 1.0, proba
    out = {
        "torch_version": report.get("torch_version"),
        "cuda": report.get("cuda"),
        "coordnet_torch_oof_auroc": torch_auc,
        "coordnet_sklearn_mlp_oof_auroc": sk_mlp,
        "extratrees_oof_auroc": sk_et,
        "donor_torch_oof_auroc": donor_auc,
        "donor_sklearn_logreg_oof_auroc": sk_donor,
        "posenet_torch_oof_auroc": pose_auc,
        "posenet_sklearn_oof_auroc": pose_sk.get("oof_auroc"),
        "torch_beats_sklearn_mlp": bool(torch_auc > sk_mlp),
        "torch_beats_extratrees": bool(torch_auc > sk_et),
        "torch_beats_random": bool(torch_auc > 0.5),
        "torch_beats_donor": bool(torch_auc > donor_auc),
        "dummy_predict_proba": float(proba),
        "checkpoint_ok": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("torch", torch.__version__)


if __name__ == "__main__":
    main()
