#!/usr/bin/env python3
"""Train multiple MetGeoDock ML scorers (no Vina features).

Affinity models: ridge / rf / extratrees / hgb / svr / mlp / xgb on pActivity
with GroupKFold by target_id.

Pose-rank models: same regressors on is_native in [0,1], plus logreg.
The best pose-rank model by grouped AP is written as pose_rank_model.joblib.
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
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.ml import FEATURE_COLUMNS, MODEL_NAMES, build_model, save_model, train_grouped  # noqa: E402

LOG = logging.getLogger("train_ml_scorer")


def _complete(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for c in FEATURE_COLUMNS:
        if c not in out.columns:
            out[c] = 0.0
    keep = FEATURE_COLUMNS + [c for c in out.columns if c not in FEATURE_COLUMNS]
    out = out[keep]
    return out.replace([np.inf, -np.inf], np.nan).fillna({c: 0.0 for c in FEATURE_COLUMNS})


def _available_models() -> list[str]:
    names = list(MODEL_NAMES)
    try:
        import xgboost  # noqa: F401
    except ImportError:
        names = [n for n in names if n != "xgb"]
    return names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, default=Path("processed/ml/native_features.csv"))
    p.add_argument("--rank", type=Path, default=Path("processed/ml/pose_rank_features.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("processed/ml"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log", type=Path, default=Path("logs/train_ml_scorer.log"))
    args = p.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "feature_columns": FEATURE_COLUMNS,
        "vina_fused": False,
    }
    models = _available_models()

    native = _complete(pd.read_csv(args.native)) if args.native.exists() else pd.DataFrame()
    aff = native.dropna(subset=["pactivity"]) if len(native) and "pactivity" in native.columns else pd.DataFrame()
    aff = aff[np.isfinite(aff["pactivity"])] if len(aff) else aff
    if len(aff) >= 40 and aff["target_id"].nunique() >= 5:
        best = None
        aff_table = []
        for name in models:
            try:
                model, tr, oof = train_grouped(aff, group_col="target_id", model_name=name, seed=args.seed)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("affinity %s failed: %s", name, exc)
                continue
            LOG.info("affinity %s mae=%s r2=%s", name, tr.cv_mae, tr.cv_r2)
            rec = {"name": name, "mae": tr.cv_mae, "r2": tr.cv_r2, "n": tr.n_samples, "groups": tr.n_groups}
            aff_table.append(rec)
            if best is None or rec["mae"] < best["mae"]:
                best = rec
                save_model(model, args.out_dir / "affinity_model.joblib", tr)
                oof.to_csv(args.out_dir / "affinity_oof.csv", index=False)
        report["affinity"] = {"best": best, "all": aff_table}
    else:
        report["affinity"] = {"skipped": True, "n_with_pactivity": int(len(aff))}

    rank = _complete(pd.read_csv(args.rank)) if args.rank.exists() else pd.DataFrame()
    if len(rank) >= 50:
        y = rank["is_native"].to_numpy(dtype=int)
        groups = rank["target_id"].astype(str).to_numpy()
        from sklearn.model_selection import GroupKFold

        n_groups = int(np.unique(groups).size)
        x = np.nan_to_num(rank[FEATURE_COLUMNS].to_numpy(dtype=float), nan=0.0, posinf=10.0, neginf=-10.0)
        rank_table = []
        best_rank = None
        best_oof = None
        best_model = None
        for name in list(models) + ["logreg"]:
            try:
                gkf = GroupKFold(n_splits=max(2, min(5, n_groups)))
                oof = np.zeros(len(y), dtype=float)
                for tr, va in gkf.split(x, y, groups):
                    model = build_model(name)
                    if name == "logreg":
                        model.fit(x[tr], y[tr])
                        oof[va] = model.predict_proba(x[va])[:, 1]
                    else:
                        model.fit(x[tr], y[tr].astype(float))
                        oof[va] = model.predict(x[va])
                auroc = float(roc_auc_score(y, oof))
                ap = float(average_precision_score(y, oof))
            except Exception as exc:  # noqa: BLE001
                LOG.warning("pose-rank %s failed: %s", name, exc)
                continue
            rec = {"name": name, "auroc": auroc, "average_precision": ap, "n": int(len(rank)), "n_native": int(y.sum())}
            rank_table.append(rec)
            LOG.info("pose-rank %s auroc=%s ap=%s", name, auroc, ap)
            if best_rank is None or rec["average_precision"] > best_rank["average_precision"]:
                best_rank = rec
                best_oof = oof
                final = build_model(name)
                if name == "logreg":
                    final.fit(x, y)
                else:
                    final.fit(x, y.astype(float))
                best_model = final
        if best_model is not None and best_oof is not None:
            save_model(best_model, args.out_dir / "pose_rank_model.joblib")
            rank.assign(native_score_oof=best_oof).to_csv(args.out_dir / "pose_rank_oof.csv", index=False)
        report["pose_rank"] = {"best": best_rank, "all": rank_table}
    else:
        report["pose_rank"] = {"skipped": True, "n": int(len(rank))}

    (args.out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path("results").mkdir(exist_ok=True)
    Path("results/ml_train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("wrote %s", args.out_dir / "train_report.json")


if __name__ == "__main__":
    main()
