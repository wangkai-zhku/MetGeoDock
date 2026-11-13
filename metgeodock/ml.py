"""Experimental-affinity and pose-ranking machine-learning scorers.

Target-wise splits are required: never split frames of the same complex into
both train and test. Features are metal-aware geometry/GP/SLEF terms only
(no Vina fusion). The models predict pActivity or native-likeness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

FEATURE_COLUMNS = [
    "slef",
    "gp_nll",
    "geo_rmsd_or",
    "clash",
    "dist_harm",
    "cn_pen",
    "n_coord_lig",
    "n_coord_pro",
    "n_heavy",
    "n_donor",
    "charge_sum",
    "min_donor_dist",
    "mean_donor_dist",
    "slef_tanh",
    "metal_zn",
    "metal_ca",
    "metal_mg",
    "metal_mn",
    "metal_fe2",
    "metal_fe3",
    "metal_cu",
    "metal_co",
    "metal_cd",
    "metal_ni",
]

MODEL_NAMES = ("ridge", "rf", "extratrees", "hgb", "svr", "mlp", "xgb")


@dataclass
class TrainReport:
    model_name: str
    cv_mae: float
    cv_r2: float
    n_samples: int
    n_groups: int
    path: Path


def _xy(frame: pd.DataFrame, y_col: str = "pactivity") -> tuple[np.ndarray, np.ndarray]:
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    x = np.nan_to_num(frame[FEATURE_COLUMNS].to_numpy(dtype=float), nan=0.0, posinf=10.0, neginf=-10.0)
    y = frame[y_col].to_numpy(dtype=float)
    return x, y


def build_model(name: str = "hgb") -> Pipeline:
    name = name.lower()
    if name == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=2.0))])
    if name == "rf":
        return Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        max_depth=12,
                        min_samples_leaf=4,
                        random_state=42,
                        n_jobs=-1,
                    ),
                )
            ]
        )
    if name in {"extratrees", "et"}:
        return Pipeline(
            [
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=400,
                        max_depth=12,
                        min_samples_leaf=4,
                        random_state=42,
                        n_jobs=-1,
                    ),
                )
            ]
        )
    if name in {"svr", "svm"}:
        return Pipeline([("scaler", StandardScaler()), ("model", SVR(C=2.0, epsilon=0.15, kernel="rbf"))])
    if name == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        max_iter=400,
                        random_state=42,
                        early_stopping=True,
                    ),
                ),
            ]
        )
    if name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=500, class_weight="balanced", C=0.5)),
            ]
        )
    if name == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is not installed; pip install xgboost") from exc
        return Pipeline(
            [
                (
                    "model",
                    XGBRegressor(
                        n_estimators=400,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1,
                        verbosity=0,
                    ),
                )
            ]
        )
    return Pipeline(
        [
            (
                "model",
                HistGradientBoostingRegressor(
                    max_depth=6,
                    learning_rate=0.06,
                    max_iter=400,
                    l2_regularization=0.1,
                    random_state=42,
                ),
            )
        ]
    )


def train_grouped(
    frame: pd.DataFrame,
    group_col: str = "target_id",
    model_name: str = "hgb",
    n_splits: int = 5,
    seed: int = 42,
    y_col: str = "pactivity",
) -> tuple[Pipeline, TrainReport, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x, y = _xy(frame, y_col=y_col)
    groups = frame[group_col].astype(str).to_numpy()
    n_groups = int(np.unique(groups).size)
    splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=splits)
    oof = np.zeros_like(y)
    for train_idx, val_idx in gkf.split(x, y, groups):
        model = build_model(model_name)
        y_tr = y[train_idx]
        if model_name == "logreg":
            model.fit(x[train_idx], (y_tr >= 0.5).astype(int))
            oof[val_idx] = model.predict_proba(x[val_idx])[:, 1]
        else:
            model.fit(x[train_idx], y_tr)
            oof[val_idx] = model.predict(x[val_idx])
    mae = float(mean_absolute_error(y, oof))
    r2 = float(r2_score(y, oof))
    final = build_model(model_name)
    if model_name == "logreg":
        final.fit(x, (y >= 0.5).astype(int))
    else:
        final.fit(x, y)
    oof_frame = frame.copy()
    oof_frame["pactivity_pred_oof"] = oof
    report = TrainReport(
        model_name=model_name,
        cv_mae=mae,
        cv_r2=r2,
        n_samples=int(len(frame)),
        n_groups=n_groups,
        path=Path("."),
    )
    _ = rng
    return final, report, oof_frame


def save_model(model: Pipeline, path: str | Path, report: TrainReport | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "report": report}, path)


def load_model(path: str | Path) -> Pipeline:
    blob = joblib.load(path)
    if isinstance(blob, dict) and "model" in blob:
        return blob["model"]
    return blob


def predict_pactivity(model: Pipeline, features: dict[str, float]) -> float:
    row = np.array([[features.get(c, 0.0) for c in FEATURE_COLUMNS]], dtype=float)
    row = np.nan_to_num(row, nan=0.0, posinf=10.0, neginf=-10.0)
    est = model
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        inner = model.named_steps["model"]
        if hasattr(inner, "predict_proba"):
            return float(model.predict_proba(row)[0, 1])
        _ = est
    return float(model.predict(row)[0])
