"""PyTorch neural-network scorers for pose ranking and ligand VS.

CoordNet classifies ChEMBL actives vs inactives from metal-warhead probes
and Morgan bits. PoseNet scores docked poses. Checkpoints are ``.pt`` dicts
with state_dict + train-only mean/std. Does not use scikit-learn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .fingerprint import COORDNET_COLUMNS, ligand_feature_vector
from .io import Molecule

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COORDNET = _ROOT / "processed" / "ml" / "coordnet.pt"
DEFAULT_POSENET = _ROOT / "processed" / "ml" / "posenet.pt"
DEFAULT_POSESCORE = _ROOT / "processed" / "ml" / "posescore.pt"
DEVICE = torch.device("cpu")
POSENET_COLUMNS: tuple[str, ...] = (
    "gp_nll",
    "geo_rmsd_or",
    "clash",
    "dist_harm",
    "cn_pen",
    "n_coord_lig",
    "n_coord_pro",
    "n_heavy",
    "n_donor",
    "min_donor_dist",
    "mean_donor_dist",
    "hybrid_total",
)


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: tuple[int, ...] = (128, 64, 32), dropout: float = 0.20):
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def coordnet_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = [c for c in COORDNET_COLUMNS if c not in frame.columns]
    for c in missing:
        frame[c] = 0.0
    return np.nan_to_num(frame[list(COORDNET_COLUMNS)].to_numpy(dtype=float), nan=0.0)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.clip(std, 1.0e-6, None)


def save_nn(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)


def load_nn(path: str | Path | None = None, kind: str = "coordnet"):
    if path is None:
        p = DEFAULT_COORDNET if kind == "coordnet" else DEFAULT_POSENET
    else:
        p = Path(path)
    if p.suffix == ".joblib" or (not p.exists() and p.with_suffix(".pt").exists()):
        p = p if p.suffix == ".pt" else p.with_suffix(".pt")
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def _bundle_predict_logit(bundle: dict, x: np.ndarray) -> np.ndarray:
    mean = np.asarray(bundle["mean"], dtype=float)
    std = np.asarray(bundle["std"], dtype=float)
    xs = standardize(x, mean, std)
    model = MLP(int(xs.shape[1]), tuple(bundle.get("hidden", (128, 64, 32))), float(bundle.get("dropout", 0.20)))
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(xs, dtype=torch.float32))
    return logits.cpu().numpy()


def predict_active_proba(bundle: dict, feats: dict[str, float]) -> float:
    row = pd.DataFrame([{c: float(feats.get(c, 0.0)) for c in COORDNET_COLUMNS}])
    x = coordnet_matrix(row)
    logit = float(_bundle_predict_logit(bundle, x)[0])
    return float(1.0 / (1.0 + np.exp(-logit)))


def posenet_rank_value(score, ligand: Molecule, model_path: str | Path | None = None) -> float:
    """Lower better. Missing checkpoint falls back to MetGeo total."""
    path = Path(model_path) if model_path else DEFAULT_POSENET
    bundle = load_nn(path, kind="posenet") if path.exists() else None
    if bundle is None:
        return float(score.total)
    cols = list(bundle.get("columns") or POSENET_COLUMNS)
    feats = {
        "gp_nll": float(score.gp),
        "geo_rmsd_or": float(score.geo),
        "clash": float(score.clash),
        "dist_harm": float(score.dist),
        "cn_pen": float(score.cn),
        "n_coord_lig": float(score.n_coord_lig),
        "n_coord_pro": float(score.n_coord_pro),
        "n_heavy": float(ligand.heavy_mask().sum()),
        "n_donor": float(ligand.donor_indices().size),
        "min_donor_dist": float(score.terms.get("min_donor_dist", 9.0)),
        "mean_donor_dist": float(score.terms.get("mean_donor_dist", 9.0)),
        "hybrid_total": float(score.total),
    }
    x = np.array([[float(feats.get(c, 0.0)) for c in cols]], dtype=float)
    logit = float(_bundle_predict_logit(bundle, x)[0])
    score.terms["posenet_logit"] = logit
    return float(-logit)


def coordnet_score_ligand(
    metal: str,
    smiles: str | None = None,
    ligand: Molecule | None = None,
    metal_xyz: np.ndarray | None = None,
    model_path: str | Path | None = None,
) -> float:
    """Higher is better (P(active)). Missing model returns NaN."""
    bundle = load_nn(model_path, kind="coordnet")
    if bundle is None:
        return float("nan")
    feats = ligand_feature_vector(metal, smiles=smiles, ligand=ligand, metal_xyz=metal_xyz)
    return predict_active_proba(bundle, feats)
