"""Same-ligand pose ranking (including water) for MetGeoDock.

GPDOCK dumps sampled poses without a rank score. PoseScore is trained on
those OUTPUT_DOCK models vs crystal RMSD and applied as a listwise rerank
on MetGeoDock samples. Lower ``rank_value`` is better.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fingerprint import probe_counts
from .forcefield import displaced_water_count
from .io import Molecule
from .metals import METAL_ORDER, metal_from_name
from .nnscore import MLP, load_nn, standardize
from .scoring import HybridScore

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSESCORE = _ROOT / "processed" / "ml" / "posescore.pt"
DEFAULT_RANKER = _ROOT / "processed" / "ml" / "ranker_best.pt"

POSE_COLUMNS: tuple[str, ...] = (
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
    "map_overlap",
    "contact",
    "centroid_pen",
    "n_water",
    "n_other_metal",
    "n_water_displaced",
    "water_min_dist",
    "semi_flex",
    "probe_hydroxamate",
    "probe_carboxylate",
    "probe_imidazole",
    "probe_thiol",
    "probe_phosphate",
) + tuple(f"metal_{m}" for m in METAL_ORDER)


def pose_feature_vector(score: HybridScore, ligand: Molecule, metal: str | None = None) -> dict[str, float]:
    t = score.terms
    probes = probe_counts(None, ligand)
    n_disp = int(t.get("n_water_displaced", 0))
    wmin = float(t.get("water_min_dist", 9.0))
    if "n_water_displaced" not in t or "water_min_dist" not in t:
        n_disp, wmin = displaced_water_count(
            ligand,
            np.asarray(t.get("water_xyz_ref", []), dtype=float).reshape(-1, 3)
            if "water_xyz_ref" in t
            else np.zeros((0, 3)),
        )
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
        "min_donor_dist": float(t.get("min_donor_dist", 9.0)),
        "mean_donor_dist": float(t.get("mean_donor_dist", 9.0)),
        "hybrid_total": float(score.total),
        "map_overlap": float(t.get("map_overlap", 0.0)),
        "contact": float(t.get("contact", 0.0)),
        "centroid_pen": float(t.get("centroid_pen", 0.0)),
        "n_water": float(t.get("n_water", 0.0)),
        "n_other_metal": float(t.get("n_other_metal", 0.0)),
        "n_water_displaced": float(n_disp),
        "water_min_dist": float(wmin),
        "semi_flex": float(t.get("semi_flex", 0.0)),
        "probe_hydroxamate": float(probes.get("hydroxamate", 0.0)),
        "probe_carboxylate": float(probes.get("carboxylate", 0.0)),
        "probe_imidazole": float(probes.get("imidazole", 0.0)),
        "probe_thiol": float(probes.get("thiol", 0.0)),
        "probe_phosphate": float(probes.get("phosphate", 0.0)),
    }
    metal_name = ""
    if metal:
        try:
            metal_name = metal_from_name(metal)
        except Exception:
            metal_name = str(metal).lower()
    for name in METAL_ORDER:
        feats[f"metal_{name}"] = 1.0 if name == metal_name else 0.0
    return feats


def analytic_pose_score(feats: dict[str, float]) -> float:
    """Lower better. MetGeo plus map / leftover-water corrections."""
    return (
        float(feats.get("hybrid_total", 0.0))
        - 0.20 * float(feats.get("map_overlap", 0.0))
        + 0.08 * max(0.0, 2.20 - float(feats.get("water_min_dist", 9.0)))
    )


def posescore_logit(feats: dict[str, float], model_path: str | Path | None = None) -> float:
    """Higher means more likely RMSD < 2 A. NaN if no checkpoint."""
    path = Path(model_path) if model_path else DEFAULT_POSESCORE
    if not path.exists():
        return float("nan")
    bundle = load_nn(path, kind="posenet")
    if bundle is None:
        return float("nan")
    cols = list(bundle.get("columns") or POSE_COLUMNS)
    x = np.array([[float(feats.get(c, 0.0)) for c in cols]], dtype=float)
    mean = np.asarray(bundle["mean"], dtype=float)
    std = np.asarray(bundle["std"], dtype=float)
    xs = standardize(x, mean, std)
    hidden = tuple(bundle.get("hidden", (64, 32)))
    model = MLP(int(xs.shape[1]), hidden, float(bundle.get("dropout", 0.10)))
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    import torch

    with torch.no_grad():
        logit = float(model(torch.tensor(xs, dtype=torch.float32))[0].cpu())
    return logit


def rank_value(
    score: HybridScore,
    ligand: Molecule,
    model_path: str | Path | None = None,
    use_nn: bool = False,
    metal: str | None = None,
    use_ranker: bool = False,
) -> float:
    """Lower better. Ranker checkpoint if requested; else analytic or old BCE net."""
    feats = pose_feature_vector(score, ligand, metal=metal)
    analytic = analytic_pose_score(feats)
    score.terms["posescore_analytic"] = float(analytic)
    if use_ranker:
        from .rankers import load_ranker, predict_rank

        bundle = load_ranker(model_path)
        if bundle is not None:
            val = float(predict_rank(bundle, feats))
            score.terms["ranker_score"] = val
            return val
    logit = posescore_logit(feats, model_path=model_path if not use_ranker else None)
    if np.isfinite(logit):
        score.terms["posescore_logit"] = float(logit)
        if use_nn:
            return float(-logit)
    return float(analytic)
