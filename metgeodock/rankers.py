"""PyTorch pose-rankers for same-ligand MetGeoDock samples.

All checkpoints store architecture name + train-only mean/std. Inference
returns a **lower-better** rank score. No scikit-learn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .nnscore import MLP, DEVICE, load_nn, save_nn, standardize

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RANKER = _ROOT / "processed" / "ml" / "ranker_best.pt"


class ResidualMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 96, depth: int = 3, dropout: float = 0.10):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.ReLU())
        blocks: list[nn.Module] = []
        for _ in range(max(1, depth)):
            blocks.append(
                nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for block in self.blocks:
            h = torch.relu(h + block(h))
        return self.out(h).squeeze(-1)


class FiLMRanker(nn.Module):
    """Metal one-hot FiLM-conditions a residual MLP (custom architecture)."""

    def __init__(self, d_in: int, n_metal: int, hidden: int = 96, dropout: float = 0.10):
        super().__init__()
        self.n_metal = n_metal
        self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.gamma = nn.Linear(n_metal, hidden)
        self.beta = nn.Linear(n_metal, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, metal_oh: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        g = torch.tanh(self.gamma(metal_oh))
        b = self.beta(metal_oh)
        h = torch.relu(g * h + b)
        return self.out(h).squeeze(-1)


class GeoListResidual(nn.Module):
    """MetGeo residual: s = hybrid_z + max_delta * tanh(net(x)).

    Bounded correction so the net cannot throw away MetGeo, which is the
    failure mode of a replacement classifier trained on GPDOCK poses.
    """

    def __init__(self, d_in: int, hybrid_index: int, hidden: int = 96, depth: int = 3, dropout: float = 0.10):
        super().__init__()
        self.hybrid_index = int(hybrid_index)
        self.net = ResidualMLP(d_in, hidden=hidden, depth=depth, dropout=dropout)
        self.max_delta = nn.Parameter(torch.tensor(1.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hybrid = x[:, self.hybrid_index]
        delta = self.max_delta.clamp(0.05, 4.0) * torch.tanh(self.net(x))
        return hybrid + delta


class LinearRanker(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.lin = nn.Linear(d_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).squeeze(-1)


def build_model(arch: str, d_in: int, extra: dict | None = None) -> nn.Module:
    extra = extra or {}
    if arch == "linear":
        return LinearRanker(d_in)
    if arch == "mlp":
        hidden = tuple(extra.get("hidden", (64, 32)))
        return MLP(d_in, hidden, float(extra.get("dropout", 0.10)))
    if arch in {"residual", "deepres"}:
        depth = 4 if arch == "deepres" else int(extra.get("depth", 3))
        hidden = int(extra.get("hidden", 128 if arch == "deepres" else 96))
        return ResidualMLP(d_in, hidden=hidden, depth=depth, dropout=float(extra.get("dropout", 0.10)))
    if arch == "film":
        return FiLMRanker(d_in, n_metal=int(extra.get("n_metal", 10)), hidden=int(extra.get("hidden", 96)))
    if arch == "geolist":
        return GeoListResidual(
            d_in,
            hybrid_index=int(extra.get("hybrid_index", 0)),
            hidden=int(extra.get("hidden", 96)),
            depth=int(extra.get("depth", 3)),
            dropout=float(extra.get("dropout", 0.10)),
        )
    raise ValueError(f"unknown arch {arch}")


def _metal_slice(columns: list[str]) -> tuple[list[int], list[int]]:
    metal_idx = [i for i, c in enumerate(columns) if str(c).startswith("metal_")]
    other = [i for i in range(len(columns)) if i not in metal_idx]
    return other, metal_idx


def forward_ranker(model: nn.Module, x: torch.Tensor, columns: list[str], arch: str) -> torch.Tensor:
    if arch == "film":
        _other, metal_idx = _metal_slice(columns)
        if not metal_idx:
            oh = torch.zeros((x.shape[0], 10), dtype=x.dtype, device=x.device)
        else:
            oh = x[:, metal_idx]
        return model(x, oh)
    return model(x)


def predict_rank(bundle: dict, feats: dict[str, float]) -> float:
    """Lower-better rank score."""
    cols = list(bundle.get("columns") or [])
    x = np.array([[float(feats.get(c, 0.0)) for c in cols]], dtype=float)
    mean = np.asarray(bundle["mean"], dtype=float)
    std = np.asarray(bundle["std"], dtype=float)
    xs = torch.tensor(standardize(x, mean, std), dtype=torch.float32)
    arch = str(bundle.get("arch", "mlp"))
    extra = dict(bundle.get("extra") or {})
    model = build_model(arch, int(xs.shape[1]), extra)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    with torch.no_grad():
        s = forward_ranker(model, xs, cols, arch)
        val = float(s.reshape(-1)[0].cpu())
    kind = str(bundle.get("score_kind", "energy"))
    if kind in {"neg_logit", "logit"}:
        return float(-val)
    return float(val)


def load_ranker(path: str | Path | None = None) -> dict | None:
    p = Path(path) if path else DEFAULT_RANKER
    if not p.exists():
        return None
    return load_nn(p, kind="posenet")


def save_ranker(bundle: dict, path: Path) -> None:
    save_nn(bundle, path)
