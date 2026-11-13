#!/usr/bin/env python3
"""Publication figures from code-computed CSV/JSON summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for plotting") from exc


def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_inventory(summary: dict, path: Path) -> None:
    metals = [r["metal"] for r in summary.get("per_metal", [])]
    counts = [r["n_complete"] for r in summary.get("per_metal", [])]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(metals, counts, color="#215C8A")
    ax.set_ylabel("Complete complexes")
    ax.set_xlabel("Metal ion")
    ax.set_title("GPDOCK crystal set indexed by MetGeoDock")
    _savefig(fig, path)


def fig_redock(frame: pd.DataFrame, path: Path) -> None:
    ok = frame[frame.get("error", pd.Series([""] * len(frame))).fillna("") == ""]
    methods = [m for m in ("metgeo", "geo", "gp", "dist", "clash", "hybrid", "vina") if f"{m}_success_top1" in ok.columns] or ["metgeo"]
    labels = methods
    vals = []
    for m in methods:
        col = f"{m}_success_top1"
        vals.append(float(ok[col].mean()) if col in ok and len(ok) else np.nan)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(labels, vals, color=["#1B4F72", "#7B241C", "#196F3D", "#B9770E"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Top-1 success (RMSD ≤ 2 Å)")
    ax.tick_params(axis="x", rotation=20)
    _savefig(fig, path)


def fig_vs(frame: pd.DataFrame, path: Path) -> None:
    methods = [m for m in ("metgeo", "geo", "gp", "dist", "hybrid", "vina") if f"{m}_auroc" in frame.columns] or ["metgeo"]
    labels = methods
    auroc = [float(frame[f"{m}_auroc"].mean()) if f"{m}_auroc" in frame else np.nan for m in methods]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(labels, auroc, color="#1B4F72")
    ax.axhline(0.5, ls="--", color="0.4", lw=1)
    ax.set_ylabel("Mean AUROC")
    ax.set_ylim(0, 1)
    _savefig(fig, path)


def fig_qm(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    if "mean_abs_dev_from_mode" in frame.columns:
        for metal, sub in frame.groupby("metal"):
            ax.scatter(
                sub["rmsd_or"],
                sub["mean_abs_dev_from_mode"],
                label=metal,
                s=28,
                alpha=0.85,
            )
        ax.set_xlabel("Coordination RMSD_OR")
        ax.set_ylabel("Mean |d − GPDOCK mode| (Å)")
        ax.legend(ncols=5, fontsize=7, frameon=False)
    _savefig(fig, path)


def fig_ablation(path_json: Path, path: Path) -> None:
    blob = json.loads(path_json.read_text(encoding="utf-8"))
    methods = blob.get("methods", {})
    order = [
        ("geo_rmsd_or", "RMSD_OR"),
        ("gp_nll", "GPDOCK −log P"),
        ("clash", "Clash"),
        ("vina_total", "Vina-like"),
        ("hybrid_total", "Default hybrid"),
        ("learned_logreg", "Learned logreg"),
        ("native_score_oof", "ExtraTrees pose-ranker"),
    ]
    labels = [lab for key, lab in order if key in methods]
    aps = [methods[key]["ap"] for key, lab in order if key in methods]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    colors = ["#7F8C8D", "#B9770E", "#196F3D", "#7B241C", "#5D6D7E", "#1B4F72", "#6C3483"][: len(labels)]
    ax.bar(labels, aps, color=colors)
    ax.set_ylabel("Average precision (native pose)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    _savefig(fig, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory-json", type=Path, default=Path("processed/crystal/inventory_summary.json"))
    p.add_argument("--redock-csv", type=Path, default=Path("results/redock_benchmark.csv"))
    p.add_argument("--vs-csv", type=Path, default=Path("results/vs_benchmark.csv"))
    p.add_argument("--qm-csv", type=Path, default=Path("processed/qm/clusters/cluster_index.csv"))
    p.add_argument("--fig-dir", type=Path, default=Path("figures"))
    p.add_argument("--ablation-json", type=Path, default=Path("results/pose_rank_ablation.json"))
    args = p.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    if args.inventory_json.exists():
        fig_inventory(json.loads(args.inventory_json.read_text(encoding="utf-8")), args.fig_dir / "fig1_inventory.png")
    if args.redock_csv.exists():
        fig_redock(pd.read_csv(args.redock_csv), args.fig_dir / "fig2_redock_success.png")
    if args.vs_csv.exists():
        fig_vs(pd.read_csv(args.vs_csv), args.fig_dir / "fig3_vs_auroc.png")
    if args.qm_csv.exists():
        fig_qm(pd.read_csv(args.qm_csv), args.fig_dir / "fig4_qm_geometry.png")
    if args.ablation_json.exists():
        fig_ablation(args.ablation_json, args.fig_dir / "fig2_pose_rank_ap.png")


if __name__ == "__main__":
    main()
