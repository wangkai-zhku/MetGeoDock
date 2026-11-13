#!/usr/bin/env python3
"""Compare MetGeoDock water/multi-metal protocols to MetalloDock capabilities.

MetalloDock numbers are literature citations (Zhang 2026) plus the published
preprocessing rule (remove water and extraneous metals). This script does not
re-run MetalloDock and does not invent a MetalloDock keep-water success rate.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--literature", type=Path, default=Path("configs/metallodock_literature.yaml"))
    p.add_argument("--ours", type=Path, default=Path("results/water_multimetal_redock.json"))
    p.add_argument("--csv", type=Path, default=Path("results/water_multimetal_redock.csv"))
    p.add_argument("--inventory", type=Path, default=Path("processed/water_multimetal/build_summary.json"))
    p.add_argument("--out", type=Path, default=Path("results/water_multimetal_vs_metallodock.json"))
    args = p.parse_args()
    lit = yaml.safe_load(args.literature.read_text(encoding="utf-8"))
    ours = json.loads(args.ours.read_text(encoding="utf-8")) if args.ours.exists() else None
    inv = json.loads(args.inventory.read_text(encoding="utf-8")) if args.inventory.exists() else None
    protocol = lit.get("protocol") or {}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "literature_citation": lit.get("citation"),
        "doi": lit.get("doi"),
        "capability": {
            "MetalloDock": {
                "keep_coordinating_water": False,
                "reason_water": protocol.get("water")
                or "GitHub README: Remove water/heteroatoms/extraneous metals",
                "keep_extraneous_metals": False,
                "reason_metal": protocol.get("extra_metals")
                or "GitHub README: Remove water/heteroatoms/extraneous metals",
                "prepared_bimetal_top1": lit.get("methods", {}).get("MetalloDock_bimetal"),
                "prepared_bimetal_note": (
                    "74.4% is reported after keeping the intended metal ions in a "
                    "prepared bimetal subset, not by ingesting leftover waters or "
                    "unrelated extra ions."
                ),
            },
            "MetGeoDock": {
                "keep_coordinating_water": True,
                "keep_extraneous_metals": True,
                "mechanism_water": (
                    "HOH/WAT in the receptor occupy vacant polyhedron vertices; "
                    "ligand donors are placed on remaining sites."
                ),
                "mechanism_metal": (
                    "Other pocket metals occupy vertices; every pocket metal is "
                    "tried as a docking center and poses are merged by MetGeo score."
                ),
            },
        },
        "inventory": inv,
        "metgeodock_ablation": ours.get("by_protocol") if ours else None,
        "by_metal": {},
        "water_keep_beats_strip_top3": None if ours is None else ours.get("water_keep_beats_strip_top3"),
        "multi_keep_beats_strip_top3": None if ours is None else ours.get("multi_keep_beats_strip_top3"),
    }
    if args.csv.exists():
        frame = pd.read_csv(args.csv)
        ok = frame[frame["error"].fillna("") == ""] if "error" in frame.columns else frame
        by_metal: dict[str, dict] = {}
        for (proto, metal), sub in ok.groupby(["protocol", "metal"]):
            by_metal.setdefault(str(proto), {})[str(metal)] = {
                "n": int(len(sub)),
                "top1": float(sub["success_top1"].mean()),
                "top3": float(sub["success_top3"].mean()),
                "sampled": float(sub["success_sampled"].mean()),
            }
        report["by_metal"] = by_metal
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
