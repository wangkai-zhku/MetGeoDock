#!/usr/bin/env python3
"""Assemble MetGeoDock computed success vs MetalloDock literature numbers.

Literature rates are read from configs/metallodock_literature.yaml (Zhang 2026).
MetGeoDock rates are read from results JSON produced by our docking scripts.
Does not invent MetGeoDock numbers.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _load_success(path: Path) -> dict | None:
    if not path.exists():
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    if "metgeodock_success" in blob:
        return blob["metgeodock_success"]
    rates = blob.get("success_rates") or {}
    if "metgeo" in rates:
        return rates["metgeo"]
    if "ff" in rates:
        return rates["ff"]
    return rates.get("hybrid")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--literature", type=Path, default=Path("configs/metallodock_literature.yaml"))
    p.add_argument("--public10", type=Path, default=Path("results/metallodock_public_redock.json"))
    p.add_argument("--timesplit", type=Path, default=Path("results/timesplit_full_redock.json"))
    p.add_argument("--rcsb2022", type=Path, default=Path("results/metallodock_541_redock.json"))
    p.add_argument("--crystal", type=Path, default=Path("results/redock_summary.json"))
    p.add_argument("--out", type=Path, default=Path("results/metallodock_comparison.json"))
    args = p.parse_args()
    lit = yaml.safe_load(args.literature.read_text(encoding="utf-8"))
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "literature": lit,
        "metgeodock": {
            "public10": _load_success(args.public10),
            "time_split": _load_success(args.timesplit),
            "rcsb_2022": _load_success(args.rcsb2022),
            "crystal_sample": _load_success(args.crystal),
        },
        "beats_metallodock_public10_top1": None,
        "beats_metallodock_public10_top3": None,
        "beats_metallodock_timesplit_top1": None,
        "beats_metallodock_timesplit_top3": None,
        "beats_metallodock_rcsb2022_top1": None,
        "beats_metallodock_rcsb2022_top3": None,
    }
    md = float(lit["methods"]["MetalloDock"])
    p10 = report["metgeodock"]["public10"]
    ts = report["metgeodock"]["time_split"]
    rcsb = report["metgeodock"]["rcsb_2022"]
    if p10 and p10.get("top1") is not None:
        report["beats_metallodock_public10_top1"] = bool(float(p10["top1"]) > md)
    if p10 and p10.get("top3") is not None:
        report["beats_metallodock_public10_top3"] = bool(float(p10["top3"]) > md)
    if ts and ts.get("top1") is not None:
        report["beats_metallodock_timesplit_top1"] = bool(float(ts["top1"]) > md)
    if ts and ts.get("top3") is not None:
        report["beats_metallodock_timesplit_top3"] = bool(float(ts["top3"]) > md)
    if rcsb and rcsb.get("top1") is not None:
        report["beats_metallodock_rcsb2022_top1"] = bool(float(rcsb["top1"]) > md)
    if rcsb and rcsb.get("top3") is not None:
        report["beats_metallodock_rcsb2022_top3"] = bool(float(rcsb["top3"]) > md)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metgeodock"], indent=2))
    print("literature_MetalloDock", md)
    print("beats_public10_top1", report["beats_metallodock_public10_top1"])
    print("beats_public10_top3", report["beats_metallodock_public10_top3"])
    print("beats_timesplit_top1", report["beats_metallodock_timesplit_top1"])
    print("beats_timesplit_top3", report["beats_metallodock_timesplit_top3"])
    print("beats_rcsb2022_top1", report["beats_metallodock_rcsb2022_top1"])
    print("beats_rcsb2022_top3", report["beats_metallodock_rcsb2022_top3"])


if __name__ == "__main__":
    main()
