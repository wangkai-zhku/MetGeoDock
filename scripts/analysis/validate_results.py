#!/usr/bin/env python3
"""Sanity bounds for MetGeoDock outputs (no raw-array dumps)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def check_inventory(path: Path, report: dict) -> None:
    frame = pd.read_csv(path)
    report["inventory"] = {
        "n_rows": int(len(frame)),
        "complete_frac": float(frame["complete"].mean()) if "complete" in frame else None,
        "n_metal": int(frame["metal"].nunique()) if "metal" in frame else 0,
        "ok_ten_metals": bool(set(frame["metal"].unique()) >= {"zn", "ca", "mg", "mn", "fe2", "fe3", "cu", "co", "cd", "ni"})
        if "metal" in frame
        else False,
    }


def check_redock(path: Path, report: dict) -> None:
    frame = pd.read_csv(path)
    ok = frame[frame.get("error", pd.Series([""] * len(frame))).fillna("") == ""]
    flags = []
    if len(ok) == 0:
        flags.append("no_successful_redock")
    for col in ok.columns:
        if col.endswith("_rmsd"):
            if (ok[col] < -1e-6).any():
                flags.append(f"negative_rmsd:{col}")
            if (ok[col] > 50).any():
                flags.append(f"unphysical_rmsd:{col}")
    report["redock"] = {"n_ok": int(len(ok)), "flags": flags}


def check_vs(path: Path, report: dict) -> None:
    frame = pd.read_csv(path)
    flags = []
    for col in frame.columns:
        if col.endswith("_auroc"):
            if ((frame[col] < -1e-6) | (frame[col] > 1.0001)).any():
                flags.append(f"auroc_out_of_range:{col}")
    report["vs"] = {"n": int(len(frame)), "flags": flags}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, default=Path("processed/crystal/inventory.csv"))
    p.add_argument("--redock", type=Path, default=Path("results/redock_benchmark.csv"))
    p.add_argument("--vs", type=Path, default=Path("results/vs_benchmark.csv"))
    p.add_argument("--out", type=Path, default=Path("results/validation_report.json"))
    args = p.parse_args()
    report: dict = {}
    if args.inventory.exists():
        check_inventory(args.inventory, report)
    if args.redock.exists():
        check_redock(args.redock, report)
    if args.vs.exists():
        check_vs(args.vs, report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    flags = []
    for block in report.values():
        flags.extend(block.get("flags", []) if isinstance(block, dict) else [])
    if flags:
        raise SystemExit("validation flags: " + ",".join(flags))


if __name__ == "__main__":
    main()
