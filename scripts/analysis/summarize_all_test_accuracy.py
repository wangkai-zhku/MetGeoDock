#!/usr/bin/env python3
"""Pool MetGeoDock redock success rates across all finished test CSVs.

Does not re-dock. Unique-PDB union prefers timesplit_full then RCSB-2022 then
public-10. Literature 78.6% is cited only; this is not the unpublished 541 list.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _ok(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "error" in frame.columns:
        err = frame["error"].fillna("").astype(str)
        frame = frame[err.isin(("", "nan"))].copy()
    return frame


def _rates(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "top1": None, "top3": None, "top5": None, "sampled": None}
    out = {"n": int(len(frame))}
    for name, col in (
        ("top1", "success_top1"),
        ("top3", "success_top3"),
        ("top5", "success_top5"),
        ("sampled", "success_sampled"),
    ):
        if col not in frame.columns:
            out[name] = None
            continue
        out[name] = float(pd.to_numeric(frame[col], errors="coerce").fillna(0).mean())
    return out


def _load(path: Path, set_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame = _ok(frame)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["set_name"] = set_name
    if "pdb_id" in frame.columns:
        frame["pdb_key"] = frame["pdb_id"].astype(str).str.lower()
    else:
        frame["pdb_key"] = ""
    return frame


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public10", type=Path, default=Path("results/metallodock_public_redock.csv"))
    p.add_argument("--timesplit", type=Path, default=Path("results/timesplit_full_redock.csv"))
    p.add_argument("--rcsb", type=Path, default=Path("results/metallodock_541_redock.csv"))
    p.add_argument("--out", type=Path, default=Path("results/all_test_accuracy.json"))
    args = p.parse_args()
    parts = [
        ("public10", args.public10),
        ("timesplit_full", args.timesplit),
        ("rcsb2022", args.rcsb),
    ]
    frames = {name: _load(path, name) for name, path in parts}
    by_set = {name: _rates(fr) for name, fr in frames.items()}
    # Unique PDB: timesplit (GPDOCK site files) first, then RCSB ASU, then public-10.
    ordered = []
    for name in ("timesplit_full", "rcsb2022", "public10"):
        fr = frames[name]
        if fr.empty:
            continue
        ordered.append(fr)
    if ordered:
        cat = pd.concat(ordered, ignore_index=True)
        uniq = cat.drop_duplicates("pdb_key", keep="first")
    else:
        uniq = pd.DataFrame()
    pooled_rows = pd.concat([fr for fr in frames.values() if not fr.empty], ignore_index=True) if any(len(fr) for fr in frames.values()) else pd.DataFrame()
    by_metal = {}
    if len(uniq) and "metal" in uniq.columns:
        for metal, sub in uniq.groupby(uniq["metal"].astype(str).str.lower()):
            by_metal[str(metal)] = _rates(sub)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "threshold_angstrom": 2.0,
        "by_set": by_set,
        "pooled_all_rows": _rates(pooled_rows),
        "unique_pdb": _rates(uniq),
        "unique_pdb_by_metal": by_metal,
        "unique_pdb_n_keys": int(uniq["pdb_key"].nunique()) if len(uniq) else 0,
        "preference": "timesplit_full > rcsb2022 > public10",
        "literature_metallodock_top1": 0.786,
        "note": (
            "unique_pdb is the overall test accuracy on distinct PDB IDs. "
            "Pooled_all_rows double-counts a PDB if it appears in more than one CSV. "
            "Not the unpublished MetalloDock 541 list."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
