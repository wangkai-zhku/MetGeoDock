#!/usr/bin/env python3
"""Estimate agri VS wall time from library size and the last completed log.

Does not re-dock. Parses logs/benchmark_agri_vs.log for the n_lib=168 run.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ INFO (?P<body>.*)$"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lib", type=Path, default=Path("processed/agri_vs/library_all.csv"))
    p.add_argument("--summary", type=Path, default=Path("processed/agri_vs/build_summary.json"))
    p.add_argument("--log", type=Path, default=Path("logs/benchmark_agri_vs.log"))
    p.add_argument("--out", type=Path, default=Path("results/agri_vs_runtime.json"))
    args = p.parse_args()
    lib = pd.read_csv(args.lib)
    n_mol = int(len(lib))
    n_target = 3
    n_dock = n_mol * n_target
    meta = json.loads(args.summary.read_text(encoding="utf-8")) if args.summary.exists() else {}
    events = []
    start_idx = None
    raw = args.log.read_text(encoding="utf-8", errors="replace").splitlines() if args.log.exists() else []
    for i, line in enumerate(raw):
        m = LINE.match(line)
        if not m:
            continue
        ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
        body = m.group("body")
        if "n_lib=168" in body and body.startswith("hppd "):
            start_idx = len(events)
        events.append({"ts": ts, "body": body})
    run = events[start_idx:] if start_idx is not None else []
    by_t: dict[str, list[float]] = {}
    prev = None
    for ev in run:
        mm = re.match(r"^(hppd|gs|igpd) (\S+) label=", ev["body"])
        if not mm:
            if ev["body"].startswith("wrote "):
                prev = ev
            continue
        tid = mm.group(1)
        if prev is not None:
            dt = (ev["ts"] - prev["ts"]).total_seconds()
            if 0.0 < dt < 600.0:
                by_t.setdefault(tid, []).append(dt)
        prev = ev
    per_target = {}
    for tid, dts in by_t.items():
        s = pd.Series(dts)
        per_target[tid] = {
            "n_timed": int(len(s)),
            "median_s": float(s.median()),
            "mean_s": float(s.mean()),
            "p90_s": float(s.quantile(0.90)),
            "sum_s": float(s.sum()),
        }
    if run:
        wall_s = (run[-1]["ts"] - run[0]["ts"]).total_seconds()
    else:
        wall_s = None
    medians = [v["median_s"] for v in per_target.values()] if per_target else []
    typical_s = float(pd.Series(medians).median()) if medians else None
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_unique_molecules": n_mol,
        "n_targets": n_target,
        "n_dock_jobs": n_dock,
        "protocol": "same 168 molecules docked to hppd, gs, igpd; coord_water=2, n_spin=6",
        "observed_last_run_wall_s": wall_s,
        "observed_last_run_wall_min": None if wall_s is None else float(wall_s / 60.0),
        "observed_last_run_wall_h": None if wall_s is None else float(wall_s / 3600.0),
        "per_target_ligand_s": per_target,
        "estimate_if_rerun_median_s": None if typical_s is None else float(typical_s * n_dock),
        "estimate_if_rerun_median_min": None if typical_s is None else float(typical_s * n_dock / 60.0),
        "estimate_if_rerun_median_h": None if typical_s is None else float(typical_s * n_dock / 3600.0),
        "library_actives": {
            tid: {"n_actives": t.get("n_actives"), "n_decoys": t.get("n_decoys")}
            for tid, t in (meta.get("targets") or {}).items()
        },
        "note": "Wall time is from the completed 168-ligand run in logs/benchmark_agri_vs.log. Median estimate uses per-ligand intervals in that run.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
