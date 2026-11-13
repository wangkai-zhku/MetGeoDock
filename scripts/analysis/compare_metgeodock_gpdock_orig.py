#!/usr/bin/env python3
"""Join MetGeoDock ablation on GPDOCK original SW/MM sets with GPDOCK Y/N labels.

GPDOCK folder success is a sampled (any-pose) label, not top-1 ranking.
This script only cites numbers that exist on disk.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def _rate(frame: pd.DataFrame, col: str) -> float | None:
    if frame.empty or col not in frame.columns:
        return None
    return float(frame[col].mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, default=Path("processed/gpdock_orig/index_summary.json"))
    p.add_argument("--gpdock-rmsd", type=Path, default=Path("results/gpdock_orig_rmsd.json"))
    p.add_argument("--ours", type=Path, default=Path("results/gpdock_orig_metgeodock.json"))
    p.add_argument("--ours-csv", type=Path, default=Path("results/gpdock_orig_metgeodock.csv"))
    p.add_argument("--literature", type=Path, default=Path("configs/metallodock_literature.yaml"))
    p.add_argument("--out", type=Path, default=Path("results/gpdock_orig_comparison.json"))
    args = p.parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8")) if args.index.exists() else {}
    gpdock = json.loads(args.gpdock_rmsd.read_text(encoding="utf-8")) if args.gpdock_rmsd.exists() else None
    ours = json.loads(args.ours.read_text(encoding="utf-8")) if args.ours.exists() else None
    lit = yaml.safe_load(args.literature.read_text(encoding="utf-8")) if args.literature.exists() else {}
    matched = None
    if args.ours_csv.exists():
        frame = pd.read_csv(args.ours_csv)
        ok = frame[frame.get("error", pd.Series([""] * len(frame))).fillna("") == ""]
        matched = {}
        for proto, sub in ok.groupby("protocol"):
            rec = {
                "n": int(len(sub)),
                "top1": _rate(sub, "success_top1"),
                "top3": _rate(sub, "success_top3"),
                "sampled": _rate(sub, "success_sampled"),
            }
            if "gpdock_label" in sub.columns:
                rec["n_gpdock_Y"] = int((sub["gpdock_label"] == "Y").sum())
                labs = sub.drop_duplicates(["pdb_id", "rec_id"]) if "rec_id" in sub.columns else sub
                rec["gpdock_folder_success_same_complexes"] = (
                    float((labs["gpdock_label"] == "Y").mean()) if len(labs) else None
                )
            matched[str(proto)] = rec
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpdock_original_sets": index.get("by_subset"),
        "gpdock_folder_note": (
            "OUTPUT_DOCK Y/N matches sampled RMSD<2 A (any pose), not top-1 ranking. "
            "Cite folder rates as GPDOCK sampling success on SW/SM/MM."
        ),
        "gpdock_recomputed_rmsd": gpdock,
        "metgeodock_on_same_layout": ours.get("by_protocol") if ours else None,
        "metgeodock_vs_gpdock_label": matched,
        "metallodock": {
            "keep_water": False,
            "keep_extra_metals": False,
            "prepared_bimetal_top1": (lit.get("methods") or {}).get("MetalloDock_bimetal"),
        },
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("gpdock_folder_note", "metgeodock_on_same_layout", "metgeodock_vs_gpdock_label")}, indent=2))


if __name__ == "__main__":
    main()
