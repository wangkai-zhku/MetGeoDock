#!/usr/bin/env python3
"""Named-pesticide ranks and per-target enrichment from agri_vs.csv.

Does not re-dock. Rank is MetGeo score (lower better), ties broken by
centroid_pen then -contact, matching pose sort keys.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _ok(frame: pd.DataFrame) -> pd.DataFrame:
    if "error" not in frame.columns:
        return frame
    err = frame["error"].fillna("").astype(str)
    return frame[err.isin(("", "nan"))].copy()


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["contact"] = pd.to_numeric(out.get("contact"), errors="coerce").fillna(0.0)
    out["centroid_pen"] = pd.to_numeric(out.get("centroid_pen"), errors="coerce").fillna(0.0)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.sort_values(["score", "centroid_pen"], ascending=[True, True], kind="mergesort")
    out["rank"] = range(1, len(out) + 1)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("results/agri_vs.csv"))
    p.add_argument("--vs-json", type=Path, default=Path("results/agri_vs.json"))
    p.add_argument("--out", type=Path, default=Path("results/agri_vs_named_ranks.json"))
    args = p.parse_args()
    frame = _ok(pd.read_csv(args.csv))
    vs = json.loads(args.vs_json.read_text(encoding="utf-8")) if args.vs_json.exists() else {}
    named = {}
    for tid, sub in frame.groupby("target"):
        ranked = _rank(sub)
        cols = ["rank", "ligand_id", "name", "source", "score", "pchembl"]
        act = ranked[ranked["label"] == 1][cols].copy()
        act["pchembl"] = pd.to_numeric(act["pchembl"], errors="coerce")
        act = act.where(act.notna(), None)
        named[str(tid)] = {
            "n": int(len(ranked)),
            "n_actives": int((ranked["label"] == 1).sum()),
            "active_ranks": json.loads(act.to_json(orient="records")),
            "best_active_rank": int(act["rank"].min()) if len(act) else None,
            "n_actives_in_top5": int(((ranked["rank"] <= 5) & (ranked["label"] == 1)).sum()),
            "n_actives_in_top10": int(((ranked["rank"] <= 10) & (ranked["label"] == 1)).sum()),
            "top10": ranked.head(10)[["rank", "name", "label", "score"]].to_dict(orient="records"),
        }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "by_target": named,
        "agri_vs_metrics": vs.get("by_target"),
        "note": "Rank uses MetGeo score; GS is the pesticide-box success case.",
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
