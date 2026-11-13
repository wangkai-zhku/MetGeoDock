#!/usr/bin/env python3
"""Check that ranking emits all ligands by default, best-to-worst.

Does not mutate data/. Writes results/ranking_validation.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.ranking import take_top  # noqa: E402


def main() -> None:
    raw = pd.DataFrame(
        {
            "ligand_id": ["c", "a", "b", "fail"],
            "score": [2.5, 0.1, 1.2, float("nan")],
            "motif": ["td", "oh", "tbp", ""],
        }
    )
    all_rows = take_top(raw, top_n=0)
    top2 = take_top(raw, top_n=2)
    top100 = take_top(raw, top_n=100)
    checks = {
        "default_n": int(len(all_rows)),
        "default_is_all": bool(len(all_rows) == 4),
        "best_first_id": str(all_rows.iloc[0]["ligand_id"]),
        "best_first_ok": bool(all_rows.iloc[0]["ligand_id"] == "a"),
        "worst_finite_last_before_nan": str(all_rows.iloc[2]["ligand_id"]),
        "nan_last": bool(pd.isna(all_rows.iloc[3]["score"])),
        "top2_n": int(len(top2)),
        "top2_ids": top2["ligand_id"].tolist(),
        "top100_equals_all": bool(len(top100) == 4),
        "columns_lead_with_rank_score": all_rows.columns[:3].tolist() == ["rank", "ligand_id", "score"],
        "ranks_are_1_based": all_rows["rank"].tolist() == [1, 2, 3, 4],
    }
    ok = all(
        [
            checks["default_is_all"],
            checks["best_first_ok"],
            checks["nan_last"],
            checks["top2_n"] == 2,
            checks["top2_ids"] == ["a", "b"],
            checks["top100_equals_all"],
            checks["columns_lead_with_rank_score"],
            checks["ranks_are_1_based"],
        ]
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ok": bool(ok),
        "checks": checks,
    }
    out = Path("results/ranking_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
