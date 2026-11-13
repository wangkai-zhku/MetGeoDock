#!/usr/bin/env python3
"""Driver for the MetGeoDock analysis pipeline. Logs the exact command sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="smaller samples for a smoke run")
    p.add_argument("--skip-chembl", action="store_true")
    args = p.parse_args()
    py = sys.executable
    per = 4 if args.quick else 12
    rec = 2 if args.quick else 4
    decoys = 9 if args.quick else 19
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    Path("logs/run_pipeline_meta.txt").write_text(
        f"utc={stamp}\npython={py}\nquick={args.quick}\n",
        encoding="utf-8",
    )
    run([py, "scripts/analysis/index_crystal_set.py"])
    if not args.skip_chembl:
        run([py, "scripts/data/fetch_chembl_activities.py", "--skip-existing"])
        run([py, "scripts/data/fetch_bindingdb.py"])
    run([py, "scripts/data/fetch_metallodock_testset.py"])
    run([py, "scripts/data/fetch_rcsb_dates.py"])
    run([py, "scripts/analysis/benchmark_redock.py", "--per-metal", str(per)])
    run([py, "scripts/analysis/benchmark_vs.py", "--receptors-per-metal", str(rec), "--decoys", str(decoys)])
    run([py, "scripts/analysis/benchmark_metallodock_set.py"])
    run([py, "scripts/analysis/build_ml_features.py", "--per-metal", str(per)])
    run([py, "scripts/analysis/train_ml_scorer.py"])
    run([py, "scripts/analysis/ablation_pose_rank.py"])
    run([py, "scripts/qm/prepare_qmmm_clusters.py", "--per-metal", "2" if args.quick else "3"])
    run([py, "scripts/qm/run_xtb_clusters.py"])
    run([py, "scripts/analysis/plot_figures.py"])
    run([py, "scripts/analysis/validate_results.py"])


if __name__ == "__main__":
    main()
