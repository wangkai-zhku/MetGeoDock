#!/usr/bin/env bash
# Retrain ML runs + plot curves. From repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${ROOT}/envs/metgeodock/bin/python"
if [[ ! -x "$PY" ]]; then PY="$(command -v python3)"; fi

echo "== pose_rank_v1 =="
"$PY" scripts/ml/train_pose_rank_run.py --epochs 40 --seed 42

echo "== coordnet_agri_v1 =="
"$PY" scripts/ml/train_coordnet_agri_run.py --epochs 40 --seed 42

echo "== plot D11-D14 =="
"$PY" scripts/manuscript/figures/plot_ml_run_curves.py

echo "done. see processed/ml_runs/ and figures/results/D11*.png"
