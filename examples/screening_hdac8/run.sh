#!/usr/bin/env bash
# Screen a 20-molecule library against HDAC8 (Zn hydrolase, open 2V5W receptor).
# 8 known ChEMBL inhibitors + 12 decoys; runtime about 2-3 min on one CPU core.
set -euo pipefail
cd "$(dirname "$0")/../.."

metgeodock screen \
  --protein examples/screening_hdac8/receptor.pdb \
  --library examples/screening_hdac8/library.sdf \
  --metal zn --scorer metgeo \
  --coord-water 2 --n-spin 4 --max-poses 4 \
  --out results/example_hdac8_scores.csv

echo
echo "Top of the ranked library (lower score = better):"
head -6 results/example_hdac8_scores.csv
echo
echo "Compare with examples/screening_hdac8/labels.csv to see where the actives land."
