#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
metgeodock dock \
  --protein examples/minimal/protein.pdb \
  --ligand examples/minimal/ligand.pdb \
  --metal zn \
  --coord-water 2 --n-spin 8 --max-poses 20 \
  --reference examples/minimal/ligand_ref.pdb \
  --out processed/dock/example_minimal --write-poses
