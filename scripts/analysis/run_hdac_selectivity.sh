#!/usr/bin/env bash
# HDAC subtype-selectivity validation: score the same selective inhibitors
# against all six subtypes (HDAC1/2/3/4/6/8).
#
# The whole point is a within-molecule comparison: identical ligand, six real
# receptors. Decoy choice plays no role, so decoy bias cannot explain the result.
#
#   bash run_hdac_selectivity.sh            # all three methods
#   bash run_hdac_selectivity.sh mg         # MetalGeoDock only
#   bash run_hdac_selectivity.sh vina       # Vina only
#   bash run_hdac_selectivity.sh mdl        # MetalloDock only
set -u
R=/home/wangkai/cursor-task/metalloprotein-chem
PY=$R/envs/metgeodock/bin/python
REC=$R/processed/hdac_receptors
LIG=$R/processed/hdac_selectivity_ligands.csv
export PATH=$HOME/cursor-task/PPO-cursor/env/conda/envs/ppo/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

declare -A PDB=( [HDAC1]=5ICN [HDAC2]=3MAX [HDAC3]=4A69 [HDAC4]=2VQM [HDAC6]=5EDU [HDAC8]=1T64 )
SUBS="HDAC1 HDAC2 HDAC3 HDAC4 HDAC6 HDAC8"
NPARTS=4

run_mg () {
  for s in $SUBS; do
    for p in $(seq 0 $((NPARTS-1))); do
      nohup "$PY" "$R/scripts/analysis/screen_thermo_ef.py" \
        --target "$s" --protein "$REC/${PDB[$s]}_h.pdb" --metal zn \
        --parts $NPARTS --part "$p" --common-set "$LIG" \
        --out-dir "$R/results/hdac_sel_mg" \
        > "$R/logs/hdac_sel_mg/${s}_p${p}.log" 2>&1 &
    done
  done
  wait
  echo "$(date '+%H:%M:%S') MetalGeoDock done"
}

run_vina () {
  for s in $SUBS; do
    for p in $(seq 0 $((NPARTS-1))); do
      nohup "$PY" "$R/scripts/analysis/vina_baseline.py" \
        --target "$s" --protein "$REC/${PDB[$s]}_h.pdb" --metal zn \
        --parts $NPARTS --part "$p" --common-set "$LIG" \
        --exhaustiveness 4 --box 22 --cpu 1 \
        --out-dir "$R/results/hdac_sel_vina" \
        > "$R/logs/hdac_sel_vina/${s}_p${p}.log" 2>&1 &
    done
  done
  wait
  echo "$(date '+%H:%M:%S') Vina done"
}

run_mdl () {
  for s in $SUBS; do
    for p in $(seq 0 $((NPARTS-1))); do
      nohup "$PY" "$R/scripts/analysis/screen_metallodock.py" \
        --target "$s" --protein "$REC/${PDB[$s]}_h.pdb" \
        --parts $NPARTS --part "$p" --common-set "$LIG" \
        --out-dir "$R/results/hdac_sel_mdl" \
        > "$R/logs/hdac_sel_mdl/${s}_p${p}.log" 2>&1 &
    done
  done
  wait
  echo "$(date '+%H:%M:%S') MetalloDock done"
}

mkdir -p "$R/logs/hdac_sel_mg" "$R/logs/hdac_sel_vina" "$R/logs/hdac_sel_mdl" \
         "$R/results/hdac_sel_mg" "$R/results/hdac_sel_vina" "$R/results/hdac_sel_mdl"

case "${1:-all}" in
  mg)   run_mg;;
  vina) run_vina;;
  mdl)  run_mdl;;
  *)    run_mg; run_vina; run_mdl;;
esac
echo "$(date '+%H:%M:%S') HDAC SELECTIVITY ALL DONE"
