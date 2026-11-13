#!/usr/bin/env bash
# AutoDock Vina on the cross-family-10 benchmark, water-free receptors.
#
# Uses the same fixed ligand sets as MetalGeoDock (results/crossfamily10/<t>_div.csv)
# and the trimmed, water-stripped receptors in processed/receptors10/.
# vina_baseline.py also filters HETATM HOH defensively before the obabel step.
#
#   bash run_cf10_vina.sh          # all batches
#   bash run_cf10_vina.sh 1        # batch 1 only
set -u
R=/home/wangkai/cursor-task/metalloprotein-chem
PY=$R/envs/metgeodock/bin/python
REC=$R/processed/receptors10
OUT=$R/results/cf10_vina
mkdir -p "$OUT" "$R/logs/cf10vina"
export PATH=$HOME/cursor-task/PPO-cursor/env/conda/envs/ppo/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

declare -A PDB=( [HDAC1]=5ICN [HDAC2]=3MAX [CA2]=3CAJ [MMP2]=1QIB [ALOX5]=3O8Y
                 [ARG2]=1PQ3 [CAPN1]=2ARY [DBH]=4ZEL [CDK2]=1HCK [METAP2]=1B6A [urease]=1FWJ )
declare -A MET=( [HDAC1]=zn [HDAC2]=zn [CA2]=zn [MMP2]=zn [ALOX5]=fe2
                 [ARG2]=mn [CAPN1]=ca [DBH]=cu [CDK2]=mg [METAP2]=co [urease]=ni )

run_batch () {
  local list="$1"
  for t in $list; do
    for p in 0 1 2 3; do
      nohup "$PY" "$R/scripts/analysis/vina_baseline.py" \
        --target "$t" --protein "$REC/${PDB[$t]}_h.pdb" --metal "${MET[$t]}" \
        --common-set "$R/results/crossfamily10/${t}_div.csv" \
        --parts 4 --part "$p" --exhaustiveness 4 --box 22 --cpu 1 \
        --out-dir "$OUT" > "$R/logs/cf10vina/${t}_p${p}.log" 2>&1 &
    done
    echo "$(date '+%H:%M:%S') launched vina $t (4 parts)"
  done
  wait
  echo "$(date '+%H:%M:%S') batch finished: $list"
}

case "${1:-all}" in
  1) run_batch "MMP2 CA2 HDAC1";;
  2) run_batch "CAPN1 METAP2 CDK2 HDAC2";;
  3) run_batch "ALOX5 DBH ARG2 urease";;
  *) run_batch "MMP2 CA2 HDAC1"
     run_batch "CAPN1 METAP2 CDK2 HDAC2"
     run_batch "ALOX5 DBH ARG2 urease";;
esac
echo "$(date '+%H:%M:%S') VINA ALL DONE"
