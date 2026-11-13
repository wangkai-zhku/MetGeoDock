#!/usr/bin/env bash
# MetalloDock on the cross-family-10 benchmark, same ligand sets and receptors
# as MetalGeoDock and Vina (results/crossfamily10/<t>_div.csv + processed/receptors10/).
#
# The pretrained weights live in processed/metallodock_ref/trained_models/ so they
# survive a reboot (upstream hardcoded /tmp/mdl/).
#
#   bash run_cf10_mdl.sh        # all batches
#   bash run_cf10_mdl.sh 2      # batch 2 only
set -u
R=/home/wangkai/cursor-task/metalloprotein-chem
PY=$R/envs/metgeodock/bin/python
REC=$R/processed/receptors10
OUT=$R/results/cf10_mdl
mkdir -p "$OUT" "$R/logs/cf10mdl"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

declare -A PDB=( [HDAC1]=5ICN [HDAC2]=3MAX [CA2]=3CAJ [MMP2]=1QIB [ALOX5]=3O8Y
                 [ARG2]=1PQ3 [CAPN1]=2ARY [DBH]=4ZEL [CDK2]=1HCK [METAP2]=1B6A [urease]=1FWJ )

run_batch () {
  local list="$1"
  for t in $list; do
    for p in 0 1 2 3; do
      nohup "$PY" "$R/scripts/analysis/screen_metallodock.py" \
        --target "$t" --protein "$REC/${PDB[$t]}_h.pdb" \
        --common-set "$R/results/crossfamily10/${t}_div.csv" \
        --parts 4 --part "$p" --out-dir "$OUT" \
        > "$R/logs/cf10mdl/${t}_p${p}.log" 2>&1 &
    done
    echo "$(date '+%H:%M:%S') launched metallodock $t (4 parts)"
  done
  wait
  echo "$(date '+%H:%M:%S') batch finished: $list"
}

case "${1:-all}" in
  1) run_batch "MMP2 CA2 HDAC1 HDAC2";;
  2) run_batch "CAPN1 METAP2 CDK2";;
  3) run_batch "ALOX5 DBH ARG2 urease";;
  *) run_batch "MMP2 CA2 HDAC1 HDAC2"
     run_batch "CAPN1 METAP2 CDK2"
     run_batch "ALOX5 DBH ARG2 urease";;
esac
echo "$(date '+%H:%M:%S') METALLODOCK ALL DONE"
