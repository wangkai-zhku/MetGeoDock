#!/usr/bin/env bash
# Resume the cross-family-10 MetalGeoDock screen on the trimmed receptors.
#
# One run of screen_thermo_ef.py writes every term the three scoring variants
# need (total = geometry, g_buried = burial, n_rotatable/buried_area = CAT), so
# a single docking pass yields MetGeoDock / +burial / +burial+CAT at once.
#
# Targets here are the six whose receptor changed in prepare_receptors10.sh
# (oligomer trimmed to one biological unit) or that never finished:
#   HDAC1  CAPN1  HDAC2   -> batch 1
#   ALOX5  DBH    ARG2    -> batch 2
# MMP2 / CA2 / METAP2 / CDK2 / urease keep their existing results: their
# receptors are single-chain (or the urease alpha/beta/gamma unit) and were
# rebuilt bit-identically.
#
#   bash run_cf10_remaining.sh            # run both batches
#   bash run_cf10_remaining.sh 1          # batch 1 only
set -u
R=/home/wangkai/cursor-task/metalloprotein-chem
PY=$R/envs/metgeodock/bin/python
REC=$R/processed/receptors10
OUT=$R/results/cf10_thermo
mkdir -p "$OUT" "$R/logs/cf10"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

declare -A PDB=( [HDAC1]=5ICN [HDAC2]=3MAX [CA2]=3CAJ [MMP2]=1QIB [ALOX5]=3O8Y
                 [ARG2]=1PQ3 [CAPN1]=2ARY [DBH]=4ZEL [CDK2]=1HCK [METAP2]=1B6A [urease]=1FWJ )
declare -A MET=( [HDAC1]=zn [HDAC2]=zn [CA2]=zn [MMP2]=zn [ALOX5]=fe2
                 [ARG2]=mn [CAPN1]=ca [DBH]=cu [CDK2]=mg [METAP2]=co [urease]=ni )

run_batch () {
  local list="$1"
  for t in $list; do
    for p in 0 1 2 3; do
      nohup "$PY" "$R/scripts/analysis/screen_thermo_ef.py" \
        --target "$t" --protein "$REC/${PDB[$t]}_h.pdb" --metal "${MET[$t]}" \
        --parts 4 --part "$p" --out-dir "$OUT" \
        --common-set "$R/results/crossfamily10/${t}_div.csv" \
        > "$R/logs/cf10/${t}_p${p}.log" 2>&1 &
    done
    echo "$(date '+%H:%M:%S') launched $t (4 parts)"
  done
  wait
  echo "$(date '+%H:%M:%S') batch finished: $list"
}

case "${1:-all}" in
  1) run_batch "HDAC1 CAPN1 HDAC2";;
  2) run_batch "ALOX5 DBH ARG2";;
  *) run_batch "HDAC1 CAPN1 HDAC2"; run_batch "ALOX5 DBH ARG2";;
esac
echo "$(date '+%H:%M:%S') ALL DONE"
