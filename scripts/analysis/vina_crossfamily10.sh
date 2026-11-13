#!/usr/bin/env bash
# AutoDock Vina on the cross-family benchmark. One group at a time (same reason as the
# MetalGeoDock script: thread oversubscription, not CPU count, was the limit).
set -u
R=/home/wangkai/cursor-task/metalloprotein-chem
PY=$R/envs/metgeodock/bin/python
REC=$R/processed/receptors10
OUT=$R/results/cf10_vina
mkdir -p "$OUT" "$R/logs/cf10vina"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

declare -A PDB=( [HDAC1]=5ICN [HDAC2]=3MAX [CA2]=3CAJ [MMP2]=1QIB [ALOX5]=3O8Y
                 [ARG2]=1PQ3 [CAPN1]=2ARY [DBH]=4ZEL [CDK2]=1HCK [METAP2]=1B6A [urease]=1FWJ )
declare -A MET=( [HDAC1]=zn [HDAC2]=zn [CA2]=zn [MMP2]=zn [ALOX5]=fe2
                 [ARG2]=mn [CAPN1]=ca [DBH]=cu [CDK2]=mg [METAP2]=co [urease]=ni )

case "${1:-0}" in
  0) LIST="MMP2 CA2 HDAC1";; 1) LIST="CAPN1 METAP2 CDK2";; 2) LIST="ALOX5 DBH urease";;
  3) LIST="ARG2";; 4) LIST="HDAC2";; *) LIST="";;
esac

for t in $LIST; do
  for p in 0 1 2 3; do
    nohup "$PY" "$R/scripts/analysis/vina_baseline.py" \
      --target "$t" --protein "$REC/${PDB[$t]}_h.pdb" --metal "${MET[$t]}" \
      --common-set "$R/results/crossfamily10/${t}_div.csv" \
      --parts 4 --part "$p" --exhaustiveness 4 --box 22 \
      --out-dir "$OUT" > "$R/logs/cf10vina/${t}_p${p}.log" 2>&1 &
  done
  echo "launched vina $t"
done
