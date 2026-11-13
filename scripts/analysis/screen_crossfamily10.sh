#!/usr/bin/env bash
# Screen the cross-family benchmark with MetalGeoDock.
#
# Concurrency is capped on purpose: each Python process spawns ~54 threads through numpy/BLAS,
# so 40 processes produced 2,160 threads on 48 cores and the machine thrashed. We pin every
# process to one thread and run GROUPS targets at a time.
#
#   bash screen_crossfamily10.sh <group>          # 0,1,2 or "all"
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

GROUP0="MMP2 CA2 HDAC1"          # small receptors, fast
GROUP1="CAPN1 METAP2 CDK2"       # medium
GROUP2="ALOX5 DBH urease"        # large receptors
GROUP3="ARG2"
GROUP4="HDAC2"

case "${1:-all}" in
  0) LIST="$GROUP0";; 1) LIST="$GROUP1";; 2) LIST="$GROUP2";;
  3) LIST="$GROUP3";; 4) LIST="$GROUP4";; *) LIST="$GROUP0 $GROUP1 $GROUP2 $GROUP3 $GROUP4";;
esac

for t in $LIST; do
  for p in 0 1 2 3; do
    nohup "$PY" "$R/scripts/analysis/screen_thermo_ef.py" \
      --target "$t" --protein "$REC/${PDB[$t]}_h.pdb" --metal "${MET[$t]}" \
      --parts 4 --part "$p" --out-dir "$OUT" \
      --common-set "$R/results/crossfamily10/${t}_div.csv" \
      > "$R/logs/cf10/${t}_p${p}.log" 2>&1 &
  done
  echo "launched $t"
done
