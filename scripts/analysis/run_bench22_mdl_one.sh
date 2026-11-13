#!/usr/bin/env bash
# 单靶点 MetalloDock 打分（供 xargs -P 并行调度）。
# 用法：bash scripts/analysis/run_bench22_mdl_one.sh <TARGET>
set -u
ROOT=/home/wangkai/cursor-task/metalloprotein-chem
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/home/wangkai/cursor-task/PPO-cursor/env/conda/envs/ppo/bin:$PATH
PY=envs/metgeodock/bin/python
OUTD=results/bench22
COMMON=$OUTD/common
T="${1:?需要靶点名}"
CS="$COMMON/$T.csv"
PRO="$OUTD/mdl_raw/$T.csv"

rec_of() {
  case "$1" in
    CA2)             echo processed/receptors10/3CAJ_h.pdb ;;
    CA12)            echo processed/receptors22/CA12_h.pdb ;;
    HDAC1)           echo processed/receptors10/5ICN_h.pdb ;;
    HDAC2)           echo processed/receptors10/3MAX_h.pdb ;;
    HDAC3)           echo processed/hdac_receptors/4A69_h.pdb ;;
    HDAC6)           echo processed/hdac_receptors/5EDU_CD2_h.pdb ;;
    HDAC8)           echo processed/hdac_receptors/2V5W_A_h.pdb ;;
    MMP1)            echo processed/receptors22/MMP1_cat_h.pdb ;;
    MMP2)            echo processed/receptors10/1QIB_h.pdb ;;
    MMP3)            echo processed/receptors22/MMP3_h.pdb ;;
    MMP9)            echo processed/receptors22/MMP9_cat_h.pdb ;;
    MMP13)           echo processed/receptors22/MMP13_h.pdb ;;
    ALOX5)           echo processed/receptors10/3O8Y_h.pdb ;;
    ARG1)            echo processed/receptors22/ARG1_h.pdb ;;
    ARG2)            echo processed/receptors10/1PQ3_h.pdb ;;
    urease)          echo processed/receptors10/1FWJ_h.pdb ;;
    METAP2)          echo processed/receptors10/1B6A_h.pdb ;;
    CYP1A2)          echo processed/receptors22/CYP1A2_h.pdb ;;
    CYP2E1)          echo processed/receptors22/CYP2E1_h.pdb ;;
    CYP19A1)         echo processed/receptors22/CYP19A1_h.pdb ;;
    EGLN2)           echo processed/receptors22/EGLN2_h.pdb ;;
    HIV1_integrase)  echo processed/receptors22/HIV1_cat_h.pdb ;;
    *) echo "" ;;
  esac
}

metal_of() {
  case "$1" in
    ALOX5)              echo fe2 ;;
    ARG1|ARG2)          echo mn ;;
    urease)             echo ni ;;
    METAP2)             echo co ;;
    CYP1A2|CYP2E1|CYP19A1|EGLN2) echo fe3 ;;
    HIV1_integrase)     echo mg ;;
    *)                  echo zn ;;
  esac
}

[ -f "$CS" ] || { echo "[$T] 缺 common set，跳过"; exit 0; }
REC=$(rec_of "$T")
[ -n "$REC" ] && [ -f "$REC" ] || { echo "[$T] 缺受体，跳过"; exit 1; }
# 完成判据用「有效 score 条数」，与 run_bench22_baselines.sh 保持一致
n_ok=$(awk -F, 'NR>1 && $3!=""' "$PRO" 2>/dev/null | wc -l)
if [ "$n_ok" -ge 130 ]; then
  echo "[$T] 已完成（$n_ok 条有效分数），跳过"; exit 0
fi

mkdir -p logs/bench22_mdl
echo "=== $T (metallodock) 开始 $(date +%H:%M:%S) ==="
nice -n 10 "$PY" -u scripts/analysis/screen_metallodock.py \
  --target "$T" --protein "$REC" --common-set "$CS" --out-dir "$OUTD/mdl_raw" \
  --metal "$(metal_of "$T")" \
  > "logs/bench22_mdl/${T}.log" 2>&1
rc=$?
rows=$([ -s "$PRO" ] && echo "$(($(wc -l < "$PRO") - 1))" || echo 0)
n_ok=$(awk -F, 'NR>1 && $3!=""' "$PRO" 2>/dev/null | wc -l)
echo "=== $T (metallodock) 结束 rc=$rc 行数=$rows/150 有效=$n_ok $(date +%H:%M:%S) ==="
[ "$rc" -eq 0 ] && [ "$n_ok" -ge 130 ]
