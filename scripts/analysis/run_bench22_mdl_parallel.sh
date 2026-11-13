#!/usr/bin/env bash
# 22 靶点 MetalloDock 并行批处理（按靶点并行，每靶点内部串行）。
#
# 用法：nohup nice -n 10 bash scripts/analysis/run_bench22_mdl_parallel.sh 6 > logs/mdl_batch.log 2>&1 &
# 断点续跑：单靶点脚本自带「有效分数 ≥130 条则跳过」的判断，可反复执行。
set -u
ROOT=/home/wangkai/cursor-task/metalloprotein-chem
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
P="${1:-6}"
ORDER="CA2 CA12 HDAC1 HDAC2 HDAC3 HDAC6 HDAC8 MMP1 MMP2 MMP3 MMP9 MMP13 ALOX5 ARG1 ARG2 urease METAP2 CYP1A2 CYP2E1 CYP19A1 EGLN2 HIV1_integrase"

echo "=== MetalloDock 批处理：22 靶点，$P 并发，开始 $(date +%F' '%T) ==="
printf '%s\n' $ORDER | xargs -P "$P" -I{} bash scripts/analysis/run_bench22_mdl_one.sh {}
echo "=== 批处理结束 $(date +%F' '%T) ==="

echo "--- 结果统计（有效 score 条数 / 150）---"
for T in $ORDER; do
  f="results/bench22/mdl_raw/${T}.csv"
  if [ -s "$f" ]; then
    n=$(awk -F, 'NR>1 && $3!=""' "$f" | wc -l)
    printf "%s=%s " "$T" "$n"
  else
    printf "%s=缺失 " "$T"
  fi
done
echo
