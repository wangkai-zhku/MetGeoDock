#!/usr/bin/env bash
# 按 MetalloDock 官方虚拟筛选口径重跑 22 靶点（1:10 配体集，批处理）。
#
# 为什么只跑 1:10：1:10 的配体集前 150 个与 1:5 集完全一致（已逐条校验），
# 所以 1:5 的结果可以从 1:10 结果里筛出前 150 个得到 —— 同一批计算、口径更严，
# 还省掉 22×150 次重复对接（约 9 小时）。
#
# 用法：nohup nice -n 10 bash scripts/analysis/mdl_official_batch.sh 8 > logs/mdl_official_batch.log 2>&1 &
#   参数 = 并发靶点数（每并发约 1.5 GB 内存）
set -u
ROOT=/home/wangkai/cursor-task/metalloprotein-chem
cd "$ROOT" || exit 1

# 单实例锁：同名任务已在运行时直接退出，避免两个程序跑同一个任务浪费算力
exec 9>"$ROOT/logs/.lock_mdl_batch"
if ! flock -n 9; then
  echo "[mdl_batch] 已有实例在运行，退出。" >&2
  exit 1
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
P="${1:-8}"
PY=envs/metgeodock/bin/python
ORDER="CA2 CA12 HDAC1 HDAC2 HDAC3 HDAC6 HDAC8 MMP1 MMP2 MMP3 MMP9 MMP13 ALOX5 ARG1 ARG2 urease METAP2 CYP1A2 CYP2E1 CYP19A1 EGLN2 HIV1_integrase"

one() {
  T="$1"
  echo "=== $T 开始 $(date +%H:%M:%S) ==="
  nice -n 10 "$PY" -u scripts/analysis/mdl_official_bench22.py \
    --set 1to10 --targets "$T" --batch 30 > "logs/mdl_official_$T.log" 2>&1
  n=$(awk -F, 'NR>1 && $3!=""' "results/mdl_official/1to10/${T}.csv" 2>/dev/null | wc -l)
  echo "=== $T 结束 $(date +%H:%M:%S) 有效 $n/1100 ==="
}
export -f one
export PY

echo "=== MetalloDock 官方口径批处理开始 $(date +%F' '%T)（$P 并发）==="
printf '%s\n' $ORDER | xargs -P "$P" -I{} bash -c 'one {}'
echo "=== 全部结束 $(date +%F' '%T) ==="

echo "--- 各靶点有效条数 ---"
for T in $ORDER; do
  f="results/mdl_official/1to10/${T}.csv"
  n=$(awk -F, 'NR>1 && $3!=""' "$f" 2>/dev/null | wc -l)
  printf "%s=%s " "$T" "$n"
done
echo
