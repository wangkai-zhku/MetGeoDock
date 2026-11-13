#!/usr/bin/env bash
# 顺序执行三件事，避免抢资源：
#   1) 等 MetalloDock 官方口径重跑 + Vina 脱盐结束；
#   2) 跑 agri_vs 重跑（12 workers，约 1 小时，断点续跑）；
#   3) 跑 multipose top-1 vs top-3 对比（36 workers，约 18 小时）。
#
# 进程名用变量拼接，避免 pgrep -f 匹配到本脚本自身。
set -u
ROOT=/home/wangkai/cursor-task/metalloprotein-chem
cd "$ROOT" || exit 1

# 单实例锁：同名任务已在运行时直接退出，避免两个程序跑同一个任务浪费算力
exec 9>"$ROOT/logs/.lock_wait_all"
if ! flock -n 9; then
  echo "[wait_all] 已有实例在运行，退出。" >&2
  exit 1
fi
PAT_MDL="mdl_official_bench""22.py"
PAT_VINA="run_vina_desalt""_fix.py"

echo "=== 阶段1：等待前序任务结束 $(date +%F' '%T) ==="
while :; do
  n1=$(pgrep -f "$PAT_MDL" 2>/dev/null | wc -l)
  n2=$(pgrep -f "$PAT_VINA" 2>/dev/null | wc -l)
  if [ "$n1" -eq 0 ] && [ "$n2" -eq 0 ]; then break; fi
  echo "  $(date +%H:%M:%S) MetalloDock=$n1 Vina脱盐=$n2，等待…"
  sleep 120
done

echo "=== 阶段1.5：Vina 脱盐剩余靶点（2 并发，低内存）$(date +%F' '%T) ==="
nice -n 10 envs/metgeodock/bin/python -u scripts/analysis/run_vina_desalt_fix.py \
  --procs 2 >> logs/vina_desalt_fix3.log 2>&1
echo "=== 阶段1.5 结束 $(date +%F' '%T) ==="

echo "=== 阶段2：agri_vs 重跑（12 workers）$(date +%F' '%T) ==="
nice -n 10 envs/metgeodock/bin/python -u scripts/analysis/benchmark_agri_vs_parallel.py \
  --workers 12 >> logs/agri_vs_v2.log 2>&1
echo "=== 阶段2 结束 $(date +%F' '%T) ==="

echo "=== 全部排队任务完成 $(date +%F' '%T) ==="
