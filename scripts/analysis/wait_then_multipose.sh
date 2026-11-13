#!/usr/bin/env bash
# 等前序任务（MetalloDock 官方口径重跑、Vina 脱盐）结束后，自动启动 22 靶点 top-1 vs top-3 对比。
#
# 注意：这里用变量拼接进程名，避免 pgrep -f 匹配到本脚本自身（此前踩过 pkill -f 自杀的坑）。
#
# 用法：nohup bash scripts/analysis/wait_then_multipose.sh 36 > logs/wait_multipose.log 2>&1 &
set -u
ROOT=/home/wangkai/cursor-task/metalloprotein-chem
cd "$ROOT" || exit 1
W="${1:-36}"
PAT_MDL="mdl_official_bench""22.py"
PAT_VINA="run_vina_desalt""_fix.py"

echo "=== 等待前序任务结束 $(date +%F' '%T) ==="
while :; do
  n1=$(pgrep -f "$PAT_MDL" 2>/dev/null | wc -l)
  n2=$(pgrep -f "$PAT_VINA" 2>/dev/null | wc -l)
  if [ "$n1" -eq 0 ] && [ "$n2" -eq 0 ]; then break; fi
  echo "  $(date +%H:%M:%S) MetalloDock=$n1 Vina脱盐=$n2，继续等待…"
  sleep 120
done
echo "=== 前序任务已结束，启动 multipose $(date +%F' '%T)（$W workers）==="
nice -n 10 envs/metgeodock/bin/python -u scripts/analysis/run_multipose_bench22.py \
  --workers "$W" --all
echo "=== multipose 全部完成 $(date +%F' '%T) ==="
