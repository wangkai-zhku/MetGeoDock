#!/usr/bin/env bash
# 22 靶点 benchmark 断点续跑（2026-10-28 因内存耗尽被杀后的收尾）。
#
# 四个阶段顺序执行，每个阶段都自带断点判断：整脚本可以反复重跑，已完成的靶点会跳过。
#   阶段1  MetalGeoDock 完整打分（metalgeo_full_*）补 4 个缺口：CYP2E1 空文件 + 3 个未跑靶点
#   阶段2  Vina baseline 重跑 3 个空文件（ARG2 / EGLN2 / HIV1_integrase）
#   阶段3  MetalGeoDock 旧打分（metalgeo_* / base_*）补齐 3 个残行靶点
#   阶段4  MetalloDock 第三方法全 22 靶点（按靶点并行）
#
# 资源：整机 48 核 / 31 GB，上次 16 workers + 双流水线并行被 systemd-oomd 杀掉。
#       实测 MetalGeoDock 每个 worker 仅 ~175 MB，故其两段提到 12 并发；
#       MetalloDock（torch，每靶点 0.6-2 GB）保守用 6 并发。同一时刻只跑一个阶段。
#
# 用法：nohup nice -n 10 bash scripts/analysis/resume_bench22.sh > logs/resume_bench22.log 2>&1 &
set -u
ROOT=/home/wangkai/cursor-task/metalloprotein-chem
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/home/wangkai/cursor-task/PPO-cursor/env/conda/envs/ppo/bin:$PATH
PY=envs/metgeodock/bin/python
OUTD=results/bench22
WM=12   # MetalGeoDock worker 数
W=6     # MetalloDock 并发靶点数
ORDER="CA2 CA12 HDAC1 HDAC2 HDAC3 HDAC6 HDAC8 MMP1 MMP2 MMP3 MMP9 MMP13 ALOX5 ARG1 ARG2 urease METAP2 CYP1A2 CYP2E1 CYP19A1 EGLN2 HIV1_integrase"

banner() { echo; echo "########## $* ##########"; date +"开始 %F %T"; }

# ---------------------------------------------------------------- 阶段1
banner "阶段1 MetalGeoDock 完整打分（metalgeo_full）补缺口"
"$PY" -u scripts/analysis/run_bench22.py metalgeo \
  --prefix metalgeo_full --score-mode full --workers "$WM" \
  --targets CYP2E1,CYP19A1,EGLN2,HIV1_integrase

# ---------------------------------------------------------------- 阶段3
banner "阶段3 MetalGeoDock 旧打分补齐残行（metalgeo_*，并同步 base_*）"
"$PY" -u scripts/analysis/run_bench22.py metalgeo \
  --prefix metalgeo --score-mode base --workers "$WM" \
  --targets CYP19A1,EGLN2,HIV1_integrase
for T in CYP19A1 EGLN2 HIV1_integrase; do
  # 历史上 base_* 就是 metalgeo_* 的副本，这里保持两族同步
  cp -f "$OUTD/metalgeo_${T}.csv" "$OUTD/base_${T}.csv" && echo "[$T] 已同步 base_${T}.csv"
done

# ---------------------------------------------------------------- 阶段2（移到阶段3 之后：先补快的）
banner "阶段2 Vina baseline 补 4 个坏/空结果"
# 只跑真正缺的靶点：MMP9（旧文件 148 行 affinity 全 0.0 的坏数据）、
# ARG2 / EGLN2 / HIV1_integrase（0 字节）。
# MMP2 不重跑：113 条有效结果已在 results/bench22/vina_MMP2.csv；旧 vina_raw 在昨晚
# 被误重跑截断，但重跑需 6-8 小时（37 个配体反复 600 s 超时），性价比不成立。
nice -n 10 bash scripts/analysis/run_bench22_baselines.sh vina \
  "MMP9 ARG2 EGLN2 HIV1_integrase"

# ---------------------------------------------------------------- 阶段4
banner "阶段4 MetalloDock 第三方法全 22 靶点（$W 并发）"
printf '%s\n' $ORDER | xargs -P "$W" -I{} bash scripts/analysis/run_bench22_mdl_one.sh {}
echo "MetalloDock 并行批结束"

# ---------------------------------------------------------------- 汇总
banner "四阶段结束"
for p in metalgeo_full metalgeo base; do
  echo "--- ${p}_* 行数（目标 150）---"
  for T in $ORDER; do
    f="$OUTD/${p}_${T}.csv"
    [ -s "$f" ] && printf "%s=%s " "$T" "$(($(wc -l < "$f") - 1))" || printf "%s=缺失 " "$T"
  done
  echo
done
echo "--- vina_raw 行数 ---"
for T in $ORDER; do
  f="$OUTD/vina_raw/${T}_p0.csv"
  [ -s "$f" ] && printf "%s=%s " "$T" "$(($(wc -l < "$f") - 1))" || printf "%s=缺失 " "$T"
done
echo
echo "--- mdl_raw 行数 ---"
for T in $ORDER; do
  f="$OUTD/mdl_raw/${T}.csv"
  [ -s "$f" ] && printf "%s=%s " "$T" "$(($(wc -l < "$f") - 1))" || printf "%s=缺失 " "$T"
done
echo
