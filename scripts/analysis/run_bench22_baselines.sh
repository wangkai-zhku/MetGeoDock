#!/usr/bin/env bash
# 22 靶点 benchmark：Vina 与 MetalloDock 批处理。
# 使用与 MetalGeoDock 完全相同的 common set（results/bench22/common/<T>.csv），
# 保证三方法在同一批配体、同一标签上对比。
#
# 用法：
#   bash scripts/analysis/run_bench22_baselines.sh vina
#   bash scripts/analysis/run_bench22_baselines.sh metallodock
#   bash scripts/analysis/run_bench22_baselines.sh vina "MMP9 ARG2 EGLN2 HIV1_integrase"   # 只跑指定靶点
set -u
ROOT=/home/wangkai/cursor-task/metalloprotein-chem
cd "$ROOT" || exit 1
export PATH=/home/wangkai/cursor-task/PPO-cursor/env/conda/envs/ppo/bin:$PATH
export OMP_NUM_THREADS=1

METHOD="${1:-vina}"
ONLY="${2:-}"          # 可选：空格分隔的靶点白名单
COMMON=results/bench22/common
OUTD=results/bench22
PY=envs/metgeodock/bin/python

declare -A REC=(
  [CA2]=processed/receptors10/3CAJ_h.pdb
  [CA12]=processed/receptors22/CA12_h.pdb
  [HDAC1]=processed/receptors10/5ICN_h.pdb
  [HDAC2]=processed/receptors10/3MAX_h.pdb
  [HDAC3]=processed/hdac_receptors/4A69_h.pdb
  [HDAC6]=processed/hdac_receptors/5EDU_CD2_h.pdb
  [HDAC8]=processed/hdac_receptors/2V5W_A_h.pdb
  [MMP1]=processed/receptors22/MMP1_h.pdb
  [MMP2]=processed/receptors10/1QIB_h.pdb
  [MMP3]=processed/receptors22/MMP3_h.pdb
  [MMP9]=processed/receptors22/MMP9_h.pdb
  [MMP13]=processed/receptors22/MMP13_h.pdb
  [ALOX5]=processed/receptors10/3O8Y_h.pdb
  [ARG1]=processed/receptors22/ARG1_h.pdb
  [ARG2]=processed/receptors10/1PQ3_h.pdb
  [urease]=processed/receptors10/1FWJ_h.pdb
  [METAP2]=processed/receptors10/1B6A_h.pdb
  [CYP1A2]=processed/receptors22/CYP1A2_h.pdb
  [CYP2E1]=processed/receptors22/CYP2E1_h.pdb
  [CYP19A1]=processed/receptors22/CYP19A1_h.pdb
  [EGLN2]=processed/receptors22/EGLN2_h.pdb
  [HIV1_integrase]=processed/receptors22/HIV1_integrase_h.pdb
)

ORDER="CA2 CA12 HDAC1 HDAC2 HDAC3 HDAC6 HDAC8 MMP1 MMP2 MMP3 MMP9 MMP13 ALOX5 ARG1 ARG2 urease METAP2 CYP1A2 CYP2E1 CYP19A1 EGLN2 HIV1_integrase"

for T in $ORDER; do
  CS="$COMMON/$T.csv"
  [ -f "$CS" ] || { echo "[$T] 缺 common set，跳过"; continue; }
  if [ -n "$ONLY" ]; then
    case " $ONLY " in *" $T "*) ;; *) continue ;; esac
  fi
  # 完成判据必须落在真实输出文件上，且按「有效分数条数」而非行数：
  # vina 曾写出 148 行 affinity 全 0.0（无 error 标记）的坏数据，行数判断会误判为完成。
  if [ "$METHOD" = "vina" ]; then
    PRO="$OUTD/vina_raw/${T}_p0.csv"
    n_ok=$(awk -F, 'NR>1 && $3!="" && $3+0!=0' "$PRO" 2>/dev/null | wc -l)
  else
    PRO="$OUTD/mdl_raw/${T}.csv"
    n_ok=$(awk -F, 'NR>1 && $3!=""' "$PRO" 2>/dev/null | wc -l)
  fi
  if [ "$n_ok" -ge 130 ]; then
    echo "[$T] 已完成（$n_ok 条有效分数），跳过"; continue
  fi
  echo "=== $T ($METHOD) 现有 $n_ok 条有效分数，开跑 ==="
  if [ "$METHOD" = "vina" ]; then
    $PY -u scripts/analysis/vina_baseline.py \
      --target "$T" --protein "${REC[$T]}" --common-set "$CS" \
      --out-dir "$OUTD/vina_raw" --exhaustiveness 4 --cpu 4 2>&1 | tail -2
  else
    $PY -u scripts/analysis/screen_metallodock.py \
      --target "$T" --protein "${REC[$T]}" --common-set "$CS" \
      --out-dir "$OUTD/mdl_raw" 2>&1 | tail -2
  fi
done
echo "=== $METHOD 全部完成 ==="
