#!/usr/bin/env bash
# 22 靶点 benchmark，actives:decoys = 1:10 重评（三方法全跑）。
#
# 与 1:5 版完全隔离：配体集 processed/ligand_sets22_1to10，结果 results/bench22_1to10。
# 旧的 results/bench22（1:5）不动。
#
# 已预填 bench22 算过的 150 个配体（scripts/analysis/seed_bench22_1to10.py），
# 所以每靶点只需补算 950 个，跑完自动汇总。
#
# 资源：整机 48 核 / 31 GB。
#   MetalGeoDock 每 worker ~175 MB（纯 CPU）→ 可高并发
#   MetalloDock   每并发 ~1.3 GB（含 torch）→ 控制并发数
#   三方法同批并行会争抢 CPU，故按「占用核心」分配。
#
# 用法：
#   nohup nice -n 10 bash scripts/analysis/run_bench22_1to10.sh 24 8 3 > logs/bench22_1to10.log 2>&1 &
#   参数：MetalGeoDock workers / MetalloDock 并发 / Vina 并行靶点数
set -u
ROOT=/home/wangkai/cursor-task/metalloprotein-chem
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/home/wangkai/cursor-task/PPO-cursor/env/conda/envs/ppo/bin:$PATH
PY=envs/metgeodock/bin/python
SETS=processed/ligand_sets22_1to10
OUTD=results/bench22_1to10
COMMON=$OUTD/common
W_MG="${1:-24}"
W_MDL="${2:-8}"
P_VINA="${3:-3}"
ORDER="CA2 CA12 HDAC1 HDAC2 HDAC3 HDAC6 HDAC8 MMP1 MMP2 MMP3 MMP9 MMP13 ALOX5 ARG1 ARG2 urease METAP2 CYP1A2 CYP2E1 CYP19A1 EGLN2 HIV1_integrase"

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
export -f rec_of metal_of
export PY SETS OUTD COMMON

banner() { echo; echo "########## $* ##########"; date +"开始 %F %T"; }

# ---------------------------------------------------------------- 1) MetalGeoDock
run_mg() {
  banner "1) MetalGeoDock（$W_MG workers）"
  "$PY" -u scripts/analysis/run_bench22.py metalgeo \
    --prefix metalgeo_full --score-mode full \
    --sets-dir "$SETS" --out-dir "$OUTD" --dec-cap 1000 --workers "$W_MG"
}

# ---------------------------------------------------------------- 2) Vina
run_vina_one() {
  T="$1"; REC=$(rec_of "$T")
  PRO="$OUTD/vina_raw/${T}_p0.csv"
  n_ok=$(awk -F, 'NR>1 && $3!="" && $3+0!=0' "$PRO" 2>/dev/null | wc -l)
  if [ "$n_ok" -ge 1000 ]; then echo "[$T] vina 已完成 ($n_ok)，跳过"; return 0; fi
  echo "=== $T (vina) 现有 $n_ok 有效，开跑 $(date +%H:%M:%S) ==="
  nice -n 10 "$PY" -u scripts/analysis/vina_baseline.py --target "$T" --protein "$REC" \
    --common-set "$COMMON/$T.csv" --out-dir "$OUTD/vina_raw" \
    --metal "$(metal_of "$T")" --exhaustiveness 4 --cpu 4 \
    > "logs/bench22_1to10_vina_$T.log" 2>&1
  echo "=== $T (vina) 结束 $(date +%H:%M:%S) ==="
}
export -f run_vina_one

run_vina() {
  banner "2) Vina（$P_VINA 靶点并行 × 4 核）"
  printf '%s\n' $ORDER | xargs -P "$P_VINA" -I{} bash -c 'run_vina_one {}'
}

# ---------------------------------------------------------------- 3) MetalloDock
run_mdl_one() {
  T="$1"; REC=$(rec_of "$T")
  PRO="$OUTD/mdl_raw/${T}.csv"
  n_ok=$(awk -F, 'NR>1 && $3!=""' "$PRO" 2>/dev/null | wc -l)
  if [ "$n_ok" -ge 1000 ]; then echo "[$T] mdl 已完成 ($n_ok)，跳过"; return 0; fi
  echo "=== $T (metallodock) 现有 $n_ok 有效，开跑 $(date +%H:%M:%S) ==="
  nice -n 10 "$PY" -u scripts/analysis/screen_metallodock.py --target "$T" --protein "$REC" \
    --common-set "$COMMON/$T.csv" --out-dir "$OUTD/mdl_raw" --metal "$(metal_of "$T")" \
    > "logs/bench22_1to10_mdl_$T.log" 2>&1
  echo "=== $T (metallodock) 结束 $(date +%H:%M:%S) ==="
}
export -f run_mdl_one

run_mdl() {
  banner "3) MetalloDock（$W_MDL 并发）"
  printf '%s\n' $ORDER | xargs -P "$W_MDL" -I{} bash -c 'run_mdl_one {}'
}

# ---------------------------------------------------------------- 汇总
run_final() {
  banner "4) 转换 + 汇总指标"
  "$PY" -u scripts/analysis/bench22_finalize.py --bench-dir "$OUTD" \
    --methods metalgeo_full,vina,metallodock
}

# 三方法并行：MetalGeoDock 与 MetalloDock 同时起跑，Vina 在 MetalloDock 结束后接上
run_mg > logs/bench22_1to10_mg.log 2>&1 &
MG_PID=$!
( run_mdl > logs/bench22_1to10_mdl.log 2>&1; run_vina ) &
MDL_PID=$!
wait $MG_PID $MDL_PID
run_final
banner "全部结束"
for m in metalgeo_full vina_raw mdl_raw; do
  echo "--- $m ---"
  for T in $ORDER; do
    case "$m" in
      metalgeo_full) f="$OUTD/${m}_${T}.csv"; col=3 ;;
      vina_raw)      f="$OUTD/$m/${T}_p0.csv"; col=3 ;;
      mdl_raw)       f="$OUTD/$m/${T}.csv"; col=3 ;;
    esac
    n=$(awk -F, -v c=$col 'NR>1 && $c!=""' "$f" 2>/dev/null | wc -l)
    printf "%s=%s " "$T" "$n"
  done
  echo
done
