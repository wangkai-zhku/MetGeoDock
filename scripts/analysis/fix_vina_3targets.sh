#!/usr/bin/env bash
# 重跑 MMP1 / MMP9 / HIV1_integrase 的 1:10 Vina（这三个靶点此前的 affinity 全是 0.0）。
#
# 根因：这三个"裁剪版"受体文件每个原子行后多一个空行，obabel 判定 "not a valid PDB file"
# 并输出 0 字节 pdbqt；vina 对着空受体仍返回 rc=0 并打出 "1 0.0"。
# 已在 vina_baseline.py 里改为只保留 ATOM/HETATM 行（不动受体文件，避免影响已完成的
# MetalGeoDock / MetalloDock 结果）。
set -u
ROOT=/home/wangkai/cursor-task/metalloprotein-chem
cd "$ROOT" || exit 1
export PATH=/home/wangkai/cursor-task/PPO-cursor/env/conda/envs/ppo/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=envs/metgeodock/bin/python
OUTD=results/bench22_1to10

one() {
  T="$1"
  case "$T" in
    MMP1)  REC=processed/receptors22/MMP1_cat_h.pdb; MET=zn ;;
    MMP9)  REC=processed/receptors22/MMP9_cat_h.pdb; MET=zn ;;
    HIV1_integrase) REC=processed/receptors22/HIV1_cat_h.pdb; MET=mg ;;
    *) echo "未知靶点 $T"; return 1 ;;
  esac
  echo "=== $T (vina 重跑) 开始 $(date +%H:%M:%S) ==="
  nice -n 10 "$PY" -u scripts/analysis/vina_baseline.py --target "$T" --protein "$REC" \
    --common-set "$OUTD/common/$T.csv" --out-dir "$OUTD/vina_raw" \
    --metal "$MET" --exhaustiveness 4 --cpu 4 > "logs/fix_vina_$T.log" 2>&1
  n=$(awk -F, 'NR>1 && $3!="" && $3+0!=0' "$OUTD/vina_raw/${T}_p0.csv" 2>/dev/null | wc -l)
  echo "=== $T 结束 $(date +%H:%M:%S) 有效 $n/1100 ==="
}
export -f one
export PY OUTD

echo "=== 3 靶点 Vina 重跑开始 $(date +%F' '%T) ==="
printf '%s\n' MMP1 MMP9 HIV1_integrase | xargs -P 3 -I{} bash -c 'one {}'
echo "=== 全部结束 $(date +%F' '%T) ==="
