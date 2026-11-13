"""22 靶点 benchmark 汇总：把三方法输出统一成 (ligand_id,label,score) 并算全指标。

bench22_metrics.py 只认 results/bench22/<method>_<TARGET>.csv 且约定 score 越小越好
（内部取 -score 当“越大越好”）。三条线的原始输出各不相同，这里统一落盘：

    metalgeo_full_*  已由 run_bench22.py 直接写出（含化学项完整打分，score=total）
    metalgeo_*       旧版仅几何/力场打分（score=total）
    vina_raw/*_p0.csv  affinity，越小越好      -> vina_<T>.csv          score=affinity
    mdl_raw/*.csv      mdn_score 是 MDN 概率和，越大越好
                                                    -> metallodock_<T>.csv score=-mdn_score

用法：
    python scripts/analysis/bench22_finalize.py            # 转换 + 算指标
    python scripts/analysis/bench22_finalize.py --convert-only
"""
from __future__ import annotations
import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "results/bench22"   # 可被 --bench-dir 覆盖
PY = ROOT / "envs/metgeodock/bin/python"
TARGETS = [p.stem.replace("metalgeo_", "") for p in sorted(BENCH.glob("metalgeo_*.csv"))]
TARGETS = [t for t in TARGETS if not t.startswith("full_")]


def write_unified(dst: Path, rows: list[tuple[str, int, float]]):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ligand_id", "label", "score"])
        for lid, lab, sc in rows:
            w.writerow([lid, lab, f"{sc:.6e}"])


def convert_vina():
    for t in TARGETS:
        src = BENCH / "vina_raw" / f"{t}_p0.csv"
        dst = BENCH / f"vina_{t}.csv"
        if not src.exists() or not src.stat().st_size:
            # 原始文件缺失/正在重算时，不能沿用可能是坏批次的派生文件
            if dst.exists():
                vals = [float(r["score"]) for r in csv.DictReader(open(dst)) if r.get("score")]
                if vals and sum(1 for v in vals if v == 0.0) / len(vals) > 0.2:
                    dst.unlink()
                    print(f"  vina {t}: 无原始文件，且已有派生文件是坏批次 -> 已删除")
                    continue
            print(f"  vina {t}: 缺原始文件，跳过")
            continue
        rows = []
        zero = 0
        for r in csv.DictReader(open(src)):
            try:
                aff = float(r["affinity"])
                lab = int(r["label"])
            except (TypeError, ValueError, KeyError):
                continue
            if aff == 0.0:
                # 旧 vina 运行把失败配体写成 affinity=0.0 且 error 列为空（MMP9 整批如此），
                # 若当成有效分数会被排成“最差”，等于给 Vina 白送分。
                zero += 1
                continue
            rows.append((r["ligand_id"], lab, aff))
        dst = BENCH / f"vina_{t}.csv"
        if zero and (len(rows) == 0 or zero / (zero + len(rows)) > 0.2):
            # 整批 affinity=0 说明那次 vina 根本没算出分数（MMP9 就是这样），
            # 这种结果不能用，也不能留旧文件冒充“已完成”。
            dst.unlink(missing_ok=True)
            print(f"  vina {t}: ★坏批次（{zero} 条 affinity=0、{len(rows)} 条有效），已剔除")
            continue
        write_unified(dst, rows)
        print(f"  vina {t}: {len(rows)} 条" + (f"（跳过 {zero} 条 affinity=0）" if zero else ""))


def convert_mdl():
    for t in TARGETS:
        src = BENCH / "mdl_raw" / f"{t}.csv"
        if not src.exists() or not src.stat().st_size:
            print(f"  metallodock {t}: 缺原始文件，跳过")
            continue
        rows = []
        for r in csv.DictReader(open(src)):
            try:
                # mdn_score 是概率和（越大越好）；metrics 约定 score 越小越好，故取负
                rows.append((r["ligand_id"], int(r["label"]), -float(r["score"])))
            except (TypeError, ValueError, KeyError):
                continue
        write_unified(BENCH / f"metallodock_{t}.csv", rows)
        print(f"  metallodock {t}: {len(rows)} 条")


def coverage():
    print("\n=== 各方法覆盖度（目标每靶点 150 = 100 actives + 50 decoys）===")
    for m in ("metalgeo_full", "metalgeo", "vina", "metallodock"):
        got, miss = [], []
        for t in TARGETS:
            p = BENCH / f"{m}_{t}.csv"
            n = sum(1 for _ in csv.DictReader(open(p))) if p.exists() and p.stat().st_size else 0
            (got if n >= 140 else miss).append(f"{t}:{n}")
        print(f"{m:14s} 达标 {len(got):2d}/{len(TARGETS)}  不足: {' '.join(miss) or '无'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert-only", action="store_true")
    ap.add_argument("--bench-dir", default="", help="结果目录，默认 results/bench22")
    ap.add_argument("--methods", default="metalgeo_full,metalgeo,vina,metallodock")
    a = ap.parse_args()
    global BENCH
    if a.bench_dir:
        BENCH = Path(a.bench_dir)

    print("=== 统一格式 ===")
    convert_vina()
    convert_mdl()
    coverage()
    if a.convert_only:
        return
    print("\n=== 计算指标 ===")
    cmd = [str(PY), str(ROOT / "scripts/analysis/bench22_metrics.py"), "--methods", a.methods]
    if a.bench_dir:
        cmd += ["--bench-dir", a.bench_dir]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
