"""把 bench22（1:5）已算的 150 个配体分数预填进 1:10 目录，避免重复对接。

新配体集的前 150 个与旧集顺序完全一致（已校验），因此：
    results/bench22/{metalgeo_full,vina,metallodock}_<T>.csv  →  results/bench22_1to10/ 下对应格式

三种输出格式各不相同，需分别转换：
    metalgeo_full : ligand_id,label,score          直接复制
    vina          : ligand_id,label,score(=affinity)  -> vina_raw/<T>_p0.csv: ligand_id,label,affinity,error
    metallodock   : ligand_id,label,score(=-mdn)      -> mdl_raw/<T>.csv:       ligand_id,label,score(=mdn)

用法：python scripts/analysis/seed_bench22_1to10.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/bench22"
DST = ROOT / "results/bench22_1to10"
TARGETS = ["CA2", "CA12", "HDAC1", "HDAC2", "HDAC3", "HDAC6", "HDAC8",
           "MMP1", "MMP2", "MMP3", "MMP9", "MMP13",
           "ALOX5", "ARG1", "ARG2", "urease", "METAP2",
           "CYP1A2", "CYP2E1", "CYP19A1", "EGLN2", "HIV1_integrase"]


def rows(p: Path):
    return list(csv.DictReader(open(p))) if p.exists() and p.stat().st_size else []


def main():
    (DST / "vina_raw").mkdir(parents=True, exist_ok=True)
    (DST / "mdl_raw").mkdir(parents=True, exist_ok=True)
    n_mg = n_v = n_m = 0
    for t in TARGETS:
        # 1) MetalGeoDock：格式相同，直接复制
        src = SRC / f"metalgeo_full_{t}.csv"
        rs = rows(src)
        if rs:
            with open(DST / f"metalgeo_full_{t}.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["ligand_id", "label", "score"])
                for r in rs:
                    w.writerow([r["ligand_id"], r["label"], r["score"]])
            n_mg += len(rs)

        # 2) Vina：score(=affinity) -> affinity 列
        rs = rows(SRC / f"vina_{t}.csv")
        if rs:
            with open(DST / "vina_raw" / f"{t}_p0.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["ligand_id", "label", "affinity", "error"])
                for r in rs:
                    w.writerow([r["ligand_id"], r["label"], r["score"], ""])
            n_v += len(rs)

        # 3) MetalloDock：score(=-mdn) -> mdn（取负还原）
        rs = rows(SRC / f"metallodock_{t}.csv")
        if rs:
            with open(DST / "mdl_raw" / f"{t}.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["ligand_id", "label", "score", "error"])
                for r in rs:
                    w.writerow([r["ligand_id"], r["label"], f"{-float(r['score']):.6f}", ""])
            n_m += len(rs)

    print(f"预填完成：metalgeo_full {n_mg} 条、vina {n_v} 条、metallodock {n_m} 条")
    print(f"新结果目录: {DST}")


if __name__ == "__main__":
    main()
