"""对比 HDAC 受体修复前后的 MetalGeoDock 富集指标。

旧受体：results/bench22/metalgeo_full_HDAC{1,2,3,6,8}.csv
新受体：results/bench22/hdacfix_<name>.csv（prepare_hdac_fixed.py 重建）

用法：python scripts/analysis/compare_hdac_fixed.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
import numpy as np  # noqa: E402
from bench22_metrics import auroc, bedroc_tc, ef, proc_logauc  # noqa: E402

B = ROOT / "results/bench22"
PAIRS = [
    ("HDAC1", "hdacfix_HDAC1_fix.csv", "5ICN 单体 → +MTA1 +IP6"),
    ("HDAC2", "hdacfix_HDAC2_fix.csv", "3MAX 单链（保留晶体构象）"),
    ("HDAC3", "hdacfix_HDAC3_fix.csv", "4A69 apo → +SMRT DAD +InsP4"),
    ("HDAC6", "hdacfix_HDAC6_fix.csv", "5EDU CD2 双拷贝 → 5G0J CD1+CD2"),
    ("HDAC8", "hdacfix_HDAC8_2V5W.csv", "2V5W 构象（清洗后）"),
    ("HDAC8", "hdacfix_HDAC8_2V5X.csv", "2V5X 构象（另一构象）"),
]


def load(path: Path):
    y, s = [], []
    for r in csv.DictReader(open(path)):
        try:
            y.append(int(r["label"])); s.append(-float(r["score"]))
        except (TypeError, ValueError):
            continue
    return np.array(y), np.array(s)


def main():
    print(f"{'靶点':7s}{'受体版本':34s}{'n':>5s}{'AUC':>7s}{'BEDROC':>8s}{'logAUC':>8s}{'EF5%':>7s}")
    for tgt, newfile, note in PAIRS:
        for label, path in ((f"旧 {tgt}", B / f"metalgeo_full_{tgt}.csv"), ("新", B / newfile)):
            if not path.exists():
                print(f"{tgt:7s}{label + ' ' + note:34s} 缺失 {path.name}")
                continue
            y, s = load(path)
            if len(y) == 0:
                print(f"{tgt:7s}{label + ' ' + note:34s} 空文件")
                continue
            print(f"{tgt:7s}{(label + ' ' + note)[:34]:34s}{len(y):>5d}{auroc(y, s):>7.3f}"
                  f"{bedroc_tc(y, s):>8.3f}{proc_logauc(y, s):>8.3f}{ef(y, s, 0.05):>7.2f}")
        print()


if __name__ == "__main__":
    main()
