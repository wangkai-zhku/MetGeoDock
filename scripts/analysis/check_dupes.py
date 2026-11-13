"""检查是否有同一任务的多余「主进程」在跑（worker 不算重复）。

判定方式：把同一脚本的所有进程按 PID/PPID 收集起来；若某进程的 PPID 也在该集合内，
它就是父进程 fork 出来的 worker（Pool / 子 shell），不计入实例数。

用法：
    python scripts/analysis/check_dupes.py          # 报告
    python scripts/analysis/check_dupes.py --quiet  # 只在有重复时输出（退出码 1）
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

# 任务名 -> (命令行片段, 是否应恰好 1 个主进程)
TASKS = [
    ("MetalloDock 批处理", "mdl_official_batch.sh", True),
    ("MetalloDock 靶点进程", "mdl_official_bench22.py", False),
    ("Vina 脱盐调度", "run_vina_desalt_fix.py", True),
    ("Vina 子进程", "vina --receptor", False),
    ("agri_vs 调度", "benchmark_agri_vs_parallel.py", True),
    ("multipose", "run_multipose_bench22.py", True),
    ("排队调度", "wait_then_all.sh", True),
]


def snapshot():
    out = subprocess.run(["ps", "-eo", "pid,ppid,cmd"], capture_output=True, text=True).stdout
    rows = []
    for ln in out.splitlines()[1:]:
        parts = ln.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        rows.append((pid, ppid, parts[2]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    me = {0, __import__("os").getpid(), __import__("os").getppid()}
    rows = snapshot()
    dupes, lines = [], []
    lines.append(f"=== 任务实例检查 {time.strftime('%F %T')} ===")
    for name, pat, must_be_single in TASKS:
        hits = [(p, pp, c) for p, pp, c in rows
                if pat in c and p not in me and "check_dupes" not in c and "grep" not in c]
        pids = {p for p, _, _ in hits}
        mains = [(p, pp, c) for p, pp, c in hits if pp not in pids]   # PPID 不在集合内 = 主进程
        n = len(mains)
        if must_be_single:
            if n > 1:
                dupes.append((name, n, mains))
                lines.append(f"  ⚠ {name}: {n} 个主进程（重复！） {[m[0] for m in mains]}")
            elif n == 0:
                lines.append(f"  · {name}: 未运行")
            else:
                lines.append(f"  ✓ {name}: 1 个主进程 (PID {mains[0][0]})，worker {len(hits)-1} 个")
        else:
            lines.append(f"  · {name}: {n} 个进程（并行 worker，属正常）")
    if not a.quiet:
        print("\n".join(lines))
    elif dupes:
        print("\n".join(lines))
    return 1 if dupes else 0


if __name__ == "__main__":
    sys.exit(main())
