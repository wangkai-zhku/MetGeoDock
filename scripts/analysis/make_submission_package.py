"""投稿材料打包：把论文、图、原始数据、软件、文档汇总到一个新文件夹。

**安全约束：本脚本只做复制（shutil.copy2 / copytree），不含任何删除、移动操作。**
原始目录 ~/cursor-task/metalloprotein-chem/ 下的所有文件保持原样，不会被改动。

幂等：可反复运行；目标文件夹里同名文件会被覆盖（只覆盖打包产物自己）。

用法：
    python scripts/analysis/make_submission_package.py
    python scripts/analysis/make_submission_package.py --out submission_20261108
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_SUFFIX = (".tmp", ".bak", ".presalt.bak", ".presalt.bak", ".pyc")


def _sha(path: Path, limit=1 << 20) -> str:
    """小文件算 sha1，大文件只记大小（避免打包耗时）。"""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(limit)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def copy_file(src: Path, dst: Path, manifest: list) -> bool:
    """只复制单个文件；源不存在则跳过。"""
    if not src.exists() or src.name.endswith(SKIP_SUFFIX):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)                       # 仅复制
    manifest.append({"path": str(dst.relative_to(dst.parents[len(dst.parents) - 1])),
                     "src": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
                     "bytes": src.stat().st_size,
                     "sha1_16": _sha(src) if src.stat().st_size < 8 << 20 else "skipped"})
    return True


def copy_glob(src_dir: Path, dst_dir: Path, patterns: list, manifest: list) -> int:
    n = 0
    for pat in patterns:
        for src in sorted(src_dir.glob(pat)):
            if src.is_file():
                n += copy_file(src, dst_dir / src.name, manifest)
    return n


def copy_tree(src: Path, dst: Path, manifest: list, skip_dirs=("__pycache__", ".git"),
              skip_suffix=SKIP_SUFFIX) -> int:
    """递归复制目录（只复制，不删除目标里已有的其它内容）。"""
    n = 0
    if not src.exists():
        return 0
    for item in sorted(src.rglob("*")):
        if any(part in skip_dirs for part in item.parts):
            continue
        if item.suffix in skip_suffix or item.name.endswith(skip_suffix):
            continue
        if item.is_file():
            n += copy_file(item, dst / item.relative_to(src), manifest)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission_20261108")
    a = ap.parse_args()
    OUT = ROOT / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: list = []
    stats = {}

    # ---------------------------------------------------------------- 01 论文
    d = OUT / "01_manuscript"
    m = [
        (ROOT / "manuscript/MetGeoDock_manuscript.md", d / "MetGeoDock_manuscript.md"),
        (ROOT / "manuscript/过程与结论.md", d / "过程与结论.md"),
        (ROOT / "manuscript/References_plan.md", d / "References_plan.md"),
        (ROOT / "manuscript/figure_storyboard.md", d / "figure_storyboard.md"),
        (ROOT / "manuscript/submission_packages/jcim/manuscript_en.docx", d / "manuscript_en.docx"),
        (ROOT / "manuscript/submission_packages/jcim/SI.docx", d / "SI.docx"),
        (ROOT / "manuscript/submission_packages/jcim/method.docx", d / "method.docx"),
        (ROOT / "manuscript/jacs_package/MetGeoDock_manuscript_zh.docx", d / "manuscript_zh.docx"),
    ]
    stats["01_manuscript"] = sum(copy_file(s, t, manifest) for s, t in m)

    # ---------------------------------------------------------------- 02 图
    fig_root = ROOT / "figures"
    main_figs = ["fig1_scoring_overview", "fig1_screening_overview", "fig1_inventory",
                 "fig2_pose_rank_ap", "fig2_redock_success", "fig3_vs_auroc",
                 "fig4_qm_geometry", "fig_ef_curves"]
    sup_figs = ["figS_screening_suite", "figS_chem_hydrogen_ranking", "figS_entropy_suite",
                "fig_entropy_calibration", "fig_metal_site_landscape"]
    n = 0
    for stem in main_figs:
        n += copy_glob(fig_root, OUT / "02_figures/main", [f"{stem}.pdf", f"{stem}.png"], manifest)
    for stem in sup_figs:
        n += copy_glob(fig_root, OUT / "02_figures/supplementary",
                       [f"{stem}.pdf", f"{stem}.png"], manifest)
    n += copy_glob(fig_root, OUT / "02_figures/toc", ["TOC_jacs.*"], manifest)
    n += copy_glob(fig_root, OUT / "02_figures/graphical_abstract",
                   ["graphical_abstract_jacs.*"], manifest)
    n += copy_glob(fig_root, OUT / "02_figures/main", ["*.caption.txt"], manifest)
    n += copy_glob(ROOT / "scripts/figures", OUT / "02_figures/graphical_abstract",
                   ["TOC_jacs.*", "graphical_abstract_jacs.*"], manifest)
    stats["02_figures"] = n

    # ---------------------------------------------------------------- 03 源数据
    for tag, srcdir in (("bench22_1to10", ROOT / "results/bench22_1to10"),
                        ("bench22_1to5", ROOT / "results/bench22_1to5_from1to10")):
        d = OUT / "03_source_data" / tag
        n = copy_glob(srcdir, d, ["metalgeo_full_*.csv", "vina_*.csv",
                                  "metallodock_*.csv"], manifest)
        n += copy_glob(srcdir / "common", d / "common", ["*.csv"], manifest)
        stats[f"03_source_data/{tag}"] = n

    d = OUT / "03_source_data/metallodock_official"
    n = copy_glob(ROOT / "results/mdl_official/1to10", d, ["*.csv"], manifest)
    n += copy_glob(ROOT / "results/mdl_official", d, ["biglib_*.csv"], manifest)
    stats["03_source_data/metallodock_official"] = n

    d = OUT / "03_source_data/agri_vs"
    n = copy_glob(ROOT / "results", d, ["agri_vs*.csv", "agri_vs*.json"], manifest)
    stats["03_source_data/agri_vs"] = n

    d = OUT / "03_source_data/biglib"
    n = copy_glob(ROOT / "results", d, ["biglib_metrics.csv"], manifest)
    n += copy_glob(ROOT / "results/biglib_scores", d / "scores", ["*.csv"], manifest)
    stats["03_source_data/biglib"] = n

    d = OUT / "03_source_data/metrics"
    n = copy_glob(ROOT / "results", d, ["bench22_metrics_*.json", "multipose_metrics*.json",
                                        "pose_rank_ablation.json"], manifest)
    stats["03_source_data/metrics"] = n

    # multipose 预留位（明天整合）
    mp = OUT / "03_source_data/multipose"
    mp.mkdir(parents=True, exist_ok=True)
    (mp / "_待整合说明.md").write_text(
        "# multipose 数据（待整合）\n\n"
        "明天 multipose 跑完后，把结果复制进来即可（只复制，不改动原目录）：\n\n"
        "```bash\n"
        "cd ~/cursor-task/metalloprotein-chem\n"
        "cp -r results/bench22_multipose/1to10/*.csv \\\n"
        "      submission_20261108/03_source_data/multipose/\n"
        "cp results/multipose_metrics.json \\\n"
        "      submission_20261108/03_source_data/metrics/\n"
        "```\n\n"
        "然后用 `python scripts/analysis/multipose_metrics.py` 生成的表更新论文里\n"
        "top-1 vs top-k 的对比段落。\n", encoding="utf-8")

    # ---------------------------------------------------------------- 04 软件
    d = OUT / "04_software"
    n = copy_tree(ROOT / "metgeodock", d / "metgeodock", manifest)
    n += copy_glob(ROOT / "scripts/analysis", d / "scripts/analysis", ["*.py", "*.sh"], manifest)
    n += copy_glob(ROOT, d, ["pyproject.toml", "README.md", "DATA.md", "CITATION.cff",
                             "LICENSE", "GITHUB_UPLOAD.md"], manifest)
    n += copy_glob(ROOT / "tests", d / "tests", ["*.py"], manifest)
    stats["04_software"] = n

    # ---------------------------------------------------------------- 05 文档
    d = OUT / "05_documentation"
    n = copy_glob(ROOT / "reports", d, ["*.md", "*.docx", "*.json"], manifest)
    stats["05_documentation"] = n

    # ---------------------------------------------------------------- 06 cover letter
    d = OUT / "06_cover_letter"
    n = copy_glob(ROOT / "manuscript/cover_letters", d, ["*"], manifest)
    stats["06_cover_letter"] = n

    # ---------------------------------------------------------------- MANIFEST
    total = sum(v for v in stats.values())
    total_bytes = sum(x["bytes"] for x in manifest)
    mf = {"created": time.strftime("%F %T"), "source_root": str(ROOT),
          "out": str(OUT), "n_files": total, "total_MB": round(total_bytes / 1e6, 1),
          "by_dir": stats, "files": manifest,
          "note": "只复制，未改动任何原始文件"}
    (OUT / "MANIFEST.json").write_text(json.dumps(mf, indent=2, ensure_ascii=False))
    print(f"打包完成：{total} 个文件，{total_bytes/1e6:.1f} MB -> {OUT}")
    for k, v in stats.items():
        print(f"  {k}: {v} 个文件")


if __name__ == "__main__":
    main()
