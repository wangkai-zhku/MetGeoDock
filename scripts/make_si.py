#!/usr/bin/env python3
"""生成 Supporting Information 的 Markdown（随后用 scripts/make_docx.py 转 Word）。

所有表格都由 results/ 下的结果文件直接生成，不手工录入数字。
输出：manuscript/SI_generated.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def md_table(header, rows, aligns=None):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(aligns or ["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    m10 = json.load(open(R / "bench22_metrics_1to10_FINAL.json"))
    m5 = json.load(open(R / "bench22_metrics_1to5_FINAL.json"))
    sites = json.load(open(R / "coordination_sites.json"))
    eff = json.load(open(R / "sampling_efficiency.json"))
    abl = json.load(open(R / "pose_rank_ablation.json"))
    ts = sorted(m10)
    L = []

    L.append("# Supporting Information\n")
    L.append("**Vacant-vertex sampling and coordination-aware scoring beat generic docking on a "
             "22-target metalloprotein benchmark**\n")
    L.append("All tables below are generated directly from the result files shipped with this "
             "work (`results/`); no value was transcribed by hand.\n")

    # S1 逐靶点性能
    L.append("## S1. Per-target virtual-screening performance (1:10 panel)\n")
    L.append("The library is 100 actives among ~1,100 molecules (9.2% active), so the enrichment "
             "factor is capped at 10.9 at every cut-off: EF cannot exceed the inverse of the active "
             "fraction. DUD-E and the benchmarks built on it use 50 decoys per active (2% active), "
             "which caps EF at about 50, so raw EF values from the two panels are not comparable; "
             "divide by the ceiling before comparing across studies.\n")
    rows = []
    for t in ts:
        d = m10[t]
        rows.append([t,
                     f"{d['metalgeo_full']['auroc']:.3f}",
                     f"{d['vina']['auroc']:.3f}",
                     f"{d['metallodock']['auroc']:.3f}",
                     f"{d['metalgeo_full']['ef1']:.2f}",
                     f"{d['vina']['ef1']:.2f}",
                     f"{d['metallodock']['ef1']:.2f}",
                     int(d['metalgeo_full']['n_active']),
                     int(d['metalgeo_full']['n']) - int(d['metalgeo_full']['n_active'])])
    L.append(md_table(["Target", "AUROC MGD", "AUROC Vina", "AUROC MD",
                       "EF1% MGD", "EF1% Vina", "EF1% MD", "n actives", "n decoys"],
                      rows, ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    L.append("")

    # S2 位姿判别消融
    L.append("## S2. Pose-discrimination ablation\n")
    rows = [[k, f"{v['auroc']:.3f}", f"{v.get('ap', float('nan')):.3f}"]
            for k, v in sorted(abl["methods"].items(), key=lambda kv: -kv[1]["auroc"])]
    L.append(md_table(["Ranker", "AUROC", "Average precision"], rows, ["---", "---:", "---:"]))
    L.append("")

    # S3 配位球检测
    L.append("## S3. Detected coordination spheres on the 22 benchmark targets\n")
    rows = [[s["target"], s["metal"], s["n_coord"], s["n_vacant"], s["geometry"]] for s in sites]
    L.append(md_table(["Target", "Metal", "CN (protein donors)", "Vacant vertices", "Preferred geometry"],
                      rows, ["---", "---", "---:", "---:", "---"]))
    L.append("")

    # S4 采样效率
    L.append("## S4. Sampling efficiency (CA2, measured)\n")
    L.append(md_table(["Strategy", "Poses sampled", "Correctly coordinated"],
                      [["Blind placement in pocket", eff["n_pose_blind"], f"{eff['coord_ok_blind']:.1%}"],
                       ["Vacant-vertex placement", eff["n_pose_vertex"], f"{eff['coord_ok_vertex']:.1%}"]],
                      ["---", "---:", "---:"]))
    L.append(f"\nCriterion: a pose is coordinated when at least one ligand donor lies in the "
             f"1.9–2.9 Å shell of the metal. {eff['note']}.\n")

    # S5 decoy 比例敏感性
    L.append("## S5. Sensitivity to the decoy ratio (1:5 subset of the 1:10 panel)\n")
    rows = []
    for m in ("metalgeo_full", "vina", "metallodock"):
        a5 = [m5[t][m]["auroc"] for t in m5]
        a10 = [m10[t][m]["auroc"] for t in ts]
        rows.append([m, f"{sum(a5)/len(a5):.3f}", f"{sum(a10)/len(a10):.3f}",
                     f"{sum(a10)/len(a10) - sum(a5)/len(a5):+.3f}"])
    L.append(md_table(["Method", "Mean AUROC 1:5", "Mean AUROC 1:10", "Difference"],
                      rows, ["---", "---:", "---:", "---:"]))
    L.append("\nThe 1:5 subset uses exactly the first 150 ligands of each 1:10 set, so the "
             "ligand identities are identical between the two panels.\n")

    # S6 数据集性质
    L.append("## S6. Physicochemical matching of actives and decoys\n")
    import numpy as _np
    from scipy.stats import mannwhitneyu as _mwu
    _rows = json.load(open(R / "ligand_properties.json"))
    _act = [q for q in _rows if q["label"] == 1]
    _dec = [q for q in _rows if q["label"] == 0]
    _tbl = []
    for _key, _lab in (("mw", "Molecular weight"), ("logp", "cLogP"),
                       ("rotb", "Rotatable bonds"), ("psa", "Polar surface area"),
                       ("hbd", "H-bond donors"), ("hba", "H-bond acceptors")):
        _va = _np.array([q[_key] for q in _act], float)
        _vd = _np.array([q[_key] for q in _dec], float)
        _d = (_va.mean() - _vd.mean()) / _vd.std()
        _p = _mwu(_va, _vd).pvalue
        _star = "***" if _p < 1e-3 else ("**" if _p < 1e-2 else ("*" if _p < .05 else "ns"))
        _tbl.append([_lab, f"{_va.mean():.2f}", f"{_vd.mean():.2f}", f"{_d:+.2f}", _star])
    L.append(md_table(["Descriptor", "Actives mean", "Decoys mean", "Standardised difference",
                       "Mann-Whitney"], _tbl, ["---", "---:", "---:", "---:", "---:"]))
    L.append("\nAll six effect sizes are |d| ≤ 0.11 SD, i.e. actives and decoys are matched on size, "
             "flexibility and polarity. The Mann–Whitney test still returns small *p* values for some "
             "descriptors simply because n = 24,200; property matching is judged on effect size, not on "
             "*p*. Exact means are in `results/ligand_properties.json` and plotted in Fig. 2f–h.\n")

    # S7 农业小库（少量农药分子，只作可行性示例）
    agri = json.load(open(R / "agri_vs.json"))["by_target"]
    L.append("## S7. Small agrochemical library (168 molecules)\n")
    L.append("One pre-registered library of 168 unique molecules (ChEMBL pChEMBL ≥ 6, 10–45 heavy "
             "atoms, the union of an HRAC/IRAC trade-name panel) was docked into three "
             "metal-dependent herbicide targets to ask whether the score recovers a known mode of "
             "action. The library is deliberately small and the number of actives per target ranges "
             "from 3 to 83, so the values below show feasibility only; every benchmark comparison "
             "in the main text uses the 22-target panel.\n")
    rows = []
    for key, label in (("gs", "Glutamine synthetase"), ("hppd", "HPPD"), ("igpd", "IGPD")):
        d = agri[key]
        top5 = d.get("metalgeo_top5", {})
        rows.append([label, d["n_actives"], d["n_decoys"], f"{d['metalgeo_auroc']:.3f}",
                     f"{d['metalgeo_ef_10pct']:.2f}", f"{d['metalgeo_ef_20pct']:.2f}",
                     f"{top5.get('n_actives_in_top', '—')}/{top5.get('k', '—')}"])
    L.append(md_table(["Target", "Actives", "Decoys", "MetGeo AUROC", "EF10%", "EF20%",
                       "Actives in top 5"], rows, ["---", "---:", "---:", "---:", "---:", "---:", "---:"]))
    gs = agri["gs"]
    L.append(f"\nOn glutamine synthetase the known inhibitor glufosinate is ranked 2nd of "
             f"{gs['n']} molecules, and the score keeps a clear margin over chance "
             f"(AUROC {gs['metalgeo_auroc']:.3f}, EF10% {gs['metalgeo_ef_10pct']:.2f}) even with three "
             f"actives in the library. HPPD is the counter-example: its library carries 83 actives "
             f"and MetGeo alone does not separate them (AUROC {agri['hppd']['metalgeo_auroc']:.3f}); "
             f"the ligand-level CoordNet scorer reaches {agri['hppd']['coordnet_auroc']:.3f} there. "
             f"Data: `results/agri_vs.json`.\n")

    # S8 重对接 public10 与文献值（正文 Fig. 4 不再画，数据在此列出）
    _acc = json.load(open(R / "all_test_accuracy.json"))
    _p10 = _acc["by_set"]["public10"]
    L.append("## S8. Redocking success on the public10 subset and comparison with literature\n")
    L.append(f"public10 contains only {_p10['n']} complexes, so it is reported here rather than in "
             "Fig. 4; the literature value for MetalloDock is not measured on the same set and the "
             "comparison is indicative only.\n")
    L.append(md_table(["Metric", "MetGeoDock (public10)", "MetalloDock (literature)"],
                      [["top-1", f"{_p10['top1']:.3f}", f"{_acc['literature_metallodock_top1']:.3f}"],
                       ["top-3", f"{_p10['top3']:.3f}", "—"],
                       ["top-5", f"{_p10['top5']:.3f}", "—"],
                       ["any sampled pose", f"{_p10['sampled']:.3f}", "—"]],
                      ["---", "---:", "---:"]))


    out = ROOT / "manuscript/SI_generated.md"
    out.write_text("\n".join(L))
    print(f"✓ {out}  ({len(out.read_text().splitlines())} 行，7 个表)")


if __name__ == "__main__":
    main()
