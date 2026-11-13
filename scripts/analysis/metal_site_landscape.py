"""Metal-site burial vs screening performance across all metalloenzyme targets.

Hypothesis from the per-target results: MetGeoDock works where the metal site
dominates binding (exposed/shallow site, e.g. carbonic anhydrase) and fails where
binding is governed by a deep pocket (MMP, HDAC, ADAM17).

Burial is quantified as the number of protein heavy atoms within 8 A of the
catalytic metal: more neighbours = deeper/more enclosed site.

Outputs results/metal_site_landscape.csv and figures/fig_metal_site_landscape.png
"""
from __future__ import annotations

import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
HDAC = Path("/home/wangkai/cursor-task/HDAC")
XTAL = Path("/tmp/xtal_train")
SHELL = 8.0


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ef(y, s, frac):
    y = np.asarray(y); s = np.asarray(s, float)
    n, na = len(y), int(y.sum())
    if na == 0:
        return float("nan")
    k = max(1, int(round(frac * n)))
    return float((y[np.argsort(s)[:k]].sum() / k) / (na / n))


def burial(pdb: Path):
    """Protein heavy atoms within SHELL of the metal + distance to protein centroid."""
    met, prot = [], []
    for ln in pdb.read_text(errors="ignore").splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        resn = ln[17:20].strip().upper()
        if resn in {"ZN", "FE3", "CA", "MG", "MN", "MN2", "FE2", "CO", "CU", "NI", "CD"}:
            met.append(xyz)
        elif ln.startswith("ATOM") and el != "H":
            prot.append(xyz)
    if not met or not prot:
        return None
    z = met[0]
    P = np.asarray(prot, float)
    d = np.linalg.norm(P - z, axis=1)
    return {"n_within_6A": int((d <= 6).sum()), "n_within_8A": int((d <= 8).sum()),
            "n_within_10A": int((d <= 10).sum()),
            "frac_within_8A": float((d <= 8).mean())}


def main():
    # per-target performance from the cross-family benchmark
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results/crossfamily/*.csv"))):
        rows += list(csv.DictReader(open(f)))
    per = defaultdict(list)
    for r in rows:
        if r.get("E") in (None, "", "nan"):
            continue
        per[(r["metal"], r["target_name"])].append((float(r["E"]), int(r["label"])))
    # HDAC subtypes (Zn family, already scored)
    hdac = {}
    nc_a = {}
    for t in ("HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC6", "HDAC7", "HDAC8"):
        f = HDAC / "docking/results" / f"{t}.csv"
        if f.exists():
            lst = []
            for r in csv.DictReader(open(f)):
                if r.get("score") in (None, "", "nan"):
                    continue
                lst.append((float(r["score"]), 1 if r["ligand_id"].split("_")[1].startswith("A") else 0))
            if len({x[1] for x in lst}) > 1:
                hdac[t] = lst
                nc_a[t] = HDAC / "structures/receptors_h" / f"{t}.pdb"

    # map target name -> its own pdb (not "any pdb of the same metal")
    pdb_by_target = {}
    for r in csv.DictReader(open(ROOT / "processed/chembl/curated/all_targets_molecules.csv")):
        if r.get("pdb") and r["pdb"] not in ("", "None"):
            pdb_by_target.setdefault(r["target_name"], r["pdb"])
    pdb_map = {}
    for (m, name) in per:
        pdb = pdb_by_target.get(name)
        if pdb and Path(f"{XTAL}/{pdb}_h.pdb").exists():
            pdb_map[(m, name)] = pdb

    out = []
    for (m, name), lst in sorted(per.items()):
        y = [p[1] for p in lst]; s = [p[0] for p in lst]
        if len(set(y)) < 2 or len(lst) < 15:
            continue
        pdb = pdb_map.get((m, name))
        b = burial(Path(f"{XTAL}/{pdb}_h.pdb")) if pdb and Path(f"{XTAL}/{pdb}_h.pdb").exists() else None
        rec = {"metal": m, "target": name, "n": len(lst), "n_act": sum(y),
               "auroc": round(auroc(y, s), 3), "ef1": round(ef(y, s, 0.01), 2),
               "ef5": round(ef(y, s, 0.05), 2), "family": "cross-family"}
        if b:
            rec.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()})
        out.append(rec)

    for t, lst in hdac.items():
        y = [p[1] for p in lst]; s = [p[0] for p in lst]
        b = burial(nc_a[t])
        rec = {"metal": "zn", "target": t, "n": len(lst), "n_act": sum(y),
               "auroc": round(auroc(y, s), 3), "ef1": round(ef(y, s, 0.01), 2),
               "ef5": round(ef(y, s, 0.05), 2), "family": "HDAC"}
        if b:
            rec.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()})
        out.append(rec)

    keys = ["metal", "family", "target", "n", "n_act", "auroc", "ef1", "ef5",
            "n_within_6A", "n_within_8A", "n_within_10A", "frac_within_8A"]
    dest = ROOT / "results/metal_site_landscape.csv"
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in out])

    print(f"{'metal':6s} {'family':12s} {'target':12s} {'AUROC':>6s} {'buried8A':>9s} {'frac8A':>7s}")
    for r in sorted(out, key=lambda x: -(x.get("auroc") or 0)):
        print(f"{r['metal']:6s} {r['family']:12s} {r['target']:12s} {r['auroc']:6.3f} "
              f"{r.get('n_within_8A',''):>9} {r.get('frac_within_8A',''):>7}")
    print(f"\nWROTE {dest}  n={len(out)}")

    # correlation between burial and AUROC
    pairs = [(r["n_within_8A"], r["auroc"]) for r in out if r.get("n_within_8A")]
    if len(pairs) > 4:
        from scipy import stats
        rho, p = stats.spearmanr([a for a, _ in pairs], [b for _, b in pairs])
        print(f"Spearman(burial, AUROC) = {rho:+.3f} (p={p:.3g}, n={len(pairs)})")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        colors = {"zn": "#1f77b4", "ca": "#ff7f0e", "mg": "#2ca02c", "mn": "#d62728",
                  "fe2": "#9467bd", "fe3": "#8c564b", "co": "#e377c2", "ni": "#7f7f7f",
                  "cu": "#bcbd22", "cd": "#17becf"}
        for r in out:
            if not r.get("n_within_8A"):
                continue
            marker = "^" if r["family"] == "HDAC" else "o"
            ax.scatter(r["n_within_8A"], r["auroc"], s=46, marker=marker,
                       color=colors.get(r["metal"], "grey"), alpha=0.85,
                       edgecolor="k", linewidth=0.4)
        ax.axhline(0.5, ls="--", lw=1, color="k", alpha=0.6)
        ax.text(ax.get_xlim()[0], 0.505, "random", fontsize=8)
        ax.set_xlabel("protein heavy atoms within 8 Å of the metal (site burial)")
        ax.set_ylabel("AUROC")
        ax.set_title("MetGeoDock enrichment vs metal-site burial", fontsize=10)
        ax.grid(alpha=0.25)
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker="o", ls="", color="grey", label="cross-family"),
                   Line2D([0], [0], marker="^", ls="", color="grey", label="HDAC (Zn)")]
        ax.legend(handles=handles, fontsize=8, frameon=False)
        fig.tight_layout()
        (ROOT / "figures").mkdir(exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(ROOT / "figures" / f"fig_metal_site_landscape.{ext}", dpi=300, bbox_inches="tight")
        print("WROTE figures/fig_metal_site_landscape.{png,pdf}")
    except Exception as exc:
        print("plot failed:", exc)


if __name__ == "__main__":
    main()
