"""Enrichment-factor curve: EF as a function of the ranked fraction.

The single most informative plot for the HDAC benchmark: it shows that enrichment
is a spike at the very top rather than a plateau, which is why EF1% can be 2-4
while BEDROC(alpha=20) stays near random.
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TARGETS = ["HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC6", "HDAC7", "HDAC8"]


def ef_curve(y, s, fracs):
    y = np.asarray(y)
    s = np.asarray(s, float)
    n, na = len(y), int(y.sum())
    out = []
    for f in fracs:
        k = max(1, int(round(f * n)))
        hits = int(y[np.argsort(s)[:k]].sum())
        out.append((hits / k) / (na / n) if na and k else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actives", type=Path, required=True)
    ap.add_argument("--decoys", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="HDAC benchmark")
    a = ap.parse_args()

    labels = {r["ligand_id"]: int(r["label"]) for r in csv.DictReader(open(a.labels))}
    fracs = np.logspace(np.log10(0.002), np.log10(0.30), 40)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    curves = []
    for t in TARGETS:
        f = a.actives / f"{t}.csv"
        if not f.exists():
            continue
        act = [float(r["score"]) for r in csv.DictReader(open(f))
               if labels.get(r["ligand_id"]) == 1 and r.get("score") not in (None, "", "nan")]
        dec = []
        for g in sorted(glob.glob(str(a.decoys / f"{t}_part*.csv"))):
            for r in csv.DictReader(open(g)):
                if r.get("score") not in (None, "", "nan"):
                    dec.append(float(r["score"]))
        if len(act) < 10 or len(dec) < 100:
            continue
        y = np.array([1] * len(act) + [0] * len(dec))
        s = np.array(act + dec, float)
        c = ef_curve(y, s, fracs)
        curves.append(c)
        ax.plot(fracs * 100, c, lw=0.9, alpha=0.55, label=t)
    if curves:
        m = np.nanmean(np.array(curves, float), axis=0)
        ax.plot(fracs * 100, m, lw=2.6, color="k", label=f"mean (n={len(curves)})")
    ax.axhline(1.0, ls="--", lw=1, color="grey", label="random")
    ax.set_xscale("log")
    ax.set_xlabel("top % of ranked library")
    ax.set_ylabel("enrichment factor (EF)")
    ax.set_title(a.title, fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(a.out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {a.out}.png/.pdf  (targets={len(curves)})")


if __name__ == "__main__":
    main()
