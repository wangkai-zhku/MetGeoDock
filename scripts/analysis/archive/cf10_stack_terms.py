"""Add physics-motivated terms on top of the existing MetGeoDock score.

Baseline (unchanged, as currently implemented):
    full = pct(total) + pct(g_buried) - pct(n_rot * clip(buried/600))

Candidate additions use equal-weight rank sums. Directions come from physics:
coordination-chemistry terms (chelation, donor count, soft-donor count) are
"more is better"; size/polarity are "less is better". No weight is fitted to
activity labels; the only choices are which physics term goes in, and every
group below is a named chemical concept rather than a searched subset.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
spec = importlib.util.spec_from_file_location("fe", "/tmp/cf10_feature_expand.py")
fe = importlib.util.module_from_spec(spec)
sys.modules["fe"] = fe
spec.loader.exec_module(fe)

DATA = fe.build()
TARGETS = list(DATA)


def pct(v):
    v = np.asarray(v, float)
    return rankdata(v, method="average") / (len(v) + 1.0)


def col(d, name):
    v = np.array([r.get(name, np.nan) for r in d], float)
    if not np.isfinite(v).any():
        return np.zeros(len(d))
    return np.where(np.isfinite(v), v, np.nanmedian(v))


def baseline(d):
    g = pct(col(d, "total"))
    gb = pct(col(d, "g_buried"))
    cat = pct(col(d, "n_rotatable") * np.clip(col(d, "buried_area") / 600.0, 0, 1))
    return g + gb - cat


def add(base, d, terms, weight=1.0):
    """terms: [(feature, +1 if higher-is-better else -1)]; equal weight."""
    s = base.copy()
    for name, sign in terms:
        s = s + weight * (-sign) * pct(col(d, name))
    return s


GROUPS = {
    "baseline (full)":            [],
    "+ chelate_tot":              [("chelate_tot", +1)],
    "+ chelate6":                 [("chelate6", +1)],
    "+ chelate5":                 [("chelate5", +1)],
    "+ chelate_tot+6":            [("chelate_tot", +1), ("chelate6", +1)],
    "+ 配位化学组":                [("chelate_tot", +1), ("n_donor", +1), ("nS", +1)],
    "+ 配位化学组+磺酰胺":         [("chelate_tot", +1), ("n_donor", +1), ("nS", +1),
                                   ("z_sulfonamide", +1)],
    "+ 密度组":                   [("d_buried_per_heavy", +1), ("d_hbond_per_heavy", +1),
                                   ("d_hydroph_per_heavy", +1)],
    "+ 配位+密度":                [("chelate_tot", +1), ("n_donor", +1), ("nS", +1),
                                   ("d_buried_per_heavy", +1), ("d_hbond_per_heavy", +1)],
    "+ 形状(fsp3)":               [("fsp3", +1)],
    "+ 螯合+形状":                [("chelate_tot", +1), ("fsp3", +1)],
    "仅 chelate_tot":             [("chelate_tot", +1)],
}

print("=" * 108)
print("在 MetGeoDock full 基础上叠加物理项（等权秩和，方向由物理定，不拟合标签）")
print("=" * 108)
print(f"{'组合':<22}" + "".join(f"{t[:8]:>10}" for t in TARGETS) + f"{'宏平均':>9}{'vs基线':>8}")
base_aurocs = {}
rows = []
for label, terms in GROUPS.items():
    au = {}
    for t in TARGETS:
        d = DATA[t]
        y = np.array([r["y"] for r in d])
        s = baseline(d) if not terms else add(baseline(d), d, terms)
        au[t] = fe.auroc(y, s)
    base_aurocs.setdefault("b", {}) if False else None
    rows.append((label, au))
    if label.startswith("baseline"):
        base = au
for label, au in rows:
    mean = float(np.mean(list(au.values())))
    delta = mean - float(np.mean(list(base.values())))
    print(f"{label:<22}" + "".join(f"{au[t]:>10.3f}" for t in TARGETS)
          + f"{mean:>9.3f}{delta:>+8.3f}")

print()
print("=" * 108)
print("逐项贡献扫描：full + 单个物理项（等权）")
print("=" * 108)
CANDS = [("chelate_tot", +1), ("chelate6", +1), ("chelate5", +1), ("n_donor", +1),
         ("nS", +1), ("z_sulfonamide", +1), ("z_amine", +1), ("z_hydroxyl", +1),
         ("fsp3", +1), ("d_buried_per_heavy", +1), ("d_hbond_per_heavy", +1),
         ("d_hydroph_per_heavy", +1), ("nheavy", -1), ("molwt", -1), ("tpsa", -1),
         ("nhba", -1), ("q_maxpos", -1)]
print(f"{'候选':<22}" + "".join(f"{t[:8]:>10}" for t in TARGETS) + f"{'宏平均':>9}{'vs基线':>8}")
out = []
for name, sign in CANDS:
    au = {}
    for t in TARGETS:
        d = DATA[t]
        y = np.array([r["y"] for r in d])
        s = add(baseline(d), d, [(name, sign)])
        au[t] = fe.auroc(y, s)
    m = float(np.mean(list(au.values())))
    out.append((m, name, au))
for m, name, au in sorted(out, reverse=True):
    print(f"{name:<22}" + "".join(f"{au[t]:>10.3f}" for t in TARGETS)
          + f"{m:>9.3f}{m - float(np.mean(list(base.values()))):>+8.3f}")
