#!/usr/bin/env python3
"""Derive MetGeo force-field parameters from GPDOCK histograms.

For each metal–element pair, treat the published 0.1 Å histogram as P(d).
The harmonic force constant is k = RT / variance(d) with RT = 0.592 kcal/mol
at 298 K. Mode distance is the bin center of argmax(P). Soft walls sit at the
GPDOCK DIS_MIN / DIS_MAX cutoffs. Writes processed/ff/metalgeo_params.json.
Does not use SLEF.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from metgeodock.metals import (
    DIS_MAX_N,
    DIS_MAX_O,
    DIS_MAX_S,
    DIS_MIN_N,
    DIS_MIN_O,
    DIS_MIN_S,
    METAL_ORDER,
    PROB_N,
    PROB_O,
    PROB_S,
)

RT_KCAL = 0.592
BIN_CENTERS = 1.7 + (np.arange(17) + 0.5) * 0.1
TABLES = {"O": PROB_O, "N": PROB_N, "S": PROB_S}
CUT_MIN = {"O": DIS_MIN_O, "N": DIS_MIN_N, "S": DIS_MIN_S}
CUT_MAX = {"O": DIS_MAX_O, "N": DIS_MAX_N, "S": DIS_MAX_S}


def pair_params(metal_index: int, element: str) -> dict:
    hist = np.asarray(TABLES[element][metal_index], dtype=float)
    p = hist / max(float(hist.sum()), 1.0e-12)
    mu = float((p * BIN_CENTERS).sum())
    var = float((p * (BIN_CENTERS - mu) ** 2).sum())
    var = max(var, 0.01)
    mode = int(np.argmax(hist))
    return {
        "mean_angstrom": mu,
        "variance_angstrom2": var,
        "k_kcal_per_a2": float(RT_KCAL / var),
        "mode_angstrom": float(BIN_CENTERS[mode]),
        "p_mode": float(hist[mode]),
        "dmin_angstrom": float(CUT_MIN[element][metal_index]),
        "dmax_angstrom": float(CUT_MAX[element][metal_index]),
        "histogram": hist.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("processed/ff/metalgeo_params.json"))
    args = p.parse_args()
    pairs = {}
    for i, metal in enumerate(METAL_ORDER):
        pairs[metal] = {el: pair_params(i, el) for el in ("O", "N", "S")}
    blob = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "GPDOCK histograms transcribed from gpml_dock.c (Wang, Brief. Bioinform. 2023)",
        "rt_kcal_per_mol": RT_KCAL,
        "temperature_K": 298.15,
        "formula": "k = RT / Var(d) ; E_coord = 0.5 k (d - d_mode)^2 - ln(P(d)/P_mode)",
        "pairs": pairs,
        "vdw_sigma_angstrom": {
            "C": 1.70,
            "N": 1.55,
            "O": 1.52,
            "S": 1.80,
            "P": 1.80,
            "F": 1.47,
            "CL": 1.75,
            "BR": 1.85,
            "I": 1.98,
            "H": 1.10,
        },
        "weights": {
            "coord": 1.0,
            "steric": 12.0,
            "contact": 0.02,
            "geo": 2.0,
            "uncoord": 12.0,
        },
        "rank_weights": {
            "coord": 0.35,
            "geo": 0.0,
            "cpen": 0.0,
            "contact": 0.0,
        },
        "steric_cutoff_angstrom": 2.30,
        "contact_lo_angstrom": 3.10,
        "contact_hi_angstrom": 4.60,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
