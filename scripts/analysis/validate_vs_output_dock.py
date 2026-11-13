#!/usr/bin/env python3
"""Validate MetGeoDock against GPDOCK OUTPUT_DOCK poses as the reference.

OUTPUT_DOCK is not re-docked. Crystal RMSD of those poses was computed by
scripts/analysis/score_gpdock_output.py (read-only). This script joins the
same complexes used in results/gpdock_orig_metgeodock.csv.

GPDOCK reference protocol:
  SW  best of out / wat_1 / wat_2  (GPDOCK water docking)
  MM  out                          (GPDOCK multi-metal docking)

Success is heavy-atom RMSD <= threshold. Does not mutate OUTPUT_DOCK.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _prep_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["pdb_id"] = out["pdb_id"].astype(str).str.upper()
    out["rec_id"] = out["rec_id"].astype(str)
    out["metal"] = out["metal"].astype(str).str.lower()
    return out


def _rates(ok: pd.DataFrame, prefix: str) -> dict:
    n = int(len(ok))
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        f"{prefix}_top1": float(ok[f"{prefix}_success_top1"].mean()),
        f"{prefix}_sampled": float(ok[f"{prefix}_success_sampled"].mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpdock-csv", type=Path, default=Path("results/gpdock_orig_rmsd.csv"))
    p.add_argument("--ours-csv", type=Path, default=Path("results/gpdock_orig_metgeodock.csv"))
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=Path("results/validate_vs_output_dock.json"))
    p.add_argument("--out-csv", type=Path, default=Path("results/validate_vs_output_dock.csv"))
    args = p.parse_args()

    gpdock = _prep_keys(pd.read_csv(args.gpdock_csv, dtype={"rec_id": str}))
    ours = _prep_keys(pd.read_csv(args.ours_csv, dtype={"rec_id": str}))
    gpdock = gpdock[gpdock["error"].fillna("") == ""].copy()
    ours = ours[ours["error"].fillna("") == ""].copy()

    keys = ["metal", "pdb_id", "rec_id"]
    sw_ref = (
        gpdock[gpdock["subset"] == "SW"]
        .groupby(keys, as_index=False)
        .agg(
            gpdock_top1_rmsd=("top1_rmsd", "min"),
            gpdock_sampled_rmsd=("best_sampled_rmsd", "min"),
            gpdock_label=("gpdock_label", "first"),
            n_gpdock_files=("protocol", "size"),
        )
    )
    mm_ref = (
        gpdock[(gpdock["subset"] == "MM") & (gpdock["protocol"] == "out")]
        .groupby(keys, as_index=False)
        .agg(
            gpdock_top1_rmsd=("top1_rmsd", "min"),
            gpdock_sampled_rmsd=("best_sampled_rmsd", "min"),
            gpdock_label=("gpdock_label", "first"),
            n_gpdock_files=("protocol", "size"),
        )
    )
    sw_ref["subset"] = "SW"
    mm_ref["subset"] = "MM"
    ref = pd.concat([sw_ref, mm_ref], ignore_index=True)
    ref["gpdock_success_top1"] = (ref["gpdock_top1_rmsd"] <= args.threshold).astype(int)
    ref["gpdock_success_sampled"] = (ref["gpdock_sampled_rmsd"] <= args.threshold).astype(int)

    proto_subset = {
        "water_keep": "SW",
        "water_strip": "SW",
        "multi_keep": "MM",
        "multi_strip": "MM",
    }
    ours["subset"] = ours["protocol"].map(proto_subset)
    paired = ours.merge(ref, on=keys + ["subset"], how="left", suffixes=("", "_ref"))
    paired.to_csv(args.out_csv, index=False)

    by_protocol = {}
    for proto, sub in paired.groupby("protocol"):
        ok = sub.dropna(subset=["gpdock_sampled_rmsd"])
        rec = {
            "n_ours": int(len(sub)),
            "n_paired": int(len(ok)),
            "metgeodock_top1": float(ok["success_top1"].mean()) if len(ok) else None,
            "metgeodock_top3": float(ok["success_top3"].mean()) if len(ok) else None,
            "metgeodock_sampled": float(ok["success_sampled"].mean()) if len(ok) else None,
            "gpdock_top1": float(ok["gpdock_success_top1"].mean()) if len(ok) else None,
            "gpdock_sampled": float(ok["gpdock_success_sampled"].mean()) if len(ok) else None,
            "gpdock_folder_Y": float((ok["gpdock_label"] == "Y").mean()) if len(ok) else None,
        }
        if len(ok):
            rec["sampled_ours_minus_gpdock"] = rec["metgeodock_sampled"] - rec["gpdock_sampled"]
            rec["top1_ours_minus_gpdock"] = rec["metgeodock_top1"] - rec["gpdock_top1"]
            rec["n_ours_sampled_win"] = int(
                (ok["success_sampled"] > ok["gpdock_success_sampled"]).sum()
            )
            rec["n_gpdock_sampled_win"] = int(
                (ok["gpdock_success_sampled"] > ok["success_sampled"]).sum()
            )
            rec["n_sampled_tie"] = int(
                (ok["success_sampled"] == ok["gpdock_success_sampled"]).sum()
            )
        by_protocol[str(proto)] = rec

    y_ok = (
        gpdock[gpdock["gpdock_label"] == "Y"].groupby(["subset", "metal", "pdb_id", "rec_id"])["success_sampled"].max()
    )
    n_ok = (
        gpdock[gpdock["gpdock_label"] == "N"].groupby(["subset", "metal", "pdb_id", "rec_id"])["success_sampled"].max()
    )
    label_check = {
        "Y_complexes_sampled_success": float(y_ok.mean()) if len(y_ok) else None,
        "N_complexes_sampled_success": float(n_ok.mean()) if len(n_ok) else None,
        "n_Y": int(len(y_ok)),
        "n_N": int(len(n_ok)),
        "Y_is_sampled_reference": bool(y_ok.mean() >= 0.90) if len(y_ok) else False,
    }

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference": "OUTPUT_DOCK GPDOCK poses; RMSD vs ORI_PDB_LIGAND crystal ligands",
        "threshold_A": args.threshold,
        "note": (
            "OUTPUT_DOCK Y/N is GPDOCK sampling success (any pose RMSD<=2 A), "
            "not a ranked top-1. Compare MetGeoDock sampled to GPDOCK sampled; "
            "compare MetGeoDock top-1/top-3 to GPDOCK file-order top-1 only as a "
            "ranking diagnostic."
        ),
        "output_dock_label_check": label_check,
        "head_to_head_same_complexes": by_protocol,
        "csv": str(args.out_csv),
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
