#!/usr/bin/env python3
"""Recompute RMSD of GPDOCK OUTPUT_DOCK poses versus ORI crystal ligands.

Does not mutate ORI_PDB_LIGAND or OUTPUT_DOCK. Ligand RMSD ignores HOH atoms
in *_out_wat_*.pdb. Success is heavy-atom RMSD <= threshold.

GPDOCK folder Y/N is compared to the recomputed top-1 and best-sampled rates
so the original split can be cited only if it matches RMSD.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.dock import redock_rmsd  # noqa: E402
from metgeodock.io import WATER_RESNAMES, iter_pdb_models, load_molecule, select_atoms  # noqa: E402

LOG = logging.getLogger("score_gpdock_output")


def _ligand_only(mol):
    keep = [i for i in range(len(mol)) if str(mol.resnames[i]).strip().upper() not in WATER_RESNAMES]
    if not keep:
        return mol
    return select_atoms(mol, keep, name=mol.name)


def _score_pose_file(pose_path: str, crystal) -> dict:
    path = Path(pose_path)
    top1 = float("nan")
    best = float("nan")
    n = 0
    try:
        for i, pose in iter_pdb_models(path):
            lig = _ligand_only(pose)
            rmsd = redock_rmsd(lig, crystal)
            if not np.isfinite(rmsd):
                continue
            n += 1
            if n == 1:
                top1 = rmsd
            best = rmsd if not np.isfinite(best) else min(best, rmsd)
    except Exception as exc:  # noqa: BLE001
        return {"n_models": 0, "top1_rmsd": float("nan"), "best_sampled_rmsd": float("nan"), "error": str(exc)}
    return {"n_models": n, "top1_rmsd": float(top1), "best_sampled_rmsd": float(best), "error": ""}


def _rates(frame: pd.DataFrame, threshold: float) -> dict:
    ok = frame[frame["error"].fillna("") == ""] if "error" in frame.columns else frame
    if ok.empty:
        return {"n": 0, "top1": None, "sampled": None}
    top1 = ok["top1_rmsd"].to_numpy(dtype=float)
    best = ok["best_sampled_rmsd"].to_numpy(dtype=float)
    return {
        "n": int(len(ok)),
        "n_parse_fail": int(len(frame) - len(ok)),
        "top1": float(np.mean(top1 <= threshold)),
        "sampled": float(np.mean(best <= threshold)),
        "top1_median_rmsd": float(np.nanmedian(top1)),
        "sampled_median_rmsd": float(np.nanmedian(best)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joined", type=Path, default=Path("processed/gpdock_orig/inventory_joined.csv"))
    p.add_argument("--output-inv", type=Path, default=Path("processed/gpdock_orig/inventory_output.csv"))
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--limit", type=int, default=0, help="optional cap per subset for smoke tests")
    p.add_argument("--out-csv", type=Path, default=Path("results/gpdock_orig_rmsd.csv"))
    p.add_argument("--out-json", type=Path, default=Path("results/gpdock_orig_rmsd.json"))
    p.add_argument("--log", type=Path, default=Path("logs/score_gpdock_output.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    joined = pd.read_csv(args.joined, dtype={"rec_id": str})
    out = pd.read_csv(args.output_inv, dtype={"rec_id": str})
    work = out.merge(
        joined[["subset", "metal", "pdb_id", "rec_id", "ligand_path", "gpdock_label"]],
        on=["subset", "metal", "pdb_id", "rec_id"],
        how="left",
        suffixes=("", "_join"),
    )
    if args.limit:
        parts = []
        for subset, sub in work.groupby("subset"):
            parts.append(sub.head(args.limit))
        work = pd.concat(parts, ignore_index=True)
    LOG.info("n_pose_files=%s", len(work))
    crystal_cache: dict[str, object] = {}
    rows = []
    for k, rec in enumerate(work.itertuples(index=False), start=1):
        lig_path = str(rec.ligand_path)
        if lig_path not in crystal_cache:
            try:
                crystal_cache[lig_path] = load_molecule(lig_path)
            except Exception as exc:  # noqa: BLE001
                crystal_cache[lig_path] = exc
        crystal = crystal_cache[lig_path]
        if isinstance(crystal, Exception):
            row = {
                "subset": rec.subset,
                "metal": rec.metal,
                "pdb_id": rec.pdb_id,
                "rec_id": rec.rec_id,
                "protocol": rec.protocol,
                "gpdock_label": rec.gpdock_label,
                "n_models": 0,
                "top1_rmsd": float("nan"),
                "best_sampled_rmsd": float("nan"),
                "error": f"ligand:{crystal}",
            }
        else:
            scored = _score_pose_file(rec.pose_path, crystal)
            row = {
                "subset": rec.subset,
                "metal": rec.metal,
                "pdb_id": rec.pdb_id,
                "rec_id": rec.rec_id,
                "protocol": rec.protocol,
                "gpdock_label": rec.gpdock_label,
                **scored,
            }
        row["success_top1"] = int(np.isfinite(row["top1_rmsd"]) and row["top1_rmsd"] <= args.threshold)
        row["success_sampled"] = int(
            np.isfinite(row["best_sampled_rmsd"]) and row["best_sampled_rmsd"] <= args.threshold
        )
        rows.append(row)
        if k % 500 == 0:
            LOG.info("[%s/%s] %s %s %s", k, len(work), rec.subset, rec.pdb_id, rec.protocol)
            pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_csv, index=False)
    report: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "threshold_A": args.threshold,
        "n_files": int(len(frame)),
        "by_subset_protocol": {},
        "folder_label_vs_rmsd": {},
        "sw_best_of_water_protocols": None,
    }
    for (subset, proto), sub in frame.groupby(["subset", "protocol"]):
        report["by_subset_protocol"][f"{subset}:{proto}"] = _rates(sub, args.threshold)
    for (subset, lab), sub in frame.groupby(["subset", "gpdock_label"]):
        report["folder_label_vs_rmsd"][f"{subset}:{lab}"] = _rates(sub, args.threshold)
    sw = frame[frame["subset"] == "SW"]
    if len(sw):
        agg = (
            sw.groupby(["metal", "pdb_id", "rec_id", "gpdock_label"], dropna=False)
            .agg(
                top1_rmsd=("top1_rmsd", "min"),
                best_sampled_rmsd=("best_sampled_rmsd", "min"),
                n_files=("protocol", "size"),
                error=("error", lambda s: next((x for x in s if x), "")),
            )
            .reset_index()
        )
        report["sw_best_of_water_protocols"] = _rates(agg, args.threshold)
        yrate = None
        labs = sw[["metal", "pdb_id", "rec_id", "gpdock_label"]].drop_duplicates()
        if len(labs):
            yrate = float((labs["gpdock_label"] == "Y").mean())
        report["sw_folder_complex_success"] = yrate
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("%s", json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
