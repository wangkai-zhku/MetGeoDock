#!/usr/bin/env python3
"""Index GPDOCK raw crystals (ORI_PDB_LIGAND) and original docking poses (OUTPUT_DOCK).

Subsets (GPDOCK layout, not renamed):
  SW  single-metal sites with a crystallographic HOH file
  SM  single-metal sites without a HOH file
  MM  multi-metal sites (protein PDB keeps extra ions; no HOH file)

OUTPUT_DOCK/{SW,SM,MM}/{Y,N}/ holds GPDOCK poses. Y/N is the original
success split. SW also has *_out_wat_1.pdb and *_out_wat_2.pdb (1- and
2-water docking). Does not mutate ORI_PDB_LIGAND or OUTPUT_DOCK.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys_path_note = str(ROOT)

import sys

sys.path.insert(0, sys_path_note)

from metgeodock.io import read_pdb  # noqa: E402

LOG = logging.getLogger("index_gpdock_sets")
NAME_RE = re.compile(
    r"^(?P<metal>[a-z0-9]+)_(?P<pdb>[0-9][a-z0-9]{3})_(?P<kind>pro|lig|hoh)_(?P<rec>\d+)$",
    re.I,
)
OUT_RE = re.compile(
    r"^(?P<metal>[a-z0-9]+)_(?P<pdb>[0-9][a-z0-9]{3})_pro_(?P<rec>\d+)_out(?P<wat>_wat_[12])?$",
    re.I,
)


def _index_ori(raw_root: Path, subset: str) -> pd.DataFrame:
    folder = raw_root / subset
    grouped: dict[tuple, dict] = {}
    for path in folder.iterdir():
        if not path.is_file():
            continue
        m = NAME_RE.match(path.stem)
        if not m:
            continue
        key = (m.group("metal").lower(), m.group("pdb").upper(), m.group("rec"))
        grouped.setdefault(key, {})[m.group("kind").lower()] = str(path)
    rows = []
    for (metal, pdb_id, rec_id), parts in grouped.items():
        rows.append(
            {
                "subset": subset,
                "metal": metal,
                "pdb_id": pdb_id,
                "rec_id": rec_id,
                "protein_path": parts.get("pro", ""),
                "ligand_path": parts.get("lig", ""),
                "water_path": parts.get("hoh", ""),
                "complete": int("pro" in parts and "lig" in parts),
                "has_water": int("hoh" in parts),
            }
        )
    return pd.DataFrame(rows)


def _index_output(out_root: Path, subset: str) -> pd.DataFrame:
    rows = []
    for label in ("Y", "N"):
        folder = out_root / subset / label
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.suffix.lower() != ".pdb":
                continue
            m = OUT_RE.match(path.stem)
            if not m:
                continue
            wat = m.group("wat") or ""
            protocol = "out"
            n_wat = 0
            if wat == "_wat_1":
                protocol = "wat_1"
                n_wat = 1
            elif wat == "_wat_2":
                protocol = "wat_2"
                n_wat = 2
            rows.append(
                {
                    "subset": subset,
                    "gpdock_label": label,
                    "protocol": protocol,
                    "n_dock_water": n_wat,
                    "metal": m.group("metal").lower(),
                    "pdb_id": m.group("pdb").upper(),
                    "rec_id": m.group("rec"),
                    "pose_path": str(path),
                    "nbytes": int(path.stat().st_size),
                }
            )
    return pd.DataFrame(rows)


def _n_metal(path: str) -> int:
    try:
        mol = read_pdb(path)
    except Exception:
        return -1
    return int(mol.metal_indices().size)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ori", type=Path, default=Path("ORI_PDB_LIGAND"))
    p.add_argument("--output-dock", type=Path, default=Path("OUTPUT_DOCK"))
    p.add_argument("--out-dir", type=Path, default=Path("processed/gpdock_orig"))
    p.add_argument("--log", type=Path, default=Path("logs/index_gpdock_sets.log"))
    p.add_argument("--count-metals", action="store_true", default=True)
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    ori_parts = []
    out_parts = []
    by_subset = {}
    for subset in ("SW", "SM", "MM"):
        ori = _index_ori(args.ori, subset)
        out = _index_output(args.output_dock, subset)
        ori_parts.append(ori)
        out_parts.append(out)
        n_y = int((out["gpdock_label"] == "Y").sum()) if len(out) else 0
        n_n = int((out["gpdock_label"] == "N").sum()) if len(out) else 0
        complexes_y = out[out["gpdock_label"] == "Y"][["metal", "pdb_id", "rec_id"]].drop_duplicates() if len(out) else out
        complexes_n = out[out["gpdock_label"] == "N"][["metal", "pdb_id", "rec_id"]].drop_duplicates() if len(out) else out
        n_cx_y = int(len(complexes_y))
        n_cx_n = int(len(complexes_n))
        n_cx = n_cx_y + n_cx_n
        rate = (n_cx_y / n_cx) if n_cx else None
        by_subset[subset] = {
            "n_ori_complete": int((ori["complete"] == 1).sum()) if len(ori) else 0,
            "n_ori_with_water": int((ori["has_water"] == 1).sum()) if len(ori) else 0,
            "n_output_files": int(len(out)),
            "n_output_files_Y": n_y,
            "n_output_files_N": n_n,
            "n_complex_Y": n_cx_y,
            "n_complex_N": n_cx_n,
            "gpdock_folder_success": rate,
            "n_ori_by_metal": ori.groupby("metal").size().to_dict() if len(ori) else {},
            "n_complex_Y_by_metal": complexes_y.groupby("metal").size().to_dict() if len(complexes_y) else {},
        }
        LOG.info("%s ori=%s out_files=%s Y_cx=%s N_cx=%s rate=%s", subset, len(ori), len(out), n_cx_y, n_cx_n, rate)
    ori_all = pd.concat(ori_parts, ignore_index=True)
    out_all = pd.concat(out_parts, ignore_index=True)
    keys = ["subset", "metal", "pdb_id", "rec_id"]
    labels = (
        out_all[keys + ["gpdock_label"]]
        .drop_duplicates(keys)
        .rename(columns={"gpdock_label": "gpdock_label"})
    )
    merged = ori_all.merge(labels, on=keys, how="left")
    if args.count_metals:
        mm = merged[merged["subset"] == "MM"].copy()
        LOG.info("count metals on MM n=%s", len(mm))
        n_metals = []
        for i, rec in mm.iterrows():
            n_metals.append(_n_metal(rec.protein_path))
            if len(n_metals) % 400 == 0:
                LOG.info("MM metals %s / %s", len(n_metals), len(mm))
        mm = mm.assign(n_metal=n_metals)
        mm.to_csv(args.out_dir / "inventory_mm.csv", index=False)
        by_subset["MM"]["n_metal_value_counts"] = mm["n_metal"].value_counts().sort_index().to_dict()
        by_subset["MM"]["n_metal_ge2"] = int((mm["n_metal"] >= 2).sum())
        by_subset["MM"]["n_metal_eq1"] = int((mm["n_metal"] == 1).sum())
    ori_all.to_csv(args.out_dir / "inventory_ori.csv", index=False)
    out_all.to_csv(args.out_dir / "inventory_output.csv", index=False)
    merged.to_csv(args.out_dir / "inventory_joined.csv", index=False)
    for subset in ("SW", "SM", "MM"):
        merged[merged["subset"] == subset].to_csv(args.out_dir / f"inventory_{subset.lower()}.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ori": str(args.ori),
        "output_dock": str(args.output_dock),
        "meaning": {
            "SW": "single-metal crystals with HOH files; GPDOCK also wrote wat_1/wat_2 poses",
            "SM": "single-metal crystals without HOH files",
            "MM": "multi-metal crystals (extra ions kept in the protein PDB)",
            "Y": "GPDOCK original success folder",
            "N": "GPDOCK original failure folder",
        },
        "by_subset": by_subset,
        "n_ori": int(len(ori_all)),
        "n_output_files": int(len(out_all)),
    }
    (args.out_dir / "index_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
