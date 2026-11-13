#!/usr/bin/env python3
"""Fetch RCSB deposition/release dates and build a MetalloDock-style time split.

MetalloDock (Zhang et al., JACS 2026) used a time-stratified split of 8836
metalloprotein complexes (train 7453 / val 842 / test 541). The public GitHub
release does not list the 541 IDs. This script dates OUR crystal inventory
(ORI_PDB_LIGAND, read-only) and labels the most recent 15% as test, the next
10% as val, remainder train — a time split of the same kind, not a claim that
the PDB IDs match the paper SI.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

LOG = logging.getLogger("fetch_rcsb_dates")
GRAPHQL = "https://data.rcsb.org/graphql"


def _post(ids: list[str]) -> list[dict]:
    query = """
    query ($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        rcsb_accession_info {
          deposit_date
          initial_release_date
        }
      }
    }
    """
    payload = json.dumps({"query": query, "variables": {"ids": ids}}).encode("utf-8")
    req = Request(
        GRAPHQL,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "MetGeoDock/0.1"},
        method="POST",
    )
    with urlopen(req, timeout=90) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("data", {}).get("entries") or []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, default=Path("processed/crystal/inventory.csv"))
    p.add_argument("--out", type=Path, default=Path("processed/splits/time_split.csv"))
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--log", type=Path, default=Path("logs/fetch_rcsb_dates.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    inv = pd.read_csv(args.inventory)
    inv = inv[inv["complete"] == 1].copy()
    pdbs = sorted({str(x).upper() for x in inv["pdb_id"].dropna()})
    rows = []
    for i in range(0, len(pdbs), args.batch):
        chunk = pdbs[i : i + args.batch]
        LOG.info("RCSB %s-%s / %s", i + 1, i + len(chunk), len(pdbs))
        try:
            entries = _post(chunk)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("batch fail %s: %s", chunk[:3], exc)
            time.sleep(2.0)
            continue
        for e in entries:
            info = e.get("rcsb_accession_info") or {}
            rows.append(
                {
                    "pdb_id": str(e.get("rcsb_id") or "").upper(),
                    "deposit_date": info.get("deposit_date"),
                    "release_date": info.get("initial_release_date"),
                }
            )
        time.sleep(0.15)
    dates = pd.DataFrame(rows)
    if dates.empty:
        raise SystemExit("no RCSB dates returned")
    dates["release_date"] = pd.to_datetime(dates["release_date"], errors="coerce")
    dates = dates.dropna(subset=["release_date"]).sort_values("release_date")
    n = len(dates)
    n_test = max(1, int(round(args.test_frac * n)))
    n_val = max(1, int(round(args.val_frac * n)))
    split = ["train"] * n
    split[-n_test:] = ["test"] * n_test
    split[-(n_test + n_val) : -n_test] = ["val"] * n_val
    dates["time_split"] = split
    merged = inv.merge(dates, on="pdb_id", how="left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_unique_pdb_dated": int(n),
        "n_test_pdb": int((dates["time_split"] == "test").sum()),
        "n_val_pdb": int((dates["time_split"] == "val").sum()),
        "n_train_pdb": int((dates["time_split"] == "train").sum()),
        "test_release_min": str(dates.loc[dates["time_split"] == "test", "release_date"].min()),
        "test_release_max": str(dates.loc[dates["time_split"] == "test", "release_date"].max()),
        "protocol": "sort unique PDBs by RCSB initial_release_date; last 15% test, previous 10% val",
        "metallodock_note": "MetalloDock paper n_test=541 on 8836 complexes; this is the same split *style* on our inventory",
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("%s", json.dumps(summary))


if __name__ == "__main__":
    main()
