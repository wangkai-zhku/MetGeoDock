#!/usr/bin/env python3
"""Download the public MetalloDock (JACS 2026) demo/test complexes.

Zhang et al. report a 541-complex time-split test set; the GitHub release
only ships 10 PDB IDs in dataset/test_pdbid.csv plus pocket/ligand files.
This script copies those files into processed/metallodock_ref/ (never into
ORI_PDB_LIGAND). The full 541 list is in the paper SI and is not redistributed
here; scripts/data/fetch_rcsb_dates.py builds a MetalloDock-style time split
on our own crystal inventory for the remaining comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

LOG = logging.getLogger("fetch_metallodock")
API = "https://api.github.com/repos/SII-ZhangHui/MetalloDock/contents"
RAW = "https://raw.githubusercontent.com/SII-ZhangHui/MetalloDock/main"
PDB_IDS = ("8elk", "8a7k", "8jao", "8bdb", "8qvu", "8hyp", "8one", "7oyo", "8hz7", "8olm")


def _get(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "MetGeoDock/0.1"})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def _download(rel: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob = _get(f"{RAW}/{rel}")
    dest.write_bytes(blob)
    return len(blob)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("processed/metallodock_ref"))
    p.add_argument("--log", type=Path, default=Path("logs/fetch_metallodock_testset.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    files = []
    csv_rel = "dataset/test_pdbid.csv"
    n = _download(csv_rel, args.out_dir / csv_rel)
    files.append({"path": csv_rel, "bytes": n})
    for pdb in PDB_IDS:
        listing = json.loads(_get(f"{API}/dataset/metallo_complex/{pdb}"))
        for item in listing:
            if item["type"] != "file":
                continue
            name = item["name"]
            if name.endswith((".pdb", ".sdf")) and (
                "ligand" in name or "pro_pocket" in name or name == f"{pdb}.pdb"
            ):
                rel = item["path"]
                n = _download(rel, args.out_dir / rel)
                files.append({"path": rel, "bytes": n})
                LOG.info("wrote %s (%s bytes)", rel, n)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://github.com/SII-ZhangHui/MetalloDock",
        "citation": "Zhang et al., J. Am. Chem. Soc. 2026, 148, 3086-3101, doi:10.1021/jacs.5c15876",
        "n_pdb": len(PDB_IDS),
        "pdb_ids": list(PDB_IDS),
        "n_files": len(files),
        "note": "GitHub ships 10 complexes; paper test set n=541 is in SI.",
    }
    (args.out_dir / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("done files=%s", len(files))


if __name__ == "__main__":
    main()
