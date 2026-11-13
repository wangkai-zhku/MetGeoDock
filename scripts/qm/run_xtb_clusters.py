#!/usr/bin/env python3
"""Optional GFN2-xTB single points on prepared metal clusters.

Skips cleanly if the xtb binary is absent. Writes energies to
processed/qm/xtb_energies.csv. Does not invent numbers.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("run_xtb")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cluster-dir", type=Path, default=Path("processed/qm/clusters"))
    p.add_argument("--out-csv", type=Path, default=Path("processed/qm/xtb_energies.csv"))
    p.add_argument("--log", type=Path, default=Path("logs/run_xtb_clusters.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    xtb = shutil.which("xtb")
    if not xtb:
        LOG.warning("xtb not on PATH; writing empty table")
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["xyz", "energy_hartree", "status"]).to_csv(args.out_csv, index=False)
        return
    rows = []
    work = args.cluster_dir / "xtb_runs"
    work.mkdir(parents=True, exist_ok=True)
    for xyz in sorted(args.cluster_dir.glob("*.xyz")):
        run = work / xyz.stem
        run.mkdir(exist_ok=True)
        dest = run / xyz.name
        dest.write_bytes(xyz.read_bytes())
        try:
            proc = subprocess.run(
                [xtb, dest.name, "--gfn", "2", "--chrg", "2"],
                cwd=run,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            energy = None
            for line in proc.stdout.splitlines():
                if "TOTAL ENERGY" in line.upper() and "Eh" in line:
                    parts = line.split()
                    for tok in parts:
                        try:
                            energy = float(tok)
                            break
                        except ValueError:
                            continue
            rows.append({"xyz": str(xyz), "energy_hartree": energy, "status": "ok" if energy is not None else "parse_fail"})
        except Exception as exc:  # noqa: BLE001
            rows.append({"xyz": str(xyz), "energy_hartree": None, "status": str(exc)})
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    LOG.info("wrote %s rows=%s", args.out_csv, len(rows))


if __name__ == "__main__":
    main()
