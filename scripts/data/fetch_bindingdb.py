#!/usr/bin/env python3
"""Download BindingDB affinities for UniProt IDs in configs/metal_targets.yaml.

Raw TSV/JSON is stored under processed/bindingdb/raw/; curated tables go to
processed/bindingdb/curated/. Does not write into ORI_PDB_LIGAND.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import yaml

LOG = logging.getLogger("fetch_bindingdb")
BDB = "https://bindingdb.org/rest/getLigandsByUniprots"


def _get(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "MetGeoDock/0.1", "Accept": "application/json"})
    with urlopen(req, timeout=120) as resp:
        return resp.read()


def _pactivity(ki_nm: float | None, ic50_nm: float | None, kd_nm: float | None, ec50_nm: float | None) -> float:
    vals = [v for v in (ki_nm, ic50_nm, kd_nm, ec50_nm) if v is not None and v > 0]
    if not vals:
        return float("nan")
    nm = min(vals)
    return float(-math.log10(nm * 1.0e-9))


def parse_records(payload) -> list[dict]:
    if isinstance(payload, list):
        recs = payload
    elif isinstance(payload, dict):
        block = (
            payload.get("getLindsByUniprotsResponse")
            or payload.get("getLigandsByUniprotsResponse")
            or payload
        )
        recs = block.get("affinities") if isinstance(block, dict) else block
        if isinstance(recs, dict):
            recs = recs.get("affinities") or [recs]
    else:
        recs = []
    return [r for r in recs if isinstance(r, dict)] if isinstance(recs, list) else []


def record_pactivity(r: dict) -> float:
    atype = str(r.get("affinity_type") or "").lower()
    val = _to_nM(r.get("affinity") or r.get("ki") or r.get("Ki") or r.get("ic50") or r.get("IC50"), r.get("units") or "nM")
    if val is None:
        return float("nan")
    if atype in {"ki", "ic50", "kd", "ec50", ""} or True:
        return float(-math.log10(val * 1.0e-9)) if val > 0 else float("nan")
    return float("nan")


def _to_nM(value, units: str | None) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    u = (units or "nM").strip().lower()
    if u in {"nm", "nanomolar"}:
        return x
    if u in {"um", "µm", "micromolar"}:
        return x * 1.0e3
    if u in {"mm", "millimolar"}:
        return x * 1.0e6
    if u in {"m", "molar"}:
        return x * 1.0e9
    if u in {"pm", "picomolar"}:
        return x * 1.0e-3
    return x


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/metal_targets.yaml"))
    p.add_argument("--out-dir", type=Path, default=Path("processed/bindingdb"))
    p.add_argument("--cutoff", type=int, default=10000, help="BindingDB affinity cutoff in nM")
    p.add_argument("--sleep", type=float, default=0.4)
    p.add_argument("--log", type=Path, default=Path("logs/fetch_bindingdb.log"))
    p.add_argument("--from-raw", action="store_true", help="parse already-downloaded JSON only")
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw_dir = args.out_dir / "raw"
    cur_dir = args.out_dir / "curated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cur_dir.mkdir(parents=True, exist_ok=True)
    exclude_tokens = [s.lower() for s in cfg.get("exclude_notes_containing", ["excluded"])]
    all_mol = []
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "targets": []}
    seen_up = set()
    for metal, block in cfg["metals"].items():
        for tgt in block.get("targets", []):
            note = str(tgt.get("note") or "")
            if any(tok in note.lower() for tok in exclude_tokens):
                continue
            uniprot = tgt.get("uniprot")
            if not uniprot or uniprot in seen_up:
                continue
            seen_up.add(uniprot)
            q = urlencode(
                {
                    "uniprot": uniprot,
                    "cutoff": args.cutoff,
                    "code": 0,
                    "response": "application/json",
                }
            )
            raw_path = raw_dir / f"{uniprot}.json"
            if args.from_raw:
                if not raw_path.exists():
                    summary["targets"].append({"metal": metal, "uniprot": uniprot, "error": "missing_raw"})
                    continue
                blob = raw_path.read_bytes()
                LOG.info("parse raw %s %s", metal, uniprot)
            else:
                url = f"{BDB}?{q}"
                LOG.info("BindingDB %s %s %s", metal, tgt.get("name"), uniprot)
                try:
                    blob = _get(url)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("fail %s: %s", uniprot, exc)
                    summary["targets"].append({"metal": metal, "uniprot": uniprot, "error": str(exc)})
                    continue
                raw_path.write_bytes(blob)
            try:
                payload = json.loads(blob.decode("utf-8"))
            except json.JSONDecodeError:
                summary["targets"].append({"metal": metal, "uniprot": uniprot, "error": "non_json"})
                continue
            recs = parse_records(payload)
            rows = []
            for r in recs:
                pact = record_pactivity(r)
                if not math.isfinite(pact):
                    continue
                rows.append(
                    {
                        "uniprot": uniprot,
                        "metal": metal,
                        "target_name": tgt.get("name"),
                        "target_id": tgt.get("chembl_id"),
                        "pdb": tgt.get("pdb"),
                        "smiles": r.get("smile") or r.get("smiles") or r.get("canonical_smiles"),
                        "ligand_id": r.get("monomerid") or r.get("bdb_id") or r.get("ligand"),
                        "pactivity": pact,
                        "source": "bindingdb",
                    }
                )
            frame = pd.DataFrame(rows)
            if not frame.empty:
                g = (
                    frame.groupby(["uniprot", "smiles"], as_index=False)
                    .agg(pactivity=("pactivity", "median"), n_measurements=("pactivity", "size"), ligand_id=("ligand_id", "first"))
                )
                g["metal"] = metal
                g["target_name"] = tgt.get("name")
                g["target_id"] = f"BDB_{uniprot}"
                g["pdb"] = tgt.get("pdb")
                g.to_csv(cur_dir / f"{metal}_{uniprot}.csv", index=False)
                all_mol.append(g)
                n_mol = int(len(g))
            else:
                n_mol = 0
            summary["targets"].append({"metal": metal, "uniprot": uniprot, "name": tgt.get("name"), "n_molecules": n_mol})
            if not args.from_raw:
                time.sleep(args.sleep)
    if all_mol:
        cat = pd.concat(all_mol, ignore_index=True)
        cat.to_csv(cur_dir / "all_targets_molecules.csv", index=False)
        summary["n_molecules_total"] = int(len(cat))
        summary["n_targets_ok"] = int(cat["target_id"].nunique())
    (args.out_dir / "fetch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("done molecules=%s", summary.get("n_molecules_total", 0))


if __name__ == "__main__":
    main()
