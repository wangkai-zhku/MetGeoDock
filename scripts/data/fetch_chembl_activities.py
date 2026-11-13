#!/usr/bin/env python3
"""Download and curate ChEMBL bioactivities for GPDOCK metal-ion targets.

Curation follows the ChEMBL modelling checklist: SINGLE PROTEIN (by using
explicit chembl_ids), binding assays, '=' relation, pChEMBL present,
confidence_score >= 8, drop flagged invalid/duplicate rows, median per
molecule. Raw JSON pages are stored under processed/chembl/raw/; curated
tables go to processed/chembl/curated/. Does not write into ORI_PDB_LIGAND.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import yaml

LOG = logging.getLogger("fetch_chembl")
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"


def _get_json(url: str, retries: int = 4) -> dict:
    last = None
    for i in range(retries):
        try:
            req = Request(url, headers={"Accept": "application/json", "User-Agent": "MetGeoDock/0.1"})
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def fetch_activities(chembl_id: str, raw_dir: Path, page_limit: int = 1000) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        q = urlencode(
            {
                "target_chembl_id": chembl_id,
                "assay_type": "B",
                "limit": page_limit,
                "offset": offset,
                "format": "json",
            }
        )
        payload = _get_json(f"{CHEMBL}/activity.json?{q}")
        page = payload.get("activities") or []
        rows.extend(page)
        page_meta = payload.get("page_meta") or {}
        total = int(page_meta.get("total_count") or len(rows))
        offset += page_limit
        if offset >= total or not page:
            break
        time.sleep(0.15)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{chembl_id}_activity.json").write_text(json.dumps({"n": len(rows), "activities": rows}), encoding="utf-8")
    return rows


def fetch_assay_confidence(target_chembl_id: str, cache: Path) -> dict[str, int]:
    """One paginated assay listing per target instead of N single-assay GETs."""
    out: dict[str, int] = {}
    if cache.exists():
        out.update(json.loads(cache.read_text(encoding="utf-8")))
    offset = 0
    while True:
        q = urlencode(
            {
                "target_chembl_id": target_chembl_id,
                "limit": 1000,
                "offset": offset,
                "format": "json",
            }
        )
        payload = _get_json(f"{CHEMBL}/assay.json?{q}")
        assays = payload.get("assays") or []
        for a in assays:
            aid = a.get("assay_chembl_id")
            if aid:
                out[aid] = int(a.get("confidence_score") or 0)
        page_meta = payload.get("page_meta") or {}
        total = int(page_meta.get("total_count") or offset + len(assays))
        offset += 1000
        if offset >= total or not assays:
            break
        time.sleep(0.1)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def curate(rows: list[dict], conf: dict[str, int], min_conf: int) -> pd.DataFrame:
    kept = []
    drop = Counterish()
    for r in rows:
        if r.get("potential_duplicate"):
            drop["duplicate"] += 1
            continue
        if r.get("data_validity_comment"):
            drop["invalid"] += 1
            continue
        if r.get("standard_relation") not in ("=", None) and r.get("standard_relation") != "=":
            drop["relation"] += 1
            continue
        if r.get("standard_relation") != "=":
            drop["relation"] += 1
            continue
        if r.get("pchembl_value") in (None, ""):
            drop["no_pchembl"] += 1
            continue
        assay = r.get("assay_chembl_id")
        if conf.get(assay, 0) < min_conf:
            drop["low_confidence"] += 1
            continue
        kept.append(
            {
                "molecule_chembl_id": r.get("molecule_chembl_id"),
                "target_chembl_id": r.get("target_chembl_id"),
                "assay_chembl_id": assay,
                "standard_type": r.get("standard_type"),
                "standard_value": r.get("standard_value"),
                "standard_units": r.get("standard_units"),
                "pchembl_value": float(r["pchembl_value"]),
                "document_chembl_id": r.get("document_chembl_id"),
                "canonical_smiles": r.get("canonical_smiles"),
                "confidence_score": conf.get(assay, 0),
            }
        )
    drop["kept_measurements"] = len(kept)
    frame = pd.DataFrame(kept)
    return frame, drop.as_dict()


class Counterish(dict):
    def __getitem__(self, k):
        return super().get(k, 0)

    def __setitem__(self, k, v):
        super().__setitem__(k, v)

    def as_dict(self):
        return dict(self)


def median_per_molecule(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    g = (
        frame.groupby(["target_chembl_id", "molecule_chembl_id"], as_index=False)
        .agg(
            pactivity=("pchembl_value", "median"),
            pactivity_spread=("pchembl_value", lambda s: float(s.max() - s.min()) if len(s) else 0.0),
            n_measurements=("pchembl_value", "size"),
            n_documents=("document_chembl_id", "nunique"),
            standard_types=("standard_type", lambda s: "|".join(sorted(set(map(str, s))))),
            canonical_smiles=("canonical_smiles", "first"),
        )
    )
    g["inconsistent"] = g["pactivity_spread"] >= 1.0
    return g


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/metal_targets.yaml"))
    p.add_argument("--out-dir", type=Path, default=Path("processed/chembl"))
    p.add_argument("--min-confidence", type=int, default=8)
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--log", type=Path, default=Path("logs/fetch_chembl_activities.log"))
    p.add_argument("--max-targets", type=int, default=0, help="debug cap; 0 = all")
    p.add_argument("--skip-existing", action="store_true", help="skip targets that already have curated CSV")
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
    conf_cache = raw_dir / "assay_confidence.json"

    all_mol = []
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "targets": []}
    n_done = 0
    seen = set()
    exclude_tokens = [s.lower() for s in cfg.get("exclude_notes_containing", ["excluded"])]
    for metal, block in cfg["metals"].items():
        for tgt in block.get("targets", []):
            note = str(tgt.get("note") or "")
            if any(tok in note.lower() for tok in exclude_tokens):
                continue
            chembl_id = tgt["chembl_id"]
            key = (metal, chembl_id)
            if key in seen:
                continue
            seen.add(key)
            curated_path = cur_dir / f"{metal}_{chembl_id}.csv"
            if args.skip_existing and curated_path.exists():
                mol = pd.read_csv(curated_path)
                if not mol.empty:
                    all_mol.append(mol)
                summary["targets"].append(
                    {
                        "metal": metal,
                        "chembl_id": chembl_id,
                        "name": tgt.get("name"),
                        "skipped_existing": True,
                        "n_molecules": int(len(mol)),
                    }
                )
                continue
            LOG.info("fetch %s %s %s", metal, tgt.get("name"), chembl_id)
            try:
                rows = fetch_activities(chembl_id, raw_dir)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("skip %s: %s", chembl_id, exc)
                summary["targets"].append({"metal": metal, "chembl_id": chembl_id, "error": str(exc)})
                continue
            conf = fetch_assay_confidence(chembl_id, conf_cache)
            meas, drop = curate(rows, conf, args.min_confidence)
            mol = median_per_molecule(meas)
            mol["metal"] = metal
            mol["target_name"] = tgt.get("name")
            mol["uniprot"] = tgt.get("uniprot")
            mol["pdb"] = tgt.get("pdb")
            mol["target_id"] = chembl_id
            if not mol.empty:
                mol.to_csv(cur_dir / f"{metal}_{chembl_id}.csv", index=False)
                all_mol.append(mol)
            summary["targets"].append(
                {
                    "metal": metal,
                    "chembl_id": chembl_id,
                    "name": tgt.get("name"),
                    "n_raw": len(rows),
                    "n_measurements": int(len(meas)),
                    "n_molecules": int(len(mol)),
                    "drop": drop,
                }
            )
            n_done += 1
            if args.max_targets and n_done >= args.max_targets:
                break
            time.sleep(args.sleep)
        if args.max_targets and n_done >= args.max_targets:
            break
    if all_mol:
        cat = pd.concat(all_mol, ignore_index=True)
        cat.to_csv(cur_dir / "all_targets_molecules.csv", index=False)
        summary["n_molecules_total"] = int(len(cat))
        summary["n_targets_ok"] = int(cat["target_id"].nunique())
    (args.out_dir / "fetch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("done molecules=%s", summary.get("n_molecules_total", 0))


if __name__ == "__main__":
    main()
