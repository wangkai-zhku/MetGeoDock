#!/usr/bin/env python3
"""Build a unified agrochemical library for three metalloprotein herbicide targets.

Pre-registered filters only (see configs/agri_vs.yaml). One library is docked
to every target. Does not mutate all-agricultural-dock.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metgeodock.io import (  # noqa: E402
    AMINO_ACIDS,
    METAL_RESNAMES,
    read_pdb,
    select_atoms,
    write_pdb,
)

LOG = logging.getLogger("build_agri_vs_library")
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"


def _get_json(url: str, retries: int = 3) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "MetGeoDock/agri-vs"})
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as handle:
                return json.loads(handle.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.2 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def n_heavy(smiles: str) -> int:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return int(mol.GetNumHeavyAtoms())


def chembl_smiles(name: str, fallback: dict[str, str]) -> str | None:
    if name in fallback:
        return fallback[name]
    q = urllib.parse.quote(name)
    url = f"{CHEMBL}/molecule/search.json?q={q}&limit=5"
    try:
        blob = _get_json(url)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("chembl name miss %s: %s", name, exc)
        return fallback.get(name)
    for mol in blob.get("molecules") or []:
        smi = ((mol.get("molecule_structures") or {}).get("canonical_smiles")) or ""
        if smi:
            return str(smi)
    return fallback.get(name)


def chembl_activities(target_id: str, min_p: float, require_p: bool, min_h: int, max_h: int, limit: int) -> list[dict]:
    offset = 0
    rows: list[dict] = []
    discarded_p = 0
    while offset < 1000:
        url = (
            f"{CHEMBL}/activity.json?target_chembl_id={target_id}"
            f"&pchembl_value__gte={min_p}&standard_relation=%3D&limit=50&offset={offset}"
        )
        blob = _get_json(url)
        acts = blob.get("activities") or []
        if not acts:
            break
        for rec in acts:
            if rec.get("data_validity_comment"):
                continue
            smi = rec.get("canonical_smiles") or ""
            if not smi:
                continue
            if require_p and "P" not in smi:
                discarded_p += 1
                continue
            nh = n_heavy(smi)
            if nh < min_h or nh > max_h:
                continue
            rows.append(
                {
                    "molecule_chembl_id": rec.get("molecule_chembl_id"),
                    "smiles": smi,
                    "pchembl": float(rec.get("pchembl_value")),
                    "standard_type": rec.get("standard_type"),
                    "target_chembl_id": target_id,
                }
            )
        offset += 50
        time.sleep(0.12)
        if not (blob.get("page_meta") or {}).get("next"):
            break
    LOG.info("%s fetched=%s discarded_noP=%s", target_id, len(rows), discarded_p)
    if not rows:
        return []
    frame = pd.DataFrame(rows).sort_values("pchembl", ascending=False)
    frame = frame.drop_duplicates("molecule_chembl_id", keep="first")
    return frame.head(int(limit)).to_dict(orient="records")


def prepare_pocket(pdb_path: Path, metal: str, radius: float, dest: Path) -> dict:
    """Atom-wise crop; protein + metals only (no pocket HOH)."""
    import numpy as np

    prot = read_pdb(pdb_path)
    idx = prot.metal_indices(metal=metal)
    if idx.size == 0:
        idx = prot.metal_indices()
    if idx.size == 0:
        raise ValueError(f"no metal in {pdb_path}")
    origin = prot.xyz[int(idx[0])]
    dist = np.linalg.norm(prot.xyz - origin, axis=1)
    keep = []
    for i, di in enumerate(dist):
        if di > float(radius):
            continue
        res = str(prot.resnames[i]).strip().upper()
        if res in AMINO_ACIDS or res in METAL_RESNAMES:
            keep.append(i)
    pocket = select_atoms(prot, keep, name=f"{pdb_path.stem}_pocket")
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_pdb(pocket, dest)
    return {
        "n_atoms": int(len(pocket)),
        "n_metal": int(pocket.metal_indices().size),
        "pdb": str(pdb_path),
        "pocket": str(dest),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/agri_vs.yaml"))
    p.add_argument("--out-dir", type=Path, default=Path("processed/agri_vs"))
    p.add_argument("--log", type=Path, default=Path("logs/build_agri_vs_library.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    agri = Path(cfg["agri_root"])
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    min_h = int(cfg.get("min_heavy_atoms", 10))
    max_h = int(cfg.get("max_heavy_atoms", 45))
    cap = int(cfg.get("max_chembl_actives", 80))
    min_p = float(cfg.get("pchembl_active", 6.0))
    fallback = dict(cfg.get("fallback_smiles") or {})

    by_tid: dict[str, list[dict]] = {tid: [] for tid in (cfg.get("targets") or {})}
    named_miss: list[str] = []
    panel_rows: list[dict] = []
    panel = cfg.get("commercial_panel") or {}
    hrac_to_target = {"F2_hppd": "hppd", "H_gs": "gs"}
    for group, names in panel.items():
        tgt = hrac_to_target.get(str(group))
        for name in names:
            smi = chembl_smiles(str(name), fallback)
            time.sleep(0.06)
            if not smi or not (min_h <= n_heavy(smi) <= max_h):
                named_miss.append(str(name))
                LOG.warning("panel miss %s", name)
                continue
            rec = {
                "ligand_id": str(name).replace(" ", "_"),
                "name": str(name),
                "smiles": smi,
                "source": f"hrac_{group}",
                "pchembl": float("nan"),
            }
            panel_rows.append(rec)
            if tgt:
                rec2 = dict(rec)
                rec2["source"] = f"hrac_active_{tgt}"
                by_tid[tgt].append(rec2)
            LOG.info("panel %s group=%s", name, group)

    pockets = {}
    chembl_meta = {}
    for tid, spec in (cfg.get("targets") or {}).items():
        tdir = out / tid
        tdir.mkdir(parents=True, exist_ok=True)
        pockets[tid] = prepare_pocket(
            agri / spec["pdb"], spec["metal"], float(cfg.get("pocket_A", 15.0)), tdir / "protein_pocket.pdb"
        )
        LOG.info("%s pocket %s", tid, pockets[tid])
        ids = list(spec.get("chembl_target_ids") or [])
        require_p = bool(spec.get("require_phosphorus"))
        fetched: list[dict] = []
        for cid in ids:
            try:
                fetched.extend(chembl_activities(cid, min_p, require_p, min_h, max_h, cap))
            except Exception as exc:  # noqa: BLE001
                LOG.warning("chembl %s: %s", cid, exc)
        if fetched:
            frame = pd.DataFrame(fetched).sort_values("pchembl", ascending=False).drop_duplicates("smiles")
            frame = frame.head(cap)
            for rec in frame.to_dict(orient="records"):
                by_tid[tid].append(
                    {
                        "ligand_id": rec["molecule_chembl_id"],
                        "name": rec["molecule_chembl_id"],
                        "smiles": rec["smiles"],
                        "source": f"chembl_{rec['target_chembl_id']}",
                        "pchembl": rec["pchembl"],
                    }
                )
        for name in spec.get("named_actives") or []:
            smi = chembl_smiles(str(name), fallback)
            time.sleep(0.06)
            if not smi:
                named_miss.append(str(name))
                continue
            if not (min_h <= n_heavy(smi) <= max_h):
                continue
            by_tid[tid].append(
                {
                    "ligand_id": str(name).replace(" ", "_"),
                    "name": str(name),
                    "smiles": smi,
                    "source": "named_active",
                    "pchembl": float("nan"),
                }
            )
        chembl_meta[tid] = {"chembl_target_ids": ids, "n_active_rows": len(by_tid[tid])}

    # Unique SMILES union.
    smiles_map: dict[str, dict] = {}
    for rec in panel_rows:
        smiles_map.setdefault(rec["smiles"], dict(rec))
    labels = {tid: set() for tid in by_tid}
    pchembl = {tid: {} for tid in by_tid}
    for tid, recs in by_tid.items():
        for rec in recs:
            smi = rec["smiles"]
            labels[tid].add(smi)
            if smi not in smiles_map:
                smiles_map[smi] = dict(rec)
            val = rec.get("pchembl")
            if val is not None and val == val:
                prev = pchembl[tid].get(smi)
                pchembl[tid][smi] = max(float(val), float(prev)) if prev is not None else float(val)

    rows = []
    for smi, rec in smiles_map.items():
        row = {
            "ligand_id": rec["ligand_id"],
            "name": rec["name"],
            "smiles": smi,
            "source": rec["source"],
            "n_heavy": n_heavy(smi),
        }
        for tid, spec in (cfg.get("targets") or {}).items():
            lab = int(smi in labels[tid])
            row[f"label_{tid}"] = lab
            row[f"pchembl_{tid}"] = pchembl[tid].get(smi, float("nan"))
            row["metal"] = spec["metal"]
        rows.append(row)
    uni = pd.DataFrame(rows)
    uni.to_csv(out / "library_all.csv", index=False)
    for tid, spec in (cfg.get("targets") or {}).items():
        sub = uni.copy()
        sub["target"] = tid
        sub["metal"] = spec["metal"]
        sub["label"] = sub[f"label_{tid}"].astype(int)
        sub["pchembl"] = sub[f"pchembl_{tid}"]
        sub.to_csv(out / tid / "library.csv", index=False)
        LOG.info("%s n=%s n_act=%s", tid, len(sub), int(sub["label"].sum()))

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": (
            "Unified library: ChEMBL pChEMBL>=6 (unique SMILES, 10-45 heavy atoms, "
            "optional phosphorus filter) UNION HRAC/IRAC commercial panel. "
            "Same molecules docked to every target. Decoys = non-actives in this library "
            "(cross-target + other MoA). No post-docking exclusions."
        ),
        "n_unique": int(len(uni)),
        "n_named_miss": named_miss,
        "targets": {
            tid: {
                **spec,
                "pocket": pockets[tid],
                "n_actives": int((uni[f"label_{tid}"] == 1).sum()) if len(uni) else 0,
                "n_decoys": int((uni[f"label_{tid}"] == 0).sum()) if len(uni) else 0,
                **chembl_meta.get(tid, {}),
            }
            for tid, spec in (cfg.get("targets") or {}).items()
        },
        "filters": {"min_heavy_atoms": min_h, "max_heavy_atoms": max_h, "max_chembl_actives": cap, "pchembl_active": min_p},
    }
    (out / "build_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("created_utc", "n_unique", "n_named_miss", "filters")}, indent=2))
    print(json.dumps({tid: {"n_actives": t["n_actives"], "n_decoys": t["n_decoys"]} for tid, t in summary["targets"].items()}, indent=2))


if __name__ == "__main__":
    main()
