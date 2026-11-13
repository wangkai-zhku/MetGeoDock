#!/usr/bin/env python3
"""Rebuild MetalloDock-style time-split test complexes from RCSB.

SI (ja5c15876_si_001.docx) states the time-split uses year 2022 as the
temporal boundary but does not tabulate the 541 PDB IDs. This script:

1. Searches RCSB for X-ray entries released >= 2022-01-01 containing
   Zn/Mg/Mn/Fe/Co/Ni/Ca (7 metals in the paper).
2. Downloads PDBs into processed/metallodock_541/raw (not data/).
3. Extracts metal + coordinating organic ligand as protein/ligand PDBs.

Does not mutate ORI_PDB_LIGAND or data/.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from metgeodock.io import METAL_RESNAMES, Molecule, read_pdb, write_pdb
from metgeodock.metals import metal_from_name

LOG = logging.getLogger("build_metallodock_2022_set")
SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
FILES = "https://files.rcsb.org/download"
METAL_COMPS = ("ZN", "MG", "MN", "FE", "FE2", "CO", "NI", "CA")
ADDITIVES = {
    "HOH", "DOD", "SO4", "GOL", "EDO", "PEG", "PGE", "PG4", "1PE", "MPD",
    "DMS", "ACT", "FMT", "MES", "TRS", "EPE", "CIT", "TLA", "MLI", "IMD", "BME",
    "NA", "K", "CL", "BR", "IOD", "NO3", "AZI", "CO3", "NH4", "ACY", "ACE",
    "NAG", "BGC", "MAN", "GLC", "FUC", "GAL", "NDG", "BMA", "UNX", "UNL",
}
NUCLEIC = {
    "DA", "DC", "DG", "DT", "DU", "A", "C", "G", "U", "I", "DI",
    "5MC", "5MU", "PSU", "H2U", "OMC", "OMG", "1MA",
}


def _post(payload: dict) -> dict:
    req = Request(
        SEARCH,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "MetGeoDock/0.1"},
        method="POST",
    )
    with urlopen(req, timeout=120) as resp:
        raw = resp.read()
        if not raw:
            return {"total_count": 0, "result_set": []}
        return json.loads(raw.decode("utf-8"))


def _get_bytes(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "MetGeoDock/0.1"})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def download_missing(ids: list[str], raw_dir: Path, workers: int = 12) -> None:
    missing = [pdb for pdb in ids if not (raw_dir / f"{pdb.lower()}.pdb").exists()]
    LOG.info("download missing %s workers=%s", len(missing), workers)
    if not missing:
        return

    def _one(pdb: str) -> tuple[str, str | None]:
        dest = raw_dir / f"{pdb.lower()}.pdb"
        try:
            dest.write_bytes(_get_bytes(f"{FILES}/{pdb}.pdb"))
            return pdb, None
        except Exception as exc:  # noqa: BLE001
            return pdb, str(exc)

    n_ok = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(_one, pdb) for pdb in missing]
        for i, fut in enumerate(as_completed(futs), start=1):
            pdb, err = fut.result()
            if err:
                LOG.warning("download fail %s %s", pdb, err)
            else:
                n_ok += 1
            if i % 100 == 0:
                LOG.info("downloaded %s/%s ok=%s", i, len(missing), n_ok)
    LOG.info("download done ok=%s / %s", n_ok, len(missing))


def search_ids(min_date: str, max_date: str, resolution: float) -> list[str]:
    metal_nodes = [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id",
                "operator": "exact_match",
                "value": m,
            },
        }
        for m in METAL_COMPS
    ]
    query = {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {"type": "group", "logical_operator": "or", "nodes": metal_nodes},
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_accession_info.initial_release_date",
                    "operator": "range",
                    "value": {"from": min_date, "to": max_date, "include_lower": True, "include_upper": True},
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {"attribute": "exptl.method", "operator": "exact_match", "value": "X-RAY DIFFRACTION"},
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal",
                    "value": resolution,
                },
            },
        ],
    }
    ids: list[str] = []
    start = 0
    rows = 10000
    while True:
        payload = {
            "query": query,
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": start, "rows": rows},
                "results_content_type": ["experimental"],
            },
        }
        blob = _post(payload)
        chunk = [str(x["identifier"]).upper() for x in blob.get("result_set") or []]
        ids.extend(chunk)
        total = int(blob.get("total_count") or 0)
        LOG.info("RCSB search %s-%s / %s", start + 1, start + len(chunk), total)
        if not chunk or start + len(chunk) >= total:
            break
        start += len(chunk)
        time.sleep(0.3)
    return sorted(set(ids))


def _residue_key(mol: Molecule, i: int) -> tuple[str, int]:
    return str(mol.resnames[i]).strip().upper(), int(mol.resseqs[i])


def extract_sites(mol: Molecule, max_d: float = 2.90) -> list[dict]:
    metal_idx = [
        i
        for i, res in enumerate(mol.resnames)
        if str(res).strip().upper() in METAL_RESNAMES
    ]
    sites = []
    for mi in metal_idx:
        mxyz = mol.xyz[mi]
        d = np.linalg.norm(mol.xyz - mxyz, axis=1)
        close = np.where((d > 1.0e-6) & (d <= max_d))[0]
        lig_res: dict[tuple[str, int], list[int]] = {}
        n_pro = 0
        for j in close:
            resn = str(mol.resnames[j]).strip().upper()
            el = str(mol.elements[j]).upper()[:1]
            if resn in METAL_RESNAMES or resn in ADDITIVES:
                continue
            if el not in {"O", "N", "S"}:
                continue
            # protein residue vs ligand: amino acids counted as protein
            from metgeodock.io import AMINO_ACIDS

            if resn in AMINO_ACIDS:
                n_pro += 1
                continue
            if resn in NUCLEIC:
                continue
            lig_res.setdefault(_residue_key(mol, j), []).append(int(j))
        if n_pro < 2 or not lig_res:
            continue
        # largest coordinating ligand residue
        key = max(lig_res, key=lambda k: len(lig_res[k]))
        if key[0] in NUCLEIC or key[0] in ADDITIVES:
            continue
        lig_atoms = [
            i
            for i in range(len(mol))
            if (str(mol.resnames[i]).strip().upper(), int(mol.resseqs[i])) == key
        ]
        n_heavy = int(sum(str(mol.elements[i]).upper() != "H" for i in lig_atoms))
        if n_heavy < 6 or n_heavy > 80:
            continue
        try:
            metal_name = metal_from_name(str(mol.resnames[mi]))
        except Exception:  # noqa: BLE001
            continue
        sites.append(
            {
                "metal_index": int(mi),
                "metal": metal_name,
                "ligand_resname": key[0],
                "ligand_resseq": key[1],
                "ligand_atoms": lig_atoms,
                "n_protein_coord": int(n_pro),
                "n_heavy": n_heavy,
            }
        )
    return sites


def subset(mol: Molecule, indices: list[int], name: str) -> Molecule:
    idx = np.asarray(indices, dtype=int)
    return Molecule(
        name=name,
        xyz=mol.xyz[idx].copy(),
        elements=mol.elements[idx].copy(),
        names=mol.names[idx].copy(),
        resnames=mol.resnames[idx].copy(),
        resseqs=mol.resseqs[idx].copy(),
        charges=mol.charges[idx].copy(),
        atom_types=mol.atom_types[idx].copy(),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-date", default="2022-01-01")
    p.add_argument("--max-date", default="2024-12-31")
    p.add_argument("--resolution", type=float, default=2.5)
    p.add_argument("--out-dir", type=Path, default=Path("processed/metallodock_541"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--reuse-ids", action="store_true")
    p.add_argument("--log", type=Path, default=Path("logs/build_metallodock_2022_set.log"))
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )
    raw_dir = args.out_dir / "raw"
    cx_dir = args.out_dir / "complexes"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cx_dir.mkdir(parents=True, exist_ok=True)
    ids_path = args.out_dir / "rcsb_ids.json"
    if args.reuse_ids and ids_path.exists():
        ids = [str(x).upper() for x in json.loads(ids_path.read_text(encoding="utf-8"))["ids"]]
        LOG.info("reuse %s ids from %s", len(ids), ids_path)
    else:
        ids = search_ids(args.min_date, args.max_date, args.resolution)
    if args.limit:
        ids = ids[: args.limit]
    LOG.info("candidate pdb %s", len(ids))
    ids_path.write_text(
        json.dumps({"n": len(ids), "ids": ids, "min_date": args.min_date, "max_date": args.max_date}, indent=2),
        encoding="utf-8",
    )
    download_missing(ids, raw_dir, workers=args.workers)
    rows = []
    for k, pdb in enumerate(ids, start=1):
        pdb_path = raw_dir / f"{pdb.lower()}.pdb"
        if not pdb_path.exists():
            continue
        try:
            mol = read_pdb(pdb_path)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("parse fail %s %s", pdb, exc)
            continue
        sites = extract_sites(mol)
        if k % 50 == 0:
            LOG.info("[%s/%s] %s sites=%s kept=%s", k, len(ids), pdb, len(sites), len(rows))
        for si, site in enumerate(sites, start=1):
            rec_id = f"{pdb.lower()}_{site['metal']}_{si}"
            dest = cx_dir / rec_id
            dest.mkdir(parents=True, exist_ok=True)
            lig_set = set(site["ligand_atoms"])
            pro_idx = [i for i in range(len(mol)) if i not in lig_set]
            protein = subset(mol, pro_idx, f"{rec_id}_pro")
            ligand = subset(mol, site["ligand_atoms"], f"{rec_id}_lig")
            pro_path = dest / f"{rec_id}_pro.pdb"
            lig_path = dest / f"{rec_id}_lig.pdb"
            write_pdb(protein, pro_path, record="ATOM")
            write_pdb(ligand, lig_path, record="HETATM")
            rows.append(
                {
                    "pdb_id": pdb,
                    "rec_id": rec_id,
                    "metal": site["metal"],
                    "ligand_resname": site["ligand_resname"],
                    "n_protein_coord": site["n_protein_coord"],
                    "n_heavy": site["n_heavy"],
                    "protein_path": str(pro_path),
                    "ligand_path": str(lig_path),
                    "source_pdb": str(pdb_path),
                }
            )
    inv = pd.DataFrame(rows)
    inv_path = args.out_dir / "inventory.csv"
    inv.to_csv(inv_path, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "SI: time-split uses year 2022 as temporal boundary; PDB IDs not tabulated in SI tables",
        "min_date": args.min_date,
        "max_date": args.max_date,
        "resolution_max": args.resolution,
        "n_rcsb_hits": len(ids),
        "n_complexes": int(len(inv)),
        "n_pdb": int(inv["pdb_id"].nunique()) if len(inv) else 0,
        "by_metal": inv["metal"].value_counts().to_dict() if len(inv) else {},
        "inventory": str(inv_path),
        "literature_n_test": 541,
    }
    (args.out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("summary %s", json.dumps(summary))


if __name__ == "__main__":
    main()
