#!/usr/bin/env python3
"""Parse MetalloDock SI DOCX for a 541-PDB test list and dataset protocol.

SI (ja5c15876_si_001.docx) is free ACS supporting text. This script:

* inventories ZIP members (xml / emf / embeddings)
* extracts Word text + table cells
* regex-scans for PDB-like tokens
* writes processed/metallodock_ref/dataset/test_541.csv if a credible list
  appears, otherwise writes a protocol-only report.

Does not mutate data/ or ORI_PDB_LIGAND.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
PDB_RE = re.compile(r"\b([0-9][A-Za-z0-9]{3})\b")
# tokens that look like PDB but are SI numbers / figure codes
FALSE_PDB = {
    "8836",
    "7453",
    "3D7H",
    "5VPT",
    "2022",
    "2023",
    "2024",
    "2025",
    "2026",
    "3086",
    "3101",
}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _texts(root: ET.Element) -> list[str]:
    out: list[str] = []
    for node in root.iter(f"{{{W_NS}}}t"):
        if node.text:
            out.append(node.text)
        if node.tail:
            out.append(node.tail)
    return out


def _tables(root: ET.Element) -> list[list[list[str]]]:
    tables = []
    for tbl in root.iter(f"{{{W_NS}}}tbl"):
        rows = []
        for tr in tbl.iter(f"{{{W_NS}}}tr"):
            cells = []
            for tc in tr.findall(f"{{{W_NS}}}tc"):
                cells.append(" ".join(_texts(tc)).strip())
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def inspect_docx(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        ext = Counter(Path(n).suffix.lower() or "(none)" for n in names)
        xml_names = [n for n in names if n.endswith(".xml")]
        blobs: list[str] = []
        table_preview: list[dict] = []
        n_tables = 0
        for xml_name in xml_names:
            if "media" in xml_name:
                continue
            raw = zf.read(xml_name)
            try:
                root = ET.fromstring(raw)
            except ET.ParseError:
                continue
            texts = _texts(root)
            if texts:
                blobs.append(" ".join(texts))
            tables = _tables(root)
            n_tables += len(tables)
            for i, rows in enumerate(tables, start=1):
                n_rows = len(rows)
                n_cols = max(len(r) for r in rows)
                head = [" | ".join(c[:80] for c in r) for r in rows[:4]]
                table_preview.append(
                    {
                        "xml": xml_name,
                        "table_index": i,
                        "n_rows": n_rows,
                        "n_cols": n_cols,
                        "head": head,
                    }
                )
        embed = [n for n in names if n.startswith("word/embeddings/") or n.startswith("word/oleObject")]
        media = [n for n in names if n.startswith("word/media/")]
    blob = "\n".join(blobs)
    raw_ids = sorted({m.group(1).upper() for m in PDB_RE.finditer(blob)})
    ids = [x for x in raw_ids if x not in FALSE_PDB and not x.isdigit()]
    protocol_hits = []
    for line in re.split(r"[\n.]", blob):
        low = line.lower()
        if any(k in low for k in ("time-split", "time split", "temporal", "2022", "541", "train", "test set")):
            s = " ".join(line.split())
            if s:
                protocol_hits.append(s[:400])
    return {
        "si_path": str(path),
        "si_bytes": path.stat().st_size,
        "n_zip_members": len(names),
        "zip_ext_counts": dict(ext),
        "n_xml": len(xml_names),
        "n_media": len(media),
        "n_embeddings": len(embed),
        "embeddings": embed,
        "n_tables": n_tables,
        "table_preview": table_preview,
        "n_text_chars": len(blob),
        "text_head": blob[:2500],
        "protocol_lines": protocol_hits[:40],
        "pdb_like_raw": raw_ids,
        "pdb_like_filtered": ids,
        "n_pdb_like_raw": len(raw_ids),
        "n_pdb_like_filtered": len(ids),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--si", type=Path, default=Path("processed/metallodock_ref/si/ja5c15876_si_001.docx"))
    p.add_argument("--out", type=Path, default=Path("processed/metallodock_ref/dataset/test_541.csv"))
    p.add_argument("--report", type=Path, default=Path("processed/metallodock_ref/si/si_parse_report.json"))
    args = p.parse_args()
    if not args.si.exists():
        raise SystemExit(f"SI file not found: {args.si}")
    info = inspect_docx(args.si)
    info["created_utc"] = datetime.now(timezone.utc).isoformat()
    ids = info["pdb_like_filtered"]
    # A real 541-list would be hundreds of IDs, not a handful of false hits.
    credible = len(ids) >= 400
    info["credible_541_list"] = bool(credible)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(info, indent=2), encoding="utf-8")
    if credible:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"pdb_id": ids, "source": "metallodock_si"}).to_csv(args.out, index=False)
        info["out"] = str(args.out)
        info["n_pdb"] = len(ids)
    else:
        info["out"] = None
        info["n_pdb"] = 0
        info["note"] = (
            "SI tables do not enumerate the 541 test PDB IDs; "
            "protocol uses year 2022 as the temporal boundary."
        )
    args.report.write_text(json.dumps(info, indent=2), encoding="utf-8")
    slim = {
        "si_path": info["si_path"],
        "si_bytes": info["si_bytes"],
        "n_tables": info["n_tables"],
        "n_embeddings": info["n_embeddings"],
        "n_pdb_like_raw": info["n_pdb_like_raw"],
        "n_pdb_like_filtered": info["n_pdb_like_filtered"],
        "pdb_like_filtered": info["pdb_like_filtered"],
        "credible_541_list": info["credible_541_list"],
        "protocol_lines": info["protocol_lines"],
        "table_shapes": [
            {"n_rows": t["n_rows"], "n_cols": t["n_cols"], "head0": (t["head"][0] if t["head"] else "")}
            for t in info["table_preview"]
        ],
        "out": info.get("out"),
        "report": str(args.report),
    }
    print(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
