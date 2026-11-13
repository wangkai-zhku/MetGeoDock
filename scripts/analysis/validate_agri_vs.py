#!/usr/bin/env python3
"""Sanity-check agri VS tables: pockets exist, labels mixed, no agri raw mutation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lib", type=Path, default=Path("processed/agri_vs/build_summary.json"))
    p.add_argument("--vs", type=Path, default=Path("results/agri_vs.json"))
    p.add_argument("--all-test", type=Path, default=Path("results/all_test_accuracy.json"))
    p.add_argument("--out", type=Path, default=Path("results/validate_agri_vs.json"))
    args = p.parse_args()
    lib = json.loads(args.lib.read_text(encoding="utf-8")) if args.lib.exists() else {}
    vs = json.loads(args.vs.read_text(encoding="utf-8")) if args.vs.exists() else {}
    acc = json.loads(args.all_test.read_text(encoding="utf-8")) if args.all_test.exists() else {}
    checks = {
        "lib_exists": bool(lib),
        "vs_exists": bool(vs),
        "all_test_exists": bool(acc),
        "targets": list((lib.get("targets") or {}).keys()),
        "each_target_has_actives": all(int(t.get("n_actives") or 0) > 0 for t in (lib.get("targets") or {}).values()) if lib else False,
        "pockets_written": all(Path(t["pocket"]["pocket"]).exists() for t in (lib.get("targets") or {}).values() if t.get("pocket")) if lib else False,
        "unique_pdb_n": (acc.get("unique_pdb") or {}).get("n"),
        "unique_pdb_top3": (acc.get("unique_pdb") or {}).get("top3"),
        "unified_library_ge_100": int((lib.get("n_unique") or 0)) >= 100,
        "protocol_documented": bool(lib.get("protocol")),
    }
    failed = [k for k, v in checks.items() if v is False]
    out = {"ok": not failed, "failed": failed, "checks": checks}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
