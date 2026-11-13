#!/usr/bin/env python3
"""Write interpreter + package versions used for agri VS / all-test summary."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _ver(mod: str) -> str | None:
    try:
        m = __import__(mod)
        return str(getattr(m, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("logs/agri_vs_env.txt"))
    args = p.parse_args()
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False) if __import__("shutil").which("git") else None
    blob = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git": git.stdout.strip() if git is not None and git.returncode == 0 else "",
        "numpy": _ver("numpy"),
        "pandas": _ver("pandas"),
        "rdkit": _ver("rdkit"),
        "torch": _ver("torch"),
        "yaml": _ver("yaml"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(blob, indent=2))


if __name__ == "__main__":
    main()
