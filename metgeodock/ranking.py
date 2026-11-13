"""Rank docking / screening hits and optionally write pose files.

Default: emit **all** scored molecules, sorted best-to-worst (lower score
is better). ``top_n`` of 100 or 1000 truncates; ``top_n <= 0`` keeps all.
Pose coordinates are written only when ``write_poses`` is True.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import write_pdb

LEADING_COLS = ("rank", "ligand_id", "score")


def take_top(frame: pd.DataFrame, top_n: int = 0, score_col: str = "score") -> pd.DataFrame:
    """Sort by score (lower = better binding) and optionally keep the top ``n``.

    ``top_n <= 0`` (default) keeps the full ranked table: every ligand that
    was scored, best first, worst last. Failed rows (NaN score) go last.
    """
    if frame.empty:
        return frame.reset_index(drop=True)
    if "rank" in frame.columns:
        frame = frame.drop(columns=["rank"])
    ranked = frame.sort_values(score_col, kind="mergesort", na_position="last")
    ranked = ranked.reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    if top_n and top_n > 0:
        ranked = ranked.iloc[: int(top_n)].copy()
    front = [c for c in LEADING_COLS if c in ranked.columns]
    rest = [c for c in ranked.columns if c not in front]
    return ranked.loc[:, front + rest].reset_index(drop=True)


def write_rank_table(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
    else:
        frame.to_csv(path, index=False)
    return path


def write_pose_files(rows: list[dict], out_dir: str | Path) -> list[str]:
    """Write PDB files for rows that carry a ``ligand`` object."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for row in rows:
        lig = row.get("ligand")
        if lig is None:
            continue
        rank = int(row.get("rank", len(paths) + 1))
        dest = out / f"pose_{rank:04d}.pdb"
        write_pdb(lig, dest)
        paths.append(str(dest))
    return paths
