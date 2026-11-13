#!/usr/bin/env bash
# Build a Windows-ready copy of the project (no symlinks, no venv, no scratch files).
#
#   bash scripts/package_for_windows.sh /path/to/output_dir
#
# What is copied:
#   - the installable package and its metadata (metgeodock, pyproject, README, LICENSE)
#   - all reproduction scripts, tests, configs, examples
#   - manuscript sources, rebuilt docx, figures, Chinese report
#   - result tables the manuscript cites + the curated ChEMBL table the scripts read
# What is NOT copied:
#   - envs/ (3.8 GB venv), ORI_PDB_LIGAND/ (3.9 GB), OUTPUT_DOCK/ (2.4 GB)
#   - docking/biglib*/ (184 MB of decoy SDF), logs/, build/, *.egg-info/
#   - processed/bindingdb (33 MB) and processed/metallodock_ref (61 MB)
#
# Symlinked directories are dereferenced only where they hold data the scripts need.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="${1:-$SRC/../metgeodock-windows}"

echo "source: $SRC"
echo "dest  : $DST"
rm -rf "$DST"
mkdir -p "$DST"

copy() {  # copy() <relative path>
  local rel="$1"
  if [ -e "$SRC/$rel" ]; then
    mkdir -p "$DST/$(dirname "$rel")"
    rsync -a --no-links --exclude '__pycache__' "$SRC/$rel" "$DST/$(dirname "$rel")/"
    echo "  + $rel"
  fi
}

echo "[1/6] package + metadata"
for p in metgeodock pyproject.toml README.md LICENSE DATA.md; do copy "$p"; done

echo "[2/6] code"
for p in scripts tests configs examples; do copy "$p"; done

echo "[3/6] manuscript + reports + figures + submission package"
for p in manuscript reports figures dataset_release; do copy "$p"; done

echo "[4/6] result tables cited by the paper"
copy results

echo "[5/6] curated inputs the scripts read"
for p in processed/chembl/curated processed/maps processed/ff processed/crystal \
         processed/splits processed/agri_vs processed/gpdock_orig processed/ml_runs; do
  copy "$p"
done

echo "[6/7] submission assets (TOC, graphical abstract, cover letters)"
mkdir -p "$DST/submission"
for f in figures/TOC_jacs.png figures/TOC_jacs.pdf \
         figures/graphical_abstract_jacs.png figures/graphical_abstract_jacs.pdf \
         manuscript/cover_letters/cover_letter_jacs.docx \
         manuscript/cover_letters/cover_letter_jacs.txt \
         manuscript/cover_letters/cover_letter_jctc.docx \
         manuscript/cover_letters/cover_letter_jctc.txt; do
  if [ -f "$SRC/$f" ]; then
    cp "$SRC/$f" "$DST/submission/$(basename "$f")"
    echo "  + $f"
  fi
done
# 平铺一份，避免子目录混乱
cp "$SRC/figures/TOC_jacs.png" "$DST/submission/TOC.png" 2>/dev/null || true
cp "$SRC/figures/graphical_abstract_jacs.png" "$DST/submission/graphical_abstract.png" 2>/dev/null || true

echo "[7/7] sanity"
# anything still a symlink?
LINKS=$(find "$DST" -type l 2>/dev/null | head -5 || true)
if [ -n "$LINKS" ]; then
  echo "  ! residual symlinks (will break on Windows):"
  echo "$LINKS"
else
  echo "  no symlinks"
fi
# line endings: make shell/py scripts LF and add a .gitattributes hint
cat > "$DST/.gitattributes" <<'EOF'
* text=auto eol=lf
*.png binary
*.pdf binary
*.docx binary
*.npz binary
*.joblib binary
EOF
echo "  wrote .gitattributes (forces LF for text files)"

du -sh "$DST" | awk '{print "total size: "$1}'
echo
echo "Copy this folder to Windows, then:"
echo "  python -m venv .venv && .venv\\Scripts\\activate"
echo "  pip install -e \".[chem]\""
echo "  metgeodock --version"
