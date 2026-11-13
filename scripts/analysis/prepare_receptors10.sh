#!/usr/bin/env bash
# Build (once) the docking receptors for the cross-family-10 benchmark.
#
# Receptors live in processed/receptors10/ and are NOT rebuilt on every run:
#   <PDB>.pdb     raw RCSB entry
#   <PDB>_h.pdb   protonated, single biological unit, crystallographic waters REMOVED
#
# Two policies are applied here:
#
# 1. Oligomer trimming. Structures solved with several copies of the same chain
#    are reduced to one copy; hetero-oligomers keep the whole functional unit.
#    Every metal's first coordination shell sits inside a single chain (checked),
#    so trimming does not break a site.
#       ARG2   1PQ3  A-F identical (24-329)            -> A
#       CAPN1  2ARY  A/B identical (33-354)            -> A
#       HDAC2  3MAX  A/B/C copies (12-378)             -> B (4 coord vs 3 in A)
#       ALOX5  3O8Y  two copies                        -> A
#       DBH    4ZEL  two copies                        -> A
#       HDAC1  5ICN  A(165-340) binder, B(8-376) enzyme -> B
#       urease 1FWJ  A(1-100) B(1-101) C(2-567) = alpha/beta/gamma unit -> all
#       MMP2/CA2/CDK2/METAP2: single chain, untouched
#
# 2. Water removal. Pocket waters are dropped for every method. MetalGeoDock
#    already docks water-free (dock_ligand(keep_water=False) -> strip_waters);
#    Vina's obabel pdbqt conversion writes every HETATM O as an OA acceptor and
#    would otherwise block the site (see scripts/analysis/vina_baseline.py).
#
# Idempotent: existing *_h.pdb are kept. Use FORCE=1 to rebuild everything.
#
#   bash prepare_receptors10.sh
#   FORCE=1 bash prepare_receptors10.sh
set -u
R=/home/wangkai/cursor-task/metalloprotein-chem
OUT=$R/processed/receptors10
PY=$R/envs/metgeodock/bin/python
mkdir -p "$OUT"

declare -A PDB=( [HDAC1]=5ICN [HDAC2]=3MAX [CA2]=3CAJ [MMP2]=1QIB [ALOX5]=3O8Y
                 [ARG2]=1PQ3 [CAPN1]=2ARY [DBH]=4ZEL [CDK2]=1HCK [METAP2]=1B6A [urease]=1FWJ )
declare -A MET=( [HDAC1]=zn [HDAC2]=zn [CA2]=zn [MMP2]=zn [ALOX5]=fe2
                 [ARG2]=mn [CAPN1]=ca [DBH]=cu [CDK2]=mg [METAP2]=co [urease]=ni )
# chains to keep; empty = keep all (single chain or hetero-oligomer unit)
declare -A KEEP=( [HDAC1]=B [HDAC2]=B [ALOX5]=A [ARG2]=A [CAPN1]=A [DBH]=A
                  [CA2]= [MMP2]= [CDK2]= [METAP2]= [urease]=ABC )

for t in HDAC1 HDAC2 CA2 MMP2 ALOX5 ARG2 CAPN1 DBH CDK2 METAP2 urease; do
  pdb=${PDB[$t]}
  raw=$OUT/$pdb.pdb
  hol=$OUT/${pdb}_h.pdb

  if [ ! -s "$raw" ]; then
    echo "[$t] downloading $pdb"
    "$PY" - "$pdb" "$raw" <<'EOF' || { echo "[$t] download FAILED"; continue; }
import sys, urllib.request
pid, dest = sys.argv[1], sys.argv[2]
d = urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.pdb", timeout=120).read()
open(dest, "wb").write(d)
print(f"  {pid}: {len(d)} bytes")
EOF
  fi

  if [ "${FORCE:-0}" = "1" ] || [ ! -s "$hol" ]; then
    echo "[$t] preparing $pdb (chains='${KEEP[$t]:-all}')"
    site=$OUT/${pdb}_site.pdb
    "$PY" - "$raw" "$site" "${KEEP[$t]:-}" <<'EOF' || { echo "[$t] trim FAILED"; continue; }
import sys
from pathlib import Path
src, dst, keep = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
WATER = {"HOH", "WAT", "H2O", "DOD"}
METAL = {"ZN", "ZN2", "CA", "CA2", "C0", "MG", "MG2", "MN", "MN2", "MN3",
         "FE", "FE2", "FE3", "CU", "CU1", "CU2", "CO", "CO2", "CD", "CD2",
         "CAD", "NI", "NI2"}
out, kept, dropped = [], 0, 0
for ln in src.read_text(errors="ignore").splitlines():
    if not ln.startswith(("ATOM", "HETATM")):
        continue
    if keep and ln[21] not in keep:
        dropped += 1
        continue
    if ln.startswith("HETATM"):
        resn = ln[17:20].strip().upper()
        if resn in WATER or resn not in METAL:
            continue
    out.append(ln)
    kept += 1
dst.write_text("\n".join(out) + "\nEND\n")
print(f"  site: {kept} atoms kept, {dropped} dropped from other chains")
EOF
    "$PY" -m metgeodock.cli prepare --pdb "$site" --out "$hol" \
        --metal "${MET[$t]}" --no-keep-water --report "$OUT/${pdb}_prepare.json" \
        || { echo "[$t] prepare FAILED"; continue; }
    rm -f "$site"
  else
    echo "[$t] ${pdb}_h.pdb present, skipped"
  fi
done

echo
echo "receptors in $OUT:"
ls -la "$OUT"/*_h.pdb
