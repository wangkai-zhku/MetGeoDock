#!/usr/bin/env bash
# Receptors for the HDAC subtype-selectivity validation (HDAC1/2/3/4/6/8).
#
# Same two policies as prepare_receptors10.sh:
#   1. oligomer trimmed to one copy (catalytic Zn located first, see below)
#   2. crystallographic waters removed, metal kept
#
# Catalytic site identification (first coordination shell, <=2.9 A):
#   5ICN HDAC1  Zn B:401   Asp176 Asp264 His178        keep B (A is a bound partner)
#   3MAX HDAC2  Zn A/B/C:379  Asp181 Asp269 His183     keep B (3 copies)
#   4A69 HDAC3  Zn A/B:500  Asp170 Asp259 His172       keep A (B is a copy; C/D are the DAD helix)
#   2VQM HDAC4  Zn A:1411   Asp196 Asp290 His198       keep A; Zn A:1412 is a Cys4 structural zinc,
#                                                      so detect_coordination must pick 1411
#   5EDU HDAC6  Zn A:901    Asp649 Asp742 His651       keep A (B is a copy; CD1 catalytic domain)
#   1T64 HDAC8  Zn A:388    Asp178 Asp267 His180       keep A (B is a copy; the Ca ions are free)
#
#   bash prepare_hdac_receptors.sh         # build missing
#   FORCE=1 bash prepare_hdac_receptors.sh # rebuild all
set -u
R=/home/wangkai/cursor-task/metalloprotein-chem
OUT=$R/processed/hdac_receptors
PY=$R/envs/metgeodock/bin/python
mkdir -p "$OUT"

declare -A KEEP=( [5ICN]=B [3MAX]=B [4A69]=A [2VQM]= [5EDU]=A [1T64]=A )
declare -A METAL=( [5ICN]=zn [3MAX]=zn [4A69]=zn [2VQM]=zn [5EDU]=zn [1T64]=zn )

for pid in 5ICN 3MAX 4A69 2VQM 5EDU 1T64; do
  raw=$OUT/$pid.pdb
  hol=$OUT/${pid}_h.pdb
  [ -s "$raw" ] || { echo "[$pid] 缺少原始结构 $raw"; continue; }
  if [ "${FORCE:-0}" = "1" ] || [ ! -s "$hol" ]; then
    echo "[$pid] preparing (chains='${KEEP[$pid]:-all}')"
    site=$OUT/${pid}_site.pdb
    "$PY" - "$raw" "$site" "${KEEP[$pid]:-}" <<'EOF' || { echo "[$pid] trim FAILED"; continue; }
import sys
from pathlib import Path
src, dst, keep = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
WATER = {"HOH", "WAT", "H2O", "DOD"}
METAL = {"ZN","ZN2","CA","CA2","MG","MG2","MN","MN2","MN3","FE","FE2","FE3",
         "CU","CU1","CU2","CO","CO2","CD","CD2","NI","NI2"}
out, kept, dropped = [], 0, 0
for ln in src.read_text(errors="ignore").splitlines():
    if not ln.startswith(("ATOM", "HETATM")):
        continue
    if keep and ln[21] not in keep:
        dropped += 1; continue
    if ln.startswith("HETATM"):
        rn = ln[17:20].strip().upper()
        if rn in WATER or rn not in METAL:
            continue
    out.append(ln); kept += 1
dst.write_text("\n".join(out) + "\nEND\n")
print(f"  site: {kept} atoms kept, {dropped} dropped from other chains")
EOF
    "$PY" -m metgeodock.cli prepare --pdb "$site" --out "$hol" \
        --metal "${METAL[$pid]}" --no-keep-water --report "$OUT/${pid}_prepare.json" \
        || { echo "[$pid] prepare FAILED"; continue; }
    rm -f "$site"
  else
    echo "[$pid] ${pid}_h.pdb present, skipped"
  fi
done

echo
echo "=== 受体金属位点复核 ==="
"$PY" - <<'EOF'
from pathlib import Path
import numpy as np
METAL={"ZN","ZN2","CA","CA2","MG","MG2","MN","FE","FE2","CU","CO","NI","CD"}
for pid in ("5ICN","3MAX","4A69","2VQM","5EDU","1T64"):
    f=Path(f"/home/wangkai/cursor-task/metalloprotein-chem/processed/hdac_receptors/{pid}_h.pdb")
    if not f.exists(): continue
    prot=[];met=[]
    for ln in f.read_text(errors="ignore").splitlines():
        if not ln.startswith(("ATOM","HETATM")): continue
        rn=ln[17:20].strip().upper()
        try: xyz=np.array([float(ln[30:38]),float(ln[38:46]),float(ln[46:54])])
        except ValueError: continue
        if ln.startswith("ATOM"): prot.append((ln[21],rn,int(ln[22:26]),xyz))
        elif rn in METAL: met.append((rn,ln[21],int(ln[22:26]),xyz))
    n_zn=sum(1 for m in met if m[0].startswith("ZN"))
    first=[m for m in met if m[0].startswith("ZN")][:1]
    desc="-"
    if first:
        _,ch,ri,xyz=first[0]
        near=sorted({f"{n}{i}" for c,n,i,x in prot if np.linalg.norm(x-xyz)<=2.9})
        desc=f"Zn {ch}:{ri} -> {', '.join(near)}"
    print(f"  {pid}: 原子={len(prot)} Zn={n_zn} 金属总={len(met)} | {desc}")
EOF
