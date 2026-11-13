#!/usr/bin/env bash
# Wait for all entropy_all_targets jobs, then run the ZBG calibration and build the
# Chinese Word report. Safe to run repeatedly.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=envs/metgeodock/bin/python

echo "[$(date +%H:%M:%S)] waiting for entropy_all_targets jobs..."
while pgrep -f "entropy_all_targets.py" > /dev/null 2>&1; do sleep 20; done

echo "[$(date +%H:%M:%S)] all jobs done, checking data quality"
$PY - <<'EOF'
import csv, glob
oe=on=tot=0
bad=[]
for f in sorted(glob.glob("results/entropy_big/*.csv")):
    rows=list(csv.DictReader(open(f)))
    if not rows: continue
    e=sum(1 for r in rows if r.get("E_best","nan") not in ("nan","","None"))
    n=sum(1 for r in rows if r.get("n_clusters","0") not in ("0","","0.0"))
    tot+=len(rows); oe+=e; on+=n
    if e<10 or n<10: bad.append(f"{rows[0]['metal']}_{rows[0]['target']}")
print(f"  E valid {oe}/{tot} | n_clusters>0 {on}/{tot}")
if bad: print(f"  still bad ({len(bad)}): {' '.join(bad[:12])}")
EOF

echo "[$(date +%H:%M:%S)] running ZBG calibration"
$PY scripts/analysis/zbg_calibration.py 2>&1 | tee results/zbg_calibration.log | head -40

echo "[$(date +%H:%M:%S)] building Word report"
$PY scripts/analysis/make_report_docx.py 2>&1 | tail -3

echo "[$(date +%H:%M:%S)] DONE"
