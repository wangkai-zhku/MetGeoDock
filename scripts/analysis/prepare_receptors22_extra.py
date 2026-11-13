"""准备 22 靶点数据集中缺失的 11 个受体。

已有（processed/receptors10 + processed/hdac_receptors，可直接复用）：
    HDAC1 5ICN, HDAC2 3MAX, CA2 3CAJ, MMP2 1QIB, ALOX5 3O8Y,
    ARG2 1PQ3, METAP2 1B6A, urease 1FWJ, HDAC3 4A69, HDAC6 5EDU(CD2), HDAC8 1T64

本脚本补：
    CA12 1JD0 (zn), MMP1 966C (zn), MMP3 1HY7 (zn), MMP9 1GKC (zn),
    MMP13 830C (zn), ARG1 2AEB (mn), CYP1A2 2HI4 (fe3/heme),
    CYP2E1 3E6I (fe3/heme), CYP19A1 3EQM (fe3/heme),
    EGLN2 3H4K (fe2), HIV1_integrase 3LPT (mg)
"""
from __future__ import annotations
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
OUTDIR = ROOT / "processed/receptors22"
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("CA12", "1JD0", "zn", "Zn 3: His94 His96 His119 (CA 同家族)"),
    ("MMP1", "966C", "zn", "catalytic Zn"),
    ("MMP3", "1HY7", "zn", "catalytic Zn"),
    ("MMP9", "1GKC", "zn", "catalytic Zn"),
    ("MMP13", "830C", "zn", "catalytic Zn"),
    ("ARG1", "2AEB", "mn", "binuclear Mn"),
    ("CYP1A2", "2HI4", "fe3", "heme Fe(III)"),
    ("CYP2E1", "3E6I", "fe3", "heme Fe(III)"),
    ("CYP19A1", "3EQM", "fe3", "heme Fe(III)"),
    ("EGLN2", "3H4K", "fe2", "non-heme Fe(II)"),
    ("HIV1_integrase", "3LPT", "mg", "Mg2+ cofactor"),
]

HEME_ATOMS = {"FE", "F", "FE2", "FE3"}


def download(pid: str) -> Path:
    dst = OUTDIR / f"{pid}.pdb"
    if dst.exists() and dst.stat().st_size > 10000:
        return dst
    req = urllib.request.Request(f"https://files.rcsb.org/download/{pid}.pdb",
                                 headers={"User-Agent": "curl/8"})
    data = urllib.request.urlopen(req, timeout=120).read()
    dst.write_bytes(data)
    return dst


def extract_metal_site(pid: str, metal: str) -> Path | None:
    """对血红素蛋白：把 HEM 的 FE 单独提成 HETATM，并保留 HEM 其余原子（作为辅基）。
    对普通金属蛋白：直接保留金属 HETATM。"""
    src = OUTDIR / f"{pid}.pdb"
    dst = OUTDIR / f"{pid}_site.pdb"
    WATER = {"HOH", "WAT", "H2O", "DOD"}
    lines = src.read_text(errors="ignore").splitlines()
    keep_heme_res = bool(metal.startswith("fe"))
    out, nmetal, nheme = [], 0, 0
    for ln in lines:
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        rn = ln[17:20].strip().upper()
        if rn in WATER:
            continue
        if ln.startswith("HETATM"):
            if rn in ("ZN", "ZN2", "MN", "MN2", "MG", "MG2", "CA", "CA2", "NI", "NI2",
                      "CO", "CO2", "CU", "CU1", "FE", "FE2", "FE3"):
                nmetal += 1
                out.append(ln)
            elif rn == "HEM" and keep_heme_res:
                nheme += 1
                out.append(ln)
            continue
        out.append(ln)
    if not out:
        return None
    dst.write_text("\n".join(out) + "\nEND\n")
    print(f"  {pid}: {len(out)} 原子 (金属 {nmetal}, HEM {nheme})", flush=True)
    return dst


def prepare(pid: str, metal: str) -> bool:
    site = extract_metal_site(pid, metal)
    if site is None:
        return False
    h = OUTDIR / f"{pid}_h.pdb"
    rep = OUTDIR / f"{pid}_prepare.json"
    cmd = [str(ROOT / "envs/metgeodock/bin/python"), "-m", "metgeodock.cli", "prepare",
           "--pdb", str(site), "--out", str(h), "--metal", metal,
           "--no-keep-water", "--report", str(rep)]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    ok = h.exists() and h.stat().st_size > 10000
    if ok:
        n = sum(1 for _ in h.open())
        coord = [l for l in r.stdout.splitlines() if "coord ->" in l]
        print(f"  {pid}: prepared {n} 行  {coord[0].strip()[:80] if coord else ''}", flush=True)
    else:
        print(f"  {pid}: FAILED  {r.stdout[-300:]} {r.stderr[-300:]}", flush=True)
    return ok


def main():
    man = {}
    for name, pid, metal, note in TARGETS:
        print(f"[{name}] {pid} ({metal})", flush=True)
        try:
            download(pid)
            ok = prepare(pid, metal)
            man[name] = {"pdb": pid, "metal": metal, "note": note, "prepared": ok}
        except Exception as e:
            print(f"  {name}: 异常 {type(e).__name__} {e}", flush=True)
            man[name] = {"pdb": pid, "metal": metal, "note": note, "prepared": False}
    (OUTDIR / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False))
    ok = sum(1 for v in man.values() if v["prepared"])
    print(f"\n完成 {ok}/{len(TARGETS)}；清单写入 {OUTDIR/'manifest.json'}")


if __name__ == "__main__":
    main()
