"""Locate the subtype-discriminating residues between HDAC subtypes.

Method
------
1. superimpose every subtype onto HDAC1 using the metal + Calpha of the three
   first-shell residues (RMSD 0.15-0.20 A, so the catalytic site is aligned and
   all remaining difference is genuine pocket difference),
2. rasterise each pocket into a 1 A occupancy grid centred on the metal,
3. take the voxels that are open in one subtype and blocked in the other,
4. attribute every such voxel to its nearest protein residue, in each structure.

The result is a ranked list of residues that physically differ between subtypes,
split by whether the subtype has EXTRA space (open here, blocked there) or LESS
space, plus the distance of each residue from the metal. That distance profile is
what tells us whether differences in the coordination layer matter at all.
"""
from __future__ import annotations
import itertools, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC8": "1T64"}
MET = {"ZN", "CA", "MG", "MN", "CO", "NI", "FE"}
RMAX, STEP, PROBE = 13.0, 1.0, 1.4


def load(pid):
    prot, met = [], []
    for ln in (ROOT / f"processed/hdac_receptors/{pid}_h.pdb").read_text(errors="ignore").splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        rn = ln[17:20].strip().upper()
        el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        an = ln[12:16].strip()
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        if ln.startswith("ATOM") and el != "H":
            prot.append((rn, ln[22:26].strip(), an, xyz))
        elif rn in MET and rn.startswith("ZN"):
            met.append(xyz)
    return prot, met[0]


def anchors(prot, m):
    sh = {}
    for rn, ri, an, xyz in prot:
        if np.linalg.norm(xyz - m) <= 3.0 and rn in ("HIS", "ASP", "GLU", "CYS"):
            sh.setdefault((rn, ri), None)
    order = {"HIS": 0, "ASP": 1, "GLU": 1, "CYS": 2}
    keys = sorted(sh, key=lambda k: (order[k[0]], k[1]))[:3]
    pts = [m]
    for rn, ri in keys:
        ca = [x for r, rr, an, x in prot if r == rn and rr == ri and an == "CA"]
        if ca:
            pts.append(ca[0])
    return np.array(pts), [f"{k[0]}{k[1]}" for k in keys]


def kabsch(P, Q):
    pc, qc = P - P.mean(0), Q - Q.mean(0)
    H = pc.T @ qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, Q.mean(0) - R @ P.mean(0)


def pocket_grid(prot, metal, R=None, t=None):
    P = np.array([p[3] for p in prot])
    m = metal
    if R is not None:
        P = (R @ P.T).T + t
        m = R @ metal + t
    n = int(2 * RMAX / STEP) + 1
    ax = np.linspace(-RMAX, RMAX, n)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    rel = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pts = rel + m
    dmin = np.empty(len(pts))
    for i in range(0, len(pts), 4000):
        blk = pts[i:i + 4000]
        dmin[i:i + 4000] = np.sqrt(((blk[:, None, :] - P[None, :, :]) ** 2).sum(-1)).min(1)
    return pts, np.linalg.norm(rel, axis=1), dmin > PROBE, P, m


def attribute(pts, protein_xyz, residues, k=1):
    """nearest residue (type+id) for each point"""
    out = []
    for i in range(0, len(pts), 2000):
        blk = pts[i:i + 2000]
        d = np.sqrt(((blk[:, None, :] - protein_xyz[None, :, :]) ** 2).sum(-1))
        idx = d.argmin(1)
        out.extend(residues[j] for j in idx)
    return out


def main():
    D = {}
    for s, pid in PDB.items():
        prot, m = load(pid)
        a, names = anchors(prot, m)
        D[s] = dict(prot=prot, metal=m, anchors=a, shell=names)

    ref = "HDAC1"
    prot_r = D[ref]["prot"]
    m_r = D[ref]["metal"]
    pts_r, r_r, open_r, P_r, m_al = pocket_grid(prot_r, m_r)
    res_r = [(p[0], p[1]) for p in prot_r]

    results = {}
    for s in PDB:
        if s == ref:
            continue
        prot, m = D[s]["prot"], D[s]["metal"]
        R, t = kabsch(D[s]["anchors"], D[ref]["anchors"])
        pts_s, r_s, open_s, P_s, _ = pocket_grid(prot, m, R=R, t=t)
        res_s = [(p[0], p[1]) for p in prot]

        diff = open_r ^ open_s
        extra_in_s = diff & open_s      # subtype has space where HDAC1 is blocked
        extra_in_r = diff & open_r      # HDAC1 has space where subtype is blocked
        pts = pts_r[diff]
        tag_extra_s = extra_in_s[diff]
        res_here = attribute(pts, P_r, res_r)
        res_there = attribute(pts, P_s, res_s)

        agg = defaultdict(lambda: {"voxels": 0, "extra": 0, "less": 0, "dists": []})
        for i, (rr, rs) in enumerate(zip(res_here, res_there)):
            key = f"{rr[0]}{rr[1]}/{rs[0]}{rs[1]}"
            a = agg[key]
            a["voxels"] += 1
            (a["extra"] if tag_extra_s[i] else a["less"])
            if tag_extra_s[i]:
                a["extra"] += 1
            else:
                a["less"] += 1
            a["dists"].append(float(np.linalg.norm(pts[i] - m_al)))
        rows = sorted(agg.items(), key=lambda kv: -kv[1]["voxels"])[:15]
        results[s] = {
            "n_diff_voxels": int(diff.sum()),
            "n_extra_in_subtype": int(extra_in_s.sum()),
            "n_extra_in_HDAC1": int(extra_in_r.sum()),
            "top_residues": [{"pair": k, "voxels": v["voxels"],
                              "extra_in_subtype": v["extra"], "extra_in_HDAC1": v["less"],
                              "median_dist_to_metal": round(float(np.median(v["dists"])), 2)}
                             for k, v in rows],
        }
        print(f"\n=== HDAC1 vs {s}   差异体素 {diff.sum()}")
        print(f"    其中 {s} 多出的空间 {extra_in_s.sum()} 体素，HDAC1 多出的空间 {extra_in_r.sum()} 体素")
        print(f"    {'残基对 (HDAC1/对方)':<26}{'体素':>6}{'多出':>6}{'缺失':>6}{'离金属(中位,A)':>15}")
        for k, v in rows[:10]:
            print(f"    {k:<26}{v['voxels']:>6}{v['extra']:>6}{v['less']:>6}"
                  f"{np.median(v['dists']):>15.1f}")

        # distance profile of the differences
        prof = []
        for lo, hi in ((0, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14)):
            prof.append(int((diff & (r_r > lo) & (r_r <= hi)).sum()))
        results[s]["diff_profile"] = prof
    print("\n" + "=" * 92)
    print("差异体素的距离剖面（按到金属的距离）—— 这条曲线决定哪一层值得描述")
    print("=" * 92)
    print(f"{'对比':<16}" + "".join(f"{f'{a}-{b}A':>9}" for a, b in
                                    ((0, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14))))
    for s, v in results.items():
        print(f"HDAC1 vs {s:<8}" + "".join(f"{x:>9}" for x in v["diff_profile"]))

    out = ROOT / "results/hdac_subtype_diff_residues.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()
