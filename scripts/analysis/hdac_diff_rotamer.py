"""Difference-region rotamer fingerprint: rotamer entropy restricted to the residues
that actually sit in the inter-subtype shape difference.

Why this is different from the earlier attempt
----------------------------------------------
The earlier structure-restricted entropy averaged over ALL ~71 pocket residues.
Most of them are conserved and contribute identical values in every subtype, so the
signal was diluted 5-10x by noise. Here the analysis is restricted to residues that
border a voxel which is OPEN in one subtype and BLOCKED in the other - i.e. exactly
where the two pockets differ. If those residues' side chains also have different
rotamer freedom, the difference is real and subtype-specific.

Outputs per subtype pair:
  n_diff_res      residues bordering the difference region
  rotamer entropy of those residues, per subtype
  shape-Jaccard of the two residue sets (do the same residues border the gap?)
  fingerprint distance between the two rotamer vectors
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/home/wangkai/cursor-task/metalloprotein-chem")
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("pm", ROOT / "scripts/analysis/pocket_rotamer_metal.py")
pm = importlib.util.module_from_spec(spec)
sys.modules["pm"] = pm
spec.loader.exec_module(pm)

PDB = {"HDAC1": "5ICN", "HDAC2": "3MAX", "HDAC3": "4A69",
       "HDAC4": "2VQM", "HDAC6": "5EDU", "HDAC8": "1T64"}
MET = {"ZN"}
RMAX, STEP, PROBE = 13.0, 1.0, 1.4
RT, EPS = 0.593, 1e6      # electrostatics off (measured harmful)
NEAR = 3.5                # a voxel "belongs" to residues within this distance


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
            prot.append((rn, ln[21] + ln[22:26].strip(), an, xyz, el))
        elif rn in MET:
            met.append(xyz)
    return prot, met[0]


def anchors(prot, m):
    sh = {}
    for rn, key, an, xyz, el in prot:
        if np.linalg.norm(xyz - m) <= 3.0 and rn in ("HIS", "ASP", "GLU", "CYS"):
            sh.setdefault((rn, key), None)
    order = {"HIS": 0, "ASP": 1, "GLU": 1, "CYS": 2}
    keys = sorted(sh, key=lambda k: (order[k[0]], k[1]))[:3]
    pts = [m]
    for rn, key in keys:
        ca = [xyz for r, k, an, xyz, el in prot if r == rn and k == key and an == "CA"]
        if ca:
            pts.append(ca[0])
    return np.array(pts)


def kabsch(P, Q):
    pc, qc = P - P.mean(0), Q - Q.mean(0)
    H = pc.T @ qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, Q.mean(0) - R @ P.mean(0)


def grid(prot, metal, R=None, t=None):
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
    return pts, rel, dmin > PROBE, P


def residues_near(pts, prot, R=None, t=None):
    """residue keys whose atoms come within NEAR of any of the given points"""
    P = np.array([p[3] for p in prot])
    if R is not None:
        P = (R @ P.T).T + t
    keys = [p[1] for p in prot]
    hit = set()
    for i in range(0, len(pts), 1500):
        blk = pts[i:i + 1500]
        d = np.sqrt(((blk[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        for j in np.where(d.min(axis=0) < NEAR)[0]:
            hit.add(keys[j])
    return hit


def rotamer_vector(prot, metal, keys):
    """S for the given residues; also returns the residue->S map"""
    by = {}
    for rn, key, an, xyz, el in prot:
        by.setdefault(key, {"rn": rn, "atoms": []})["atoms"].append((rn, key, an, xyz, el))
    allxyz = np.array([p[3] for p in prot])
    allel = [p[4] for p in prot]
    allkeys = [p[1] for p in prot]
    out = {}
    for key in keys:
        if key not in by:
            continue
        v = by[key]
        mask = np.array([k != key for k in allkeys], bool)
        r = pm.residue_entropy(v["rn"], v["atoms"], allxyz[mask],
                               [e for e, m_ in zip(allel, mask) if m_], metal, RT, EPS)
        if r is not None:
            out[key] = r["S"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HDAC1")
    ap.add_argument("--out", default="results/hdac_diff_rotamer.json")
    a = ap.parse_args()

    D = {}
    for s, pid in PDB.items():
        prot, m = load(pid)
        D[s] = {"prot": prot, "metal": m, "anchors": anchors(prot, m)}

    ref = a.ref
    aln = json.load(open(ROOT / "results/hdac_sequence_alignment.json"))
    MAPS = {k: v.get("mapping", {}) for k, v in aln["pairs"].items()}
    prot_r, m_r = D[ref]["prot"], D[ref]["metal"]
    pts_r, rel_r, open_r, P_r = grid(prot_r, m_r)

    print(f"参照 {ref}；差异区残基的 rotamer 指纹")
    print(f"{'对比':<16}{'差异体素':>9}{'对方残基':>9}{'本方残基':>9}{'Jaccard':>9}"
          f"{'本方熵':>8}{'对方熵':>8}{'ΔS':>8}")
    out = {}
    for s in PDB:
        if s == ref:
            continue
        prot_s, m_s = D[s]["prot"], D[s]["metal"]
        R, t = kabsch(D[s]["anchors"], D[ref]["anchors"])
        pts_s, rel_s, open_s, P_s = grid(prot_s, m_s, R=R, t=t)
        diff = open_r ^ open_s
        if diff.sum() == 0:
            continue
        dpts = pts_r[diff]                       # in the reference frame
        # residues bordering the difference region, in each structure's own frame
        keys_r = residues_near(dpts, prot_r)
        keys_s = residues_near(dpts, prot_s, R=R, t=t)
        Sr = rotamer_vector(prot_r, m_r, keys_r)
        Ss = rotamer_vector(prot_s, m_s, keys_s)
        # align by sequence correspondence, not by residue id
        mapping = MAPS.get(s, {})
        keys_s_aligned = {mapping[k] for k in keys_r if k in mapping}
        Ss_al = rotamer_vector(prot_s, m_s, keys_s_aligned)
        pairs = [(k, mapping[k]) for k in Sr if k in mapping and mapping[k] in Ss_al]
        if pairs:
            ent_r = float(np.sum([Sr[a] for a, _ in pairs]))
            ent_s = float(np.sum([Ss_al[b] for _, b in pairs]))
            common = [a for a, _ in pairs]
        else:
            ent_r = ent_s = float("nan"); common = []
        jac = len(keys_r & keys_s) / max(len(keys_r | keys_s), 1)
        n_common = len(common)
        out[s] = {"n_diff_voxels": int(diff.sum()), "n_res_ref": len(keys_r),
                  "n_res_other": len(keys_s), "n_common": n_common,
                  "jaccard_residues": round(jac, 3),
                  "entropy_ref": round(ent_r, 3), "entropy_other": round(ent_s, 3),
                  "delta_S": round(ent_s - ent_r, 3)}
        print(f"{ref} vs {s:<8}{diff.sum():>9}{len(keys_s):>9}{len(keys_r):>9}"
              f"{jac:>9.3f}{ent_r:>8.2f}{ent_s:>8.2f}{ent_s - ent_r:>+8.2f}")

    # does the fingerprint separate subtypes?
    print("\n差异区残基集合的重叠（Jaccard）—— 越小说明差异区的残基基础越不同")
    ks = list(out)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            print(f"  {ref} vs {ks[i]:<8} ΔS={out[ks[i]]['delta_S']:+6.2f}   "
                  f"{ref} vs {ks[j]:<8} ΔS={out[ks[j]]['delta_S']:+6.2f}")
            break
        break
    print("\n各对 vs 参照的 ΔS（针对差异区残基，不是全部口袋残基）:")
    for s, v in sorted(out.items(), key=lambda kv: -abs(kv[1]["delta_S"])):
        print(f"  {ref} vs {s:<8} ΔS={v['delta_S']:+7.3f}  "
              f"共同残基={v['n_common']:<3} 残基Jaccard={v['jaccard_residues']:.3f}")

    json.dump(out, open(ROOT / a.out, "w"), indent=2)
    print(f"\n写出 {a.out}")


if __name__ == "__main__":
    main()
