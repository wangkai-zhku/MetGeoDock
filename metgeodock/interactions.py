"""Graded protein-ligand interaction profile.

Motivation
----------
MetalGeoDock's scoring is built around the metal coordination sphere. That is its
strength and also its blind spot: isoform selectivity is usually decided by the
*other* contacts - hydrogen bonds, hydrophobic packing, pi stacking, cation-pi,
salt bridges, halogen bonds - against residues that differ between subtypes while
the catalytic machinery stays identical.

Every term here is a smooth (graded) function of geometry, never a hard cutoff, so
small pose differences produce small score differences instead of binary flips:

    graded(d; d0, s)      = exp(-((d - d0)/s)^2)
    hbond                 = graded(d; 3.0, 0.6)  x donor/acceptor complementarity
    hydrophobic           = graded(d; 4.0, 0.8)  x apolar-atom overlap
    pi_pi                 = graded(dc; 4.6, 1.2)  x orientation factor
    cation_pi             = graded(dc; 4.2, 1.0)
    salt_bridge           = graded(d; 3.8, 0.9)
    halogen               = graded(d; 3.4, 0.7)

Terms are returned as a dict so a caller can (a) use them individually or (b)
whiten them with their covariance and combine - the combination weights then come
from the descriptor covariance structure, not from activity labels.
"""
from __future__ import annotations
import numpy as np

__all__ = ["interaction_profile", "INTERACTION_TERMS"]

INTERACTION_TERMS = ["hbond", "hydrophobic", "pi_pi", "cation_pi", "salt_bridge",
                     "halogen", "polar_contact"]

# protein-side atom classes, keyed by residue type -> atom names
DONOR = {
    "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"], "ASN": ["ND2"], "GLN": ["NE2"],
    "HIS": ["ND1", "NE2"], "LYS": ["NZ"], "ARG": ["NE", "NH1", "NH2"],
    "TRP": ["NE1"], "CYS": ["SG"],
}
ACCEPTOR = {
    "ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"], "ASN": ["OD1"], "GLN": ["OE1"],
    "HIS": ["ND1", "NE2"], "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"],
    "MET": ["SD"], "CYS": ["SG"],
}
POSITIVE = {"LYS": ["NZ"], "ARG": ["NE", "NH1", "NH2"], "HIS": ["ND1", "NE2"]}
NEGATIVE = {"ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"]}
AROMATIC_RING = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
}
APOLAR = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "CYS", "GLY"}


def _graded(d, d0, s):
    """smooth kernel: 1 at d0, ~0 beyond d0 +- 2.5 s"""
    return np.exp(-((np.asarray(d, float) - d0) / s) ** 2)


def _ring_centers_normal(protein, metal_index, out):
    """centroid + normal of each aromatic side chain in the protein"""
    byres = {}
    for i in range(len(protein)):
        rn = str(protein.resnames[i]).strip().upper()
        an = str(protein.names[i]).strip().upper()
        if rn in AROMATIC_RING and an in AROMATIC_RING[rn]:
            key = (rn, int(protein.resseqs[i]))
            byres.setdefault(key, []).append(protein.xyz[i])
    for key, pts in byres.items():
        if len(pts) < 3:
            continue
        P = np.array(pts, float)
        c = P.mean(axis=0)
        u, s, vt = np.linalg.svd(P - c)
        out.append((c, vt[2] / max(np.linalg.norm(vt[2]), 1e-9)))


def _ligand_rings(ligand, rd_mol=None):
    """aromatic ring centroids + normals from the docking Molecule (topology-free:
    falls back to a geometric ring search over aromatic-looking cycles)"""
    rings = []
    if rd_mol is not None:
        for ring in rd_mol.GetRingInfo().AtomRings():
            if not all(rd_mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                continue
            pts = []
            ok = True
            for i in ring:
                p = rd_mol.GetConformer().GetAtomPosition(i)
                pts.append([p.x, p.y, p.z])
            if not ok or len(pts) < 5:
                continue
            P = np.array(pts, float)
            c = P.mean(axis=0)
            u, s, vt = np.linalg.svd(P - c)
            rings.append((c, vt[2] / max(np.linalg.norm(vt[2]), 1e-9)))
    return rings


def interaction_profile(protein, ligand, metal_index: int | None = None,
                        rd_mol=None) -> dict:
    """Graded interaction terms between a docked pose and the receptor.

    ``rd_mol`` (an RDKit mol carrying the same conformer) is optional; without it
    pi_pi / cation_pi fall back to zero because aromaticity cannot be recovered
    from heavy-atom coordinates alone.
    """
    px = np.asarray(protein.xyz, float)
    lx = np.asarray(ligand.xyz, float)
    pel = [str(e).strip().upper() for e in protein.elements]
    lel = [str(e).strip().upper() for e in ligand.elements]
    pres = [str(r).strip().upper() for r in protein.resnames]
    pname = [str(n).strip().upper() for n in protein.names]

    out = {k: 0.0 for k in INTERACTION_TERMS}

    # ---- precompute protein atom classes
    p_don = np.array([i for i in range(len(px))
                      if pname[i] in DONOR.get(pres[i], ())], int)
    p_acc = np.array([i for i in range(len(px))
                      if pname[i] in ACCEPTOR.get(pres[i], ())], int)
    p_pos = np.array([i for i in range(len(px))
                      if pname[i] in POSITIVE.get(pres[i], ())], int)
    p_neg = np.array([i for i in range(len(px))
                      if pname[i] in NEGATIVE.get(pres[i], ())], int)
    p_apo = np.array([i for i in range(len(px))
                      if pres[i] in APOLAR and pel[i] in ("C", "S")], int)
    p_pol = np.array([i for i in range(len(px)) if pel[i] in ("N", "O")], int)

    # ---- ligand atom classes
    l_pol = np.array([i for i in range(len(lx)) if lel[i] in ("N", "O")], int)
    l_apo = np.array([i for i in range(len(lx)) if lel[i] in ("C", "S", "F", "CL", "BR", "I")], int)
    l_hal = np.array([i for i in range(len(lx)) if lel[i] in ("F", "CL", "BR", "I")], int)
    l_pos = l_neg = np.array([], int)
    if rd_mol is not None:
        # formal charges are almost always 0 for neutral drugs -> use Gasteiger
        # partial charges so zwitterionic / strongly polar groups still register
        try:
            from rdkit.Chem import AllChem as _A
            _A.ComputeGasteigerCharges(rd_mol, nIter=20, throwOnParamFailure=False)
            q = np.array([a.GetDoubleProp("_GasteigerCharge")
                          if a.HasProp("_GasteigerCharge") else 0.0
                          for a in rd_mol.GetAtoms()], float)
            q = np.nan_to_num(q)
            l_pos = np.where(q > 0.30)[0]
            l_neg = np.where(q < -0.30)[0]
        except Exception:
            l_pos = np.array([a.GetIdx() for a in rd_mol.GetAtoms()
                              if a.GetFormalCharge() > 0], int)
            l_neg = np.array([a.GetIdx() for a in rd_mol.GetAtoms()
                              if a.GetFormalCharge() < 0], int)

    def mind(A, B):
        if len(A) == 0 or len(B) == 0:
            return np.full(len(A), 99.0)
        d = np.sqrt(((lx[A][:, None, :] - px[B][None, :, :]) ** 2).sum(-1))
        return d.min(axis=1)

    # hbond: ligand polar atom to complementary protein polar atom
    if len(l_pol) and (len(p_don) or len(p_acc)):
        dmin = np.minimum(mind(l_pol, p_don) if len(p_don) else 99.0,
                          mind(l_pol, p_acc) if len(p_acc) else 99.0)
        out["hbond"] = float(_graded(dmin, 3.0, 0.6).sum())

    # hydrophobic: apolar-apolar packing
    if len(l_apo) and len(p_apo):
        dmin = mind(l_apo, p_apo)
        out["hydrophobic"] = float(_graded(dmin, 4.0, 0.8).sum() / max(len(l_apo), 1) * 10.0)

    # polar contact (generic N/O proximity, catches polar-but-uncharged packing)
    if len(l_pol) and len(p_pol):
        dmin = mind(l_pol, p_pol)
        out["polar_contact"] = float(_graded(dmin, 3.4, 0.8).sum())

    # salt bridge: formal charges of the ligand vs charged residues
    if len(l_pos) and len(p_neg):
        out["salt_bridge"] = float(_graded(mind(l_pos, p_neg), 3.8, 0.9).sum())
    if len(l_neg) and len(p_pos):
        out["salt_bridge"] += float(_graded(mind(l_neg, p_pos), 3.8, 0.9).sum())

    # pi-pi and cation-pi need ring information from the RDKit conformer
    prots_rings = []
    _ring_centers_normal(protein, metal_index, prots_rings)
    lig_rings = _ligand_rings(ligand, rd_mol)
    if prots_rings and lig_rings:
        for lc, ln_ in lig_rings:
            for pc, pn in prots_rings:
                dc = float(np.linalg.norm(lc - pc))
                if dc > 8.0:
                    continue
                ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(ln_, pn)), 0, 1))))
                # parallel (0 deg) and T-shaped (90 deg) both count, worst at 45
                orient = max(np.exp(-((ang - 0) / 30.0) ** 2), np.exp(-((ang - 90) / 30.0) ** 2))
                out["pi_pi"] += float(_graded(dc, 4.6, 1.2) * orient)
    if lig_rings and len(p_pos):
        for lc, _ in lig_rings:
            d = np.linalg.norm(px[p_pos] - lc, axis=1)
            out["cation_pi"] += float(_graded(d.min(), 4.2, 1.0))
    if prots_rings and len(l_pos):
        for pc, _ in prots_rings:
            d = np.linalg.norm(lx[l_pos] - pc, axis=1)
            out["cation_pi"] += float(_graded(d.min(), 4.2, 1.0))

    # halogen bond: halogen to protein acceptor
    if len(l_hal) and len(p_acc):
        out["halogen"] = float(_graded(mind(l_hal, p_acc), 3.4, 0.7).sum())

    return out
