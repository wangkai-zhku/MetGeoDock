"""Minimal PDB / MOL2 readers. Raw files are opened read-only."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DONOR_ELEMENTS = {"O", "N", "S"}
WATER_RESNAMES = {"HOH", "WAT", "H2O", "DOD"}
AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "CYX", "CYM", "ASH", "GLH", "LYN", "MSE",
}
# Residue / atom names used in PDB and in GPDOCK/Amber ion files (C0=Ca, CAD=Cd).
METAL_RESNAMES = {
    "ZN", "ZN2", "CA", "CA2", "C0", "MG", "MG2", "MN", "MN2", "MN3",
    "FE", "FE2", "FE3", "CU", "CU1", "CU2", "CO", "CO2", "CD", "CD2",
    "CAD", "NI", "NI2",
}
METAL_ATOM_NAMES = METAL_RESNAMES | {"NA", "K", "HG", "PT", "AU"}


@dataclass
class Molecule:
    name: str
    xyz: np.ndarray
    elements: np.ndarray
    names: np.ndarray
    resnames: np.ndarray
    resseqs: np.ndarray
    charges: np.ndarray
    atom_types: np.ndarray
    path: Path | None = None
    bonds: list[tuple[int, int]] = field(default_factory=list)
    # atom index -> functional group (hydroxamate / carboxylate / thiol / ...);
    # filled for RDKit-sourced ligands only. None = chemistry layer unavailable.
    donor_groups: dict[int, str] | None = None

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def heavy_mask(self) -> np.ndarray:
        return np.array([el.upper() != "H" for el in self.elements], dtype=bool)

    def donor_indices(self) -> np.ndarray:
        return np.array(
            [i for i, el in enumerate(self.elements) if el.upper() in DONOR_ELEMENTS],
            dtype=int,
        )

    def metal_indices(self, metal: str | None = None) -> np.ndarray:
        wanted = None
        if metal:
            from .metals import PDB_RESNAMES, metal_from_name

            wanted = {x.upper() for x in PDB_RESNAMES[metal_from_name(metal)]}
        out = []
        for i, res in enumerate(self.resnames):
            resu = str(res).strip().upper()
            if resu in AMINO_ACIDS:
                continue
            if resu not in METAL_RESNAMES:
                continue
            if wanted is not None and resu not in wanted:
                continue
            out.append(i)
        return np.array(out, dtype=int)

    def is_water(self, i: int) -> bool:
        return str(self.resnames[i]).strip().upper() in WATER_RESNAMES

    def copy_with_xyz(self, xyz: np.ndarray) -> "Molecule":
        mol = Molecule(
            name=self.name,
            xyz=np.asarray(xyz, dtype=float).copy(),
            elements=self.elements.copy(),
            names=self.names.copy(),
            resnames=self.resnames.copy(),
            resseqs=self.resseqs.copy(),
            charges=self.charges.copy(),
            atom_types=self.atom_types.copy(),
            path=self.path,
            bonds=list(self.bonds),
            donor_groups=self.donor_groups,
        )
        return mol


def select_atoms(mol: Molecule, indices: list[int] | np.ndarray, name: str | None = None) -> Molecule:
    idx = np.asarray(indices, dtype=int)
    return Molecule(
        name=name or mol.name,
        xyz=mol.xyz[idx].copy(),
        elements=mol.elements[idx].copy(),
        names=mol.names[idx].copy(),
        resnames=mol.resnames[idx].copy(),
        resseqs=mol.resseqs[idx].copy(),
        charges=mol.charges[idx].copy(),
        atom_types=mol.atom_types[idx].copy(),
        path=mol.path,
        bonds=[],
    )


def concat_molecules(mols: list[Molecule], name: str) -> Molecule:
    alive = [m for m in mols if m is not None and len(m) > 0]
    if not alive:
        raise ValueError("no molecules to concatenate")
    if len(alive) == 1:
        out = select_atoms(alive[0], list(range(len(alive[0]))), name=name)
        return out
    return Molecule(
        name=name,
        xyz=np.vstack([m.xyz for m in alive]),
        elements=np.concatenate([m.elements for m in alive]),
        names=np.concatenate([m.names for m in alive]),
        resnames=np.concatenate([m.resnames for m in alive]),
        resseqs=np.concatenate([m.resseqs for m in alive]),
        charges=np.concatenate([m.charges for m in alive]),
        atom_types=np.concatenate([m.atom_types for m in alive]),
        path=alive[0].path,
        bonds=[],
    )


def strip_waters(mol: Molecule) -> Molecule:
    keep = [i for i in range(len(mol)) if not mol.is_water(i)]
    return select_atoms(mol, keep, name=f"{mol.name}_nowater")


def keep_only_metal(mol: Molecule, metal_index: int) -> Molecule:
    """Drop every metal ion except ``metal_index`` (MetalloDock-style extra-metal removal)."""
    mets = set(int(i) for i in mol.metal_indices())
    keep = [i for i in range(len(mol)) if i not in mets or i == int(metal_index)]
    return select_atoms(mol, keep, name=f"{mol.name}_1metal")


def merge_with_waters(protein: Molecule, water: Molecule | None, name: str | None = None) -> Molecule:
    """Append crystallographic waters to the receptor (read-only concat)."""
    if water is None or len(water) == 0:
        return protein
    return concat_molecules([protein, water], name=name or f"{protein.name}_hoh")


def crop_pocket(mol: Molecule, metal_xyz: np.ndarray, radius_A: float = 15.0, name: str | None = None) -> Molecule:
    """Keep residues that have any atom within ``radius_A`` of any listed metal."""
    origins = np.asarray(metal_xyz, dtype=float).reshape(-1, 3)
    if origins.size == 0:
        return mol
    dist = np.linalg.norm(mol.xyz[:, None, :] - origins[None, :, :], axis=2).min(axis=1)
    keep_res = {
        (str(mol.resnames[i]).strip().upper(), int(mol.resseqs[i]))
        for i, di in enumerate(dist)
        if di <= radius_A
    }
    idx = [
        i
        for i in range(len(mol))
        if (str(mol.resnames[i]).strip().upper(), int(mol.resseqs[i])) in keep_res
    ]
    if not idx:
        return mol
    return select_atoms(mol, idx, name=name or f"{mol.name}_pocket")


def _element_from_pdb_name(atom_name: str, element_field: str, resname: str = "") -> str:
    resu = resname.strip().upper()
    name = atom_name.strip()
    if resu in METAL_RESNAMES or name.strip().upper() in METAL_ATOM_NAMES:
        key = resu if resu in METAL_RESNAMES else name.strip().upper()
        mapping = {
            "ZN": "Zn", "ZN2": "Zn", "CA": "Ca", "CA2": "Ca", "C0": "Ca",
            "MG": "Mg", "MG2": "Mg", "MN": "Mn", "MN2": "Mn", "MN3": "Mn",
            "FE": "Fe", "FE2": "Fe", "FE3": "Fe", "CU": "Cu", "CU1": "Cu",
            "CU2": "Cu", "CO": "Co", "CO2": "Co", "CD": "Cd", "CD2": "Cd",
            "CAD": "Cd", "NI": "Ni", "NI2": "Ni",
        }
        return mapping.get(key, "X")
    el = element_field.strip()
    if el and el.upper() in {"C", "N", "O", "S", "H", "P", "F", "CL", "BR", "I"}:
        return el.capitalize() if len(el) > 1 else el.upper()
    if not name:
        return "X"
    if name[0].isdigit():
        name = name[1:]
    letters = "".join(ch for ch in name if ch.isalpha())
    if not letters:
        return "X"
    return letters[0].upper()


def _parse_pdb_atom_line(line: str) -> tuple | None:
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return None
    if len(line) < 54:
        return None
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        return None
    name = line[12:16].strip()
    resn = line[17:20].strip()
    try:
        resi = int(line[22:26])
    except ValueError:
        resi = 0
    occ = line[54:60].strip() if len(line) >= 60 else ""
    charge = float(occ) if occ else 0.0
    el = _element_from_pdb_name(name, line[76:78] if len(line) >= 78 else "", resn)
    return x, y, z, name, resn, resi, charge, el


def molecule_from_atom_rows(rows: list[tuple], name: str, path: Path | None = None) -> Molecule:
    xyz, names, resnames, resseqs, charges, elements = [], [], [], [], [], []
    for x, y, z, name_a, resn, resi, charge, el in rows:
        xyz.append((x, y, z))
        names.append(name_a)
        resnames.append(resn)
        resseqs.append(resi)
        charges.append(charge)
        elements.append(el)
    return Molecule(
        name=name,
        xyz=np.asarray(xyz, dtype=float),
        elements=np.asarray(elements),
        names=np.asarray(names),
        resnames=np.asarray(resnames),
        resseqs=np.asarray(resseqs, dtype=int),
        charges=np.asarray(charges, dtype=float),
        atom_types=np.asarray(elements),
        path=path,
    )


def iter_pdb_models(path: str | Path):
    """Yield ``(model_index, Molecule)`` for each MODEL/ENDMOL block (GPDOCK output)."""
    path = Path(path)
    model_i = 0
    rows: list[tuple] = []
    in_model = False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("MODEL"):
                in_model = True
                rows = []
                continue
            if line.startswith("ENDMOL") or line.startswith("ENDMDL"):
                if rows:
                    model_i += 1
                    yield model_i, molecule_from_atom_rows(rows, f"{path.stem}_m{model_i}", path)
                rows = []
                in_model = False
                continue
            atom = _parse_pdb_atom_line(line)
            if atom is None:
                continue
            if not in_model:
                in_model = True
            rows.append(atom)
        if rows:
            model_i += 1
            yield model_i, molecule_from_atom_rows(rows, f"{path.stem}_m{model_i}", path)


def read_pdb(path: str | Path) -> Molecule:
    path = Path(path)
    xyz, elements, names, resnames, resseqs, charges, types = [], [], [], [], [], [], []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            atom = _parse_pdb_atom_line(line)
            if atom is None:
                continue
            x, y, z, name, resn, resi, charge, el = atom
            xyz.append((x, y, z))
            elements.append(el)
            names.append(name)
            resnames.append(resn)
            resseqs.append(resi)
            charges.append(charge)
            types.append(el)
    if not xyz:
        raise ValueError(f"No ATOM/HETATM records in {path}")
    return Molecule(
        name=path.stem,
        xyz=np.asarray(xyz, dtype=float),
        elements=np.asarray(elements),
        names=np.asarray(names),
        resnames=np.asarray(resnames),
        resseqs=np.asarray(resseqs, dtype=int),
        charges=np.asarray(charges, dtype=float),
        atom_types=np.asarray(types),
        path=path,
    )


def read_mol2(path: str | Path) -> Molecule:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    xyz, elements, names, resnames, resseqs, charges, types = [], [], [], [], [], [], []
    bonds: list[tuple[int, int]] = []
    section = None
    for line in text:
        s = line.strip()
        if s.startswith("@<TRIPOS>"):
            section = s.replace("@<TRIPOS>", "").upper()
            continue
        if not s or s.startswith("#"):
            continue
        if section == "ATOM":
            parts = s.split()
            if len(parts) < 6:
                continue
            names.append(parts[1])
            xyz.append((float(parts[2]), float(parts[3]), float(parts[4])))
            sybyl = parts[5]
            types.append(sybyl)
            el = sybyl.split(".")[0]
            elements.append(el[0].upper() + el[1:].lower() if len(el) > 1 else el.upper())
            resnames.append(parts[7] if len(parts) > 7 else "LIG")
            resseqs.append(int(parts[6]) if len(parts) > 6 and parts[6].isdigit() else 1)
            charges.append(float(parts[8]) if len(parts) > 8 else 0.0)
        elif section == "BOND":
            parts = s.split()
            if len(parts) >= 4:
                bonds.append((int(parts[1]) - 1, int(parts[2]) - 1))
    if not xyz:
        raise ValueError(f"No atoms parsed from MOL2 {path}")
    return Molecule(
        name=path.stem,
        xyz=np.asarray(xyz, dtype=float),
        elements=np.asarray(elements),
        names=np.asarray(names),
        resnames=np.asarray(resnames),
        resseqs=np.asarray(resseqs, dtype=int),
        charges=np.asarray(charges, dtype=float),
        atom_types=np.asarray(types),
        path=path,
        bonds=bonds,
    )


def molecule_from_rdkit(mol, name: str, path: Path | None = None) -> Molecule:
    """Convert an RDKit mol with a conformer into a MetGeoDock Molecule."""
    if mol is None:
        raise ValueError("empty RDKit molecule")
    if mol.GetNumConformers() < 1:
        raise ValueError(f"SDF record {name} has no 3D coordinates")
    conf = mol.GetConformer()
    xyz, elements, names, resnames, resseqs, charges, types = [], [], [], [], [], [], []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        el = atom.GetSymbol()
        xyz.append((float(pos.x), float(pos.y), float(pos.z)))
        elements.append(el)
        names.append(atom.GetSymbol() + str(atom.GetIdx() + 1))
        resnames.append("LIG")
        resseqs.append(1)
        charges.append(float(atom.GetFormalCharge()))
        types.append(el)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    try:
        from .donor_chem import classify_donors

        donor_groups = classify_donors(mol)
    except Exception:  # chemistry layer must never break ligand loading
        donor_groups = None
    return Molecule(
        name=name,
        xyz=np.asarray(xyz, dtype=float),
        elements=np.asarray(elements),
        names=np.asarray(names),
        resnames=np.asarray(resnames),
        resseqs=np.asarray(resseqs, dtype=int),
        charges=np.asarray(charges, dtype=float),
        atom_types=np.asarray(types),
        path=path,
        bonds=bonds,
        donor_groups=donor_groups,
    )


def iter_sdf_molecules(path: str | Path):
    """Yield ``(ligand_id, Molecule)`` for every record in an SDF."""
    path = Path(path)
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover
        raise ImportError('RDKit is required to read SDF files. Install it with: pip install metgeodock[chem]') from exc
    suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    used: set[str] = set()
    n = 0
    for i, mol in enumerate(suppl, start=1):
        if mol is None:
            continue
        n += 1
        raw = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
        lig_id = raw or f"{path.stem}_{i}"
        if lig_id in used:
            lig_id = f"{lig_id}_{i}"
        used.add(lig_id)
        yield lig_id, molecule_from_rdkit(mol, lig_id, path)
    if n == 0:
        raise ValueError(f"No molecule parsed from SDF {path}")


def read_sdf(path: str | Path) -> Molecule:
    """Read the first molecule from an SDF via RDKit (coordinates required)."""
    path = Path(path)
    first = next(iter_sdf_molecules(path), None)
    if first is None:
        raise ValueError(f"No molecule parsed from SDF {path}")
    return first[1]


def load_molecule(path: str | Path) -> Molecule:
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".mol2":
        return read_mol2(path)
    if suf == ".sdf":
        return read_sdf(path)
    if suf in {".pdb", ".ent"}:
        return read_pdb(path)
    raise ValueError(f"Unsupported molecule format: {path}")


def write_pdb(mol: Molecule, path: str | Path, record: str = "HETATM") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (xyz, el, name, resn, resi) in enumerate(
        zip(mol.xyz, mol.elements, mol.names, mol.resnames, mol.resseqs), start=1
    ):
        aname = str(name)[:4].ljust(4)
        lines.append(
            f"{record:<6}{i:5d} {aname} {str(resn)[:3]:>3} A{int(resi):4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {str(el)[:2]:>2}\n"
        )
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")
