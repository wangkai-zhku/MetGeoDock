# Data index

Two kinds of data ship with this repository.

## 1. Benchmark set (`datasets/bench22/`)

| file | contents |
|---|---|
| `targets.csv` | 22 metalloprotein targets: metal ion, PDB entry, coordination number, geometry, ligand counts, per-target AUROC |
| `ligands.csv` | 23,920 molecules: target, ligand id, active/decoy label, SMILES, MetGeo score |

Receptors are identified by PDB entry. Download the entry, then run

```bash
metgeodock prepare --protein 1T64.pdb --out receptor_h.pdb
```

## 2. Example inputs (`examples/`)

| path | contents |
|---|---|
| `examples/minimal/` | one Zn complex (8jao): receptor, ligand and reference pose |
| `examples/screening_hdac8/` | HDAC8 receptor (2V5W, chain A, hydrogens added) and a 20-molecule library |

## What is not distributed

Raw crystal collections, decoy pools and intermediate docking output stay out of the repository —
they are large and, in the case of third-party structures, not ours to redistribute. Every number in
the paper can be regenerated from the files above plus the source structures they name.
