# Benchmark set: 22 metalloprotein targets

Everything needed to re-run the screening benchmark in the paper: the targets, the receptor for
each one, the molecules, and the score every molecule received.

| path | contents |
|---|---|
| `targets.csv` | 22 rows: `target`, `metal`, `pdb`, `receptor`, `n_coord`, `geometry`, `n_ligands`, `n_actives`, `n_decoys`, `auroc_metgeo` |
| `ligands.csv` | 23,920 rows: `target`, `ligand_id`, `label` (1 = active), `smiles`, `score_metgeo` |
| `receptors/` | one docking-ready PDB per target, hydrogens added, waters stripped |

## What is in the set

22 targets spanning six metal ions — Zn (12), Fe(II)/Fe(III) (5), Mn (2), and one each of Ni, Co
and Mg. Each carries 100 curated ChEMBL actives and roughly 1,000 property-matched decoys, a 1:10
active-to-decoy ratio. 23,920 of the 24,200 designed molecules carry a score; the missing 280 are
duplicate identifiers and docking failures, all of them decoys.

`score_metgeo` is what `metgeodock screen` writes — **lower is better**. Recomputing AUROC from the
`label` and `score_metgeo` columns reproduces the `auroc_metgeo` column exactly.

## Reproducing one target

```bash
metgeodock screen \
  --protein datasets/bench22/receptors/HDAC8.pdb \
  --library ligands.sdf \
  --metal zn \
  --out scores.csv
```

The scoring includes the hydrophobic, rotation and desolvation terms by default; `--no-chem` gives
the geometric terms alone. Only the default reproduces the values in `ligands.csv`.

For a single molecule with explicit control over the pose count:

```python
from metgeodock.io import read_pdb, molecule_from_rdkit
from metgeodock.dock import dock_ligand

protein = read_pdb("datasets/bench22/receptors/HDAC8.pdb")
result = dock_ligand(protein, ligand, metal="zn", n_spin=8, max_poses=8, include_chem=True)
```

## Receptors

`receptors/<TARGET>.pdb` is the structure the benchmark used, with hydrogens added at pH 7.4 and
crystallographic waters removed. `pdb` in `targets.csv` names the entry it came from, so the
provenance of each one is traceable; where a structure holds several copies of a chain, only the
copy carrying the catalytic metal is kept.

HDAC6 is the tandem CD1–CD2 construct of 5EDU trimmed to the CD2 domain, which is where its
catalytic zinc sits.

## A note on enrichment factors

With 100 actives among ~1,100 molecules (9.2% active), the enrichment factor cannot exceed
`1 / 0.092 = 10.9` at any cut-off. Published EF values from DUD-E-style panels are computed at a
1:50 ratio, where the same ceiling is about 50, so raw EF numbers from the two are not comparable
without dividing by the ceiling first.
