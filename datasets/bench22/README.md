# Benchmark set: 22 metalloprotein targets

Everything needed to re-run the screening benchmark described in the paper: which targets, which
molecules, and the per-ligand score each one received.

| file | rows | columns |
|---|---|---|
| `targets.csv` | 22 | `target`, `metal`, `pdb`, `n_coord`, `geometry`, `n_ligands`, `n_actives`, `n_decoys`, `auroc_metgeo` |
| `ligands.csv` | 23,920 | `target`, `ligand_id`, `label` (1 = active), `smiles`, `score_metgeo` |

## What is in the set

22 targets spanning six metal ions — Zn (12), Fe(II)/Fe(III) (5), Mn (2), Ni, Co and Mg (one each).
Each target carries 100 curated ChEMBL actives and roughly 1,000 property-matched decoys, giving a
1:10 active-to-decoy ratio. 23,920 of the 24,200 designed molecules carry a score; the remainder are
duplicate ligand identifiers and docking failures, all of them decoys.

`score_metgeo` is the MetGeo score written by `metgeodock screen` — **lower is better**. Recomputing
AUROC from the `label` and `score_metgeo` columns reproduces the per-target values in
`targets.csv`.

## Reproducing the ratio

The same actives support a 1:5 subset by keeping only the first 150 molecules of each target. The
allocation is deterministic (molecules are ordered by ChEMBL identifier), so no random seed is
involved.

## A note on enrichment factors

With 100 actives among ~1,100 molecules (9.2% active), the enrichment factor cannot exceed
`1 / 0.092 = 10.9` at any cut-off. Published EF values from DUD-E-style panels are computed at a
1:50 ratio, where the same ceiling is about 50, so raw EF numbers from the two are not comparable
without dividing by the ceiling first.

## Running the benchmark

```bash
metgeodock screen \
  --protein receptor.pdb \
  --library ligands.sdf \
  --metal zn \
  --out scores.csv
```

Receptors are not redistributed here: `pdb` in `targets.csv` gives the entry to download from the
Protein Data Bank, and `metgeodock prepare` adds hydrogens.
