# HDAC8 screening example

A 20-molecule library (8 known HDAC8 inhibitors + 12 decoys) screened against the open,
substrate-bound HDAC8 receptor (PDB 2V5W, Zn first shell intact).

```bash
bash examples/screening_hdac8/run.sh
```

Runtime: about 10 min on one CPU core. The script writes `results/example_hdac8_scores.csv`;
sort it by `score` (lower is better) and compare the `ligand_id` column against `labels.csv`.

Expected result: 6 of the 8 known inhibitors land in the top 10, the first of them at rank 1.
HDAC8 is the best-performing target of the 22-target benchmark (AUROC 0.747), so this is the
most favourable place to start.

Files:

| file | contents |
|---|---|
| `receptor.pdb` | HDAC8, 2V5W chain A, hydrogens added, waters stripped |
| `library.sdf` | 8 ChEMBL inhibitors (`HDAC8_A*`) + 12 decoys (`decoy_*`) |
| `labels.csv` | ground truth for the 20 molecules |
| `run.sh` | end-to-end screen |
