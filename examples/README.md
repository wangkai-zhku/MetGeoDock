# MetGeoDock examples

| path | purpose | runtime |
|---|---|---|
| `minimal/` | redock one Zn complex with a reference pose (PDB 8jao) | ~1 min |
| `screening_hdac8/` | screen 20 molecules (8 known inhibitors + 12 decoys) against HDAC8 | ~10 min on one core |

```bash
bash examples/minimal/run.sh              # single complex, with RMSD to the reference
bash examples/screening_hdac8/run.sh      # small library screen, ranked output
```

`screening_hdac8/` ships everything needed: the open HDAC8 receptor (PDB 2V5W), the 20-molecule SDF
and a `labels.csv` so you can check where the known inhibitors land. HDAC8 is the best-performing
target of the 22-target benchmark (AUROC 0.747), so it is a fair first test.
