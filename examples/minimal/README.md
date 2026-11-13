# Minimal redock (PDB 8jao, Zn)

Quick install check (~1 min). A public Zn complex with a reference pose, so the run also reports
RMSD to the crystal ligand.

```bash
bash examples/minimal/run.sh
# or:
metgeodock dock \
  --protein examples/minimal/protein.pdb \
  --ligand examples/minimal/ligand.pdb \
  --metal zn \
  --coord-water 2 --n-spin 8 --max-poses 20 \
  --reference examples/minimal/ligand_ref.pdb \
  --out processed/dock/example_minimal --write-poses
```

For the paper VS protocol, use ``examples/agri_vs_168/``.
