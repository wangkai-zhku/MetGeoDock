# MetGeoDock

[![PyPI](https://img.shields.io/pypi/v/metgeodock)](https://pypi.org/project/metgeodock/)
[![Python](https://img.shields.io/pypi/pyversions/metgeodock)](https://pypi.org/project/metgeodock/)
[![Downloads](https://img.shields.io/pypi/dm/metgeodock)](https://pypi.org/project/metgeodock/)
[![License](https://img.shields.io/badge/license-academic%20%2F%20non--commercial-blue)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/wangkai-zhku/MetGeoDock?style=social)](https://github.com/wangkai-zhku/MetGeoDock)

Geometry-driven molecular docking and virtual screening for **metalloproteins**. The sampler places
ligand donors on the vacant vertices of the metal coordination sphere instead of searching the whole
pocket, and the scoring function reads coordination geometry directly. Ten ions are supported: Zn,
Ca, Mg, Mn, Fe(II), Fe(III), Cu, Co, Cd and Ni.

```python
from metgeodock import dock_ligand
result = dock_ligand(protein, ligand, metal="zn")
```

## Install

```bash
pip install metgeodock                 # from PyPI; core: numpy, scipy, pandas, pyyaml
pip install "metgeodock[chem]"         # + RDKit (SDF/MOL2 input, donor chemistry)
pip install "metgeodock[chem,ml,prep]" # + torch/scikit-learn, OpenMM (H rebuild)
```

Protein PDB parsing, scoring and PDB-ligand docking work with numpy alone; SDF input raises an
`ImportError` that names the extra to install.

```bash
metgeodock --version
metgeodock dock --help
```

## How it works

| Step | What happens |
|---|---|
| Coordination sphere | metal site detected; first-shell protein donors define the **occupied** directions |
| Vacant vertices | the complement of those directions gives **1–5 open vertices**, set by the detected polyhedron |
| Sampling | ligand donors are matched onto open vertices (Hungarian assignment on unit vectors); coordinating waters optional |
| Scoring | MetGeo geometry term + hydrophobic burial + rotation, contact and desolvation terms, all with physical constants |

Two scorers ship with the package:

| Scorer | What it is | When to use |
|---|---|---|
| `metgeo` (default) | geometric + thermodynamic terms, no weights fitted to activity data | structure docking, pose ranking |
| `coordnet` | small neural network on warhead and ligand descriptors | prioritising a compound library before docking |

Both thermodynamic terms are on by default and cost about 0.22 s per ligand; `--no-thermo` reproduces
the purely geometric score.

## Performance

Virtual screening on a 22-target metalloprotein benchmark (2,200 actives, 22,000 property-matched
decoys):

| Method | mean AUROC | targets above 0.5 |
|---|---|---|
| **MetGeoDock** | **0.573** | **18 / 22** |
| AutoDock Vina | 0.513 | 12 / 22 |
| MetalloDock | 0.503 | 10 / 22 |

MetGeoDock ranks higher than Vina on **16 of 22** targets, and higher than MetalloDock on 16 of 22.
Paired differences: +0.070 [+0.015, +0.120] against MetalloDock (*p* = 0.025) and +0.059
[−0.004, +0.124] against Vina (*p* = 0.074).

Redocking from crystal-quality input, 1,974 distinct PDB entries:

| top-1 | top-3 | top-5 | any sampled pose |
|---|---|---|---|
| 50.5% | 61.9% | 64.9% | 72.8% |

Pose recognition on a 702-pose split (78 native): AUROC **0.919** for the full scorer, 0.826 for a
learned linear combination of the same descriptors, 0.434 for hand-set weights.

## Quick start

```bash
# redock one complex (PDB 8jao, with RMSD to the crystal pose)
bash examples/minimal/run.sh

# screen a 20-molecule library against HDAC8 (~10 min on one core)
bash examples/screening_hdac8/run.sh
```

Both examples ship their own receptor and ligands under `examples/`.

Defaults: `--no-keep-water`, `--coord-water 2`, `--multi-metal`, `--rerank metgeo`. Lower `score` is
better. Use `--top-n 100` to truncate the output.

## Benchmark data

`datasets/bench22/` carries the 22-target screening benchmark: `targets.csv` (metal, PDB entry,
coordination number, per-target AUROC) and `ligands.csv` (23,920 molecules with SMILES, active/decoy
label and MetGeo score). Receptors are named by PDB entry rather than redistributed; `metgeodock
prepare` adds hydrogens once you have them.

## Citation

If you use MetGeoDock, please cite the paper (see `CITATION.cff`). The geometry-matching idea builds
on GM-DockZn (*Bioinformatics* 2020) and GPDOCK (*Brief. Bioinform.* 2023).

## License

**Free for academic and non-commercial research.** Teaching, academic research and non-commercial
evaluation are permitted without fee; redistribution and modification are allowed under the same
terms. **Commercial use requires a separate licence** — contact the authors. See `LICENSE` for the
full text.
