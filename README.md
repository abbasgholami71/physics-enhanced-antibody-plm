# Antibody–RBD Interface Analysis with PLMs, Structure Modeling, and MD

Comparative study of the SARS‑CoV‑2 RBD and the CR3022 antibody interface using protein language models (ESM‑2, AntiBERTy), structure prediction/refinement (AlphaFold, Boltz‑2), and molecular dynamics (GROMACS). This repository evaluates and contrasts predictions across data‑driven models and physics‑based simulations.

Badges:
- ![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
- ![GROMACS](https://img.shields.io/badge/GROMACS-2021%2B-336699.svg)

## Highlights
- PLM-based interface propensities using:
  - ESM‑2 (general protein LM)
  - AntiBERTy (antibody-focused LM)
- Structure modeling and selection using:
  - AlphaFold/ColabFold and Boltz‑2
- Interface characterization:
  - Contacts, H‑bonds, SASA/BSA, residue‑level maps and summaries
- Molecular dynamics:
  - Clean, reproducible GROMACS pre‑processing → equilibration → production → analyses
- End-to-end comparison:
  - Contrast PLM‑derived insights against modeled structures and MD‑averaged interface behavior

## Repository structure
```
physics-enhanced-antibody-plm/
├─ structure/     # PDB files for antibody (H/L), RBD, and modeled complexes
├─ data/          # Sequences for antibody, RBD, and complex metadata
├─ scripts/       # PLM utilities, interface analyses, and MD post-processing
└─ simulation/    # GROMACS inputs and a clean MD workflow
```

- structure/
  - Example naming: antibody_H.pdb, antibody_L.pdb, rbd.pdb, complex_af2_v1.pdb, complex_boltz2_v1.pdb
- data/
  - Sequences (FASTA or txt) for antibody chains and RBD; optional complex or mapping files
- scripts/
  - Notebooks and/or Python scripts to run PLMs (ESM‑2, AntiBERTy), detect interface residues, compute contact/H‑bond/BSA metrics, and analyze MD trajectories (RMSD/RMSF, SASA, clustering)
- simulation/
  - GROMACS topology/parameter files (.top, .itp, .mdp) and a streamlined workflow for pre‑, run‑, and post‑processing

## Getting started

### Prerequisites
- Python 3.9+
- JupyterLab/Notebook
- GROMACS 2021+ (ensure a consistent precision build)
- Optional: CUDA‑enabled PyTorch for faster PLM inference

Recommended Python packages (conda or pip):
- Core: numpy, scipy, pandas, matplotlib, seaborn, tqdm, pyyaml, click
- Bio/simulation I/O: biopython, MDAnalysis, mdtraj
- ML: torch (CPU or CUDA build), fair‑esm (ESM‑2), transformers (AntiBERTy via HF)
- Utilities: pdb‑tools (optional)

Example environment setup:
```bash
# Create and activate environment
conda create -n rbd-cr3022 python=3.10 -y
conda activate rbd-cr3022

# Scientific Python + bio/sim I/O
conda install -c conda-forge numpy scipy pandas matplotlib seaborn tqdm biopython -y
conda install -c conda-forge mdanalysis mdtraj -y

# PyTorch (choose CPU/GPU per your system)
# See https://pytorch.org/get-started/locally/ for the right command; CPU example:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# PLMs
pip install fair-esm transformers

# Optional helpers
pip install pdb-tools pyyaml click
```

GROMACS installation (conda-forge example):
```bash
conda install -c conda-forge gromacs -y
```

Note: For GPU‑accelerated MD, prefer a system/module install matched to your drivers.

## Typical workflow

1) Prepare sequences
- Place the antibody heavy/light and RBD sequences in data/ (FASTA or txt).
- Keep chain labels consistent (e.g., H, L for antibody; R for RBD).

2) Run protein language models (PLMs)
- Use notebooks/scripts in scripts/ to compute residue‑level features:
  - ESM‑2 embeddings and attention/propensity features
  - AntiBERTy paratope/interface propensities for antibody chains
- Save outputs (e.g., CSV/NPZ) with residue indices, chain IDs, and scores.

3) Structure modeling
- AlphaFold/ColabFold: generate complex or separate chain models; save PDBs to structure/.
- Boltz‑2: generate/refine alternative conformations; name clearly (e.g., complex_boltz2_v1.pdb).

4) Interface analysis (static structures)
- Identify interface residues by distance cutoff and/or buried surface area (BSA).
- Compute:
  - Contacts (heavy atom/side‑chain)
  - Hydrogen bonds and salt bridges
  - SASA and BSA per chain/residue
  - Contact/score maps and summary tables
- Store results as CSV and figures (PNG/SVG), e.g., results/structure/.

5) Molecular dynamics (GROMACS)
- Use simulation/ workflow for:
  - Pre‑processing: cleanup, protonation, box/solvation, ion addition, topology
  - Energy minimization and equilibration (NVT → NPT)
  - Production run
  - Post‑processing: RMSD/RMSF, H‑bonds, SASA/BSA, interface contacts, clustering

Illustrative GROMACS sequence:
```bash
# Pre-processing
gmx pdb2gmx -f structure/complex_af2_v1.pdb -o sim/processed.gro -p sim/topol.top -i sim/posre.itp
gmx editconf -f sim/processed.gro -o sim/boxed.gro -c -d 1.0 -bt cubic
gmx solvate  -cp sim/boxed.gro -cs spc216.gro -p sim/topol.top -o sim/solv.gro
gmx grompp   -f simulation/minim.mdp -c sim/solv.gro -p sim/topol.top -o sim/em.tpr
gmx mdrun    -deffnm sim/em

# Equilibration
gmx grompp -f simulation/nvt.mdp -c sim/em.gro -p sim/topol.top -o sim/nvt.tpr
gmx mdrun  -deffnm sim/nvt
gmx grompp -f simulation/npt.mdp -c sim/nvt.gro -p sim/topol.top -o sim/npt.tpr
gmx mdrun  -deffnm sim/npt

# Production
gmx grompp -f simulation/md.mdp -c sim/npt.gro -p sim/topol.top -o sim/md.tpr
gmx mdrun  -deffnm sim/md
```

6) MD analysis and cross‑comparison
- Extract MD‑averaged interface metrics (contacts, H‑bonds, SASA/BSA) and stability (RMSF).
- Compare PLM propensities vs:
  - Interface residues from structures
  - Contact persistence and dynamics from MD
- Summarize alignment/misalignment to assess predictive value.


## Notes and scope
- This repository is a comparison/benchmark of methods; it does not train or propose a new PLM.
- CR3022–RBD system is used as a concrete case study; scripts are adaptable to other antibody–antigen pairs.


## Contact
- Maintainer: @abbasgholami71
- For questions or collaboration, please open an issue.

## Acknowledgments
Thanks to the authors and maintainers of ESM‑2, AntiBERTy, AlphaFold/ColabFold, GROMACS, and the broader protein ML and simulation communities.
