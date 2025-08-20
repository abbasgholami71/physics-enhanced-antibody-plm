# Physics-Enhanced Antibody PLM
Framework enhancing protein language models with physics-based simulations for improved biomolecular interaction prediction. This repository integrates state-of-the-art protein language models (ESM-2, AntiBERTy), structure prediction (AlphaFold, Boltz-2), and molecular dynamics (GROMACS) to analyze the interface of the SARS-CoV-2 RBD and the CR3022 antibody.

Badges:
- ![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
- ![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange.svg)
- ![GROMACS](https://img.shields.io/badge/GROMACS-2021%2B-336699.svg)

## Highlights
- Protein language models:
  - ESM-2 and AntiBERTy embeddings and interface propensity analyses
- Structure modeling:
  - Complex and component modeling using AlphaFold and Boltz-2
- Interface analysis:
  - Residue-level contacts, hydrogen bonds, SASA/BSA, interface maps, and scoring
- Molecular dynamics:
  - Clean, reproducible GROMACS workflow for pre-processing, running, and post-processing MD
- Reproducibility:
  - Scripted analyses and organized inputs/outputs for end-to-end reproducibility

## Repository structure
```
physics-enhanced-antibody-plm/
├─ structure/     # Structure files (PDB) for antibody, RBD, and complexes
├─ data/          # Sequences (FASTA/CSV/JSON) for antibody, RBD, and complexes
├─ scripts/       # PLM utilities, structural interface analyses, and MD analytics
└─ simulation/    # MD inputs (topologies, parameters) and workflow scripts
```

- structure/
  - Contains PDB files for isolated chains (antibody heavy/light, RBD) and modeled complexes
  - Suggested naming: antibody_(H|L).pdb, rbd.pdb, complex_*.pdb
- data/
  - Contains sequences (FASTA or simple text) for antibody chains, RBD, and complex metadata
- scripts/
  - Utilities to run protein language models (ESM-2, AntiBERTy)
  - Interface residue detection, contact maps, and structural comparison
  - MD trajectory analysis (RMSD/RMSF, H-bonds, SASA/BSA, contacts, clustering)
- simulation/
  - GROMACS-ready topology, parameter files, and a clean workflow for pre-, run-, and post-processing

## Getting started

### Prerequisites
- Python 3.9+
- Conda or venv for environment management
- GROMACS 2021+ (compiled with double or single precision consistently)
- Optional: CUDA-enabled PyTorch for accelerated PLM inference

Recommended Python packages (install via conda or pip):
- pytorch (CPU or CUDA), torchvision, torchaudio
- biopython, mdanalysis, mdtraj
- numpy, scipy, pandas, scikit-learn
- matplotlib, seaborn
- tqdm, click, pyyaml
- fair-esm (for ESM-2)
- transformers (for AntiBERTy via Hugging Face)
- pdb-tools (optional utilities)

Example environment setup:
```bash
# Create and activate environment
conda create -n pe-aplm python=3.10 -y
conda activate pe-aplm

# Install core scientific stack
conda install -c conda-forge numpy scipy pandas scikit-learn matplotlib seaborn tqdm biopython -y
conda install -c conda-forge mdanalysis mdtraj -y

# Install PyTorch (choose CUDA/CPU build as appropriate)
# See https://pytorch.org/get-started/locally/ for the exact command for your system
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install PLM toolkits
pip install fair-esm transformers

# Optional utilities
pip install pdb-tools pyyaml click
```

GROMACS installation:
- Install system-wide or via conda-forge:
```bash
conda install -c conda-forge gromacs -y
```

Note: If you rely on GPU-accelerated MD or specific force fields, prefer a system or module installation that matches your hardware.

## Typical workflow

1) Prepare sequences
- Place antibody chain sequences and RBD sequence files in data/.
- Use FASTA or plain text; ensure headers and chain identifiers are clear.

2) Run protein language models
- Use scripts in scripts/ to compute embeddings and residue-level interface propensities from:
  - ESM-2: embeddings, attention-based contact/propensity features
  - AntiBERTy: antibody-specific features for paratope propensity
- Output tips:
  - Save per-residue CSV/NPZ with positions, chain IDs, embeddings, and scores under data/ or a results/ subfolder you create.

3) Structural modeling
- AlphaFold/ColabFold: Generate complex or separated chain models; place resulting PDBs under structure/.
- Boltz-2: Generate or refine conformations as applicable; keep variants named clearly (e.g., complex_af2_v1.pdb, complex_boltz2_v1.pdb).

4) Interface analysis
- Use scripts in scripts/ to:
  - Define interface residues via distance cutoff or buried surface area (BSA)
  - Compute contacts (heavy atom or sidechain), hydrogen bonds, salt bridges
  - Compute SASA and BSA per chain; generate contact maps and interface summaries
- Save outputs as CSV and figures (PNG/SVG) in an analysis/ folder (recommendation).

5) Molecular dynamics (GROMACS)
- The simulation/ folder provides a clean workflow for:
  - Pre-processing: structure cleaning, protonation, box/solvation, ion addition, topology generation
  - Energy minimization and equilibration (NVT/NPT)
  - Production runs
  - Post-processing analysis (RMSD, RMSF, H-bonds, SASA/BSA, contacts, clustering)

Example GROMACS command sequence (illustrative):
```bash
# Pre-processing
gmx pdb2gmx -f structure/complex_af2_v1.pdb -o sim/processed.gro -p sim/topol.top -i sim/posre.itp
gmx editconf -f sim/processed.gro -o sim/boxed.gro -c -d 1.0 -bt cubic
gmx solvate -cp sim/boxed.gro -cs spc216.gro -p sim/topol.top -o sim/solv.gro
gmx grompp -f simulation/minim.mdp -c sim/solv.gro -p sim/topol.top -o sim/em.tpr
gmx mdrun -deffnm sim/em

# Equilibration (NVT then NPT)
gmx grompp -f simulation/nvt.mdp -c sim/em.gro -p sim/topol.top -o sim/nvt.tpr
gmx mdrun -deffnm sim/nvt
gmx grompp -f simulation/npt.mdp -c sim/nvt.gro -p sim/topol.top -o sim/npt.tpr
gmx mdrun -deffnm sim/npt

# Production
gmx grompp -f simulation/md.mdp -c sim/npt.gro -p sim/topol.top -o sim/md.tpr
gmx mdrun -deffnm sim/md

# Post-processing examples
gmx rms -s sim/md.tpr -f sim/md.xtc -o analysis/rmsd.xvg
gmx rmsf -s sim/md.tpr -f sim/md.xtc -o analysis/rmsf.xvg -res
```

6) Cross-link PLM and physics signals
- Compare PLM-predicted paratope/epitope/interface propensities with:
  - Structural interface residues from modeling and MD-averaged contacts
  - Stability metrics (RMSF, H-bonds) around the interface
- Summarize alignment/misalignment to guide sequence or structure iterations.

## Reproducibility tips
- Seed PLM sampling and any stochastic steps where applicable
- Record software versions (Python, PyTorch, GROMACS, force fields)
- Keep a YAML or JSON config per experiment (input sequences, PLM model/size, structure model variant, MD settings)
- Save intermediate artifacts (embeddings, selections, masks) and final outputs

## Results and artifacts
- Consider organizing outputs as:
```
results/
  plm/
    esm2_embeddings/
    antiberty_scores/
  structure/
    interface_summaries.csv
    contact_maps/
  md/
    run_001/
      analysis/
        rmsd.csv
        rmsf.csv
        hbonds.csv
        sasa_bsa.csv
        interface_contacts.csv
```
- Figures: store in results/*/figures with clear filenames and captions in a RESULTS.md if desired.

## Data conventions
- Chain IDs: use consistent chain labels (e.g., H/L for antibody heavy/light; R for RBD)
- Numbering: align residue numbering across sequence, structure, and MD
- Units: distances in Å, time in ns, temperature in K

## Citations and references
If you use this repository, please also cite the relevant tools:
- ESM-2: Lin et al., and the facebookresearch/esm project
- AntiBERTy: antibody-specific protein language model (cite the original AntiBERTy paper and model repository)
- AlphaFold/ColabFold: Jumper et al., Nature 2021; Mirdita et al., Nat Methods 2022 (for ColabFold)
- GROMACS: Abraham et al., SoftwareX 2015 (or the latest GROMACS reference)
- Any force fields used (e.g., AMBER, CHARMM)

Add exact references and DOIs according to what you used in your experiments.

## Contributing
- Issues and pull requests are welcome.
- For substantial changes, please open an issue to discuss what you would like to change.
- Add tests or example data where appropriate, and document new scripts in scripts/.

## License
Specify your license (e.g., MIT, Apache-2.0) in a LICENSE file.

## Contact
- Maintainer: @abbasgholami71
- For questions or collaboration inquiries, please open an issue in this repository.

## Acknowledgments
- Thanks to the authors and maintainers of ESM-2, AntiBERTy, AlphaFold/ColabFold, and GROMACS.
- This work builds on the broader community’s efforts in protein ML and molecular simulation.

---
Tip: If your structures or trajectories are large, consider using Git LFS and/or publishing processed data and checkpoints as releases.
