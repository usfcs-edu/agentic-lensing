---
name: reference-paper-corpus
description: "Where Huang's lensing papers and the SpectrumFM DOE Genesis proposal live in Greg Benson's local repo"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 95b34e74-4971-42df-affe-07a9467ad83a
---

Repo root: `/Users/gbenson/sync/research/agentic-lensing/` (git, on `main`).

- **Papers** (16 PDFs): `papers/`
  - Methodology-lineage: `Huang_2020_DECaLS_lenses.pdf`, `Huang_2021_DESI_legacy_lenses.pdf`, `Storfer_2024_DR9_new_lenses.pdf`, `Inchausti_2025_DR10_two_architectures.pdf`, `Gu_2022_GIGA_Lens.pdf`, `Huang_2025a_DESI_Foundry_I.pdf`, `Huang_2025b_DESI_Foundry_II.pdf`
  - Discovery/follow-up: `Sheu_2023_lensed_supernovae.pdf`, `Sheu_2024a_variable_lensed_quasars.pdf`, `Sheu_2024b_carousel_lens.pdf`, `Cikota_2023_Einstein_cross.pdf`, `Dawes_2022_multiply_lensed_quasars.pdf`, `Hsu_2025_pairwise_spectroscopic.pdf`, `Silver_2025_ML_driven_discoveries.pdf`, `Lin_2025_DESI_Foundry_IV.pdf`, `Agarwal_2025_DESI_Foundry_III.pdf`
- **Proposals**: `proposals/doe_genesis_spectrumfm_project_narrative_v7.docx` (the DOE Genesis SpectrumFM project narrative)
- **Plans output**: `plans/` (created May 25 2026 for the onboarding-report deliverable)
- **Scripts**: `scripts/` (build_onboarding_docx.py and related)

To read .docx files, prefer `pandoc <file> -o /tmp/<name>.md` (then Read the markdown) over direct .docx parsing.

External resources to bookmark when working on this codebase:
- Xiaosheng Huang faculty page: https://www.usfca.edu/faculty/xiaosheng-huang
- Personal site: https://highredshift.github.io/
- Neural-lens project page: https://sites.google.com/usfca.edu/neuralens
- DESI data portal: https://data.desi.lbl.gov/

Related: [[project-huang-lensing]], [[project-spectrumfm]].
