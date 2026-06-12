# Foundry I Reproduction Reports

LaTeX sources for the two Foundry-I reproduction reports (Benson & Huang):

- **`main.tex` → `main.pdf` (13 pp)** — the **final-state report** (June
  2026): final posterior vs Huang+2025a Table 3, corrected data/PSF/noise
  treatment, uncertainty budget, open findings (scale bimodality,
  correlated-noise likelihood), and guidance for the gigalens code base.
  This is the document served on the project site.
- **`evolution.tex` → `evolution.pdf` (33 pp)** — the complete
  **development record** (v1 → v13, May–June 2026), preserved as written:
  the HMC investigation, the Perlmutter campaign, the rigor revision, and
  every superseded claim with its retraction.

Built against the shared template at `reproductions/tech-report.sty`. Same
build flow as the Phase 2 (Hsu 2025) and Phase 3a (Huang 2020) reports.

## Directory contents

| File / dir       | Purpose |
| ---------------- | ------- |
| `main.tex`       | Final-state report. Single-column `article`; uses `../../tech-report.sty`. |
| `evolution.tex`  | Full development record (the pre-June-2026 `main.tex`, renamed; history via `git log --follow`). |
| `references.bib` | BibTeX entries shared by both documents (each verified against arXiv / ADS). A few entries carry inline `% TODO verify …` comments where a Bibcode/DOI could not be confirmed; address these if you ever promote a report to a journal submission. |
| `Makefile`       | `make pdf` builds **both** PDFs via `pdflatex`/`bibtex` 4-pass; `make clean` removes aux artefacts; `make distclean` removes the PDFs too. |
| `figures/`       | Symlink to `../figs` — PNGs referenced by `\includegraphics` in both documents. |

## Build

```bash
cd /raid/benson/git/agentic-lensing/reproductions/foundry-i/papers
make pdf
```

Works with any TeX Live ≥ 2022. No special class file needed (we switched
off `aastex701` to plain `article`), no TL2026 PATH pin in the Makefile.

## What's in the final-state report (`main.pdf`)

The end state of the reproduction of Huang et al. 2025a, *DESI Strong Lens
Foundry I: HST Observations and Modeling with GIGA-Lens*
(arXiv:2502.03455). Headline: on the corrected native-scale product
(`data/cutout_v2d.npz`) preconditioned HMC gives **gamma =
1.433 [1.400, 1.468]** (published 1.372 ± 0.023, consistent at <2σ, with
±0.1-class model and scale systematics), theta_E = 2.6551 (0.33% from
published), R̂_max = 1.077, gamma ESS = 5,714, and the published
inner-critical-curve topology recovered. The report also documents the two
load-bearing defects found en route (the gigalens PSF kernel-sampling
convention; wing-contaminated sky noise calibration), seven upstream stack
issues with fixes, and the compute cost (44 of 200 budgeted A100-hours).

The reproduction scripts live in the parent directory
(`reproductions/foundry-i/`): the final-state pipeline is
`40_make_cutout_v2.py`–`46_noise_audit.py` + `slurm/`, against the
vendored `gigalens-sean` `multinode-2025` library in `vendor/`. The
campaign job log is `../PERLMUTTER_CAMPAIGN.md`.

## Reproducing the headline numbers

```python
import json
d = json.load(open('../data/hmc_v13_v2d_diag.json'))['mass_table']
for k, v in d.items():
    print(f"{k}: median={v['median']:+.4f}  [{v['lo']:+.4f}, {v['hi']:+.4f}]  "
          f"Rhat={v['rhat']:.3f}  ESS={v['ess']:.0f}")
```

The posterior itself is `../data/hmc_v13_v2d.npz` (48 chains × 8,000
draws), produced by `43_hmc_paper_scale.py` on one Perlmutter A100 node.

## History

`evolution.tex` is the original report, developed May–June 2026 (and before
that in two-column AASTeX 7.0.1, recoverable via `git log`); it was renamed
from `main.tex` in June 2026 when the final-state report replaced it as the
site-served document. Its "headline" v10 SVI result and the v11 NUTS
second-mode estimate are superseded by the final-state posterior above.
