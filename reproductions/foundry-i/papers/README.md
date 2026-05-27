# Foundry I (Phase 1) Internal Reproduction Report

LaTeX source for the Phase 1 internal tech-report "Public-Data Reproduction
of GIGA-Lens Modeling for the DESI Strong Lens Foundry Demonstration System
DESI-165.4754-06.0423" (Benson & Huang, May 2026).

Built against the shared template at `reproductions/tech-report.sty`. Same
build flow as the Phase 2 (Hsu 2025) and Phase 3a (Huang 2020) reports.

## Directory contents

| File / dir       | Purpose |
| ---------------- | ------- |
| `main.tex`       | Single-column `article` LaTeX source. Uses `../../tech-report.sty`. |
| `references.bib` | BibTeX entries (each verified against arXiv / ADS). A few entries carry inline `% TODO verify …` comments where a Bibcode/DOI could not be confirmed; address these if you ever promote the report to a journal submission. |
| `Makefile`       | `make pdf` builds `main.pdf` via `pdflatex`/`bibtex` 4-pass; `make clean` removes aux artefacts; `make distclean` removes the PDF too. |
| `figures/`       | PNGs referenced by `\includegraphics` in `main.tex`. |

## Build

```bash
cd /raid/benson/git/agentic-lensing/reproductions/foundry-i/papers
make pdf
```

Works with any TeX Live ≥ 2022. No special class file needed (we switched
off `aastex701` to plain `article`), no TL2026 PATH pin in the Makefile.

## What's in the report

A reproduction of Huang et al. 2025a, *DESI Strong Lens Foundry I: HST
Observations and Modeling with GIGA-Lens* (arXiv:2502.03455). The
reproduction itself was carried out in the parent directory
(`reproductions/foundry-i/`); the scripts are `01_download_hst.py`
through `25_fit_nuts_v11f.py`. The headline result is a 10,000-sample
Gaussian variational posterior in `data/svi_v10_posterior_mass.npz`
that recovers all six mass-parameter sign quadrants of Huang+2025a
Table 3, matches `theta_E` to 3.0%, `e1` to 2.5%, and the external
shear position angle to within 1 degree. A v11 NUTS chain
(`data/nuts_v11f_posterior_mass.npz`) provides a second-mode point
estimate that matches `theta_E` to 1.6%.

## History

The earlier version of this report (`commit 851ce04` through `bd74c8a`)
used two-column AASTeX 7.0.1 with `\deluxetable*` tables and
`aasjournalv7` bibliography style. It was converted to the
`reproductions/tech-report.sty` template in May 2026 (`commit` TBD)
as part of standardising the Phase 1/2/3a internal-report format.
The AASTeX source is recoverable via `git log -- main.tex` if anyone
ever wants to revive the journal-submission format.

## Reproducing the numerical results

The report's Table 1 numbers are produced by:

```python
import numpy as np
d = np.load('../data/svi_v10_posterior_mass.npz')
for k in ('theta_E', 'gamma', 'e1', 'e2', 'gamma1', 'gamma2'):
    a = d[k]
    print(f"{k}: median={np.median(a):+.4f}  "
          f"16/84%=({np.percentile(a, 16):+.4f}, {np.percentile(a, 84):+.4f})")
```

This is the only computation performed by the writeup itself; the
heavy lifting (200-chain multi-start MAP + 200-particle SVI on 2× L4
GPUs) was done by the upstream fit scripts.
