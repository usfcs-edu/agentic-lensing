# Foundry I Reproduction Paper

LaTeX source for the manuscript "Public-Data Reproduction of GIGA-Lens
Modeling for the DESI Strong Lens Foundry Demonstration System
DESI-165.4754-06.0423" (Benson & Huang, in preparation; to be submitted to ApJ).

## Directory contents

| File / dir          | Purpose |
| ------------------- | ------- |
| `main.tex`          | AASTeX 6.3.1 two-column manuscript source. |
| `references.bib`    | BibTeX entries (each verified against arXiv / ADS). A few entries carry inline `% TODO verify …` comments where a Bibcode/DOI could not be confirmed via the web; address these before journal submission. |
| `Makefile`          | `make pdf` builds `main.pdf`; `make clean` removes auxiliary artifacts; `make distclean` removes the PDF too. Uses `latexmk` if available, otherwise falls back to a `pdflatex`/`bibtex`/`pdflatex`/`pdflatex` 4-pass. |
| `figures/`          | Symlink to `../figs/` (the reproduction's PNG output directory). Contains `cutout_preview.png`, `psf_comparison.png`, `nearby_galaxy_detection.png`, `map_v2_residual.png`, `map_v4_residual.png`, `map_residual.png`. |

## Build

```bash
cd /raid/benson/git/agentic-lensing/reproductions/foundry-i/papers
make pdf
```

The build expects `aastex631.cls` to be installed. If you see
`! LaTeX Error: File 'aastex631.cls' not found`, install the AAS
publisher class:

- Debian / Ubuntu: `sudo apt-get install texlive-publishers`
- Fedora / RHEL / AlmaLinux: `sudo dnf install texlive-collection-publishers`
- macOS (MacTeX): `sudo tlmgr install aastex`
- TeX Live manual: `tlmgr install aastex`

If `latexmk` is also unavailable, the Makefile will fall back to the
classic 4-pass build using only `pdflatex` and `bibtex`.

## What's in the paper

The paper is a reproduction of Huang et al. 2025a, "DESI Strong Lens
Foundry I: HST Observations and Modeling with GIGA-Lens"
(arXiv:2502.03455). The reproduction itself was carried out in the
parent directory (`reproductions/foundry-i/`); the scripts are
`01_download_hst.py` through `19_svi_v10_paper_mode_empirical.py`,
and the headline mass-parameter posterior is in
`data/svi_v10_posterior_mass.npz` (a 10,000-sample variational
Gaussian posterior). The paper documents the full pipeline,
the v1->v10 internal ablation that led to the working recipe,
four upstream issues filed against gigalens 2.0 / JAX 0.6.2,
and the headline result: all six mass-parameter sign quadrants
match Huang+2025a Table 3, with theta_E reproduced to 3.0% and the
external-shear position angle to within 1 degree.

## Reproducing the numerical results

The paper's Table 1 numbers are produced by:

```python
import numpy as np
d = np.load('../data/svi_v10_posterior_mass.npz')
for k in ('theta_E', 'gamma', 'e1', 'e2', 'gamma1', 'gamma2'):
    a = d[k]
    print(f"{k}: median={np.median(a):+.4f}  "
          f"16/84%=({np.percentile(a, 16):+.4f}, {np.percentile(a, 84):+.4f})")
```

This is the only computation performed by the writeup itself; the
heavy lifting (200-chain multi-start MAP + 200-particle SVI on 2x L4
GPUs) was done by the upstream fit scripts.
