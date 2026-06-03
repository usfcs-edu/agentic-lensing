# Dawes 2022/2023 (ApJS 269:61) reproduction — multiply lensed & binary quasar search

Algorithmic reproduction of the autocorrelation candidate-search pipeline in
Dawes, Huang, Storfer, et al. 2023, *Finding Multiply Lensed and Binary Quasars
in the DESI Legacy Imaging Surveys* (ApJS 269:61). Visual-inspection grading is
**out of scope** (handled the same way as the hsu-2025 reproduction).

## The algorithm (paper Section 3)

"Autocorrelation" = group quasar targets within an angular separation cut and
recursively link overlapping pairs into systems (doubles -> triples -> quads).
This is exactly **Friends-of-Friends (FoF)** with a fixed angular linking
length. Dawes first search a **10″** radius, then halve to **<5″** because
>95% of the "discoverable known" systems have images separated by <5″. They run
on the **DESI Quasar Sample (~5M photometric, RF-selected QSO targets)** over
19,000 deg². They report **100% recovery** of the **94 "discoverable known"**
systems (the ~30% of ~300 known lenses with ≥2 objects in the QSO sample), and
**436 new candidates** (102 A + 118 B + 216 C; published Type col: 432 Double +
4 Quad; text: 24 quads identified by the algorithm). Gaia EDR3 cuts
**PMSIG < 8** and **PXSIG < 3.5** reject Milky-Way stellar contaminants.

## Proxy caveat (important)

Dawes' ~5M **photometric** QSO target catalog is **not on disk**. We use the
most directly available proxy: the **DESI DR1 spectroscopic QSO sample** from
`zall-pix-iron.fits` (`SPECTYPE=='QSO' & ZWARN==0 & ZCAT_PRIMARY`) =
**1,645,843 spectra**. This is a related but **not identical** sample: ~1/3 the
size and selected differently. Most published candidate *images* are faint,
blended lensed images that DESI never spectroscopically confirmed as QSOs — only
**138/436** published candidates have **any** DR1 QSO within 5″, and only
**58/436** have **≥2 distinct** DR1 QSOs (i.e. are "discoverable in our proxy").
Absolute candidate counts therefore cannot match 436; the **algorithm** and the
**recovery of discoverable systems** are what we reproduce faithfully.

## Pipeline

| script | what |
|---|---|
| `00_fetch_published_catalog.py` | download Dawes Table 2 from VizieR `J/ApJS/269/61` (875 rows → 436 systems) → `data/dawes2023_vizier_table2.csv` |
| `01_build_qso_sample_and_fof.py` | build DR1 QSO sample, spherimatch FoF at 5″ and 10″ → `data/qso_groups_{5,10}arcsec.parquet`, `data/fof_stats.json` |
| `02_recover_published_candidates.py` | nearest-group recovery of the 436 published positions → `data/recovery.json` |
| `03_gaia_pm_px_cuts.py` | Gaia EDR3 PMSIG/PXSIG stellar rejection on resolved candidate images → `data/gaia_matches.parquet`, `data/gaia_cuts.json` (LONG; checkpointed/resumable) |
| `04_make_figures.py` | separation histogram + recovery-offset histogram → `figs/` |
| `05_refine_candidates.py` | collapse spectroscopic duplicates, keep resolved (0.3″≤sep≤5″) candidate systems → `data/resolved_candidates.parquet`, `data/refined_stats.json` |
| `06_recovery_ceiling.py` | proxy-aware **conditional** recovery (analog of 94/94) → `data/recovery_ceiling.json` |

venv: `/home/benson/.venvs/hsu/bin/python` (spherimatch, astropy, astroquery, pdfminer.six).

## Results vs published

| metric | published (Dawes) | ours (DR1 QSO proxy) |
|---|---|---|
| QSO sample | ~5,000,000 photometric targets | 1,645,843 spectroscopic QSOs |
| FoF groups @5″ (raw) | — | 10,917 |
| FoF groups @10″ (raw) | — | 12,896 |
| recommendations @10″ | >27,000 quasar targets | 25,962 spectra |
| recommendations @5″ | ~6,000 quasar targets | 21,965 spectra |
| **resolved** candidate systems @5″ (0.3–5″) | 436 candidates | 1,548 |
| double : (triple+quad) ratio (resolved) | ~21:1 (text: 24 quads) | 1529 : 19 ≈ 80:1 |
| **raw recovery of 436** (centroid <3″) | n/a | **61/436 = 14.0%** (proxy-limited) |
| **conditional recovery** (≥2 distinct DR1 QSOs) | **94/94 = 100%** (their sample) | **58/58 = 100.0%** |
| median offset of recovered | — | 0.139″ |
| Gaia info for ≥1 image | 380/436 ≈ 87% | see `data/gaia_cuts.json` |

**Headline:** every published candidate that is *discoverable in our proxy*
(has ≥2 distinct DR1 spectroscopic QSOs within 5″) is correctly grouped by our
FoF implementation — **58/58 = 100%**, the apples-to-apples analog of the
paper's 94/94 = 100%. Recovered matches sit a median **0.139″** from the
published centroid. The raw 14% and the discrepancies in absolute counts /
double:quad ratio are driven entirely by the proxy sample (spectroscopic, ~1/3
size, different selection), not by the algorithm.

### Notes on the count discrepancies
- The raw FoF groups are dominated by pairs at ~0″ separation: distinct
  `TARGETID`s at the *same* sky position (the same physical QSO observed under
  multiple DESI surveys/programs, or Tractor splits). Dawes' photometric target
  catalog has one detection per source, so they never see this pile-up.
  `05_refine_candidates.py` collapses these (dedup <0.2″, require 0.3–5″
  resolved separation), yielding 1,548 resolved systems.
- The double:quad ratio cannot match because spectroscopic confirmation of
  *both/all* faint images of a quad is rare in DR1: we find only 1 quad among
  resolved systems, vs the paper's 24 (mostly Tractor-deblended from photometric
  doubles, which we have no access to).
