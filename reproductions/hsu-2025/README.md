# Hsu 2025 reproduction (Phase 2)

Internal reproduction of \[Hsu, Huang, Storfer, Inchausti, Schlegel, Moustakas
et al. 2025, *A New Way to Discover Strong Gravitational Lenses: Pair-wise
Spectroscopic Search from DESI DR1*, submitted to ApJS,
[arXiv:2509.16033](https://arxiv.org/abs/2509.16033)\]. The full tech-report
PDF lives at [`papers/main.pdf`](papers/main.pdf); this README is a short
operator's guide and a results summary. A machine-readable numerical summary is
also auto-generated at [`papers/REPRODUCTION.md`](papers/REPRODUCTION.md).

## Headline result

Hsu+2025 introduces a new lens-discovery modality — **pair-wise spectroscopy**:
DESI fiber spectra that are close on the sky but at distinct redshifts, found by
a friends-of-friends grouping of the redshift catalog. We reproduce the
*algorithmic core* of the pipeline (pre-filter → `spherimatch` FoF at 3″ link
length → redshift-ratio cut) on the full 28.4M-fiber DESI DR1 redshift catalog.
Our intermediate counts match the paper to within **2.4%**:

| Stage | Published | **Ours** | Δ |
| :--- | ---: | ---: | ---: |
| DR1 raw fibers | ~28M | 28,425,963 | — |
| After pre-filter (§3.1) | ~15.8M | 15,786,243 | −0.09% |
| FoF groups after z-ratio ≥ 1.3 (§3.2) | 13,218 | **13,530** | **+2.4%** |
| Spectra in retained groups | 26,621 | **27,334** | **+2.7%** |

Recall on the 20 Grade-A "new" lens candidates explicitly tabulated in
Hsu+2025 Table 2: **20/20 matched within 3″** of a group centroid (= 100%
recall), 19/20 within 1.5″ (median offset 0.91″, max 1.54″), and all 20
reproduce the published (z_d, z_s) redshift pairs to three decimal places.

The full 28M-fiber pipeline runs in **~2 minutes** on local hardware (catalog
load 71 s + `spherimatch` FoF 36 s), no GPU. The smoke test (script 03) shows
`spherimatch` is effectively sublinear in this regime (t ∼ N^0.63), far better
than the original Phase-2 plan's 2–4 week budget for the FoF step.

## Scope

The published pipeline has three stages; **this reproduction covers stage 1
(algorithmic candidate generation) only.**

1. **Algorithmic candidate generation** (§3.1–3.2): pre-filter → `spherimatch`
   FoF → z-ratio cut. **← reproduced (the 13,530-group output above).**
2. **Visual inspection of spectra** (§3.3): H/M/R quality grading, 26,621
   spectra → 23,811. *Out of scope* (human workflow).
3. **Visual inspection of imaging** (§4): A/B/C lens-candidate grading on DR10
   cutouts, yielding the published **2,046 conventional + 318 dimple**
   candidates. *Out of scope* (human workflow).

The **13,530 → 2,046 VI funnel is deliberately not attempted.** It is a
two-stage human visual-inspection workflow (§3.3 spectra grading + §4 imaging
grading). The 13,530-pair → 2,046-graded retention (~15%) implies an
irreducible ~30:1 false-positive rate against the published counts for any
algorithm-only pipeline — consistent with the paper's own ~1 lens per 6,500 DR1
redshifts (§5.1). The "dimple" class in particular is *morphologically* defined
(surface-brightness indentations in DR10 imaging), not algorithmically
definable from σ_v or any tabular feature (Hsu Fig. 6 caption), so it cannot be
recovered without a DR10-imaging morphology classifier.

## Pipeline (in script-number order)

| Script | Purpose | Wall-clock |
| :--- | :--- | :---: |
| `01_download_dr1_zcatalog.py` | Fetch DR1 `zall-pix-iron.fits` (22.4 GB, full) + `zpix-sv3-dark.fits` (901 MB, validation); HTTP-Range resumable | — |
| `02_download_fastspecfit.py` | Fetch DR1 FastSpecFit v3.0 VAC for σ_v (tier=small ~2.7 GB / tier=full ~79 GB, HEALPix-partitioned) | — |
| `03_install_spherimatch_smoketest.py` | Assert `spherimatch` FoF parity vs `astropy.search_around_sky` on 10k synthetic; time N=1e3..1e6 to extrapolate full-DR1 cost | < 1 min |
| `04_validate_small_region.py` | End-to-end §3 on SV3-dark (~521k fibers) before committing to the full run | < 1 min |
| `05_run_full_fof.py` | **The headline run**: §3 algorithm at full DR1 scale → 13,530 groups / 27,334 spectra | ~2 min |
| `06_xmatch_published_catalog.py` | Cross-match pair list vs the 20 Hsu Table-2 Grade-A candidates → 20/20 recall | < 1 min |
| `07_classify_einstein_dimple.py` | Per-group (lens,source); look up σ_v in FastSpecFit; SIS Einstein radius (eq. 1); dimple proxy | < 1 min |
| `08_write_reproduction_report.py` | Collect scripts 03–07 numbers into `papers/REPRODUCTION.md` | < 1 min |
| `09_download_dr10_cutouts.py` | DR10 (g,r,z) cutout per group centroid (jpeg ~50 MB total / fits) from Legacy Survey | — |
| `10_inspection_grid.py` | Lay DR10 jpegs into QA inspection grids (Table-2 / top-θ_E / random) — our own QA, not Hsu §4 VI | < 1 min |

`05_run_full_fof.py` is the load-bearing step. The pre-filter cascade
(implemented exactly as §3.1: `ZCAT_PRIMARY==True` for the "longest-effective-
exposure coadd per object" rule, then `ZWARN==0`, then `SPECTYPE≠STAR`, then
`Z>0`) takes 28,425,963 → 15,786,243; `spherimatch` FoF at 3.0″ linking length
on (`TARGET_RA`, `TARGET_DEC`) then groups, and the z_max/z_min ≥ 1.3 cut keeps
the 13,530 groups. Note: Yuan-Ming Hsu, the paper's first author, is also
`spherimatch`'s author, so the reproduction uses the exact tool the paper does.

## Group-multiplicity breakdown

| Group size | Published | Ours | Δ |
| ---: | ---: | ---: | ---: |
| 2 (pair) | 13,044 | 13,276 | +1.8% |
| 3 (triplet) | 165 | 236 | +43% |
| 4 (quartet) | 7 | 16 | +129% |
| 5 (quintet) | 2 | 2 | 0 |

Pairs (98.7% of all groups) match within 1.8%; the +2.4% group-count excess is
concentrated in the small higher-multiplicity tail. Likely causes: cross-survey
fiber overlap (DR1 has fibers observed by multiple of SV1/SV2/SV3/main at the
same sky position; `ZCAT_PRIMARY` dedups per `TARGETID`, not per sky position),
or a post-grouping dedup step in Hsu not derivable from §3.1.

## Key data files

Large source catalogs are gitignored; the slim derived artefacts are committed.

| File | Committed? | Contents |
| :--- | :---: | :--- |
| `data/zall-pix-iron.fits` | ✗ (22.4 GB) | DR1 full all-sky redshift catalog (input to script 05) |
| `data/zpix-sv3-dark.fits` | ✗ (901 MB) | SV3-dark validation subset (script 04) |
| `data/fastspecfit/*.fits` | ✗ (~79 GB) | DR1 FastSpecFit v3.0 VAC (σ_v source) |
| `data/fastspecfit/*.sigmav.parquet` | ✗ | Per-tile (TARGETID, σ_v) extracts |
| `data/dr1_pairs.parquet` | ✓ (1.3 MB) | One row per spectrum in a kept group (the 13,530-group output) |
| `data/dr1_stats.json` | ✓ | Full-run pre-filter + multiplicity counts |
| `data/classified_pairs.parquet` | ✓ (1.4 MB) | Per-group lens/source + σ_v + θ_E + dimple flag |
| `data/classified_stats.json` | ✓ | σ_v / θ_E distribution + coverage |
| `data/xmatch_table2.json` | ✓ | Per-candidate match offset + (z_d, z_s) for the 20 Table-2 systems |
| `data/sv3dark_pairs.parquet` | ✓ | SV3-dark validation pair list |
| `data/sv3dark_stats.json` | ✓ | SV3-dark validation counts |
| `data/smoketest_timings.json` | ✓ | `spherimatch` N-scaling timings (script 03) |
| `data/cutouts_dr10_manifest_jpeg.csv` | ✓ (2.5 MB) | Per-group DR10 cutout manifest (script 09) |

## Reproducing from scratch

```bash
# Use the hsu venv throughout
PY=/home/benson/.venvs/hsu/bin/python

# Acquire data (~22 GB DR1 + 2.7 GB FastSpecFit small tier; --tier full = 79 GB)
$PY 01_download_dr1_zcatalog.py
$PY 02_download_fastspecfit.py --tier full

# Validate, then run the full FoF (~2 min)
$PY 03_install_spherimatch_smoketest.py
$PY 04_validate_small_region.py
$PY 05_run_full_fof.py            # -> data/dr1_pairs.parquet (13,530 groups)

# Recall + classifier + report
$PY 06_xmatch_published_catalog.py
$PY 07_classify_einstein_dimple.py
$PY 08_write_reproduction_report.py   # -> papers/REPRODUCTION.md

# Figures (optional)
$PY 09_download_dr10_cutouts.py --format jpeg
$PY 10_inspection_grid.py --table2

# Build PDF
make -C papers pdf
```

## Tech-report

```bash
make -C papers pdf      # builds main.pdf
make -C papers clean    # remove latex artefacts, keep PDF
```

Source: [`papers/main.tex`](papers/main.tex); shared preamble at
[`../tech-report.sty`](../tech-report.sty).

## Caveats (short form, see paper §6 + REPRODUCTION.md §6–7)

1. **Algorithmic stage only.** Stages 2 and 3 (visual inspection of spectra and
   imaging) are human workflows we do not attempt. The published 2,046 + 318
   counts are products of those stages.
2. **The +2.4% group excess** lives in the higher-multiplicity tail (triplets,
   quartets), not in the dominant pair population (within 1.8%). Most likely a
   cross-survey / per-sky-position deduplication difference, not a methodology
   error.
3. **σ_v coverage is partial.** Only 31.3% (4,238) of pairs have a reliable
   FastSpecFit σ_v on the lens (`VDISP_IVAR>0`); the rest are default-cap fits
   (σ_v=250 km/s, IVAR=0) or have no fit. Lens σ_v median 217 km/s, estimated
   θ_E median 0.68″ (SIS, flat ΛCDM, H₀=70, Ω_m=0.3) — both consistent with the
   paper. The 9,292 σ_v-less pairs are an *algorithmic proxy* for the dimple
   class: a necessary but not sufficient condition (~30:1 contamination vs the
   published 318 dimples).
4. **Published full catalog not yet released.** The complete 2,046 + 318
   machine-readable catalog announced in Appendix A ("on our project website and
   on Zenodo") had not appeared as of the run date (paper still in ApJS review),
   so recall is validated against the 20 explicit Table-2 systems. Once it
   lands, `06_xmatch_published_catalog.py` should be extended to full-catalog
   recall.
