# Sheu 2024a reproduction — targeted variable-lensed-quasar difference-imaging pipeline

Internal reproduction of \[Sheu, Huang, Cikota, Suzuki, Palmese, Schlegel &
Storfer 2024a, *A Targeted Search for Variable Gravitationally Lensed Quasars*,
[arXiv:2408.02670](https://arxiv.org/abs/2408.02670)\]. The paper PDF is at
`../../papers/Sheu_2024a_variable_lensed_quasars.pdf`.

## What this is

Sheu+2024a (Paper II) is the **variability variant** of Sheu+2023 (Paper I, the
lensed-supernova search we reproduced in `../sheu-2023/`). It reuses the same
targeted difference-imaging core but, instead of finding one-off lensed
supernovae, it detects **stochastic quasar variability at the lensed-image
positions** across ~5 yr of DECam epochs, and quantifies it with a
**magnitude-standard-deviation metric** (their Eq. 1). Run on 5807 strong lenses
+ 655 known lensed-quasar candidates, the paper reports 13 new lensed-quasar
candidates and variability confirmation for 13 known ones (their Table 2,
⟨σ⟩ ≈ 0.13–0.40 mag).

This reproduction does **not** redo that 6462-system search. It **reuses the
validated Paper-I per-system pipeline** (NOIRLab per-exposure DECam access →
reproject → median reference → from-scratch B08 differencing → SEP) and adds the
**new Paper-II variability layer** (PSF light curves at the lensed-image
positions → Eq. 1 σ), demonstrating it **MVP on one Grade-A lensed-quasar
candidate** end to end on real DECam data + CPU, plus a controlled
injected-variable-source validation of the σ metric.

### Reuse from `../sheu-2023/`
`01`–`04` are direct adaptations of the sibling scripts (only the target
coordinates change). The genuinely new, Paper-II-specific work is `05`
(variability light curves + Eq. 1 σ) and `06` (over/under-subtraction panel).

```
NOIRLab Astro Data Archive  (individual DECam g/r/z InstCal exposures)
        │  01_download_decam_exposures.py        (REUSED from Paper I)
        ▼
WCS reproject → common 801×801 @ 0.262″ grid;  median-coadd REFERENCE
        │  02_reproject_and_reference.py         (REUSED)
        ▼
B08 difference imaging   (Bramich 2008 delta-basis kernel, from scratch)
        │  03_difference_imaging_b08.py          (REUSED)
        ▼
SEP detection (1.0–2.5σ ladder) → spatial/temporal FoF grouping
        │  04_detect_and_group.py                (REUSED, confirms variable source)
        ▼
PSF photometry at the lensed-image positions → light curves → Eq.1 σ   ◄── NEW (Paper II)
           05_variability_lightcurve.py   (+ injected-variable-source validation)
           06_make_variability_figure.py  (over/under-subtraction evidence panel)
```

## Target

**DESI-038.0655-24.4942** — a Grade-A **double** lensed-quasar candidate
(Dawes+2023 / He+2023; image separation 1.54″), in the **DES footprint** (deep
DECam coverage), and one of the paper's §4.2 variability-confirmed systems.
Sheu+2024a Table 2 reports **⟨σ⟩ = 0.25 mag** and r ≈ 18.39 for this system.
We cross-matched it at 0.00″ in the on-disk Dawes catalog
(`../dawes-2022/data/dawes2023_vizier_table2.csv`).

## Headline results

**1. Real DECam data, real target — variability detected at the lensed images.**
The NOIRLab SIA cone search at DESI-038.0655-24.4942 yields ~12–16 unique
exposures/band; we pulled **32 InstCal g/r/z cutouts** (12 g, 9 r, 11 z) spanning
**MJD 56944–58454 (4.13 yr, the DES era)**, median-coadded a per-band reference,
and B08-differenced all 32 epochs. SEP detection + 0.8″/50 d FoF grouping
(≥3-rule, the Paper-I machinery) produces a noisy field, but the **location
filter isolates the variable source**: the dominant group has **20
sub-detections across 11 epochs in all three bands, at 0.25″ from the system
centre** (image A), with a second grz group at 1.71″ (image B). i.e. the
posited lensed-quasar images show repeated over/under-subtraction — exactly the
paper's Criterion-3 variability signal. See `figs/variability_panel.png`.

**2. The Eq. 1 variability σ — the headline metric.**
We resolve the Dawes **double** (image A at the system centre, image B 1.17″
away — the catalog 1.54″ is foreshortened by blending at ~1.3″ seeing), do
Gaussian-PSF photometry at both positions on every epoch (S/N > 5 cut, all 64
measurements pass), build per-band magnitude light curves, and compute the
paper's Eq. 1 magnitude standard deviation:

| image | σ (mag) | per-band (g / r / z) | N epochs |
| :---- | ------: | :------------------- | -------: |
| A     | **0.332** | 0.459 / 0.276 / 0.240 | 32 |
| B     | **0.342** | 0.357 / 0.315 / 0.348 | 32 |
| **⟨σ⟩** | **0.337** | — | — |

Sheu+2024a Table 2 reports **⟨σ⟩ = 0.25 mag** for this system. We recover the
same order, somewhat high (our blended single-Gaussian photometry adds
cross-image + lens-galaxy scatter, strongest in g where σ = 0.46; the paper's
Tractor PSF photometry deblends better). The light curves
(`figs/lightcurve_real.png`) show unmistakable ~0.5–1 mag stochastic,
band-coherent quasar variability over the 4-yr baseline.

**3. The σ metric, validated on injected truth.**
To prove the photometry → Eq. 1 → σ chain is *correct* independent of the
blended real-data photometry, `05 --mode synth` injects a point source with a
**known** per-band magnitude scatter into the real reference frames at a
blank-sky location and pushes it through the same B08 + PSF-photometry + Eq. 1:

| injected σ | realised injected (g/r/z) | recovered ⟨σ⟩ | ratio |
| ---------: | :------------------------ | ------------: | ----: |
| 0.10 | 0.105 / 0.129 / 0.084 | 0.109 | 1.09 |
| 0.20 | 0.210 / 0.257 / 0.169 | 0.216 | 1.08 |
| 0.25 | 0.156 / 0.254 / 0.210 | 0.207 | 0.83\* |
| 0.40 | 0.419 / 0.515 / 0.337 | 0.429 | 1.07 |

The recovered σ tracks the *realised* injected scatter to a few percent across
the paper's full 0.13–0.40 mag range (the 0.25/seed-0 row reads low only because
that random draw itself scattered low — realised g = 0.156, not 0.25; recovery
still matches it). The metric is honest and linear. See `figs/lightcurve_synth.png`.

## Scope discipline — what is real, what is a proxy, what is out of scope

**Real (validated):**
- Per-exposure DECam data access from NOIRLab (not coadds) — reused from Paper I.
- B08 (Bramich 2008) from-scratch difference imaging; SEP detection — reused.
- The **Paper-II Eq. 1 variability σ metric**, computed from per-epoch PSF
  photometry at both lensed-image positions with the paper's **S/N > 5** cut,
  and **quantitatively validated** against an injected variable source of known
  σ (`--mode synth`).
- The difference-imaging **over/under-subtraction** evidence at the image
  positions (`figs/variability_panel.png`), the qualitative variability signal
  the paper shows in its per-system figures.

**Proxy / simplified:**
- **Photometry.** We fit a circular-Gaussian PSF (FWHM from the per-frame
  seeing) on a small stamp at each lensed-image position; the paper uses
  Tractor/forced PSF photometry. For a double with 1.54″ separation on ~1.3″
  DECam seeing, the two images partially blend, so absolute magnitudes carry
  lens-galaxy + cross-image contamination — hence the injected-source validation
  for the quantitative σ claim.
- **Eq. 1 interpretation.** As printed, Eq. 1 omits the `1/N_b` inside the
  per-band `sqrt`, which would give σ ~√N_b larger than the 0.1–0.4 mag scale of
  Table 2. We interpret it (per the Table 2 caption, "average magnitude standard
  deviation") as the N_b-weighted mean of the per-band magnitude **standard
  deviation**, which reproduces the Table 2 scale. Both forms are in
  `data/sigma_*.csv` so the choice is transparent.
- **SFFT** (the paper's GPU Fourier subtraction, used for their published diff
  images) and the **Hu et al. 2022** second algorithm are not implemented; we
  use B08 only, as in our Paper-I reproduction.
- **Montage** is replaced by `reproject`.

**Out of scope (paper, not reproduced):**
- The 5807-strong-lens + 655-lensed-quasar-candidate search (this is one system).
- The visual location/color grading (done by eye in the paper); we target a
  pre-graded Dawes/He candidate directly.
- Note: the **public NOIRLab SIA** cone search indexes fewer exposures at this
  position (~12–16 unique per band) than the paper's full DR9+DR10 NERSC
  exposure database (it reports 31/21 for this system), so our light curves are
  shorter than theirs — still ample for a robust σ.

## Files

| script | does |
| :--- | :--- |
| `01_download_decam_exposures.py` | NOIRLab SIA discovery + full-frame fetch → covering-CCD WCS slice → 801² g/r/z cutouts (REUSED) |
| `02_reproject_and_reference.py` | reproject to common grid + median-coadd reference per band (REUSED) |
| `03_difference_imaging_b08.py` | Bramich-2008 delta-basis kernel difference imaging (REUSED) |
| `04_detect_and_group.py` | SEP detection (1.0–2.5σ) + 0.8″/50 d FoF grouping (REUSED) |
| `05_variability_lightcurve.py` | **NEW**: PSF photometry at lensed-image positions → light curves → Eq. 1 σ; `--mode synth` injected-source validation |
| `06_make_variability_figure.py` | **NEW**: reference + over/under-subtraction evidence panel |

Key outputs in `data/` (gitignored, regenerable): `exposure_manifest.csv`,
`reference_<band>.fits`, `diff/<band>/*_diff.fits`, `image_positions.csv`,
`lightcurves_<tag>.csv`, `sigma_<tag>.csv`. Figures in `figs/`.

## Reproduce

```bash
PY=/home/benson/.venvs/lens/bin/python
cd reproductions/sheu-2024a
$PY 01_download_decam_exposures.py --bands g r z --max-per-band 30
$PY 02_reproject_and_reference.py  --bands g r z
$PY 03_difference_imaging_b08.py   --bands g r z --hw 4 --ntile 2
$PY 04_detect_and_group.py         --bands g r z
$PY 05_variability_lightcurve.py   --mode both --bands g r z --synth-sigma 0.25
$PY 06_make_variability_figure.py
```

## Honest one-line summary

We adapted Sheu+2023's targeted difference-imaging pipeline into the Sheu+2024a
**variability** variant (NOIRLab per-exposure DECam → reproject → median
reference → from-scratch B08 differencing → SEP), added the new **PSF
light-curve + Eq. 1 magnitude-σ** layer, and **measured ⟨σ⟩ = 0.337 mag of
stochastic, band-coherent variability in both images of the Grade-A double
lensed-quasar candidate DESI-038.0655-24.4942** over a 4.1-yr DECam baseline
(paper Table 2: 0.25 mag) — with the σ metric **validated to a few percent**
against an injected variable source of known scatter across the paper's full
0.13–0.40 mag range. The real-data photometry is a blended single-Gaussian PSF
proxy (not Tractor), and the 6462-system search, SFFT, and Hu-2022 second
algorithm are out of scope.
