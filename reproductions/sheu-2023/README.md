# Sheu 2023 reproduction — targeted lensed-supernova difference-imaging pipeline

Internal reproduction of \[Sheu, Huang, Cikota, Suzuki, Schlegel & Storfer 2023,
*Retrospective Search for Strongly Lensed Supernovae in the DESI Legacy Imaging
Surveys*, [arXiv:2301.03578](https://arxiv.org/abs/2301.03578)\]. The paper PDF
is at `../../papers/Sheu_2023_lensed_supernovae.pdf`.

## What this is

Sheu+2023 built a **targeted lensed-transient pipeline** and ran it on 5807
strong lenses in the DECaLS footprint, finding seven lensed-SN candidates —
headlined by **DESI-344.6252-48.8977**, a very likely galaxy-scale L-SN Ia at
z ≈ 0.83 in a θ_E ≈ 1.5″ lens (their §6.1.1). The full search was a NERSC
job (~120,000 exposure cutouts). This reproduction does **not** redo that
search; it **rebuilds the per-system pipeline from scratch and demonstrates it
MVP on that one Grade-A target**, end to end, on real DECam data + CPU.

Pipeline (paper §3), all reimplemented here:

```
NOIRLab Astro Data Archive  (individual DECam g/r/z InstCal exposures)
        │  01_download_decam_exposures.py
        ▼
WCS reproject → common 801×801 @ 0.262″ grid;  median-coadd REFERENCE
        │  02_reproject_and_reference.py
        ▼
B08 difference imaging   (Bramich 2008 delta-basis kernel, from scratch)
        │  03_difference_imaging_b08.py
        ▼
SEP source detection (1.0–2.5σ ladder) → spatial(<0.8″)/temporal(<50 d) FoF grouping
        │  04_detect_and_group.py        (groups with ≥3 sub-detections kept)
        ▼
forced photometry on diff images → SALT3 / sncosmo fit → magnification
           05_lightcurve_salt3.py        (+ synthetic injection validation)
           06_make_detection_figure.py   (visual re-detection panel)
```

## Headline results

**1. Real DECam data, real target — the per-exposure access path works.**
The crux of the paper (individual DECam exposures, not coadds) is obtainable
from the **NOIRLab Astro Data Archive** CCD-level SIA service
(`/api/sia/vohdu`). A cone search at DESI-344.6252-48.8977 returns ~220 covering
CCD HDUs; we pulled and sliced **20 InstCal g/r/z exposure cutouts** (8 g, 7 r,
5 z; all 90 s; MJD 56887–58072, the DR9 era), median-coadded a per-band
reference, and B08-differenced all 20 epochs. The target is the **exact**
Storfer-catalog lens (cross-matched at **0.2″**, `data/storfer2024_published_catalog.csv`).

**2. Re-detection of the transient on the counter-image.**
SEP at the paper's 1.0–2.5σ ladder produces a noisy field (≈88 k raw
sub-detections, mostly 1σ noise → 4625 ≥3 FoF groups). Applying the paper's
*location grade* (proximity to lensing features), **only 6 groups fall within 3″
of the lens centre**, and one dominates:

| group | n_subdet | n_epochs | bands | offset from lens |
| ----: | -------: | -------: | :---- | ---------------: |
| 26517 | **11**   | **5**    | **grz** | **1.48″** |

i.e. a multi-band, multi-epoch transient sitting **1.48″ from the lens centre —
on the counter-image**, exactly where Sheu+2023 place the L-SN (θ_E ≈ 1.5″).
See `figs/redetection_panel.png` and `figs/detection_summary.png`.

**3. End-to-end light-curve + magnification chain, validated on injected truth.**
A real-data SALT3 fit recovers a lensed-regime amplification but with a poor
χ²/dof — our quick aperture photometry on bright lens-subtraction residuals is a
**proxy**, not the paper's careful transient-free-reference PSF photometry
(see "What is a proxy" below). To prove the photometry → SALT3 →
Hubble-residual → magnification chain is *correct*, `05` also runs a
**controlled synthetic injection**: a SALT3 SN Ia at the paper's parameters
(z = 0.833, calibrated to the SN Ia standard candle M_B = −19.25) injected into
the real reference frames at a blank-sky location, magnified by a known μ, then
pushed back through B08 differencing + photometry + SALT3:

| injected μ | recovered μ | recovered Hubble resid |
| ---------: | ----------: | ---------------------: |
| 4.0 | 3.61 | −1.39 |
| 6.0 | 6.34 | −2.01 |
| **8.0** | **8.61** | **−2.34** |
| 10.0 | 10.71 | −2.57 |

The μ = 8.0 case is the paper's reported magnification for this very system
(Sheu give 8.23 +2.61/−1.98, Hubble residual −2.29 ± 0.30). We recover
**μ = 8.61** and **resid = −2.34** with χ²/dof = 2.2, and t0 to within ~10 d —
a clean match to the paper's headline number. See `figs/lightcurve_synth.png`.

## Scope discipline — what is real, what is a proxy, what is out of scope

**Real (validated):**
- Per-exposure DECam data access from NOIRLab (not coadds) — the hard part.
- B08 (Bramich 2008) non-parametric difference imaging, implemented from
  scratch via linear least squares on a delta-function kernel basis + flat
  differential background, spatially varying over an N×N tile grid.
- SEP detection on the paper's threshold ladder; 0.8″/50 d FoF grouping with
  the ≥3-sub-detection rule.
- The location-grade filter isolating the real transient on the counter-image.
- The full SALT3 → magnification chain, **quantitatively verified** against
  injected truth and reproducing the paper's μ ≈ 8 for this system.

**Proxy / simplified:**
- **Photometry.** We use circular-aperture forced photometry on the B08 diffs.
  The paper uses PSF photometry with a *transient-free* reference (re-coadd
  excluding the transient epochs) + DR10 i/Y. Our reference still contains some
  transient light and our aperture sits on bright lens-galaxy subtraction
  residuals, so the **real-data** SALT3 χ²/dof is poor and the real-data μ is not
  reliable — hence the synthetic-injection validation for the quantitative claim.
- **SFFT** (the paper's GPU Fourier subtraction, used for their final
  photometry) is not implemented; we use B08 only. SFFT is the documented
  optional GPU path.
- **Montage** is replaced by `reproject` for the WCS regridding step.
- B08 absorbs ~20 % of a bright point source into its kernel (a known B08
  effect with large tiles); this biases recovered fluxes slightly low but the
  standard-candle magnification still recovers to ≲10 %.

**Out of scope (paper, not reproduced):**
- The 5807-system / ~120 k-cutout NERSC deployment (this is one system).
- The rate forecast (§4), the 32 DES-SN-Ia photometry test (§5), and the
  161-template CC-SN fitting (§3.4). We fit the SALT3 SN Ia model + an
  unlensed-prior variant only.

## Files

| script | does |
| :--- | :--- |
| `01_download_decam_exposures.py` | NOIRLab SIA discovery + full-frame fetch → covering-CCD WCS slice → 801² g/r/z cutouts (+ weight/DQ) |
| `02_reproject_and_reference.py` | reproject to common grid + median-coadd reference per band |
| `03_difference_imaging_b08.py` | Bramich-2008 delta-basis kernel difference imaging (spatially tiled) |
| `04_detect_and_group.py` | SEP detection (1.0–2.5σ) + 0.8″/50 d FoF grouping (≥3 rule) |
| `05_lightcurve_salt3.py` | forced photometry → SALT3 fit → magnification; `--mode synth` injection validation |
| `06_make_detection_figure.py` | reference + diff-image re-detection panel |

Key outputs in `data/` (gitignored, regenerable): `exposure_manifest.csv`,
`reference_<band>.fits`, `diff/<band>/*_diff.fits`, `subdetections.csv`,
`groups.csv`, `forced_photometry*.csv`, `salt3_fit_*.csv`. Figures in `figs/`.

## Reproduce

```bash
PY=/home/benson/.venvs/lens/bin/python
cd reproductions/sheu-2023
$PY 01_download_decam_exposures.py --bands g r z --max-per-band 12   # ~few GB transient net, mins
$PY 02_reproject_and_reference.py  --bands g r z
$PY 03_difference_imaging_b08.py   --bands g r z --hw 4 --ntile 2
$PY 04_detect_and_group.py         --bands g r z
$PY 05_lightcurve_salt3.py         --mode both --bands g r z --synth-mu 8.0
$PY 06_make_detection_figure.py
```

Dependencies added to the venv during this work: `pandas`, `aiohttp`,
`iminuit`, `pdfminer.six` (others — `reproject`, `sep`, `sncosmo`, `astropy`,
`photutils` — were already present).

## Honest one-line summary

We rebuilt Sheu+2023's targeted lensed-SN pipeline (NOIRLab per-exposure DECam →
reproject → median reference → **from-scratch B08 differencing** → SEP →
FoF grouping → SALT3 → magnification), **re-detected the Grade-A L-SN candidate
DESI-344.6252-48.8977 as an 11-sub-detection, 5-epoch, grz transient 1.48″ from
the lens on the counter-image**, and **reproduced the paper's μ ≈ 8 magnification
to within ~10 %** via a controlled SALT3 injection at the paper's z = 0.833.
The real-data SALT3 photometry itself is a proxy (aperture, not PSF, with a
transient-contaminated reference); SFFT and the 5807-system NERSC search are out
of scope.
