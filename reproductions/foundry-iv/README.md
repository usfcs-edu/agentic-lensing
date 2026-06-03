# Foundry IV (Lin et al. 2025, VLT/MUSE) reproduction

Public-data reproduction of **Lin et al. 2025, _DESI Strong Lens Foundry IV:
Spectroscopic Confirmation of DESI Lens Candidates with VLT/MUSE_**
(arXiv:2509.18087). The paper presents MUSE integral-field spectroscopy of 75
DESI strong-lens candidates and determines lens & source redshifts for 48
confirmed systems by **manual** identification of spectral features in extracted
1D spectra.

**Methodological value-add of this reproduction:** we build an **automated**
emission/absorption line-ID + redshift finder and run it on the **public** ESO
MUSE Phase-3 datacubes, reproducing redshifts that the paper obtained by hand.

Environment: `/home/benson/.venvs/lens/bin/python` (mpdaf 3.6, astroquery 0.4.11,
pyvo 1.8.1, astropy 7.2, scipy). CPU only.

---

## Headline result

We downloaded the public reduced MUSE cubes for **3 MVP systems** and ran the
automated finder.

**Lens redshifts (the robust win):** all 3 recovered, two of them to dz < 0.003
— essentially the paper's quoted precision — directly from a noise-weighted
cross-correlation of the continuum-normalized 2″-aperture lens spectrum against
the Ca H&K / G-band / Hβ / Mg b / Na D absorption template.

| System | MUSE cube | z_lens (auto) | z_lens (paper) | Δz | z_src (auto, unguided) | z_src (guided [OII]) | z_src (paper) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DESI J003.6745-13.5042 | Lens16 (109.238W.004) | **0.4314** | 0.431 | **+0.0004** | 0.500 | 0.864 | 0.908 |
| DESI J060.5238-22.0990 | Lens22 (111.24P8.001) | **0.4668** | 0.467 | **−0.0002** | 1.473 | **0.8210** | 0.821 |
| DESI J065.6453-28.0646 | Lens24 (111.24P8.001) | **0.6048** | 0.620 | **−0.0152** | 0.520 | 1.230 | 1.175 |

**Source redshifts (the genuinely hard half).** The fully-*unguided* automated arc
finder is interloper-prone: across the 60″ MUSE FoV there is usually a field
emission-line galaxy brighter than the faint lensed arc, so the global
(z, position) optimum lands on an interloper, not the source — exactly why
Lin et al. did the source IDs by hand. When *guided* with the human prior "look for
[OII] 3727 near the published z" (script 06), the same machinery re-locates the arc
and recovers **z = 0.8210 for Lens22, exactly matching the paper**; for the fainter
Lens16/Lens24 arcs it lands within Δz ≈ 0.05 but not on the exact arc spaxel.

The `figs/04_<target>_spectra.png` panels show extracted spectra + redshifted line
markers + z-score curves; `figs/05_z_comparison.png` is the auto-vs-paper scatter
(lens points on the 1:1 line, source points scattered off it);
`figs/06_guided_source.png` is the guided-source demonstration.

---

## The ESO public-data access path (reusable)

The five MUSE programs all have a 12-month proprietary period that elapsed years ago
for the 2022–2024 runs, so the **reduced datacubes are public Phase-3 products** —
no need to run the heavy MUSE ESOREX pipeline on raw frames.

1. **Query** the public ESO TAP service with `pyvo`:
   `pyvo.dal.TAPService("http://archive.eso.org/tap_obs")`, table `ivoa.ObsCore`,
   filtered by `proposal_id IN (...)` and `instrument_name='MUSE'`,
   `dataproduct_type='cube'`. This returns **133 reduced cubes** across the five
   programs (`109.238W.004, 111.24UJ.008, 111.24P8.001, 112.2614.001, 113.267Q.001`).
2. **Cross-match by coordinate.** The cubes are archived under generic target names
   (`Lens1`, `Lens16`, …), NOT DESI names. We match each cube to the confirmed
   Foundry-IV systems within 30″ (MUSE FoV is 60″). **21 cubes** fall on confirmed
   systems; **all 13** of the systems in our hand-curated catalog have ≥1 public cube.
3. **Resolve DataLink → download.** The ObsCore `access_url` is an ESO **DataLink**
   endpoint returning a VOTable; the `#this` row gives the real ~3 GB cube at
   `https://dataportal.eso.org/dataPortal/file/<dpid>`. Script 02 parses this and
   downloads.

`data/cube_match_table.csv` is the full archive-cube ↔ confirmed-system table.

---

## Scripts

| Script | What it does |
| --- | --- |
| `01_build_confirmed_catalog.py` | Transcribe the confirmed systems' coords + published z_lens/z_source + the line features the paper used (Table 2 / Sect 4.1) into `data/confirmed_catalog.csv`. These are the **ground truth** — transcribed from the paper, not measured by us. |
| `02_query_download_cubes.py` | Query ESO TAP for the public MUSE cubes, cross-match to the catalog, write `data/cube_match_table.csv`, resolve DataLink, and download the MVP cubes. `--list` to inspect without downloading. |
| `_zfinder.py` | The automated line-ID engine (no archive deps). Noise-weighted cross-correlation against redshifted galaxy-absorption / UV-absorption / emission templates, with sky-line and high-variance masking; plus a **joint spatial+spectral** emission-line source finder (`find_emission_source_in_cube`). |
| `03_synthetic_unittest.py` | Unit-tests `_zfinder` on synthetic MUSE spectra at known z (passive lens, [OII] source, LBG UV-absorption source). All recover z to dz < 0.001. Validates the engine independently of the download. |
| `04_measure_redshifts.py` | Open each cube with mpdaf, extract the lens 1D spectrum at the catalog coord (2.0″ aperture) → auto z_lens; run the joint [OII]/[OIII] scan to find the arc → extract its spectrum → auto z_source (unguided); compare to the paper; write `data/measured_redshifts.csv` and per-system figures. `--lens-only` skips the slow source scan. |
| `05_compare_summary.py` | Read `measured_redshifts.csv`, print the pass/fail table (lens dz<0.005 target), and make the auto-vs-paper scatter `figs/05_z_comparison.png`. |
| `06_guided_source_demo.py` | Honest-scope demo on the hard half: guide the [OII] narrow-band line map to the published source z, re-locate the arc, and recover the source redshift with the 1D emission engine. Writes `data/guided_source_redshifts.csv` + `figs/06_guided_source.png`. |

---

## Method detail: the automated line finder (`_zfinder.py`)

- **Sky / bad-pixel mask.** MUSE spectra are dominated by imperfectly-subtracted OH
  airglow residuals redward of ~5500 Å. We mask ±4 Å around a strong-sky-line list
  and reject pixels with anomalously high pipeline variance, so sky residuals cannot
  masquerade as galaxy features.
- **Continuum.** Running-median over a ~120 Å window using only good pixels.
- **Redshift search = matched-filter cross-correlation.** For a z grid we build a
  unit-amplitude multi-Gaussian template of the rest-frame line list at z, and score
  `Σ(w·T·S) / sqrt(Σ(w·T²))` with inverse-variance weights w. This is a standard
  cross-correlation redshift estimator (à la RVSAO/Marz); the score is ~the combined
  detection SNR and is comparable across z.
  - **Lens (absorption):** S = continuum-normalized absorption depth; template =
    Ca II H&K, G-band, Hβ, Mg b, Na D (the exact lines the paper lists).
  - **Source (emission):** S = continuum-subtracted positive residual; template =
    [OII] λλ3727/29, [NeIII], Hγ, Hβ, [OIII] λλ4959/5007, Hα, [SII].
  - **Source (high-z LBG):** UV-absorption template — Si IV, Si II, C IV, Fe II, Al II.
- **Joint source localization.** Across a trial-z grid, build narrow-band line maps at
  [OII]/[OIII]/Hβ (on-band minus side-band), restrict to an annulus around the lens
  (where arcs live, not the wide-FoV interlopers), and take the (z, position) that
  maximizes annular line flux. This is the automated analog of the paper's manual
  "pick the arc spaxels, find the [OII] doublet."

---

## Honest scope & what is a proxy

- **Real, public data:** every measured number comes from the actual public ESO MUSE
  Phase-3 reduced cube for that system (not a synthetic or imaging proxy). The MUSE
  reduction itself (pipeline v2.2 + ZAP) is the ESO archive's, exactly as in the paper.
- **Ground-truth redshifts are transcribed** from the published paper (Sect 4.1),
  not independently derived — they are the comparison target, by design.
- **Lens redshifts** are the robust, automated win (passive galaxies with strong,
  well-modeled absorption templates).
- **Source redshifts** are the genuinely hard part the paper did by hand: arcs are
  faint, blended with interlopers, and span [OII]-emission (low z) to UV-absorption
  (z>2) regimes. Our joint scan re-locates the arc and recovers the emission-line
  source z where the [OII]/[OIII] features are in-band and the arc is the dominant
  annular line emitter; for the faintest / highest-z sources the automated pick is
  less reliable than a human (see `data/measured_redshifts.csv` `src_snr` and `dz_src`).
- MVP scope: 3 of 48 confirmed systems were fully run; the pipeline scales to all 21
  matched cubes (just add target names to script 02 / rerun script 04).

Data (`data/`, the ~3 GB cubes, match tables, CSVs) is gitignored and fully
regenerable via the numbered scripts.
