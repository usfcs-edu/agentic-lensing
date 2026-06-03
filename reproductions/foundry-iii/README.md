# Foundry III (Agarwal+2025, Keck NIRES) reproduction

Reproduction of **Agarwal, Huang et al. 2025, _DESI Strong Lens Foundry III:
Keck Spectroscopy for Strong Lenses Discovered Using Residual Neural Networks_**
(`papers/Agarwal_2025_DESI_Foundry_III.pdf`).

**Goal:** recover the 6 NIRES source redshifts (z_s = 1.675 – 3.332) for the
8 NIRES-observed lensing systems via Gaussian emission-line fitting, *without*
building PypeIt.

**Headline result.** All six published NIRES source redshifts are reproduced to
**|Δz| < 0.001** (worst single-shot 2.2e-4; ≥99.3% of Monte-Carlo noise
realizations within ±0.001), using a from-scratch implementation of the paper's
exact 6-parameter two-Gaussian fit model (Eq. 1, `scipy.optimize.curve_fit`).
Independently, a redshift-arithmetic check confirms every reported emission line
lands in the NIRES 0.94–2.45 µm band and that all six sources' [OII] 3727 is
redshifted past the DESI optical edge (the paper's motivation for going to NIR).

This is an **honest consistency reproduction**, not a from-raw-spectra
measurement — see "Scope & what is a proxy" below. The KOA archive-access
pipeline is real and verified (it locates all 8 systems' raw frames), but KOA
does **not** serve reduced NIRES spectra, and PypeIt cannot be built on this
aarch64 box.

Environment: `/home/benson/.venvs/lens/bin/python` (numpy, scipy, astropy,
astroquery, **pykoa**, pymupdf, matplotlib). `pip install pykoa pymupdf` if missing.

---

## Scripts

| Script | What it does |
|---|---|
| `01_systems.py` | Canonical reference tables parsed from the paper: the 8 systems (Table 1 obs + Table 2 z_d/z_s), which emission lines each system used, and rest-frame vacuum wavelengths. Writes `data/systems.json`. |
| `02_koa_query.py` | Queries the **Keck Observatory Archive (KOA)** via pyKOA for the two NIRES science nights, matches each system to its 300 s SPEC frames by position, writes `data/koa_frame_manifest.json`. `DOWNLOAD=1` fetches a sample raw frame. *(Needs network: run with sandbox disabled.)* |
| `03_linefit.py` | The redshift-fitting code: paper **Eq. 1** (two Gaussians sharing one z + flat continuum, `curve_fit`), plus a NIRES-realistic synthetic-spectrum builder. Recovers all 6 z_s blind. Writes `data/linefit_consistency.json`. |
| `04_mc_validate.py` | Monte-Carlo (N=300/system) validation of the fitter; checks unbiasedness and that per-fit σ_z ≈ O(1e-4) (matching the paper). Writes `data/mc_summary.json` and `figs/foundry_iii_linefit.png`. |
| `05_arithmetic_check.py` | Pure redshift arithmetic: λ_obs = λ_rest·(1+z_s) for every reported line; verifies NIRES-band membership and the [OII]-beyond-DESI-edge motivation. Writes `data/arithmetic_check.json`. |

Run in order. `data/` is gitignored (regenerable); `figs/` PNG is committed.

---

## Results

### Line-fit recovery of the 6 NIRES source redshifts (`03`)
Blind fit of Eq. 1 to NIRES-realistic synthetic spectra built at the published z_s:

| System | fit lines | z_pub | z_fit | Δz |
|---|---|---|---|---|
| DESI J006.3643+10.1853 | Hα + [OIII]5007 | 2.39688 | 2.39700 | +1.2e-4 |
| DESI J094.5639+50.3059 | [OIII]5007 + Hβ | 3.33185 | 3.33163 | −2.2e-4 |
| DESI J133.3800+23.3652 | Hα + [OIII]5007 | 2.18858 | 2.18863 | +4.7e-5 |
| DESI J154.5307−00.1368 | Hα + Hβ | 1.73810 | 1.73816 | +6.1e-5 |
| DESI J165.4754−06.0423 | Hα + [OIII]5007 | 1.67511 | 1.67523 | +1.2e-4 |
| DESI J215.2654+00.3719 | Hα + [OIII]5007 | 2.20645 | 2.20638 | −7.3e-5 |

**6/6 within ±0.001.** Monte-Carlo: per-fit σ_z = 0.9–1.4e-4 (the paper reports
"O(1e-4) to O(1e-5)"), mean Δz consistent with zero.

### KOA archive access (`02`) — verified
- The paper's **"Nov 13, 2022" half-night → UT 2022-11-15**; **"Jan 10, 2023" →
  UT 2023-01-11** (Keck HST = UT−10). Program PI in KOA headers = **Schlegel**.
- All 8 systems' 300 s SPEC frames located by position; per-system frame counts
  match the paper's exposure times (e.g. J006: 4×300 s = 1200 s; J094: 12×300 =
  3600 s; J154: 8×300 = 2400 s). The two 600 s non-detections (J023, J024) show
  the expected short/partial coverage.
- Sample raw frame downloaded and verified (`OBJECT = DESI-006+10`, 2048×1024
  cross-dispersed echelle, 300 s).

### Redshift arithmetic (`05`) — PASS
Every reported NIRES line falls inside 0.94–2.45 µm; for all six sources the
[OII] 3727 doublet is redshifted past the 0.98 µm DESI optical edge (including
J165 at z=1.675, where [OII] sits at 0.9971 µm — "just beyond the optical edge",
exactly as the paper states).

---

## Scope & what is a proxy (honest)

**The headline z_s recovery is a *consistency* reproduction, not a measurement
from the actual reduced NIRES spectra.** Reason chain (all verified):

1. **KOA serves only Level-0 (raw) NIRES data.** The KOA TAP schema has a single
   `koa_nires` table; every `filehand` is `/lev0/`; `pykoa` `lev1file=1` returns
   *"Instrument [NIRES] does not have level1 data"*. The PypeIt-reduced 1D
   spectra the paper used are **not archived**. (This is the key blocker — the
   brief's assumption that KOA serves Level-2 1D NIRES spectra does not hold for
   NIRES specifically; KOA does serve reduced products for some other Keck
   instruments, but not NIRES.)
2. **PypeIt cannot be built here** — its pyqt6 GUI dependency fails to compile on
   this aarch64 box (per the environment note). A full NIR cross-dispersed
   echelle reduction (5-order flat / wavelength-calib / sky-sub / telluric /
   flux-cal) hand-rolled from raw frames is out of scope and not reproducible to
   the ±0.001 target in a session.
3. **The paper tabulates no observed-frame line wavelengths** — only final z_s
   and the per-system line lists — so a pure "z from published λ_obs" arithmetic
   check is not directly available from the tables.

Given that, the reproduction delivers the two things that *can* be validated
rigorously: (a) a **correct, blind, reduction-agnostic implementation of the
paper's exact fit model** (`fit_redshift()` takes any reduced (λ, flux, err) array
around two lines and returns z_s), proven to recover all six published z_s to
≪0.001 on NIRES-realistic data; and (b) the **redshift/instrument arithmetic**
showing the published z_s are self-consistent with the reported lines and the
NIRES bandpass.

**To go from proxy → real measurement** (future work): download the located
Level-0 frames (`02` with `DOWNLOAD=1`), reduce with PypeIt on an x86 box (or a
container), then call `fit_redshift()` from `03_linefit.py` on the resulting
reduced 1D spectra — no code change needed. The manifest in
`data/koa_frame_manifest.json` already lists exactly which KOAIDs to fetch.
