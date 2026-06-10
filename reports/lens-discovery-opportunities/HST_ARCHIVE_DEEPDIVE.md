# HST General Archive — Pressure-Test of the Strong-Lens Opportunity

**Supplement to** `OPPORTUNITIES.md` · **Date:** 2026-06-06 · **Method:** 4-agent research
workflow (inventory + prior-coverage + access-pipeline → adversarial red-team; archived at
`workflow-hst-deepdive.js`).

> **Headline correction.** The main report (and my first-pass chat estimate) rated the HST
> general archive as *the best drop-in by gross yield* (O(10²), "hundreds of galaxy–galaxy
> lenses never uniformly ML-swept"). **The red-team DOWNGRADED this with 6 corroborating
> sources.** The thesis has a kernel of truth but fails in its strong form on two independently
> verified fronts: (1) HST is **pencil-beam** for lens-quality imaging, and (2) the archive is
> **not virgin** — it was already crowd-scanned end-to-end. The realistic *net-new secure* yield
> is **tens to ~100–150**, not hundreds, and it comes from a *completeness re-mine of
> already-covered sky*, not from discovering untouched data.

---

## 1. The verified area accounting

| Quantity | Value | Source / basis |
|---|---|---|
| Total unique HST imaging footprint (all instruments/filters) | ~40 deg² ("~0.1% of sky") | Hubble Source Catalog / HLA union |
| Extragalactic, lens-quality ACS/WFC + WFC3 subset | ~25–35 deg² | high-latitude subset; heavy multi-filter overlap |
| **Firm** single-orbit ACS/WFC **F814W** searchable area (through 2011) | **6.03 deg²** (+1.01 deg² WFC3/IR) | Pawase et al. 2014 (MNRAS 439, 3392) |
| Defensible lens-searchable area by 2026 | **~10–15 deg²** | Pawase + ~15 yr archive growth + multi-band |
| **Contiguous deep mosaic** (the part suited to arc-finding) | **only ~2.5–3 deg²** | COSMOS 1.64 deg² dominant; CANDELS/GOODS/GEMS/AEGIS make up the rest |
| Galaxy-scale lens surface density at HST single-orbit depth | **~7–12 secure / deg²** | Pawase 6.6/deg² (F814W); Faure 2008 ~12/deg² clear arcs; Collett 2015 ~11/deg²; HAH-II ~9/deg² |
| **Total recoverable secure population in the *entire* archive** | **~250–350** | area × density — this is the *ceiling for everything*, not net-new |

The decisive fact: **HST is non-contiguous**. Most "searchable area" is scattered single-band GO
pointings, not mosaics. Only ~2.5–3 deg² is contiguous deep imaging, and the single largest
mosaic (COSMOS, 1.64 deg²) is the most-studied patch of extragalactic sky on the planet.

## 2. The prior-search ledger — the archive is *not* virgin

| Search | Method | Area / fields | Lenses |
|---|---|---|---|
| **Hubble Asteroid Hunter II** (Garvin/Kruk 2022) | citizen-science, blind, **archive-wide** | **~27 deg²** — essentially the entire large-FOV ACS+WFC3 archive 2002–2020 | 252 (198 new) |
| Pawase et al. 2014 | expert visual, blind | ~7 deg² (subset of HAH) | 49 |
| Faure 2008 | visual (pre-selected deflectors) | COSMOS 1.64 deg² | 67 |
| Jackson 2008 | complete visual | COSMOS 1.64 deg² | 2 new + candidates |
| **LensFlow / Pourrahmani 2018** | **CNN** | COSMOS 1.64 deg² | 92 (46 new) |
| COWLS 2025 | visual+ML | COSMOS-Web 0.54 deg² (**JWST**, not HST) | >100 |
| SL2S; CLASH/HFF/RELICS | targeted snapshots / cluster models | negligible blind area | ~56; cluster arcs |

**De-duplicated already-published HST-archive set: ~300–450 candidates (~100–150 secure).** Two
killers for the "never ML-searched" premise: (i) **Hubble Asteroid Hunter II already
serendipitously scanned ~60–70% of the contiguous wide-field archive**, and (ii) **COSMOS — the
single densest lens field — was already searched by a CNN** (LensFlow) *and* is now JWST-mined
(COWLS). There is no untouched HST sky of any size.

## 3. What is genuinely left (the real, smaller opportunity)

- **Never blind-searched area: ~10–20 deg², but heterogeneous and shallow** — WFPC2/pre-ACS
  fields outside HAH's selection, ~6 yr of post-2020 imaging, single-band-only fields HAH skipped
  (it required composite color PNGs), and the ~30% of frames quality-excluded from HLA/HSC. Very
  little of this is COSMOS-depth.
- **The dominant axis is completeness-depth, not virgin sky.** HAH-II was a by-eye citizen pass
  (10 non-experts per cutout, color PNGs) with low/ill-characterized completeness for **faint,
  small-separation, low-contrast** galaxy–galaxy lenses — *exactly* the regime where a CNN
  excels. HAH flagged only ~9/deg² where the depth implies 10–20+/deg², so there is real
  recoverable incompleteness on the same ~27 deg².

**Realistic net-new yield (red-team, 6 sources): tens to ~100–150 secure galaxy–galaxy lenses,**
plus a few-hundred low-confidence candidate pool with a low secure fraction (rings/mergers/pairs)
and heavy vetting cost. **O(10²) *new raw candidates* is defensible; O(10²) *new secure*
never-ML-searched lenses is optimistic.**

**Biggest risk to the value:** **Euclid is industrializing space-based lens discovery.** Euclid
Q1 alone produced **497 space-based galaxy–galaxy candidates (250 grade A) in one quarter** —
doubling all previously-known space-based candidates — and is forecast to reach >100,000. A few
hundred extra single-band HST-archive lenses is marginal against that, *except* for the
high-resolution / small-θ_E / specific-deflector niches Euclid's 0.1–0.2″ PSF can't resolve.

## 4. Concrete cutout-generation pipeline (this part is solid and cheap)

A practical, end-to-end recipe — **a few person-months for one ML scientist on a single GPU
node**; the dominant ongoing cost is candidate vetting, not compute.

1. **Enumerate footprints.** `astroquery.mast` `Observations.query_criteria(obs_collection='HST',
   dataproduct_type='image', instrument_name in {ACS/WFC, WFC3/UVIS, WFC3/IR}, filters='F814W')`
   → `s_region` polygons; build a MOC coverage map. (REST: `https://mast.stsci.edu/api/v0/`.)
2. **Pick the data product.** **HAP Multi-Visit Mosaics** (drizzled, CR-cleaned, uniform scaling,
   **Gaia-DR3 aligned**) via the **HAPcut** cutout API
   (`https://mast.stsci.edu/hapcut/api/v0.1/astrocut`); for deep contiguous multi-band fields use
   **CANDELS/COSMOS/GOODS/HFF HLSP mosaics** (`https://archive.stsci.edu/hlsp/candels`) cut
   locally. Avoid raw FLT/FLC (distortion, CRs); **do not** anchor to HLA endpoints (retired
   mid-2025).
3. **Seed deflectors.** Query **Hubble Source Catalog v3** (`https://catalogs.mast.stsci.edu/hsc/`
   via CasJobs/TAP) for extended/massive galaxies inside footprints → one cutout per candidate
   deflector (~10⁶–10⁷ cutouts), far richer than blind tiling; add a blind overlapping grid
   (64–100 px, ~50% stride) for completeness.
4. **Standardize.** Resample **every** cutout to one canonical scale + box (e.g. **0.05″/px,
   80×80 px ≈ 4″**, matched to θ_E ~ 0.5–2″); carry per-cutout PSF/filter/instrument as metadata.
5. **Bands.** Most fields are **single-band F814W** → 1-channel input (or engineered
   [raw, smoothed, high-pass] pseudo-channels); map 2–4 band treasury fields to RGB with Lupton
   asinh scaling. **Honest caveat: losing color removes the red-deflector/blue-arc discriminant
   the grz CNN relies on → higher false-positive rate**, only partly offset by HST's ~10–25×
   finer resolution.
6. **Adapt the CNN.** Warm-start from the grz ResNet (Huang 2020/2021; Storfer 2024), swap the
   input stem to the HST channel count + box, re-init first-conv/final-FC, **transfer-learn**.
   Build the training set with **`deeplenstronomy`** (ships an **`hst`** survey preset mimicking
   WFC3/F814W with a Tiny Tim PSF) — simulate galaxy–galaxy lenses at the correct ACS/WFC3
   PSF/pixel-scale and **inject onto real HST non-lens cutouts** for realistic noise. Calibrate
   the threshold on known HST lenses (SLACS/BELLS HST imaging, Faure/Pawase candidates, the HAH-II
   finds as positives).
7. **Scale.** For millions of cutouts, bulk-pull drizzled mosaics from the **MAST AWS Open Data
   bucket** (`s3://stpubdata/hst/`) and cut **locally with `astrocut` on an EC2 node in
   us-east-1** (no egress, avoids HAPcut's 5 req/s & 25 Mpx limits).

**Gotchas to budget for:** require ≥3 contributing exposures (single/2-exposure fields leave CR
residuals that mimic compact arcs); always read the WHT/weight extension and mask zero-weight
pixels (chip gaps, bleeds, diffraction spikes); track field-dependent depth/PSF; expect elevated
single-band false positives (ring galaxies, face-on spirals, mergers, edge-on disks) → plan
human/agent vetting.

## 5. Recommendation

**Re-rate the HST general archive from "best-gross-yield drop-in" to "a cheap, well-scoped
completeness re-mine worth tens-to-~100 secure lenses."** It remains *legitimate and inexpensive*
— the CNN genuinely outperforms the by-eye HAH citizen pass at faint/small-separation systems,
and the pipeline above is a few person-months on one GPU. But it is **not** a flagship, and it is
**not** virgin sky. Practically:

- **Do it if** the goal is a clean, low-cost demonstrator of the space-PSF CNN retrain (it doubles
  as the pre-positioning prototype for **Euclid DR1 / Roman**, which *is* where the 10³–10⁵
  population lives), or to target a **specific niche** Euclid can't resolve (small-θ_E, particular
  deflector classes).
- **Don't expect** it to rival the program's DESI hauls. For "new findings with existing
  techniques," the HST archive now ranks **below VHS** (genuinely fresh far-south sky), **GAMA
  DR4** (clean method fit, no de-dup problem), and the **ATLAS/ZTF** time-domain streams — and its
  best framing is as the **HST→Euclid/Roman space-CNN readiness prototype**.

---

*Sources (verified, ≥2 per headline claim): Pawase et al. 2014 (MNRAS 439, 3392); Garvin/Kruk et
al. 2022 (A&A 667, A141, Hubble Asteroid Hunter II); Faure et al. 2008 (ApJS 176, 19); Jackson
2008 (MNRAS 389, 1311); Pourrahmani et al. 2018 (LensFlow, ApJ 856, 68); COWLS 2025 (MNRAS 543,
203); Collett 2015 (ApJ 811, 20); Euclid Q1 Strong Lensing Discovery Engine (arXiv:2503.15324);
SKYSURF (AJ 164, 141). Full structured output in the workflow result.*
