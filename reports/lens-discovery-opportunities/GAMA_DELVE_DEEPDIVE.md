# GAMA DR4 & DELVE DR2 — Pressure-Test of the Two Surviving Spectro/Optical Drop-ins

**Supplement to** `OPPORTUNITIES.md` · **Date:** 2026-06-06 · **Method:** combined 2-track,
8-agent workflow (3 research + adversarial red-team per survey; archived at
`workflow-gama-delve-deepdive.js`).

> **Headline.** These were the two drop-ins that most carried the "use existing techniques now"
> recommendation after HST and VHS fell. Both shrank: **GAMA DR4 — REFUTED (6 sources)**;
> **DELVE DR2 — DOWNGRADED (5 sources)**. With this, **all four** optical/NIR/spectroscopic
> "fresh public data, existing method" leads from the first scan have been downgraded or refuted
> (see the synthesis at the end).

---

## 1. GAMA DR4 — REFUTED

**Thesis:** *GAMA's multi-pass sampling defeats the fiber-collision block, so the spectro-FoF
method drops in for 10¹–10² net-new galaxy–galaxy lenses.* The causal chain breaks at two
independent links.

**The non-sequitur.** GAMA's multi-pass tiling *does* beat the 30–40″ single-config 2dF collision
wall (>97% close-pair completeness, down to a ~3″ photometric-deblending floor — a genuine ~15–20×
improvement over SDSS 55″ / BOSS 62″). **But fiber collision was never the binding constraint for
galaxy–galaxy lensing.**

**The physics kill-shot.** Classic galaxy–galaxy lenses have θ_E ~ 0.5–2″ (SLACS mean 1.2″; source
size ~0.2″), so the lensed source sits ~1–2″ from the deflector — **below GAMA's ~3″ two-fiber
deblending floor and inside the 2″ fiber.** Deflector + source land in **one fiber**, and the arc
is never an independent catalogue target. So the **two-fiber FoF/pair channel (Hsu 2025) cannot
catch galaxy–galaxy lenses on GAMA** — it reaches only wide/group pairs (>~3–5″).

**The G-G channel that works was already mined.** The single-fiber **blended-spectrum** method
(deflector absorption + background-source emission in one fiber — the SLACS/BELLS/DESI-single-fiber
approach) is the productive G-G spectroscopic channel, and **Holwerda et al. 2015 already executed
it on GAMA** (280 blended spectra → 104 strong G-G candidates), re-vetted by Knabel et al. 2020.
*(For scale, the same single-fiber channel on DESI just produced 4,110 candidates — DESI Single
Fiber Lens Search, arXiv:2512.04275 — confirming this, not FoF pairs, is the spectroscopic G-G
engine.)*

**Yield reality.** GAMA's ~250–286 deg² is ~1/36 of Hsu's ~9,000 deg², and the r<19.8 bright /
low-z sample deletes the high-z source population that powers ~40% of Hsu's DESI pairs; the
equatorial fields (G09/G12/G15) already sit inside DESI DR1 (Hsu-searched) and KiDS/DECaLS
(CNN-searched). **Realistic net-new across all channels: ~0–10 mixed-type candidates; net-new
classic G-G via FoF ~0.**

**Honest residual framings** (do *not* pitch GAMA-FoF as a G-G engine):
1. **Wide-separation / group-scale spectroscopic lens search** — the one genuinely un-run channel
   (a rarer, different population; order tens of candidates).
2. **A clean ~98%-complete (r<19.8) redshift + velocity-dispersion confirmation layer** for vetting
   deflector/source z of imaging-CNN candidates in the KiDS/HSC fields.
3. **A low-cost testbed** to prototype the single-fiber blended-spectrum pipeline (Holwerda 2015 as
   ground truth) before scaling to DESI / 4MOST-WAVES.
   The genuine *fresh-discovery* sliver is G23 (~50 deg², south of DESI's −18° SGC limit) + part of
   G02 — low single digits.

## 2. DELVE DR2 — DOWNGRADED

**Thesis:** *Re-running the optical CNN on DELVE DR2 griz harvests net-new southern lenses because
most of its ~17–21k deg² is deep, arc-capable, and not yet in a published lens catalog.* The minor
premises are true (deep, arc-capable DECam griz; no DR2-*labeled* lens paper) but the **major
premise — "fresh sky" — is false.**

**The de-dup.** DELVE DR2 tops out at **dec ~+30** (entirely below Legacy DR10's +32 search
ceiling), and **Inchausti 2025 / Legacy DR10 already CNN-searched ~14,000 deg² south of +32** —
nearly all extragalactic sky there — and **explicitly ingested the same DELVE/DeROSITAS DECam
exposures** DELVE DR2 is built from. So **~80–88% of DELVE DR2 is the same photons, same
instrument, comparable depth the group already searched.** The far-south escape hatch fails too:
DR10 reaches ~−68° via reprocessed DES.

**Genuinely fresh-and-deep residual: only ~2,000–4,000 deg²** — the low-Galactic-latitude
10<\|b\|<~18 annulus that the Legacy extragalactic cut excludes (plus a Magellanic-periphery
sliver) — and it is the **dustiest, most crowded, highest-false-positive** sky. At observed DECam
densities (~0.05–0.14 candidates/deg²), a blanket DR2 pass on that residual yields **order tens to
a few hundred mostly-low-grade candidates, not thousands.**

**The real levers** (better than a blanket DR2 re-run):
1. **DELVE DR3 deeper-coadd faint-deflector pass** — DR3 is ~0.5–0.7 mag deeper (g~24.9/r~24.5/
   i~24.0/z~23.5), the **only genuine depth gain** and the strongest standalone case.
2. **Targeted low-\|b\| (10<\|b\|<~18) frontier search** — sky excluded by *both* the Legacy
   extragalactic cut *and* LSST's \|b\|>15 cut, so it is uniquely DELVE's (low yield, high contamination).
3. **Reduction-diversity complementary catalog** on the overlap sky — an independent DESDM/Source-
   Extractor reduction surfaces candidates Legacy's Tractor missed (different-pipeline lens searches
   overlap only ~31–70%), a modest deblending/morphology bonus that adds *no new photons*.

**Biggest risk:** **Rubin/LSST overtakes this exact sky within 2–3 years** (WFD −70°<dec<+12.5°,
\|b\|>15°; single-visit r=24.5 already matches DELVE DR2; Year-1+ coadds 1–2 mag deeper, and
systematically lens-searched). The durable DELVE niche (low-\|b\|, outside LSST's cut) is precisely
the lowest-yield sky.

---

## 3. Synthesis: what four pressure-tests revealed

| Drop-in (first-scan) | First-scan rating | Deep-dive verdict | Why it shrank |
|---|---|---|---|
| **HST general archive** | drop-in, score 68 | **DOWNGRADED** → ~55 | pencil-beam + already crowd-scanned (HAH-II); completeness re-mine, tens–150 |
| **VHS** | drop-in, score 63 | **REFUTED** | ~3–4 mag too shallow to see arcs + not actually fresh (DECam covers it) |
| **GAMA DR4** | drop-in, score 64 | **REFUTED** | two-fiber FoF can't reach G-G separations; the blended channel was already done |
| **DELVE DR2** | drop-in, score 61 | **DOWNGRADED** | ~80–88% is the same sky already searched via Legacy DR10 |

**The consistent cause:** the southern DECam optical sky **and** the DESI/SDSS spectroscopic
catalogues are far more thoroughly mined — by the group itself (Legacy DR10, Hsu 2025) and the
community (HAH-II, Holwerda 2015, DESI single-fiber, KiDS/HSC CNNs) — than a first-pass landscape
scan credits. Each alternative modality then hits a hard wall: HST is pencil-beam, ground NIR is
too shallow for arcs, GAMA spectroscopy can't resolve G-G pairs, and DELVE re-treads searched sky.
A recurring coda is **"Euclid/Rubin will overtake this within 2–3 years."**

**What this means for "new findings with existing techniques":** there is **no large fresh-public-
data harvest** reachable with the current optical-CNN / spectro-FoF methods. The honest, *modest*
residuals that survive are:
- **Time-domain diff-imaging on ATLAS / ZTF** — the variability *content* is a genuinely different
  axis from static-imaging coverage and is the most robust surviving drop-in (not yet pressure-tested).
- **DELVE DR3 deeper pass + low-\|b\| frontier**; **GAMA wide-pair / blended testbed + confirmation
  layer**; **JWST COSMOS-Web** (small, deeper); **J-PLUS/S-PLUS** narrow-band (speculative).
- The **single-fiber blended-spectrum** channel at scale — but DESI is already being mined this way
  by others (arXiv:2512.04275), so the program's edge there is method/Foundation-Model, not data.

**Where the real, large opportunities actually are** (unchanged by these dives, because they were
never "existing-technique" claims):
1. **New-tooling green fields:** VLASS / ALMA-archive / ALMACAL (virgin modalities) and the
   **SDSS/BOSS SpectrumFM re-mine** — these require a build but are genuinely un-mined.
2. **Pre-positioning for the megasurveys:** Euclid DR1 (2026-10-21), Rubin/LSST (DP2 ~2026 Q3),
   Roman (≤2027), DESI DR2 (~2027) — where the 10³–10⁵ populations live, and which keep recurring as
   "the thing that obsoletes a marginal current-data search."

**Bottom line:** the most defensible program is **(a)** a small, honestly-scoped current-data effort
led by **time-domain diff-imaging (ATLAS/ZTF)** plus the DELVE-DR3/low-\|b\| and GAMA-confirmation
niches, run as **readiness prototypes**, while **(b)** investing the real effort in the new-tooling
green fields and **(c)** pre-positioning the pipelines for Euclid/Rubin/Roman.

---

*Sources (verified, ≥2 per claim): GAMA DR4 (Driver 2022, arXiv:2203.08539); GAMA close-pair
completeness (Robotham 2010/2014); Holwerda et al. 2015 (arXiv:1503.04813); Knabel et al. 2020
(arXiv:2009.09493); DESI Single Fiber Lens Search (arXiv:2512.04275); SLACS θ_E (Bolton 2006);
Patton et al. 2016 (fiber-collision pair recovery); DELVE DR2 (Drlica-Wagner 2022,
arXiv:2203.16565); DESI Legacy DR10 / Inchausti 2025 (arXiv:2508.20087); Zaborowski 2023
(arXiv:2210.10802); Rubin/LSST footprint + forecast (arXiv:2406.08919). Full structured output in
the workflow result.*
