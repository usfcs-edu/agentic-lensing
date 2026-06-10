# VHS (VISTA Hemisphere Survey) — Pressure-Test of the NIR Strong-Lens Opportunity

**Supplement to** `OPPORTUNITIES.md` · **Date:** 2026-06-06 · **Method:** 4-agent research
workflow (footprint/depth + prior-NIR-coverage/depth + pipeline → adversarial red-team; archived
at `workflow-vhs-deepdive.js`). *(The run hit transient 529 errors mid-way and was resumed from
cache; the cached footprint/depth and prior-coverage agents are unaffected.)*

> **Headline verdict: REFUTED (8 corroborating sources).** VHS was the top "fresh far-south sky"
> drop-in in the first scan (score 63, ~10–100 yield). The pressure-test refutes **both**
> load-bearing claims: VHS is **~3–4 mag too shallow to detect lensed arcs** (it sees the
> deflector, not the lens), and the **"~60% fresh sky" is a large overstatement** — deep DECam
> optical already blankets the southern hemisphere. Realistic net-new NIR-confirmed yield: **~0 to
> low-tens**, mostly already-known or deflector-only systems. VHS is **not a blind discovery
> channel**; its only defensible role is ancillary NIR photometry layered on existing optical.

---

## 1. The depth kill-shot (the binding failure)

Lensed arcs are typically **faint blue star-forming galaxies** — bright in observed optical,
intrinsically faint in observed J/Ks — and arc detection is **surface-brightness-limited**
(lensing conserves surface brightness; magnification only buys ~√μ in S/N).

| | VHS (most of footprint) | VHS-DES (deepest) | Where ground lens-finding works |
|---|---|---|---|
| 5σ point-source depth (AB) | J~20.8, Ks~20.0 | J~21.4, H~20.7, Ks~20.3 | DES i~24, KiDS r~25, HSC i~26 |
| Blue coverage | **none** (J/H/Ks only) | none | g, r essential for blue arcs |
| Effective **extended/arc** SB limit | ~J 19–20 / Ks 18–19 | ~1 mag brighter than PS | arcs sit at i~22–24+ |
| Seeing / pixel | ~0.9–1.0″ / 0.34″/px | same | comparable (not the bottleneck) |

VHS is **~3–4 mag shallower** than the optical surveys where ground-based galaxy–galaxy
lens-finding works, *and* has no blue bands. A typical faint blue arc falls **below** the VHS
limit, so a VHS-only CNN would be **deflector-color-driven, not arc-driven** — it recovers the red
deflector and mostly re-finds systems already known from optical. Resolution is *not* the problem
(0.9–1.0″ ≈ DECam); **depth and bandpass are.**

**Confirming evidence:** there is **no published blind wide-NIR galaxy–galaxy strong-lens
discovery catalog** from VHS, VIKING, UKIDSS, or 2MASS — *ever*. All wide-NIR lensing results are
either (a) NIR **counterparts** to submm-selected lenses (VIKING ↔ Herschel-ATLAS; Bourne 2012,
Edge 2013, Ward 2022) or (b) NIR **color cuts** on optically/quasar-selected lenses (UKIDSS+SDSS
MUSCLES; 2MASS quasar excess; DES+VHS+unWISE lensed-QSO CNNs graded *on the optical*, Yue 2022).
The first genuinely NIR-*selected* blind galaxy–galaxy catalogs are **space-based** (Euclid NISP,
JWST), at ~24 AB and 0.3″ PSF — not shallow ground NIR. Holloway et al. 2023 (arXiv:2308.00851)
explicitly frame blind wide-NIR searches as "forthcoming" and show the NIR-favored population only
materializes at deep *space* NIR depth.

## 2. The fresh-sky overstatement

- **There is no "VISTA-only deep south."** VISTA (Paranal, −24.6°) and DECam (CTIO, −30.2°) both
  reach δ ≈ −90°; DECam images the LMC/SMC. So the geographic premise is false.
- **Deep DECam optical already blankets the southern hemisphere, 3–4 mag deeper than VHS:**
  **DELVE DR2** delivers ~21,000 deg² in griz (~17,000 deg² in all four; g~24.3/r~23.9/i~23.5/z~22.8
  AB) over \|b\|>10 south; **DECaPS2** fills ~2,700 deg² of the plane; **DES** ~5,000 deg²;
  **DESI Legacy DR10** ~14,000 deg² south of +32.
- **Most of it is already lens-searched** — DESI Legacy DR10 by the Huang group itself (Inchausti
  2025), plus DES Y6 and DELVE DR1 (Zaborowski 2023).
- The only VHS sky with *no* deep optical is the **\|b\|<10 Galactic plane** (VHS-GPS), which is
  dust/crowding-hostile and unusable for galaxy–galaxy lens finding.
- **Net genuinely-fresh, NIR-only, lens-usable sky: ~0–10%**, not ~60% — and that sliver is the
  unusable plane.

## 3. The fusion tension and obsolescence

- The scientifically sound design would be **NIR+optical fusion** (VHS Ks/J for the red deflector
  + a deep blue band for the arc). But this hits a **central tension**: *where deep optical
  exists, the fresh-sky claim collapses and it's already searched; where the sky is genuinely
  fresh, there is no deep blue band* (only shallow SkyMapper g~23 at ~1.5″, or Gaia). So the fusion
  degrades exactly where you'd want it.
- **Obsolescence within ~2–3 years:** **Euclid** (Q1 already 497 candidates in 63 deg² → forecast
  >100,000 over the 14,000 deg² Wide Survey, including the south, at VIS~26.5/NISP~24, 0.16–0.3″
  PSF) and **Rubin/LSST** (~120,000 galaxy-scale lenses over 18,000 deg² south, ugrizy, 30–45%
  delivered in years 1–2) will harvest this exact far-south sky vastly better, on both depth and
  resolution, before a shallow 2-band VHS search could publish.

## 4. The pipeline (feasible, but the science caps it)

Access is mature and the engineering is ~2–4 weeks (deflector seeding + cutout pipeline; bulk
tiles are tens of TB), with ~3–6 person-months to a first NIR-retrained/fusion prototype — but the
**yield is capped by physics, not engineering**.
- **Access:** VSA/WFAU Freeform SQL `http://horus.roe.ac.uk/vsa/` + MultiGetImage cutouts; NOIRLab
  Astro Data Lab TAP **(catalog only — no VHS image/SIA there)** `vhs_dr5.vhs_cat_v3`; ESO Phase 3
  tiles `http://archive.eso.org/.../phase3_main/form?phase3_collection=VHS`.
- **Product:** 1.5 deg² **tile** images (not pawprints); seed deflectors from the band-merged
  catalog (MERGEDCLASS=galaxy, bright/extended/red J−Ks), cut ~20–25″ (~60–72 px) boxes.
- **Bands:** reliably **2-band J,Ks** (Y on ~4,800 deg², H on ~2,900 deg² only). 3-channel input =
  **[J, Ks, (J−Ks)]**. This discards the grz blue-arc discriminant entirely.
- **CNN:** a **retrain** (not transfer) — the optical weights don't apply (different bands/PSF/
  pixscale). The hard part is the **positive set**: no real blind-NIR lenses exist, so paint
  simulated arcs into real VHS tiles degraded to VHS depth/PSF.
- **Gotchas:** bright/variable NIR sky, VIRCAM persistence/cross-talk, tile seams, inhomogeneous
  depth across VHS-GPS/ATLAS/DES, and arcs blending with deflectors at ~1″ seeing.

## 5. Recommendation

**Drop VHS from the drop-in shortlist as a discovery channel.** Re-rate it from YELLOW/score-63 to
**RED / low (REFUTED)**. Defensible residual roles only:
1. **Ancillary NIR photometry** layered on existing deep DECam optical — better deflector photo-z,
   red/massive early-type deflector selection, and flagging the niche red/dusty/high-z tail.
2. **Submm-lens deflector counterparts** (Herschel-ATLAS), at ~0.1–0.3/deg² — but those lenses are
   *found by submm*; VHS only supplies photometry, so it is not a blind-VHS yield.

**What VHS points to instead:** the genuine southern residual is **re-running the optical CNN on
DELVE DR2 griz** — ~17,000–21,000 deg² of deep 4-band DECam imaging, most of which is *not yet in a
published lens catalog* (only DELVE DR1 ~4,000 deg² was searched; Legacy DR10 used a bright z<20
deflector cut). That is deep, arc-capable, and a true drop-in for the existing CNN — strictly
better than a shallow NIR search of the same sky. **DELVE DR2 (already in the report at score 61)
is the right southern play; VHS is not.**

---

*Sources (verified, ≥2 per claim): VHS DR5 (ESO release description 144; McMahon 2013); VIRCAM
specs (A&A 2015); DELVE DR2 (arXiv:2203.16565); DECaPS2 (arXiv:2206.11909); SkyMapper DR2
(arXiv:2008.10359); DESI Legacy DR10 lens search / Inchausti 2025 (arXiv:2508.20087); DES Y6;
Holloway et al. 2023 NIR detectability (arXiv:2308.00851); Bourne 2012 / Edge 2013 / Ward 2022
(VIKING↔H-ATLAS counterparts); Yue 2022 (arXiv:2211.14543); Euclid Q1 Discovery Engine
(arXiv:2503.15324); Rubin lens forecast (arXiv:2406.08919). Full structured output in the workflow
result.*
