# ATLAS / ZTF Time-Domain — Pressure-Test of the Diff-Imaging Opportunity

**Supplement to** `OPPORTUNITIES.md` · **Date:** 2026-06-06 · **Method:** 4-agent workflow
(lensed-SN saturation + lensed-QSO saturation + method-fit/obsolescence → adversarial red-team;
archived at `workflow-atlas-ztf-deepdive.js`).

> **Headline verdict: DOWNGRADED (16 corroborating sources) — but the *strongest survivor* of all
> five deep-dives.** The thesis ("ATLAS/ZTF time-domain diff-imaging is a robust net-new
> discovery opportunity") fails on three of its four load-bearing axes, yet it retains a genuine
> niche (ATLAS extreme-magnification glSNe) and — uniquely among the five — a real **forward-looking
> precursor value for the Rubin era**. So it lands a clean DOWNGRADED: same tier as HST/DELVE,
> a notch above the REFUTED VHS/GAMA. **Reframe it from "harvest" to "Rubin dress-rehearsal."**

---

## 1. Lensed supernovae — DOWNGRADED (ZTF saturated; ATLAS a thin but genuine gap)

**ZTF glSN content is NOT unsearched — it is a crowded, active field:**
- A full ZTF public-archive search (Magee, Sainz de Murieta, Collett & Enzi 2023, arXiv:2303.15439)
  scanned 15,215 transients over 4 yr → **0 compelling candidates.**
- An AMPEL pipeline (Townsend et al. 2025, arXiv:2405.18589) runs **live on the alert stream**
  (2 candidate glSNe Ia).
- **The group's *own* method** (Park, Shafieloo, Kim, Linder & X. Huang 2026, arXiv:2604.27511) — a
  blended/unresolved light-curve glSN finder — is already validated on 445 ZTF SNe Ia.
- All four world-public brokers (ALeRCE/Fink/Lasair/ANTARES) classify **every** ZTF alert in real time.
- Both real ZTF glSNe — **SN Zwicky** (Goobar 2023) and **SN 2025wny** (2025) — were caught **live by
  the magnification method, not at catalogued lens positions.**

**So retrospective diff-imaging at the 5,807 DESI lens positions would duplicate, not net-add:**
the expected number of glSNe at those positions over ~5 yr is **<<1** (per-lens SN rate ~10⁻³–10⁻²/yr).
The binding constraint on the realized rate is the **m<19 spectroscopic-screening limit** (≈0.5
glSN/yr screened; Sagués Carracedo 2024), not image subtraction — so **"tens of glSNe from ZTF" is
~10× optimistic** (realized ≈1 per 4 yr).

**ATLAS is the one genuine under-searched channel** — no dedicated lensed-SN pipeline exists, and the
public forced-photometry server is open — **but** its m~19–19.5 depth caps recoverable content at
**a handful of the most extreme magnified events over the whole archive** (consistent with the
first-scan "a handful" claim), and an all-sky magnification scan out-yields a position-targeted run.

## 2. Lensed quasars — DOWNGRADED, bordering REFUTED for net-new discovery

**Hard physics wall:** lensed-quasar image separations **rarely exceed 2″**, while ZTF samples at
1.01″/px with ~2″ FWHM (ATLAS coarser) — so **~90–95% of systems are unresolved.** Variability at a
known position can confirm an AGN but **cannot establish lensing.** The one published ZTF application
(Dux et al. 2026, GraL J1651−0417) worked only because it is an anomalous ~10″ quad, and it is
**re-characterization of an already-known lens**, not discovery.
- Discovery of lensed quasars is driven by **static Gaia/Lemon astrometry**, not variability.
- The variability toolkit (Springer & Ofek 2021; Bag et al. 2021) is mature/mined; the cosmology
  subset is intensively monitored (COSMOGRAIL/TDCOSMO; 73/364 known systems already have time delays).
- **In-house ceiling:** Sheu 2024a on *deeper, partially-resolving* DECam (0.262″/px) over 6,462
  systems netted only **13 new candidates** — a shallower, unresolved ZTF cannot beat that.
- "Tens-to-hundreds net-new" is **refuted at the hundreds end** (that scale is LSST: ~1,000
  variability-detectable, Tie 2023); defensible only as "tens of variability/time-delay
  **re-characterizations of already-known lenses**" (net-new lenses ≈0). ATLAS lensed-QSO ≈ 0.

## 3. Method fit — REFUTED as "diff-imaging drops straight in"

ZTF and ATLAS **already perform image subtraction internally and publish the products** — ZTF's
alert stream emits every above-threshold difference-image source as a public alert, and both offer
user-position **forced photometry.** So the program would **not** re-run Bramich-B08 differencing;
the real discovery mechanism is **broker alert-stream classification** (SN Zwicky was flagged as a
Hubble-residual overluminous outlier in the live stream). The program's genuine additive edge is a
**thin layer**: a known-lens **watchlist** on the existing alert/forced-phot streams + a **SpectrumFM
SN-type/host-z spectral gate** to attack the m<19 screening bottleneck.

## 4. Obsolescence — a ~1–2 year window (shorter than claimed)

Rubin/LSST's public alert stream goes live ~2026 at **~44–380 glSNe/yr** and **~1,000
variability-detectable lensed QSOs** (~100× scale), collapsing the effective ZTF/ATLAS precursor
window to **~1–2 yr** (the thesis said 2–3).

## 5. Realistic net-new yield

- **Lensed SNe:** ~0–3 highly-magnified glSNe total over a realistic 1–2 yr precursor (dominated by
  ATLAS's all-sky catch of the most extreme events). Net-new at the 5,807 DESI positions via
  position-targeted diff-imaging: **<<1.** Not "tens."
- **Lensed QSOs:** net-new *confirmed* lenses beyond Gaia/Lemon ≈ single-digit-at-most, realistically
  **~0** (resolution wall + false-positive follow-up bottleneck); the defensible deliverable is "tens
  of time-delay re-characterizations of known lenses."

## 6. Recommendation — reframe to a Rubin precursor (this is the constructive part)

Do **not** pitch a Bramich diff-imaging re-run or a "tens-of-glSNe / hundreds-of-lensed-QSOs"
harvest. The defensible, genuinely useful build is a **thin, fast precursor / dress-rehearsal for the
LSST broker era**, which is the same stack that scales onto Rubin's firehose:

1. **A known-lens real-time watchlist** — cross-match the live ZTF/ATLAS (and soon Rubin) broker
   alert + forced-photometry streams against the program's ~5,807 DESI lenses + ~6,500 candidates,
   so any transient at a known-lens position triggers immediate spectroscopy.
2. **A SpectrumFM SN-type / host-redshift spectral gate** — directly attacks the m<19
   spectroscopic-screening bottleneck that caps the realized glSN rate (and is a natural SpectrumFM
   application).
3. **ATLAS extreme-magnification glSN sweep** — the one defensible *standalone current-data
   discovery sliver* (no dedicated ATLAS pipeline exists), accepting a handful of events.

This is the best-justified "do-now" item of the five pressure-tested leads precisely because its
value is **forward-looking**: it builds and validates, on cheap public precursor data, exactly the
watchlist + spectral-gate infrastructure the program will need when Rubin delivers ~100× the
lensed-transient rate in 2026+.

---

## 7. Where this leaves all five pressure-tests

| Lead | Verdict | One-line |
|---|---|---|
| HST general archive | **DOWNGRADED** | completeness re-mine, tens–150 (not virgin) |
| VHS | **REFUTED** | too shallow to see arcs + not fresh |
| GAMA DR4 | **REFUTED** | FoF can't reach G-G; blended already done |
| DELVE DR2 | **DOWNGRADED** | ~80–88% already in Legacy DR10; DR3-depth is the lever |
| **ATLAS/ZTF time-domain** | **DOWNGRADED (strongest)** | precursor/watchlist + SpectrumFM gate → Rubin dress-rehearsal; ATLAS glSN sliver |

**No large current-data harvest with existing techniques survives.** But ATLAS/ZTF uniquely yields a
*constructive* next step (the watchlist + spectral-gate precursor) rather than just a smaller number.
The real, large opportunities remain the **new-tooling green fields** (VLASS, ALMA archive/ALMACAL,
the SDSS/BOSS SpectrumFM blended-spectrum re-mine) and **pre-positioning for Euclid DR1 / Rubin /
Roman / DESI DR2** — and the ATLAS/ZTF precursor is itself a form of Rubin pre-positioning.

---

*Sources (verified, ≥2 per claim): Magee/Sainz de Murieta/Collett/Enzi 2023 (arXiv:2303.15439);
Townsend et al. 2025 (arXiv:2405.18589); Sagués Carracedo et al. 2024 (arXiv:2406.00052); Goobar et
al. 2017 (iPTF16geu, Science); Goobar et al. 2023 (SN Zwicky, Nat. Astron.); Goldstein et al. 2019
(arXiv:1809.10147); Park, Shafieloo, Kim, Linder & Huang 2026 (arXiv:2604.27511); Springer & Ofek
2021 (arXiv:2110.15315); Dux et al. 2026 (A&A, GraL J1651−0417); Sheu et al. 2024a
(arXiv:2408.02670); Tie et al. 2023 / Rubin lensed-transient forecasts; ZTF alert/forced-phot docs
(IRSA/IPAC); ATLAS forced-photometry server (fallingstar-data.com). Full structured output in the
workflow result.*
