# Where to Look Next: A Data-Landscape & Opportunity Scoping Report for Strong-Lens Discovery

**Internal scoping report — agentic-lensing program (X. Huang group)**
**Author:** Greg Benson (with a multi-agent deep-research workflow) · **Date:** 2026-06-06

> **Provenance.** This report was produced by a deterministic multi-agent research workflow:
> 8 survey-family research agents → adversarial red-team verification of every flagged
> opportunity → priority-scoring synthesis → completeness critic (51 agents, ~2.2M tokens,
> 781 web/tool calls). Every GREEN/YELLOW rating below was challenged by an independent
> skeptic agent on four axes (*already searched? public now? new sky? method fits?*); a GREEN
> survives only with ≥2 corroborating sources. Of 41 challenged opportunities, **34 were
> CONFIRMED, 6 DOWNGRADED, 1 REFUTED**. The lens-finding itself is **out of scope** — this
> report only tells you *where* the un-mined data is.

> **⚠ Pressure-test log (deeper follow-up dives).** Individual opportunities are being
> stress-tested with dedicated 4-agent deep-dives (inventory + prior-coverage + pipeline →
> adversarial red-team). These *supersede* the first-scan ratings where they disagree:
> | Dataset | Deep-dive verdict | Net-new yield (revised) | Detail |
> |---|---|---|---|
> | **HST general archive** | **DOWNGRADED** (was 68) | **tens–~150 secure** — a completeness re-mine, not virgin sky | `HST_ARCHIVE_DEEPDIVE.md` |
> | **VHS** | **REFUTED** (was 63; 8 src) | **~0–low-tens** — too shallow to see arcs (deflector-only) + not fresh | `VHS_DEEPDIVE.md` |
> | **GAMA DR4** | **REFUTED** (was 64; 6 src) | **~0 net-new G-G** — two-fiber FoF can't reach θ_E~0.5–2″; blended channel already done (Holwerda 2015) | `GAMA_DELVE_DEEPDIVE.md` |
> | **DELVE DR2** | **DOWNGRADED** (was 61; 5 src) | **tens–few-hundred low-grade** — ~80–88% already searched via Legacy DR10; DR3-depth + low-\|b\| are the real levers | `GAMA_DELVE_DEEPDIVE.md` |
> | **ATLAS / ZTF time-domain** | **DOWNGRADED — strongest survivor** (was 75/62; 16 src) | **~0–3 glSNe / 1–2 yr; net-new lensed-QSO ≈0** — ZTF glSN field saturated (incl. the group's *own* Park/Huang 2026 method); diff-imaging redundant (alerts already differenced); lensed-QSO blocked by the <2″ resolution wall. **But the only lead with a constructive reframing → a Rubin precursor.** | `ATLAS_ZTF_DEEPDIVE.md` |
>
> First-scan numbers below are kept for the audit trail but struck through / annotated where a
> deep-dive revised them. **⚠ Meta-finding — read this first:** *all five* current-data "fresh
> public data, existing method" leads I led with have now been **downgraded or refuted** (2 refuted,
> 3 downgraded). The consistent cause: the southern DECam optical sky, the DESI/SDSS spectroscopy,
> **and** the ZTF time-domain stream are far more thoroughly mined — by the group itself (Legacy
> DR10, Hsu 2025, Park/Huang 2026) and the community (HAH-II, Holwerda 2015, DESI single-fiber,
> KiDS/HSC CNNs, Magee 2023, Goobar group, 4 brokers) — than a first-pass scan credits, and each
> alternative modality hits a hard wall (HST pencil-beam; NIR too shallow for arcs; GAMA can't
> resolve G-G pairs; DELVE re-treads searched sky; ZTF saturated + diff-imaging redundant).
> **There is no large fresh-public-data harvest reachable with current techniques.** *However*,
> the **ATLAS/ZTF dive is the one that points somewhere constructive** — a known-lens **watchlist +
> SpectrumFM spectral gate** that doubles as **Rubin-era readiness** (§8a). See the per-dataset
> supplements and `ATLAS_ZTF_DEEPDIVE.md` §6–7 for the full synthesis.

---

## 1. Executive summary

The program has searched the DESI optical-imaging and DESI-DR1 spectroscopic footprints about
as hard as they can be searched with current methods (see §2). The honest finding of this scan
is that **the easy, in-footprint, same-modality opportunities are largely gone** — the major
peer surveys (HSC PDR3, KiDS DR5, DES Y6, UNIONS, Euclid Q1) are *already searched and
published*. The real opportunities fall into four buckets:

1. **Virgin off-modality reservoirs (highest novelty, needs new tooling).** Radio and
   submm/mm sky has **never had a blind catalog-scale strong-lens search** — only forecasts and
   known-lens cross-matches exist. **VLASS** (33,885 deg², public), the **ALMA Science Archive**
   + **ALMACAL** (a petabyte of interferometric data never lens-searched), and the faint-DSFG
   tail of **Herschel/SPT-3G/ACT** are genuine green fields. They require a radio/submm
   lens-finder build, but VLASS uniquely can *also radio-confirm the program's existing ~3,900
   optical candidates*.

2. **Drop-in re-points of the existing pipelines (public now).** ⚠ **Read §8a and the
   pressure-test log first — this bucket did not survive scrutiny.** All four leads I first put
   here were downgraded or refuted on deep-dives: **HST archive** (completeness re-mine, not
   virgin), **VHS** (too shallow for arcs), **GAMA DR4** (FoF can't reach galaxy–galaxy
   separations; blended channel already done), and **DELVE DR2** (~80–88% already searched via
   Legacy DR10). The honest survivors are *narrow*: **time-domain diff-imaging on ATLAS / ZTF**
   (the most robust — variability is a different axis from imaging coverage), **DELVE DR3** (a
   real *depth* gain) + the low-\|b\| frontier, **JWST COSMOS-Web** (small, deeper), and
   **J-PLUS/S-PLUS** narrow-band (speculative). **There is no large fresh-public-data harvest with
   current methods.**

3. **A foundation-model re-mine of legacy spectroscopy.** ~5M public **SDSS/BOSS/eBOSS** spectra
   cannot be pair-searched (the 55″/62″ fiber-collision radius forbids two fibers on a
   galaxy-scale pair) and the single-fiber emission-residual method is exhausted — but a
   SpectrumFM-style foundation-model re-mine of the full spectral set is an open, build-once lever.

4. **Pre-positioning for the megasurveys.** **Euclid DR1 Wide** (public 2026-10-21, ~10³–10⁴
   lenses), **Rubin/LSST** (DP2 ~2026 Q3; world-public alert stream), **Roman HLWAS** (launch
   by 2027, ~160,000-lens forecast), **DESI DR2 spectra** (~2027), and **4MOST** (the only large
   *fresh southern* spectroscopy) are the eventual prizes — worth early/joint-access inquiries now.

**Top opportunities at a glance** (priority score / tier / pipeline fit / access effort 1–5):

| # | Opportunity | Score | Tier | Fit | Effort | One-line |
|---|---|---|---|---|---|---|
| 1 | **VLASS** (VLA Sky Survey) | 78 | new-tooling | new | 2 | 33,885 deg² radio, no blind lens search ever; can also radio-confirm existing optical lenses |
| 2 | **ALMA Science Archive** + **ALMACAL** | 78/75 | new-tooling | new | 5/4 | Petabyte of mm/submm interferometry, never systematically lens-searched |
| 3 | **ATLAS/ZTF** time-domain ⚠ | ~~75/62~~→**~55/45** | drop-in | diff-img | 2 | ⚠ *downgraded (strongest survivor): ZTF glSN field saturated + diff-imaging redundant; lensed-QSO blocked by <2″ wall. Reframe as a **Rubin precursor** — known-lens watchlist + SpectrumFM gate. `ATLAS_ZTF_DEEPDIVE.md`* |
| 4 | **HST general archive** (HLA + treasury) | ~~68~~→**~55** | drop-in | CNN | 3 | ⚠ *downgraded:* not virgin (Hubble Asteroid Hunter II already scanned ~27 deg²); ~tens–150 net-new secure via completeness re-mine. See `HST_ARCHIVE_DEEPDIVE.md` |
| 5 | ~~**GAMA DR4**~~ ⚠ **REFUTED** | ~~64~~→**low** | ~~drop-in~~ | spec-FoF | 1 | *two-fiber FoF can't reach galaxy–galaxy θ_E~0.5–2″ (lands in one fiber); the blended channel that works was already done (Holwerda 2015). ~0 net-new G-G. See `GAMA_DELVE_DEEPDIVE.md`.* |
| 6 | ~~**VHS**~~ ⚠ **REFUTED** | ~~63~~→**low** | ~~drop-in~~ | CNN(NIR) | 2 | *too shallow to see arcs (deflector-only) + not actually fresh — DECam already covers it. See `VHS_DEEPDIVE.md`. Use **DELVE DR2 griz** instead.* |
| 7 | **SPT-3G / Herschel** faint-DSFG tail | 66/63 | new-tooling | new | 2–4 | Large undelivered lensed-DSFG candidate populations below the bright cut |
| 8 | **JWST COSMOS-Web** + **DELVE DR2/3** ⚠ | 62 / ~~61~~→**~45** | drop-in | CNN | 2 | JWST: deeper space re-search. *DELVE DR2 **downgraded** — ~80–88% already searched via Legacy DR10; the real levers are **DR3 depth** + the low-\|b\| frontier. See `GAMA_DELVE_DEEPDIVE.md`.* |
| — | **Euclid DR1 / Rubin / Roman / DESI-DR2** | 70–58 | future-watch | CNN/diff/FoF | 3–4 | The megasurvey prizes — pre-position with early/joint-access inquiries |

---

## 2. Scope & baseline (what is *excluded* as already-searched)

This report excludes the footprints the program has already mined, indexed in
`reproductions/REPRODUCTIONS.md`. These are the **baseline**, not opportunities:

| Already searched (baseline) | Method | Result |
|---|---|---|
| DESI Legacy Imaging DR7/8/9/10 (DECaLS+BASS+MzLS, ~14k–19k deg², grz/+i) | optical-CNN finders (Huang 2020/2021; Storfer 2024; Inchausti 2025) | ~3,900 galaxy-scale candidates |
| DESI DR1 (Iron) spectroscopy (28M fiber redshifts) | spectroscopic FoF / pair (Hsu 2025) + QSO autocorrelation (Dawes 2022) | ~2,600 candidates |
| DECam per-exposure multi-epoch over the DECaLS/DES lens footprint | difference imaging (Sheu 2023 lensed SNe; Sheu 2024a variable lensed QSOs) | time-domain confirmations |
| HST/MAST, Keck NIRES, VLT/MUSE, **Euclid Q1** | follow-up / confirmation only (Foundry I–IV); Q1 is a benchmark | not discovery |

**The three in-house methods** (used for the *fit-to-pipeline* column throughout):
1. **optical-cnn** — CNN/ViT finder on optical *grz(i)* cutouts (retrains for NIR or space PSF as a config change).
2. **spectro-fof** — Friends-of-Friends / redshift-pair search on a spectroscopic redshift catalog.
3. **diff-imaging** — difference-imaging time-domain search on multi-epoch exposures.

Anything else (radio source-matching, uv-plane lens modelling, submm DSFG selection, IFU,
astrometric multiplets, foundation-model spectral re-mine) is tagged **new-tooling**.

---

## 3. How to read this report (rating rubric)

- **RAG (verified):** **GREEN** = largely un-searched / fresh & public-now; **YELLOW** =
  partially searched, or searched-but-unpublished, or future-but-high-value, or re-searchable
  deeper; **RED** = exhaustively searched & published (or unviable), low residual.
- **Priority score (0–100)** = `0.75·opportunity + 0.25·actionability`. *opportunity* weighs
  the verified RAG (45%), genuinely-fresh-and-unsearched sky/modality (30%), and expected
  net-new yield (25%). *actionability* weighs pipeline fit (45%), access effort (25%), and
  public-availability (30%). The 0.75/0.25 weighting deliberately lets **scientific opportunity
  dominate convenience** — so an exhausted-but-trivial RED (DES Y6) lands ~32 while a
  green-but-build-heavy submm reservoir (ALMA archive) scores ~78 despite effort-5.
- **Tier:** **drop-in-now** (public + an in-house method fits + not exhausted) · **new-tooling-now**
  (public + real opportunity but needs a radio/submm/IFU/foundation-model build) ·
  **future-watch** (embargoed/future, high ceiling) · **low** (RED / exhausted / unviable).
- **Red-team verdict:** **CONFIRMED / DOWNGRADED / REFUTED**, with a corroborating-source count.

---

## 4. Master opportunity table

All 55 datasets, ranked by priority. `Avail` = public-now (●) / embargoed (◐) / future (○).
`Fit`: CNN = optical-cnn, FoF = spectro-fof, Diff = diff-imaging, New = new-tooling.

| Score | Tier | RAG | Avail | Fit | Dataset | Footprint | New-sky | Eff | Net-new yield |
|---:|---|:--:|:--:|:--:|---|---|---|:--:|---|
| 78 | new-tooling | 🟢 | ● | New | **VLASS** (VLA Sky Survey, ~3 GHz) | ~33,885 deg² | ~½ sky, 100% modality | 2 | O(10–10²) blind |
| 78 | new-tooling | 🟢 | ● | New | **ALMA Science Archive** (PI + calibrators) | tens of thousands of fields | 100% modality | 5 | O(10–10²) serendip. |
| 75 | new-tooling | 🟢 | ● | New | **ALMACAL** (calibrator blind survey) | ~0.3 deg² (1,000 sightlines) | 100% modality | 4 | O(1–10) |
| ~~75~~→**~55** | drop-in | 🟡 | ● | Diff | **ATLAS** (Tonry) all-sky forced phot ⚠ | ~30,000+ deg² | ATLAS = the one genuine glSN gap (no dedicated pipeline) | 2 | handful of *extreme-mag* glSNe over the whole archive (depth-capped) |
| 70 | future-watch | 🟡 | ○ | CNN | **Roman** HLWAS + HLTDS | ~1,700–2,000 deg² | ~50–70% | 4 | ~160,000 (forecast) |
| 69 | future-watch | 🟡 | ◐ | Diff | **Rubin/LSST** (DP1 now; DP2/DR1 future) | WFD ~18,000 deg² | ~0.9 depth/epoch | 4 | ~10⁵ full survey |
| 68 | future-watch | 🟡 | ○ | CNN | **Euclid DR1 — Wide Survey** | ~1,900 deg² | ~50–65% fresh | 3 | ~7,000 A/B (forecast) |
| ~~68~~→**55** | drop-in | 🟡 | ● | CNN | **HST general archive** (HLA + treasury) ⚠ | ~25–35 deg² (pencil-beam; ~2.5–3 contiguous) | ~10–20 deg² never-blind-searched (mostly re-mine) | 3 | tens–150 net-new secure |
| 68 | future-watch | 🟡 | ○ | FoF | **4MOST** (on VISTA) | CRS ~5,700; total ~17,000 | ~70–90% (fresh south) | 5 | 10³–10⁴ |
| 66 | new-tooling | 🟡 | ● | New | **Herschel-ATLAS** (DR1+DR2) | ~660 deg² | ~0.9 of lens pop. | 2 | O(10³) faint forecast |
| 65 | new-tooling | 🟡 | ● | New | **HerMES + HeLMS + HerS** | ~760 deg² | ~0.9 of lens pop. | 2 | O(10²–10³) faint |
| ~~64~~→**low** | ~~drop-in~~ | 🔴 | ● | FoF | **GAMA DR4** ⚠ REFUTED | ~286 deg² | n/a (FoF can't reach G-G; blended done) | 1 | ~0 net-new G-G |
| ~~63~~→**low** | ~~drop-in~~ | 🔴 | ● | CNN | **VHS** ⚠ REFUTED | ~17,000–20,000 deg² | ~0–10% (DECam covers it deeper) | 2 | ~0–low-tens (deflector-only; too shallow for arcs) |
| 63 | new-tooling | 🟡 | ● | New | **SPT-3G** emissive source catalog | ~1,500 deg² | ~0.85 vs SPT-SZ | 4 | O(10²–10³) lensed DSFGs |
| 62 | drop-in | 🟡 | ● | CNN | **JWST COSMOS-Web** (NIRCam) | 0.54 deg² | ~30–50% | 2 | 10¹–10² |
| ~~62~~→**~45** | drop-in | 🟡 | ● | Diff | **ZTF** (public DRs + alerts) ⚠ DOWNGRADED | ~30,000 deg² | saturated (Magee→0; AMPEL live; Park/Huang 2026; 4 brokers) | 2 | net-new ≈0 (glSN ~1/4yr already caught live; lensed-QSO blocked by <2″ wall) |
| ~~61~~→**~45** | drop-in | 🟡 | ● | CNN | **DELVE DR2 + DR3** ⚠ DOWNGRADED | DR3 ~20,000 deg² | ~2–4k deg² (low-\|b\|; ~80–88% already in Legacy DR10) | 2 | tens–few-hundred low-grade; DR3-depth is the real lever |
| 60 | future-watch | 🟡 | ◐ | CNN | **UNIONS** (CFIS u/r + PS/HSC g/i/z) | ugriz ~3,730 deg² | ~20–30% literal | 4 | 10²–10³ |
| 60 | drop-in | 🟡 | ● | CNN | **J-PLUS / S-PLUS** (12-band) | ~3,000–3,200 deg² each | ~20% (modality) | 2 | ~10–100 (speculative) |
| 60 | future-watch | 🟡 | ◐ | FoF | **WEAVE** spectroscopic surveys | thousands (north) | ~40–60% fresh | 4 | 10²–10³ |
| 60 | future-watch | 🟡 | ◐ | FoF | **Subaru PFS** (SSP) | Cosmo ~1,100–1,400 deg² | ~100% as spectroscopy | 4 | 10²–10³ |
| 59 | future-watch | 🟡 | ○ | FoF | **DESI-II / Spec-S5** | DESI-II ~14k; S5 wider | deeper/higher-z | 5 | 10³–10⁴ |
| 58 | future-watch | 🟡 | ◐ | FoF | **DESI DR2** spectroscopy (3-yr) | ~14,000 deg² | ~50% new objects | 4 | 10³ |
| 58 | new-tooling | 🟡 | ● | New | **ACT** extragalactic mm catalog (DR6) | DR6 ~19,000 deg² | ~0.85 of lens pop. | 4 | O(10²) |
| 57 | drop-in | 🟡 | ● | FoF | **LAMOST** DR11/DR12 | northern (δ>−10°) | ~30% fresh vs SDSS | 2 | 10¹–10² |
| 57 | drop-in | 🟡 | ● | Diff | **DECam multi-epoch** outside Sheu footprint | DES-SN ~27; DELVE-DEEP ~135 +Magellanic ~2,200 | ~0.45 | 4 | tens of lensed transients |
| 56 | drop-in | 🟡 | ● | CNN | **VISTA VIKING** | ~1,350 deg² | ~15% (modality) | 2 | ~10–100 NIR-favored |
| 56 | new-tooling | 🟡 | ● | New | **ASKAP EMU + RACS** | RACS ~34,000/band; EMU ~20,000 | ~0.6–0.7 sky | 3 | galaxy O(0–10); cluster O(10) |
| 54 | future-watch | 🟡 | ○ | CNN | **HSC-SSP PDR4** (increment over PDR3) | newly-full-depth Wide ~few×100 | low-moderate | 4 | ~10² |
| 53 | drop-in | 🟡 | ● | CNN | **UKIDSS LAS** | ~4,000 deg² | ~10–15% | 2 | ~10 |
| 53 | new-tooling | 🟡 | ● | New | **Planck PHz / PCCS2** all-sky submm | all-sky; PHz ~10,700 | ~0.6 | 3 | O(10¹–10²) residual |
| 52 | drop-in | 🟡 | ● | CNN | **VIDEO** | ~12 deg² | ~10% | 2 | ~1–10 |
| 51 | drop-in | 🟡 | ● | CNN | **VST-ATLAS DR4** | ~4,700 deg² | ~0% sky; u-band differentiator | 3 | O(10), mostly re-discoveries |
| 50 | new-tooling | 🟡 | ● | New | **LOFAR LoTSS DR2** (+ILT sub-arcsec) | ~5,700 deg² | ~0.1–0.2 sky, 100% modality | 4 | O(10²–10³) if sub-arcsec |
| 50 | new-tooling | 🟡 | ● | New | **ALMA Lensing Cluster Survey (ALCS)** | ~0.037 deg² | ~0.95 modality | 3 | O(1–10) galaxy-scale |
| 48 | new-tooling | 🟡 | ● | New | **SDSS/BOSS/eBOSS** legacy spectroscopy (+MaNGA) | ~9,400–10,000 deg² | ~0–15% residual | 1 | 10² (FM re-mine only) |
| 48 | new-tooling | 🟡 | ● | New | **MeerKAT** MIGHTEE + MALS | MIGHTEE ~20; MALS ~2,289 deg² | MALS ~0.4–0.5 | 3 | O(0–10) |
| 47 | low | 🟡 | ● | New | **Gaia GraL / GravLens** lensed-QSO | all-sky | ~40–60% point-source modality | 3 | 10²–10³ (largely worked) |
| 46 | low | 🟡 | ● | CNN | **SkyMapper** DR4 ⟶ *REFUTED→red* | ~21,000–26,000 deg² | ~0% exploitable | 2 | O(1–10) bright; ~0 net-new |
| 44 | low | 🔴 | ● | CNN | **KiDS DR5** / KiDS-Legacy | 1,347 deg² | ~30–40% but pre-searched | 1 | 10¹–10² incremental |
| 44 | new-tooling | 🟡 | ● | New | **Gaia Photometric Science Alerts** | all-sky | ~0.3 (alert modality) | 2 | 0–1 (bright limit) |
| 42 | low | 🔴 | ● | CNN | **JWST** other deep fields (JADES/PRIMER/…) | ~0.3–0.5 deg² cum. | ~80–95% but now searched | 3 | 10¹–10² (largely searched) |
| 41 | low | 🔴 | ● | CNN | **Pan-STARRS1 3π DR2** | ~30,000 deg² | overlapped + shallow | 1 | 10¹–10² incremental |
| 36 | low | 🔴 | ● | CNN | **HSC-SSP PDR3** | Wide ~670 deg² full-depth | ~10–20% | 1 | 10¹–10² incremental |
| 35 | low | 🔴 | ● | CNN | **DECaLS-south DR10 i-band** increment | ~14,000 deg² | ~0% | 1 | ~0 at bright cut |
| 35 | low | 🔴 | ● | FoF | **2dFGRS + 6dFGS** | 2dF ~1,500; 6dF ~17,000 deg² | shallow/low-z | 1 | ~10⁰ |
| 34 | low | 🔴 | ● | CNN | **HST COSMOS** (ACS F814W) | ~2 deg² | ~0% | 2 | ~10² published |
| 34 | low | 🔴 | ● | FoF | **DEVILS DR1** | ~6 deg² | ~0% | 2 | 10⁰–10¹ |
| 33 | low | 🔴 | ● | CNN | **Euclid Q1** — Deep Fields | 63.1 deg² | ~0% (exhausted) | 1 | ~500 published |
| 32 | low | 🔴 | ● | CNN | **DES Y6 / DR2** | ~5,000 deg² | ~0% | 1 | ~0 net-new |
| 32 | low | 🔴 | ● | CNN | **Southern Pan-STARRS** strip | a few thousand deg² | ~0% | 1 | ~0 net-new |
| 32 | low | 🔴 | ● | CNN | **UltraVISTA** (COSMOS) | ~1.5–1.8 deg² | ~0% | 2 | ~0–10 |
| 30 | low | 🔴 | ● | New | **SPT-SZ** DSFG sample | ~2,500 deg² | exhausted | 2 | O(10¹) residual |
| 29 | low | 🔴 | ● | New | **FIRST + NVSS** (incl. CLASS/JVAS) | FIRST ~10,575; NVSS ~33,800 deg² | low residual | 1 | O(0–1) |
| 23 | low | 🔴 | ● | New | **VVV / VVVX** | VVV ~560; VVVX ~1,700 deg² | Zone of Avoidance | 4 | ~0–1 |

---

## 5. Findings by family

### 5.1 Wide optical ground imaging
**Verdict: mostly RED — the optical-CNN home turf is the most-searched sky in the survey.**
The southern DECam optical sky (DES, DESI-Legacy DR10-south, DELVE DR1) is the *most multiply-searched*
region in this study. DES Y6 is exhausted (Jacobs 2019a/b, Rojas 2022, Space Warps ViT / O'Donnell
2025). HSC PDR3 is combed end-to-end (SuGOHI I–X, HOLISMOKES VI/XVI). KiDS DR5 is done
(Petrillo/LinKS, Li 2020/21, TEGLIE/Grespan 2024). Pan-STARRS is shallow and overlapped.

The residual opportunities here are *deeper re-searches of the same sky*, not new sky:
- **DELVE DR2/DR3 (🟡 DOWNGRADED on deep-dive, 5 sources; first-scan score 61).** *The deep-dive
  (`GAMA_DELVE_DEEPDIVE.md`) tightened this.* DELVE DR2 tops out at dec ~+30 (below Legacy DR10's
  +32 ceiling), and **Legacy DR10 / Inchausti 2025 already CNN-searched ~14,000 deg² south of +32
  *and explicitly ingested the same DELVE/DeROSITAS exposures*** — so ~80–88% of DELVE DR2 is the
  same photons the group already searched. Genuinely fresh-and-deep residual: only ~2,000–4,000 deg²
  of low-\|b\| (dusty/crowded) sky. **The real levers are (a) a DELVE DR3 *deeper-coadd* faint pass
  (DR3 is ~0.5–0.7 mag deeper — the only genuine depth gain), (b) the low-\|b\| frontier (outside
  both the Legacy and LSST \|b\|>15 cuts), and (c) a reduction-diversity catalog on overlap sky.**
  Still the best of the optical drop-ins, but a niche, not a fresh-hemisphere harvest — and Rubin
  overtakes the high-latitude part within ~2–3 yr.
- **VST-ATLAS DR4 (🟡, score 51).** ~0% fresh sky (inside the deeper DECam footprint), but its
  *u-band* is a modality differentiator; only lensed-quasar work exists (Agnello/Spiniello 2018).
- **SkyMapper DR4 (🔴 REFUTED, 8 sources).** Nominal whole-south footprint but ~1.5″ seeing /
  shallow depth make galaxy-scale lensing effectively unviable — the red-team **refuted** the
  apparent new-sky opportunity.

### 5.2 Wide-area NIR ground imaging
**Verdict: nobody has run a blind NIR lens search — and the VHS deep-dive shows *why*: NIR is too
shallow to see the arcs.** It is a real *modality gap* but for a physical reason, not an oversight.
- **VHS (🔴 REFUTED on deep-dive, 8 sources; first-scan score 63)** — *I initially rated this the
  top "fresh far-south" drop-in; the dedicated pressure-test (`VHS_DEEPDIVE.md`) refuted it on
  both legs.* (i) **Depth kill-shot:** VHS (J~20.8/Ks~20.0 AB; deepest J~21.4/Ks~20.3) is ~3–4 mag
  too shallow and has no blue bands, so faint blue arcs fall below the limit — a VHS CNN sees the
  *deflector*, not the lens (no blind wide-NIR galaxy–galaxy catalog has *ever* been published; all
  wide-NIR lensing is submm-counterpart or optical-color use). (ii) **Not fresh:** DECam reaches
  δ=−90°, and DELVE DR2 (~21k deg² griz) + DES + Legacy DR10 already blanket the southern sky 3–4 mag
  deeper *and* mostly lens-searched. Net NIR-only fresh, usable sky ~0–10% (the unusable plane);
  realistic yield ~0–low-tens. **The southern residual it points to is re-running the optical CNN on
  DELVE DR2 griz (§5.1), not a NIR search.**
- **VIKING (🟡, 8 sources)** is deep ZYJHKs but its footprint **is the KiDS sky** (already searched
  optically) — a pure modality gap, not a sky gap.
- **UKIDSS LAS / VIDEO (🟡)** sit inside SDSS/DECaLS-north and the deep fields; **VVV (🔴)** is in
  the unsearchable Zone of Avoidance; **UltraVISTA (🔴)** is the over-studied COSMOS patch.
- All retrain the optical-CNN for NIR cutouts — a **config change, not a new tool** → drop-in.

### 5.3 Space high-resolution imaging
**Verdict: mixed — the contiguous deep fields are now swept, and the HST archive is a modest
re-mine (downgraded on follow-up), not a virgin harvest.**
- **HST general archive (🟡→ DOWNGRADED on deep-dive; first-scan score 68 → ~55)** — *I initially
  rated this the best space drop-in; the dedicated pressure-test (`HST_ARCHIVE_DEEPDIVE.md`,
  6 sources) corrected that.* HST is **pencil-beam** (~2.5–3 deg² contiguous deep mosaic; whole-archive
  secure population only ~250–350) and **not virgin**: **Hubble Asteroid Hunter II** (Garvin 2022)
  already crowd-scanned ~27 deg² (198 new), and COSMOS was already CNN-searched (LensFlow 2018)
  *and* JWST-mined (COWLS). Realistic **net-new secure yield: tens–~150**, via a *completeness
  re-mine* (the CNN beats the by-eye citizen pass at faint/small-separation systems), not virgin
  discovery — and Euclid Q1 (497 candidates/quarter, forecast >100k) erodes the marginal value.
  Best framing: a **cheap HST→Euclid/Roman space-PSF CNN readiness prototype**.
- **JWST COSMOS-Web (🟡 CONFIRMED, score 62)** — re-searchable *deeper* than the COWLS visual pass.
- **JWST other deep fields (🔴 DOWNGRADED→red)** — the red-team found AnomalyMatch (2026) and
  COWLS/Nagam 2025 have now systematically swept JADES/PRIMER/NGDEEP/CEERS/UNCOVER, erasing what
  looked like the freshest space residual.
- **Gaia GraL/GravLens (🟡 DOWNGRADED, score 47)** — the astrometric lensed-quasar search is mature
  (GraL I–X, Lemon I–IV); residual is sub-arcsec confirmations needing Euclid/HST, off-dataset.
- **Euclid Q1 (🔴)** exhausted (Discovery Engine A–F); **Euclid DR1 / Roman** are future-watch (§9).

### 5.4 Optical spectroscopy
**Verdict: spectro-FoF is the wrong tool for galaxy–galaxy lenses; the productive channel is the
single-fiber *blended spectrum*, and it's already being mined.**
- **GAMA DR4 (🔴 REFUTED on deep-dive, 6 sources; first-scan score 64)** — *I first rated this the
  best legacy spectroscopy target; the deep-dive (`GAMA_DELVE_DEEPDIVE.md`) refuted it.* GAMA's
  multi-pass tiling *does* beat the fiber-collision wall (down to a ~3″ deblending floor), **but
  that's a non-sequitur for lensing:** galaxy–galaxy lenses have θ_E~0.5–2″, so the source sits
  inside one fiber — the **two-fiber FoF channel cannot reach them** (it catches only wide/group
  pairs). The channel that *does* work is the **single-fiber blended spectrum**, and **Holwerda 2015
  already did it on GAMA** (104 candidates). Net-new G-G ≈ 0. *Honest residual:* a wide/group-pair
  search, a clean redshift/σ_v confirmation layer, and a blended-pipeline testbed.
- **The real spectroscopic lesson:** the single-fiber blended channel scales — the **DESI Single
  Fiber Lens Search** (arXiv:2512.04275) just found **4,110 candidates** that way. So the program's
  spectroscopic edge is a **foundation-model blended-spectrum re-mine** (see SDSS/BOSS below), *not*
  FoF pairs on small surveys.
- **SDSS/BOSS/eBOSS (🟡 CONFIRMED, 9 sources, but new-tooling).** ~5M public spectra, **trivial
  access (effort 1)** — but the pairwise method is *blocked* by the 55″/62″ fiber-collision radius
  and the single-fiber emission-residual method is *exhausted* (SLACS, BELLS, BELLS-GALLERY, S4TM,
  SILO/Talbot 2021). The only open lever is a **foundation-model (SpectrumFM) re-mine** of the full
  spectral set → tagged new-tooling. This is the natural science target for the program's SpectrumFM.
- **LAMOST DR11/12 (🟡, score 57)** — ~30% fresh objects over SDSS NGC; low-res caps purity.
- **2dF/6dF, DEVILS (🔴)** — too shallow / too small.
- Future: **DESI DR2 (~2027), 4MOST (south, ops mid-2026), WEAVE, PFS, DESI-II/Spec-S5** (§9).

### 5.5 Radio continuum — the headline new-tooling opportunity
**Verdict: GREEN — no blind catalog-scale radio strong-lens search has ever been published.**
Everything in the radio literature is either a *forecast* (McKean 2008 LOFAR; McCarty & Connor 2024
wide-field; Rezaei 2022 ILT-ML on *simulations*) or a *known-lens cross-match* (Martinez/Connor 2024
VLASS) or *legacy lobe surveys* (CLASS/JVAS, FIRST efficient-lens survey).
- **VLASS (🟢 CONFIRMED, 5 sources, top score 78).** 33,885 deg² at 2–4 GHz, public CIRADA/CADC
  cutout + component catalogs. Two distinct levers: a radio-source ML cross-match for blind
  discovery, **and** — uniquely — a way to *radio-confirm the program's existing ~3,900 optical
  candidates*. Needs new tooling but low access effort.
- **LOFAR LoTSS DR2 (🟡, score 50).** The 6″ catalog yields ~0 galaxy-scale lenses, but the
  **sub-arcsec ILT (International LOFAR Telescope) uv-plane** path could give O(10²–10³) — this is
  the *unbuilt high-yield* radio lever (wide-area ILT product has no release date yet).
- **ASKAP EMU + RACS (🟡, score 56).** ~20k–34k deg² fresh southern radio sky; galaxy-scale yield
  is modest at current resolution, cluster/lobe yield O(10).
- **MeerKAT MIGHTEE/MALS (🟡)** — small/medium; **FIRST+NVSS/CLASS (🔴)** — legacy, low residual.

### 5.6 Submillimeter / millimeter — virgin modality + a faint residual
**Verdict: bright-source lensing is RED (famously complete); the faint tail + interferometric
archive are GREEN/YELLOW.**
- **ALMA Science Archive (🟢 CONFIRMED, 7 sources, score 78) + ALMACAL (🟢 CONFIRMED, 5 sources,
  score 75).** A petabyte of mm/submm interferometry that has **never been systematically
  lens-searched**; serendipitous lenses sit in thousands of PI fields and ~1,000 calibrator
  sightlines (which have *no proprietary period*). Highest novelty; needs uv-plane lens-finding
  (effort 4–5). Positionally irrelevant to optical pipelines → pure new modality.
- **Herschel-ATLAS / HerMES (🟡 CONFIRMED, score 66/65).** The bright S₅₀₀>100 mJy selection is
  famously complete (Negrello 2010/2017, Nayyeri 2016, Wardlow 2013), but a **faint/sub-threshold
  DSFG selector** has an O(10²–10³) forecast residual.
- **SPT-3G emissive catalog (🟡 DOWNGRADED→yellow, score 63).** Deeper than SPT-SZ; **Archipley 2024
  already extracted 4,303 "strong SMG candidates"** (51% with no prior counterpart) — see
  *searched-but-unpublished* below. **ACT DR6 (🟡, score 58)** and **Planck PHz/PCCS2 (🟡, score 53)**
  have similar faint residuals. **SPT-SZ (🔴)** is exhausted.

### 5.7 Time-domain / alert streams
**Verdict: DOWNGRADED on deep-dive — the *content* is NOT unsearched, and diff-imaging is
redundant; but this is the strongest survivor and the one with a constructive reframing
(`ATLAS_ZTF_DEEPDIVE.md`, 16 sources).**
- **ZTF (🟡→ DOWNGRADED).** The glSN alert content is *saturated*: a full-archive search (Magee
  2023) returned **0** candidates, an AMPEL pipeline runs **live**, **the group's own** blended-
  light-curve finder (Park/Huang 2026) is validated on ZTF, and 4 brokers classify every alert. Both
  real ZTF glSNe (SN Zwicky, SN 2025wny) were caught **live by the magnification method, not at lens
  positions** — so retrospective diff-imaging at the 5,807 DESI positions adds **<<1**. "Tens of
  glSNe" is ~10× optimistic (realized ~1/4 yr; the m<19 spectroscopic-screening limit binds).
  Lensed-QSO discovery is blocked by the **<2″ resolution wall** (~90–95% unresolved at ZTF's ~2″
  PSF) — net-new ≈0; the deliverable is time-delay re-characterization of *known* lenses.
- **ATLAS (🟡→ DOWNGRADED).** The one genuine *under-searched* channel (no dedicated glSN pipeline),
  but its m~19–19.5 depth caps it at **a handful of extreme-magnification glSNe over the whole
  archive** — and even those arrive via the public alert/forced-phot route, not a diff-imaging port.
- **Method fit: REFUTED as "diff-imaging drops straight in."** ZTF/ATLAS already emit
  *difference-image* alerts + forced photometry; the program would not re-run Bramich differencing.
  The genuine edge is a **known-lens watchlist + SpectrumFM spectral gate** → a **Rubin
  dress-rehearsal** (§8a), not a pixel-level harvest. Rubin (~100× rate) overtakes within ~1–2 yr.
- **DECam multi-epoch outside the Sheu footprint (🟡 CONFIRMED, 8 sources, score 57).** DES-SN deep
  fields + DELVE-DEEP + community archive — fresh epochs the program's own diff-imaging hasn't run.
- **Gaia Photometric Science Alerts (🟡, score 44)** — bright limit dominates; **Rubin/LSST** is the
  eventual prize (§9), with a **world-public alert stream** (no data-rights gate on alerts).

### 5.8 Summary of red-team outcomes
Of 41 challenged opportunities: **34 CONFIRMED, 6 DOWNGRADED** (JWST-deep-fields→red,
SPT-3G→yellow, Gaia-GraL→yellow, ATLAS→yellow, HSC-PDR4→yellow, VHS→yellow), **1 REFUTED**
(SkyMapper→red). The three surviving GREENs (VLASS, ALMA archive, ALMACAL) all carry ≥5
corroborating sources.

---

## 6. Cross-cutting analysis

### 6.1 Footprint overlap — what sky is virgin vs multiply-searched
- **Most multiply-searched:** the southern DECam optical sky. DES (~5,000 deg²) ∪ DESI-Legacy
  DR10-south (~14,000 deg², which *ingested* DELVE+DeROSITAS DECam exposures) ∪ DELVE DR1
  (~4,000 deg², Zaborowski) cover almost all of the extragalactic south. VST-ATLAS, SkyMapper,
  and the southern PS1 strip all sit *inside* this deeper footprint → ~0% new sky.
- **The single most-searched patch:** COSMOS (~2 deg²) — HST ACS, HSC-Deep, UltraVISTA, JWST
  COSMOS-Web/COWLS, PRIMER-COSMOS all overlap.
- **Genuinely virgin reservoirs:** fresh southern
  radio (**EMU/RACS** ~20,000 deg²; **ACT-DR6** south); deeper mm (**SPT-3G**); the submm
  **interferometric modality** (**ALMA/ALMACAL**, never lens-searched, positionally decoupled from
  optical); all-sky **time-domain content** (ZTF/ATLAS/Gaia/Rubin overlap static baselines
  spatially but their *variability* is unsearched); and the **future megasurveys** (Euclid Wide,
  Roman HLWAS, Rubin WFD, 4MOST-south).
- **Key dedup facts:** VIKING ≈ KiDS sky (modality-only); UKIDSS LAS ⊂ SDSS/DECaLS-north;
  DELVE DR2/3 ≈ DES ∪ Legacy-DR10 (only ~10–20% net-new); UNIONS overlaps shallow BASS/MzLS but
  GLUE I already ran the deep gri re-search, leaving only a z-band + ~1,500 deg² + CFIS-u residual.

### 6.2 Method-vs-modality matrix
| In-house method | Drop-in now (public) | Needs a build |
|---|---|---|
| **optical-cnn** (retrain for NIR/space PSF = config change) | **DELVE DR3** (depth gain) + low-\|b\| frontier, J-PLUS/S-PLUS, JWST COSMOS-Web | — |
| *…CNN candidates that failed scrutiny* | ⚠ **HST archive** downgraded (`HST_ARCHIVE_DEEPDIVE.md`); ⚠ **VHS/wide-NIR** refuted — too shallow for arcs (`VHS_DEEPDIVE.md`); ⚠ **DELVE DR2** downgraded — ~80–88% already in Legacy DR10 (`GAMA_DELVE_DEEPDIVE.md`) | — |
| **spectro-fof** (z-pairs + emission residual) | LAMOST, GAMA wide/group-pairs only, (2dF/6dF/DEVILS low) | SDSS/BOSS → **foundation-model blended-spectrum re-mine** (the productive G-G channel; cf. DESI 4,110 via arXiv:2512.04275) |
| *…spectro-fof candidate that failed* | ⚠ **GAMA DR4** refuted for G-G — FoF can't reach θ_E~0.5–2″; blended already done (Holwerda 2015) (`GAMA_DELVE_DEEPDIVE.md`) | — |
| **diff-imaging** (multi-epoch) | ATLAS, ZTF, DECam-multi-epoch outside Sheu footprint, (Rubin alert stream at release) | — |
| **new-tooling** (no current method) | — | VLASS / EMU / MeerKAT (radio source-match); LoTSS-ILT (uv-plane); ALMA/ALMACAL/ALCS (uv-plane); Herschel/SPT-3G/ACT/Planck (faint-DSFG selector); Gaia (astrometric ML) |

The highest-value *builds* are: the **VLASS cross-match** (green, + radio-confirms existing
optical lenses), the **ALMA-archive/ALMACAL** uv-plane lens-finder (green, virgin modality), the
**SPT-3G/Herschel faint-DSFG selector** (large undelivered populations), and the
**SpectrumFM re-mine of SDSS/BOSS** (turns a fiber-collision-blocked dead-end into yield).

### 6.3 Searched-but-unpublished (direct-inquiry targets)
Cases where a search likely *happened* but no public lens catalog exists — worth an email, not a pipeline:
1. **SPT-3G strong-SMG candidates** — Archipley 2024 extracted **4,303 candidates** (51% no prior
   counterpart) over 1,500 deg²; the lens-confirmed subset was never published as a lens catalog.
2. **Euclid DR1 Wide** — the Discovery Engine consortium will sweep ~1,900 deg² internally
   around the Oct-2026 release; request early/joint access to consortium grades.
3. **UNIONS final ugriz** — the GLUE/UNIONS group is actively searching the embargoed multi-band
   data; no public 5-band catalog exists. Inquire about collaboration access.
4. **HSC-SSP PDR4 increment** — SuGOHI/HOLISMOKES will run finders at release; internal lists likely precede public.
5. **ACT DR6 lensed-DSFGs** — DR6 source catalog exists (~19,000 deg²) but no systematic
   lensed-DSFG search beyond the 480 deg² brightest-30 (Gralla 2020).
6. **Planck PCCS2 faint candidates** — Trombetti 2021 / Bonato 2025 extracted faint candidates,
   ~50% still unconfirmed.
7. **Gaia GravLens FPR small-separation candidates** — GraL IX (XGBoost) lists whose sub-arcsec
   confirmations are not yet a public catalog.

---

## 7. Gaps the completeness critic found (fold these in)

The critic flagged ~16 genuinely-missed datasets/families. The strongest additions, in priority order:

| Added dataset | Family (new) | Why it matters | Rough rating |
|---|---|---|---|
| **WISE / unWISE / CatWISE2020** | space mid-IR all-sky cross-match | The *universal* lens-vetting layer: W1−W2≥0.8 isolates type-1 + obscured QSOs; W1/W2-dropouts flag hyperluminous lensed DSFGs; it is how Herschel/SPT/ACT submm candidates get counterparts. You listed everything that *needs* WISE without WISE. | HIGH (cross-match) |
| **eROSITA / SRG eRASS1 DR1** | X-ray (new family) | 12,247-system X-ray cluster catalog = deepest all-sky sample of **group/cluster-scale lenses** (giant arcs); X-ray points flag lensed quasars. DR1 public 2024; DR2 ~mid-2026. | HIGH (clusters) |
| **NOIRLab Source Catalog (NSC) DR2** | ground optical all-sky cross-match | The unified all-sky DECam catalog (3.9B objects, ~35,000 deg², ugrizY + proper motions) — the homogeneous base layer ML lens-finders actually cross-match against; PMs reject stellar contaminants. | HIGH (cross-match) |
| **JCMT S2CLS / S2COSMOS** (SCUBA-2 450/850 µm) | deep ground submm | The classic high-resolution lensed-DSFG number-count selector (bright-end S₈₅₀>10 mJy) over ~5 deg² of the canonical fields — fills the deep/high-res submm gap above wide-shallow Herschel. | MED-HIGH |
| **Space slitless grism** (HST WFC3 3D-HST/GLASS/PASSAGE; JWST NIRISS/NIRCam WFSS) | space slitless spectroscopy | A distinct blind discovery+confirmation modality: slitless spectra detect lensed-source emission lines with no slits; GLASS targets lensing clusters. | MED |
| **Euclid NISP grism (SIR)** | space NIR slitless spectroscopy | Euclid's *free* spectroscopy already gave source/deflector z for 461 Q1 lens candidates — a blind Hα-emitter lens channel over the whole Wide survey at zero extra cost. | MED |
| **Low-freq / southern radio** (GMRT TGSS 150 MHz; GLEAM/GLEAM-X; Apertif) | radio <300 MHz + south | Steep-spectrum lensed sources pop out in TGSS×NVSS×VLASS spectral-index cross-matches; GLEAM extends into the far south. | MED |
| **CFHTLS / CFHTLenS / SL2S** | deep ground optical + ML training set | The canonical deep ground lens-search dataset and the **RingFinder ground-truth training set** any new ML search needs. | MED |
| **VLT/MUSE (+KMOS) ESO archive** | optical IFU spectroscopy | The de-facto lens-confirmation channel (Foundry IV); every public MUSE cube is a blind serendipitous-lens finder (background line emitters). | MED |
| **Quaia + Gaia×WISE (CatNorth/CatSouth)** | all-sky lensed-QSO cross-match | Ready-made 1.3M-quasar parent samples for systematic lensed-quasar mining (close pairs, astrometric jitter, BP/RP+WISE color). | MED |
| **Spitzer SWIRE / SERVS** | space mid-IR deep | IRAC 3.6/4.5 µm over standard fields — deflector photo-z/stellar mass + counterpart layer for Herschel/JCMT/radio candidates. | MED-LOW |
| **BlackGEM / BG-SASS** | southern time-domain | 30,000 deg² southern 6-band survey to ~22 mag (ESO archive) — a southern complement for lensed-SN/QSO variability. | MED-LOW |
| **J-PAS / miniJPAS** | ground 56-narrow-band photo-spectroscopy | R~50 pseudo-spectra per pixel → pseudo-spectroscopic redshifts + slitless emission-line lens detection over ⅓ of the north. | MED (north, ramping) |
| **SDSS Stripe 82 deep coadds** | deep ground optical | ~300 deg² equatorial coadd ~2 mag deeper than single-epoch SDSS; outside every per-survey footprint here. | LOW |
| **Chandra CSC 2.x / 4XMM** | X-ray high-res archival | Resolves close lensed-quasar image pairs; classic route for X-ray-bright lensed AGN. | LOW-MED |
| **Simons Observatory (now) + CMB-S4** | submm/mm future | Next-gen all-sky mm source catalogs dominated by strongly-lensed DSFGs. | LOW-MED (future) |

> These were not run through the full research+red-team template (they surfaced at the critic
> stage). They are recommended as the **next scan's seed list** — particularly the three new
> *families* the original 8 missed: **X-ray** (eROSITA/Chandra/XMM), **all-sky mid-IR cross-match**
> (WISE/Spitzer), and **slitless+IFU spectroscopy** (grism / MUSE).

---

## 8. Ranked action shortlist

### Tier A — drop-in now (public data, existing method) — *post-pressure-test*
> ⚠ After **five** deep-dives, Tier A is **much smaller than the first scan implied** and is best
> treated as **prototype-scale**, not a major harvest (see §8a). Ordered by what survived scrutiny:
1. **ATLAS/ZTF → reframed as a Rubin precursor** (⚠ *downgraded, but the strongest survivor;
   `ATLAS_ZTF_DEEPDIVE.md`*). Do **not** re-run diff-imaging (ZTF/ATLAS already emit difference-image
   alerts + forced photometry; ZTF glSN search is saturated). Instead build **(a) a known-lens
   real-time watchlist** on the live ZTF/ATLAS (→Rubin) broker/forced-phot streams against the
   ~5,807 lenses + ~6,500 candidates → trigger spectroscopy on any transient at a lens; **(b) a
   SpectrumFM SN-type/host-z spectral gate** (attacks the m<19 screening bottleneck); **(c) an ATLAS
   extreme-magnification glSN sweep** (the one standalone current-data sliver). This is the
   best-justified do-now item *because it is forward-looking* — the same stack scales onto Rubin's
   ~100× rate. *Effort 2.*
2. **DELVE DR3** *deeper-coadd* faint-deflector pass (~0.5–0.7 mag deeper than DR2/Legacy) + the
   **low-\|b\| frontier** (outside both Legacy and LSST cuts). ⚠ *downgraded from DR2 "fresh
   harvest" — a niche, tens–few-hundred low-grade; `GAMA_DELVE_DEEPDIVE.md`.* *Effort 2.*
3. **JWST COSMOS-Web** → CNN re-search deeper than the COWLS visual pass (small field). *Effort 2.*
4. **J-PLUS / S-PLUS** → narrow-band CNN (novel, speculative). *Effort 2.*
5. **HST general archive** → ⚠ *downgraded* (`HST_ARCHIVE_DEEPDIVE.md`): completeness re-mine,
   tens–~150 secure, **best run as the HST→Euclid/Roman space-CNN readiness prototype**. *Effort 3.*
6. **GAMA** (wide/group-pair search + redshift/σ_v **confirmation layer** + blended-pipeline
   testbed) and **LAMOST**, **DECam-multi-epoch (outside Sheu)** — secondary / supporting.

> ⚠ **Refuted/downgraded out of the lead by the deep-dives:** **VHS + wide-NIR** (VIKING/UKIDSS/
> VIDEO — too shallow for arcs), **GAMA DR4** for galaxy–galaxy (FoF can't reach θ_E~0.5–2″;
> blended already done), and **DELVE DR2 as a "fresh harvest"** (~80–88% already in Legacy DR10).
> Details in `VHS_DEEPDIVE.md` and `GAMA_DELVE_DEEPDIVE.md`.

### 8a. Synthesis after five pressure-tests (the honest bottom line)
**Every** current-data "fresh public data + existing method" lead from the first scan was
**downgraded or refuted** (HST, VHS, GAMA, DELVE DR2, ATLAS/ZTF — 2 refuted, 3 downgraded). The
southern DECam optical sky, the DESI/SDSS spectroscopy, *and* the ZTF time-domain stream are more
thoroughly mined — by the group and the community — than a first-pass scan credits, and each
alternative modality hits a physical/coverage wall. So:
- **Small current-data program (prototype-scale, do now):** the **ATLAS/ZTF → Rubin precursor**
  (known-lens watchlist + SpectrumFM spectral gate + ATLAS extreme-mag glSN sweep) — the
  best-justified item because it is *forward-looking*; plus **DELVE DR3**-depth + low-\|b\|, **GAMA**
  confirmation/testbed, **JWST COSMOS-Web**, **J-PLUS/S-PLUS**. Realistic combined net-new: low
  hundreds *at most*, much of it transients — treat as readiness prototypes, not a harvest.
- **The real, large opportunities (unchanged — they were never "existing-technique" claims):**
  the **new-tooling green fields** — **VLASS**, the **ALMA archive / ALMACAL**, and the **SDSS/BOSS
  SpectrumFM blended-spectrum re-mine** (the genuinely un-mined, build-once levers) — and
  **pre-positioning** for **Euclid DR1 / Rubin / Roman / DESI DR2**, where the 10³–10⁵ populations
  live and which keep recurring as "the thing that obsoletes a marginal current-data search."
- **Recommended posture:** run the current-data items as **readiness prototypes** for the space-PSF
  CNN and the alert-stream diff-imaging, while investing the real effort in the new-tooling builds
  and megasurvey pre-positioning.

### Tier B — new-tooling now (public data, real opportunity, build required)
1. **VLASS** radio cross-match — *and* radio-confirm the existing ~3,900 optical lenses. *Build: radio source-match ML.*
2. **ALMA Science Archive + ALMACAL** — virgin submm interferometric modality. *Build: uv-plane lens-finder.*
3. **SPT-3G + Herschel faint-DSFG selector** — large undelivered candidate populations (start by requesting the Archipley 4,303-candidate list). *Build: faint-DSFG point-source selector.*
4. **SDSS/BOSS/eBOSS foundation-model re-mine** — the natural SpectrumFM science target (trivial access, exhausted by classical methods). *Build: spectral FM re-mine.*
5. **ASKAP EMU/RACS**, **LOFAR LoTSS-ILT** (uv-plane, highest radio ceiling), **ACT DR6**, **Planck** — secondary builds.

### Tier C — pre-position for the future (inquiries + pipeline readiness now)
1. **Euclid DR1 Wide** (public 2026-10-21) — request early/joint consortium access; ready the CNN for VIS+NISP PSF.
2. **Rubin/LSST** (DP2 ~2026 Q3; world-public alert stream) — connect to a broker now; diff-imaging is alert-stream-ready.
3. **Roman HLWAS** (launch by 2027; ~160k-lens forecast) — the largest single ceiling; ready a space-PSF CNN.
4. **DESI DR2 spectra** (~2027) — re-run the Hsu-2025 FoF at scale on day one.
5. **4MOST** (ops mid-2026) — the only large *fresh southern* spectroscopy; plan a southern FoF search.

---

## 9. Future watch-list & timeline

| Public availability | Dataset | Modality | Fit | Forecast yield | Action |
|---|---|---|---|---|---|
| **2026-07…09** (DP2) | Rubin/LSST DP2 (then DR1 ~2027) | optical time-domain + coadd | diff / CNN | ~10⁵ full survey | broker connection; alert-stream diff-imaging |
| **mid-2026** (ops) | 4MOST | optical MOS spectroscopy (south) | FoF | 10³–10⁴ | southern FoF plan |
| **2026** (rolling) | WEAVE | optical MOS spectroscopy (north) | FoF | 10²–10³ | monitor WAS public DRs |
| **2026-10-21** | Euclid DR1 Wide | space VIS+NISP imaging + grism | CNN / grism | ~7,000 A/B | early/joint consortium inquiry |
| **TBD (unknown)** | HSC-SSP PDR4 | optical grizy imaging | CNN | ~10² | monitor; SuGOHI internal lists |
| **~late 2026/2027** | Subaru PFS (SSP) | optical-NIR MOS spectroscopy | FoF | 10²–10³ | SMOKA after 18-mo proprietary |
| **launch by 2027** | Roman HLWAS + HLTDS | space NIR imaging + time-domain | CNN / diff | ~160,000 | space-PSF CNN readiness |
| **~2027** | DESI DR2 spectra (3-yr) | optical MOS spectroscopy | FoF | 10³ | day-one Hsu-2025 FoF re-run |
| **2026–2031 ops** | DESI-II / Spec-S5 | optical-NIR MOS spectroscopy | FoF | 10³–10⁴ | long-horizon |
| **now (online)** | Simons Observatory LAT | mm source catalogs | New | lensed-DSFG factory | monitor first catalogs |

---

## Appendix A — Data-access cookbook

Per-dataset access for the public-now and near-term opportunities (●=public, ◐=embargoed, ○=future).
Mechanisms: **SIA**=image cutout service · **TAP/ADQL**=table query · **bulk**=full download.

### Optical ground imaging
- **DELVE DR2/DR3** ● — NOIRLab Astro Data Lab: SIA `https://datalab.noirlab.edu/sia/delve_dr3`, TAP `https://datalab.noirlab.edu/tap`. Portal `https://datalab.noirlab.edu/data/delve`. Auth: anon for SIA/TAP, free account for bulk/notebook. Vol: DR3 ~2.6B objects, multi-TB.
- **DES Y6/DR2** ● — Astro Data Lab cutout/TAP + DESaccess bulk `https://des.ncsa.illinois.edu/easyweb/`. Portal `https://datalab.noirlab.edu/data/dark-energy-survey`. ~691M objects, ~0.5 PB images.
- **DECaLS DR10** ● — cutout `https://www.legacysurvey.org/viewer/cutout.fits?layer=ls-dr10`, docs `https://www.legacysurvey.org/dr10/`. Anon. PB-scale.
- **VST-ATLAS DR4** ● — ESO Phase 3 + WFAU OSA `http://osa.roe.ac.uk`. Free ESO portal account.
- **SkyMapper DR4** ● — SIAP `https://api.skymapper.nci.org.au` + DR4 portal `https://skymapper.anu.edu.au/data-release/dr4/`; ADL mirror. Anon cutout/cone.
- **HSC-SSP PDR3** ● — NAOJ archive `https://hsc-release.mtk.nao.ac.jp/doc/` (cutout / CAS / TAP). Free registration required.
- **Pan-STARRS1 3π DR2** ● — MAST cutout `https://ps1images.stsci.edu/cgi-bin/ps1cutouts`, catalogs `https://catalogs.mast.stsci.edu/`. Anon; CasJobs account for catalog.
- **UNIONS** ◐ — CADC `https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/community/unions/` (public CFIS u/r subset); full multi-band by CFIS MoU.
- **J-PLUS / S-PLUS** ● — `https://splus.cloud/` + `splusdata` API; J-PLUS CEFCA `https://archive.cefca.es/catalogues/jplus-dr3`. Free account for bulk.

### NIR ground imaging
- **VHS** ● — Astro Data Lab `https://datalab.noirlab.edu/vhsdr5.php` (TAP/SIA); also ESO/VSA. Anon.
- **VIKING** ● — VSA/WFAU `http://horus.roe.ac.uk/vsa/` + ESO Science Portal. Anon.
- **UKIDSS LAS** ● — WSA/WFAU `http://wsa.roe.ac.uk/` (SQL + cutout). Anon (DR11).
- **VIDEO / UltraVISTA / VVV** ● — VSA/WFAU + ESO Phase 3; COSMOS mirror `https://cosmos.astro.caltech.edu/page/optical`.

### Space imaging
- **HST general archive (HLA)** ● — `https://hla.stsci.edu/` + MAST archive request. Anon.
- **HST COSMOS** ● — IRSA `https://irsa.ipac.caltech.edu/Missions/cosmos.html`. Anon.
- **JWST COSMOS-Web** ● — bulk `https://exchg.calet.org/cosmosweb-public/DR1/`, MAST DOI 10.17909/2sx3-ad32. Anon.
- **JWST deep fields (DJA)** ● — DAWN JWST Archive `https://dawn-cph.github.io/dja/` + ASTRODEEP. Anon.
- **Euclid Q1** ● / **DR1** ○ — ESA SAS `https://easidr.esac.esa.int/sas/`; DR1 `https://www.cosmos.esa.int/web/euclid/euclid-dr1` (2026-10-21). Free SAS registration.
- **Roman** ○ — MAST `https://archive.stsci.edu/missions-and-data/roman` + Roman Research Nexus (launch by 2027).
- **Gaia GraL** ● — Gaia Archive TAP `https://gea.esac.esa.int/archive/` (FPR GravLens table) + VizieR. Anon.

### Optical spectroscopy
- **SDSS/BOSS/eBOSS (+MaNGA)** ● — SAS bulk + CAS/SkyServer TAP `https://www.sdss4.org/dr17/data_access/`; eBOSS lens VAC (SILO). Anon.
- **LAMOST DR11/12** ● — `https://www.lamost.org/dr12/` (web + bulk FITS). Free registration.
- **GAMA DR4** ● — `http://www.gama-survey.org/dr4/` (TAP + bulk). Anon.
- **DESI DR2** ◐ — `https://data.desi.lbl.gov/doc/releases/` (NERSC bulk + ADL TAP at release, ~2027).
- **WEAVE** ◐ (WAS/AIP) · **4MOST** ○ (ESO SAF) · **PFS** ◐ (SMOKA, 18-mo proprietary) · **DESI-II/Spec-S5** ○.

### Radio
- **VLASS** ● — CIRADA/CADC cutout (SIA+SODA) + QL/SE component catalogs `https://cirada.ca/vlasscatalogueql0`. Anon.
- **LOFAR LoTSS DR2** ● — `https://lofar-surveys.org/dr2_release.html` (TAP/SIA/cone); mosaics on SURF. Anon.
- **ASKAP EMU/RACS** ● — CSIRO CASDA `https://research.csiro.au/casda/` (TAP + bulk; CADC mirror). Anon for public tiles.
- **MeerKAT MALS** ● — `https://mals.iucaa.in` (image + catalog). Anon. **MIGHTEE** — partial via IDIA/ARC.
- **FIRST + NVSS** ● — `https://sundog.stsci.edu/` + NRAO/SkyView. Anon.

### Submm / mm
- **Herschel H-ATLAS / HerMES** ● — HeDaM `https://hedam.lam.fr/` + `https://www.h-atlas.org/public-data/download`. Anon.
- **SPT-SZ / SPT-3G** ● — `https://pole.uchicago.edu/public/data/` + LAMBDA. Anon.
- **ACT DR6** ● — LAMBDA `https://lambda.gsfc.nasa.gov/product/act/` + `https://act.princeton.edu/act-dr6-data-products`. Anon.
- **Planck PHz/PCCS2** ● — PLA `https://pla.esac.esa.int/` + IRSA. Anon.
- **ALMA Science Archive / ALMACAL / ALCS** ● — ASA `https://almascience.eso.org/asax/`, TAP `https://almascience.eso.org/tap`, `ALminer` Python toolkit; ALCS `https://www.ioa.s.u-tokyo.ac.jp/ALCS/alma_data.php`. Anon for public; calibrators have no proprietary period.

### Time-domain
- **ATLAS** ● — forced-photometry server `https://fallingstar-data.com/forcedphot/` (web + REST). Free registration.
- **ZTF** ● — IRSA `https://irsa.ipac.caltech.edu/Missions/ztf.html` (TAP + lightcurve API + DR bulk); IPAC forced-phot (free reg); brokers (ALeRCE/Fink/Lasair/ANTARES). Anon for DR/alerts.
- **Gaia Science Alerts** ● — `http://gsaweb.ast.cam.ac.uk/alerts/` (web + CSV) + GaiaX stream. Anon.
- **Rubin/LSST** ◐ — `https://dp1.lsst.io/` (RSP/Butler); alert stream world-public via brokers. RSP account = data rights for catalog/image; **alerts open to anyone**.
- **DECam multi-epoch** ● — NOIRLab Astro Data Archive `https://astroarchive.noirlab.edu/` (per-exposure + diff frames). Anon for released DES/DELVE; 12-mo proprietary on community exposures.

### Critic-added archives (next scan)
- **WISE/unWISE/CatWISE2020** — IRSA `https://irsa.ipac.caltech.edu/Missions/wise.html`; unWISE coadds; CatWISE2020 via IRSA/VizieR.
- **eROSITA eRASS1 DR1** — `https://erosita.mpe.mpg.de/dr1/` (western Galactic hemisphere; DR2 ~mid-2026).
- **NOIRLab Source Catalog DR2** — Astro Data Lab `https://datalab.noirlab.edu/nscdr2.php`.
- **JCMT/SCUBA-2 S2CLS** — CADC JCMT Science Archive.
- **HST/JWST grism** — MAST (3D-HST, GLASS, PASSAGE, NIRISS WFSS); **VLT/MUSE** — ESO Science Archive / AMUSED.
- **GMRT TGSS ADR1** — `https://tgssadr.strw.leidenuniv.nl/`; **GLEAM** — VizieR/MWA; **Apertif DR1** — ASTRON VO.
- **CFHTLS/CFHTLenS/SL2S** — CADC/Terapix; **Spitzer SWIRE/SERVS** — IRSA; **Quaia** — Zenodo/VizieR; **Stripe 82 coadds** — SDSS SAS.

---

## Appendix B — Selected prior-search citation index

The workflow gathered **147 unique prior-search references**. The discovery papers that set each
survey's RED/YELLOW status (the ones that matter for "has it been searched?") are below; full URLs
are in `references.bib`.

- **DES:** Jacobs 2019a (1811.03786), Jacobs 2019b (ApJS 243,17), Rojas 2022 (2109.00014), O'Donnell/Space Warps 2025 (2501.15679).
- **HSC:** SuGOHI I–X (Sonnenfeld 2018 1704.01585 … Cañameras/Jaelani 2024 2312.07333), HOLISMOKES VI (2106…) / XVI (Schuldt 2025 2503.07733), HSC Space Warps (Sonnenfeld 2020).
- **KiDS:** Petrillo/LinKS 2019 (1807…), Li 2020/2021, He 2020, TEGLIE/Grespan 2024 (2407…).
- **UNIONS:** Savary 2022 (2110.11972), GLUE I 2025 (2505.05032), late-type UNIONS 2026.
- **DELVE:** Zaborowski 2023 (2210.10802). **DECaLS/baseline:** Huang 2020 (1906.00970)/2021, Inchausti 2025 (2508.20087), Hsu 2025 (2509.16033), Sheu 2023 (2301.03578)/2024a.
- **JWST/HST:** COWLS I (Nersesian/Mahler 2025 2503.08777), Nagam 2025 (2505.17318), LensFlow/Pourrahmani 2018, Faure 2008, Hubble Asteroid Hunter II (Garvin 2022 2207.06997).
- **Euclid:** Discovery Engine A (2503.15324) / F (2603.28580), arc Mask R-CNN (2511.03064), NISP-grism Q1 (2604.02726).
- **Spectroscopy:** SLACS (Bolton 2006), BELLS (Brownstein 2012), BELLS-GALLERY (Shu 2016), S4TM (Shu 2017), SILO (Talbot 2021 2007.09006), GAMA blended (Holwerda 2015), DESI single-fiber (2512.04275).
- **Radio:** Martinez/Connor 2024 VLASS (2404.09954), CLASS (Browne 2003/Myers 2003), FIRST efficient-lens (Phillips 2000), Rezaei 2022 ILT-ML, McKean 2008 / McCarty&Connor 2024 (2412.01746, forecasts).
- **Submm:** Negrello 2010 (1011.1255)/2017, Nayyeri 2016, Wardlow 2013, Vieira 2013 (1303.2723), Everett 2020, Reuter 2020, Planck GEMS (Cañameras 2015), Archipley 2024 (SPT-3G), Gralla 2020 (ACT), Trombetti 2021 / Bonato 2025 (Planck), ALMACAL I–IX (Oteo 2016 …).
- **Time-domain:** SN Zwicky / Goobar 2023 (Nat. Astron.), DeepZipper II (Morgan 2023 2204.05924), Townsend 2025 (2405.18589), LSST glSN forecasts (Bag/Wojtak; Sagués-Carracedo 2024).
- **Forecasts/future:** Collett 2015 (OM10), Holloway 2023/2024/2025 (Rubin/Roman), Euclid revolution (2508.14624), Spec-S5 (Schlegel 2025 2503.07923).

---

*End of report. Generated 2026-06-06 by the agentic-lensing deep-research workflow
(`reports/lens-discovery-opportunities/`). Companion LaTeX/PDF: `main.tex` → `make pdf`.
Workflow script archived at `workflow.js`.*
