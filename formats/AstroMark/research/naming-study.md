# Naming the annotation standard — study and recommendation

**Date:** 2026-09-03 · **For:** Greg Benson, Xiaosheng Huang, Nate Kvinnesland ·
**Status:** recommendation for review — nothing here has been adopted.

**Question asked.** What should the new standard for annotating astronomical images be called?
It must serve strong gravitational lenses today and extend to astronomical image annotation
generally, and it must survive being said from an AAS podium, typed as a package name, and printed
in a paper title.

**Method in one line.** Two agent fleets: 4 generation angles → 25 candidates → 3 web collision-check
batches → 3 independent judging lenses, then an adversarial verification round that re-checked the
finalists and the three names the room itself had floated. Every registry and domain state below was
queried, not guessed. Full method and limitations in §10.

---

## 1. Recommendation

### **AstroMark**, with a core-plus-profiles architecture.

| Layer | Name | Notes |
|---|---|---|
| The standard | **AstroMark** | what you publish, cite and take to the AAS |
| The notation (the sign set) | **AstroMark Notation** | arrow roles and colours, mask circles, the ring, polygons, polarity |
| The file format | **AstroMark JSON** | extension `.astromark.json` |
| Core schema | `astromark/1.0` | geometry, coordinates, provenance, render contract |
| Lens vocabulary | `astromark/lens/1.0` | arcs, counter-images, deflectors, θ_E, satellites, dust lanes |
| The app | **LensMark** | keeps its name; becomes the reference implementation of the lens profile |

**The one-line case.** AstroMark is the only candidate in the field with *zero* astronomy prior art —
verified across ASCL, the arXiv API, the full IVOA document index and the astropy affiliated registry,
each with a control query proving the search path actually works — while also being self-describing
enough that it needs no explanation from a podium, and staying in the `-Mark` family so that
**nothing already built has to be renamed.**

**Why the architecture matters more than the word.** Four different things needed names and the room
had been trying to find one word for all of them (§2). Separating them means the ApJ paper is about
*the lens profile of AstroMark* rather than a lens-only format — a better framing for the paper, and
the difference between a standard that can grow and one that must be renamed at exactly the moment
adoption starts.

**Migration is mechanical.** `lensmark/1.0` → `astromark/lens/1.0`; the `-config`, `-critique`,
`-patch` and `-manifest` sibling schemas restem 1:1. `apps/LensMark` keeps its directory, its name
and its role.

### Actions this recommendation implies

| # | Action | Why | Urgency |
|---|---|---|---|
| 1 | Register **astromark.io** | verified free; a standard needs a domain it controls for JSON Schema `$id` URLs and the profile registry. `astromark.org` is taken by an unrelated stargazing-outreach site | before any public mention |
| 2 | Take the GitHub org **`astromark-std`** | `github.com/astromark` belongs to a working astronomer (Mark Booth, 10 repos incl. the AtLAST sensitivity calculator). Do not squat near him | before first push |
| 3 | Claim PyPI **`astromark`** and npm **`astromark`** | both verified free | before first release |
| 4 | Trademark clearance by counsel | **unresolved** — Justia, Trademarkia and the USPTO API all refused programmatic access, so no clearance was performed for any name in this study | before public launch |
| 5 | Read and cite **AVM** (Astronomy Visualization Metadata) | the nearest existing standard; the spec should say what AstroMark does that AVM does not | before writing the spec |
| 6 | Always write the internal capital: *AstroMark*, never *astromark* in prose | separates it from npm's `astro` namespace and from the astrology practice that owns the bare lowercase word | style rule |

### Residual risks, stated plainly

None of these is an astronomy collision; all are search-hygiene:

- **Astrology owns the bare word.** `astromark.us` is Mark Dodich's astrology practice, operating since
  1980. For a scientific standard this is an unwelcome first search result. *Mitigation and counter-evidence:*
  astropy, astroquery, astroplan, astroalign and astroML have made the `astro-` prefix thoroughly
  astronomical in practice; nobody reads `astropy` as astrology. The risk is real but small.
- **Astromart is one letter away** — the leading amateur-astronomy classifieds portal, which lands in
  precisely the audience the standard courts.
- **npm's `astro` is the Astro web framework** (5,097,604 downloads/week across 7,860 packages), so a JS
  toolchain named `astromark` risks being read as a framework integration. Matters little for a
  Python/JSON standard with a small frontend; matters more if you ever ship browser tooling.
- **Two low-star GitHub projects already use the name**: a school gradebook (4★) and a Markdown editor
  (1★). Neither is in astronomy; the Markdown editor is mildly adjacent.
- `astromark.org` and `astromark.com` are gone; `.io` is the available option.

---

## 2. Why the naming question was hard: four things need names

The workshop record (`workshops/2026-08-31-LensMark/04-annotation-standard.md` §2.11) shows the room
circling names — LensMark, LensDown, "lens markup standard", AstroMark, CosmoMark — without landing.
The reason is that four distinct things were competing for one word:

| # | Thing | State before this study |
|---|---|---|
| 1 | The **app** — hand/model/voice annotation UI, the propose/critique loop | named **LensMark**; the room decided to keep it |
| 2 | The **symbolic notation** — green = deflector, dotted ring, dashed vs dotted mask circles, planned polarity and polygons | unnamed |
| 3 | The **metadata format** — normalized-coordinate vector JSON, provenance, deterministic render contract, `schema_version: "lensmark/1.0"` | unnamed |
| 4 | The **standard as a whole** — what gets published, cited and adopted | unnamed |

Naming (4) separately from (1) is what unlocks everything else. A **general core plus domain profiles**
means the lens work is a coequal profile rather than a legacy stem:

```
astromark/1.0            core: geometry, coordinates, provenance, render contract
astromark/lens/1.0       arcs, counter-images, deflectors, theta_E, satellites, dust lanes
astromark/morphology/1.0 (later)
astromark/transient/1.0  (later)
astromark/artifact/1.0   (later)
```

---

## 3. The decision rule the study produced

The most useful finding was not a name but a rule, surfaced by the astronomy-community judge and
borne out across the whole field:

> **In astronomy, names that became *standards* are dry and self-describing — FITS, WCS, VOTable, MOC,
> SAMP, HEALPix, region files. Names that are memorable, witty or erudite belong to *tools* — SExtractor,
> DS9, TOPCAT, emcee. Tools are chosen. Standards are conformed to.**

A three-person group asking a community to conform to a format named in Latin or Greek reads as a
conceit. That single rule eliminated the entire etymology cluster the generators were proudest of —
Symbolon, Sigla, Signary, Legenda, Vidimus, Indicium, Obelus — not on prior art but on register and on
spellability after one hearing, which is the mechanism word-of-mouth adoption actually runs on.

A second rule emerged from the instrument-word cluster: **do not name the standard after a word that
already denotes something in the images being annotated.** *Lucida* is the brightest star of a
constellation. *Alidade* is the azimuth support structure of a radio dish. *Reticle* is a focal-plane
mark. Each collides with live vocabulary about the very objects the format describes.

---

## 4. Evidence for AstroMark

### 4.1 Astronomy prior art: none, and the search was controlled

| Source | Query | Result | Control |
|---|---|---|---|
| ASCL | `astromark` | **0 codes** | `cosmomc` → 17, `lenstronomy` → 15 |
| arXiv API | `all:"AstroMark"` | **0 results** | `all:"CosmoMC"` → 82 |
| IVOA document index | full 357 KB index fetched | **0** (the 47 `Mark` substring hits are all the editor Markus Demleitner) | — |
| astropy affiliated registry | 55 packages | **0** containing `mark` | — |

Spellings `astro-mark`, `astromarkup` and `astromarker` were checked separately and are equally empty.
**There is no existing astronomy annotation or markup tool by that name.** The nearest genuine prior
art is historical and differently named: Astronomical Markup Language (AML), Astronomical Instrument
Markup Language (AIML) and Remote Telescope Markup Language, all 1999-era XML efforts.

### 4.2 Namespace state

| Namespace | State | Method |
|---|---|---|
| PyPI `astromark` | **free** | JSON API → HTTP 404 |
| npm `astromark` | **free** | registry.npmjs.org → Not found |
| GitHub `astromark` | **taken** — Mark Booth, a working astronomer, 10 public repos, account since 2015 | GitHub API |
| `astromark.io` | **free** | whois → "Domain not found"; `dig` NS and A both empty |
| `astromark.org` | **registered** 2022-05-25, Tucows, Wix-hosted — "Astro Mark, Stargazing Guide" outreach site | RDAP (PIR) HTTP 200, corroborated by whois + dig |
| `astromark.com` | **taken**, parked for sale (Afternic) | dig NS |

---

## 5. The three names the room already floated

All three were checked; **none collides with anything in astronomy.** That is the only good news, and
it was not sufficient for two of them.

### LensMark, as the name of the general standard — **rejected**

Keep it for the app. Do not promote it. Two independent reasons:

1. **The stem is both too narrow and ambiguous.** It asserts gravitational lensing on every morphology,
   transient and artifact annotation the standard will eventually carry — a transient broker emitting
   `.lensmark.json` is a category error. Worse, in *any* imaging context the bare word "lens" reads as
   camera or ophthalmic glass, so for an image-annotation standard it does not even reliably signal
   *gravitational* lensing.
2. **It is the most crowded of the three, and crowded in the adjacent space:**
   - `lensmark.org` is **registered and live** (2023-04-04, Key-Systems) serving *Lensmark — Photo-monitoring
     Solution for WordPress*, a citizen-science photo-monitoring plugin built as a bachelor thesis at Bern
     University of Applied Sciences for Gantrisch Nature Park (`github.com/mrtn97/lensmark-plugin`).
   - **"LensMark – Picture Border Editor"** is a live paid iOS app (App Store id6760441266) that reads EXIF
     and renders camera metadata as designed overlays on photographs. *That is an image-annotation product
     wearing the name.*
   - Lensmark.kz — a funded contact-lens distributor in Almaty, founded 2012.
   - LensMark II — a laser inscription system for marking ophthalmic lenses (Laserop Ltd).
   - `lensmark.com` is parked for sale at HugeDomains.

   **Flag for the team:** none of this forces a rename of the research app, but the live commercial iOS
   collision is directly adjacent and is the name in this study most likely to attract a live trademark.
   Worth knowing before publication rather than after.

### CosmoMark — **rejected on semantics despite the cleanest slate**

PyPI, npm, `cosmomark.org` and `cosmomark.io` are *all free* — the best availability in the study. It
still fails, for a reason that can be quantified: ASCL lists **31 distinct code names beginning with
`Cosmo`** — CosmoMC, CosmoSIS, CosmoHammer, CosmoBolognaLib, CosmoPower, CosmoLike, CosmoNest, CosmoPMC,
CosmoRec, CosmoLattice, CosmoTransitions, CosmoSlik, CosmoCov, CosmoFlow, CosmoGRaPH, CosmoGraphNet and
more — and **every one is a cosmological *computation* code**: samplers, likelihoods, Boltzmann solvers,
emulators. An astronomer places `CosmoMark` in that family by reflex, and the `-Mark` suffix then reads
as *benchmark*, so the compound parses most naturally as "a benchmarking suite for cosmology codes."
Nothing in the name signals images, annotation or geometry — and the standard does no cosmology, so the
name over-claims one field while saying nothing about the one it serves.

### AstroMark — **recommended**, see §1 and §4.

---

## 6. Runner-up: **Markstone**

Take this if the team prefers a distinctive proper noun over a self-describing one.

**For:** PyPI, npm *and* crates.io are all free — the only candidate clear on all three. It transcribes
correctly after one hearing by natives and non-natives alike (two of the highest-frequency morphemes in
English, fully transparent orthography). Zero astronomy usage (arXiv `all:markstone` → 0). It is a
60× sharper search token than the nearest rival (GitHub code search: 1,604 hits vs 96,768 for
*indicium*). And the meaning is exactly right: Old English *mearcstān* (Bosworth-Toller, attested in
charter bounds) is a boundary stone that two parties agree on and inscribe with who set it and when —
which is precisely this file: geometry that fixes a boundary of meaning, plus sha256, author, method
and date saying who set it. Nothing in that ages.

**Against:** it carries **no astronomy signal whatsoever**, so every talk must teach the name before
using it, and it pattern-matches to a private-equity firm first. That firm brings baggage: Markstone
Capital Partners (Tel Aviv/LA, $800M 2004 vintage) carries a 2010 $18M New York Attorney General
settlement. All three obvious domains are registered — and note `markstone.org` has a **registry expiry
of 2026-09-07**, four days from this report, currently a dead placeholder host; worth watching if this
name is chosen. `github.com/markstone` is a dormant squat. Trademark: the one live US registration is
class 33 (alcoholic beverages); the Markstone Capital mark is dead since 2011; no live class 9 conflict
surfaced.

---

## 7. Killed by verification — the two names the first pass liked most

The verification round exists because the first pass ranked these #1 on two of three lenses. Both fell.

### SkyMark — **blocked**

The first pass claimed astrometry.net "owns" the word. **That claim was half wrong and the verification
says so:** `skymark` appears zero times in Lang et al. 2010 (arXiv:0910.2233, AJ 139:1782 — `quad`
appears 181 times), zero times in the astrometry.net source, and on exactly one documentation page
(`doc/readme.rst`, 8 lines, introduced 2012-05-18 by Dustin Lang). "The skymark file" would not confuse
anyone; astrometry.net says *index file* 45 times and never *skymark file*.

**But two collisions both earlier passes missed are worse:**

1. **KStars/Ekos ships the literal string `SkyMark` as a visible GUI label** —
   `kstars/ekos/align/opsastrometryindexfiles.ui` line 1326, a column header in the Astrometry Index
   Files settings dialog, with `skymarksize` as a variable in the accompanying `.cpp`. KStars is a
   mainstream cross-platform astronomy application in every Linux distribution.
2. **Starlink's IRAS90 has an application literally named SKYMARK** whose stated purpose is *"Draw
   markers at specified positions"* on sky images, with the positions logged to a reusable file.
   **That is this project's concept, under this exact name** — albeit in a long-obsolete package.

Namespace seals it: PyPI and npm free, but the GitHub org is squatted, and every obvious domain
(`.com`, `.org`, `.net`, `.io`, `.dev`, `.app`, `.ai`) is registered — only `.sh` is free. Bare-word search page 1 is 100% commercial, 0%
astronomy — Skymark Airlines dominates. The SkyMapper phonetic-confusion claim was an **overreach**
(2 vs 3 syllables, different stress and coda); the real residual is same-domain adjacency in reference
lists.

### PlateMark — **blocked**

It has the best namespace in the entire study: PyPI, npm, `.org`, `.io` and `.net` all verified free
three ways each (registry whois, RDAP, NXDOMAIN). It dies on semantics anyway.

**"Plate" in astronomy is not fading — it is growing, and every live sense is an image-metadata sense:**

- **Plate solving.** ASTAP, the most widely used astrometric solver, released v2026.09.03 describing
  itself as a "plate solver" and listing "Star annotation" and "Deep sky annotation" features in the
  same breath. This is the most damaging collision: adjacent tool, adjacent function, same word.
- **Plate scale.** Rubin/LSSTCam documentation (updated May 2026) states the 0.2 arcsec/pixel plate
  scale. Universal in instrument papers, never going away.
- **Photographic plates — measurably growing.** arXiv title/abstract counts for "photographic plate":
  **19** in 2014–2016 vs **31** in 2024–2026. DASCH finished scanning in early 2024 (429,274 plates) and
  released DR7 on 2024-12-29 with 23,574,404,199 measurements of 252,458,490 sources.
- **SDSS.** Plug plates are operationally retired, but DR19 still documents spec files as
  `FIELD(PLATE)-MJD-CATALOGID(FIBER)` and the glossary still defines "Plate."

**Why this is a permanent tax rather than a two-second correction:** a harmless homonym is one the
listener discards instantly because the wrong reading is absurd (nobody thinks Python is a snake in a
code review). Here *every* wrong reading is plausible and adjacent. And there is a sharper irony worth
recording: the format's headline property is **vector geometry in normalized coordinates — explicitly
plate-scale-independent.** Naming it "Plate"-anything primes the exact property the design was built to
repudiate.

Discoverability compounds it: "platemark" is an established art term (the ridge embossed by an intaglio
plate), and the bare word is owned by a printmaking podcast, a 1999-vintage interior-design firm on the
`.com`, and two 2026 restaurant-menu products on `.dev` and `.app`.

---

## 8. Blocked on prior art

| Name | Intended expansion | What blocks it |
|---|---|---|
| **AVAM** | Astronomical Vector Annotation Metadata | **AVM** — Astronomy Visualization Metadata, the IVOA-endorsed standard for metadata embedded in astronomical image files. One inserted letter from the single nearest existing standard in concept space; indistinguishable by ear from a podium |
| **AAM** | Astronomical Annotation Metadata | AAM = *Author Accepted Manuscript*, everyday scholarly-publishing vocabulary every astronomer meets through arXiv deposit; also one letter from AVM |
| **IMAS** | Image Markup and Annotation Standard | ITER's IMAS (Integrated Modelling & Analysis Suite), an established big-science data standard; the neighbour AIMS is already double-blocked in astronomy |
| **VERSA** | Vector-Encoded Region Standard for Astronomy | Versa Networks (SD-WAN vendor, registered marks); "vice versa" pollutes every search |
| **GLOSS** | Geometry, Labels and Overlay Semantics Standard | GLOSS = Global Sea Level Observing System (IOC/UNESCO); `gloss` is an established Haskell 2D-vector-graphics package family |
| **CAIRN** | Coordinate-Anchored Image Region Notation | **CAIRNS** — the Cluster And Infall Region Nearby Survey (Rines et al. 2003–05), one letter away in the same literature; also Cairn Research Ltd (scientific imaging hardware) and phonetically on top of Cairo |
| **Blazon** | heraldry's formal description, from which any herald re-draws the image | **blazar** — one letter, near-homophone, and one of the most-annotated source classes in high-energy astrophysics |
| **Stencil** | — | Entrenched in computational astrophysics (finite-difference stencils, 26 astro-ph abstracts); Stencil.js owns the software namespace |
| **Dioptra** | Hero of Alexandria's sighting instrument | NIST's Dioptra, an active AI/ML test platform (300★), whose audience overlaps the survey-ML community you are courting |
| **Indicium** | Latin *indicium*, the evidence of a witness | Live Rust crate (v0.6.10, 2026-07-12, 928,613 downloads); Indicium Tech, a $92.8M-ARR data/AI consultancy in your exact category with a pending USPTO Class 35 application; plural *indicia* already means the metadata block printed on a published object |

---

## 9. The rest of the field

Generated and collision-checked, not recommended. Recorded so the ground is not re-covered.

| Name | The deciding objection |
|---|---|
| **VOMA** (Vector Overlay Metadata for Astronomy) | The `VO-` prefix is IVOA's de facto namespace (VOTable, VOEvent, VOSpace, VOResource, VO-DML, VOUnits). Using it without sponsorship reads as claiming an endorsement you do not have — and invites the rename at the exact moment of adoption. Also one letter from VOMS. *An acronym additionally freezes today's implementation into the permanent name.* |
| **Reticle** | Semiconductor lithography owns it commercially (a reticle is the photomask patterning a wafer); gun-sight crosshair is the consumer reading; `github.com/reticle` is an org with 11 repos and PyPI is taken |
| **Waymark** | PyPI `waymark` actively maintained (v0.22.0); waymark.com is an AI ad-video company; Waymarking.com is the geocaching incumbent |
| **Signary** | Heard once, lands as "signatory" or "seminary"; `.sgy` is a hard clash with SEG-Y seismic data; GitHub org taken |
| **Sigla** | One vowel from *sigma*, the most-spoken word in an astronomy talk; in Italian/Spanish/Portuguese it is the ordinary word for "acronym" |
| **Legenda** | The format already contains a `legend` block; Legenda is also the Modern Humanities Research Association's academic imprint; a common noun for "caption" in five European languages |
| **Vidimus** | A conjugated verb, not a noun; PyPI taken by a conceptually adjacent provenance-attestation package |
| **Symbolon** | Strongest modern association is the Symbolon astrology card deck — the worst possible adjacency for an astronomy standard; PyPI held by a live 2026 package |
| **Lucida** | *Lucida* already means the brightest star of a constellation; the Lucida typeface family ships on every Mac and PC |
| **Obelus** | The historical obelus marks a passage as *spurious or to be deleted* — a negative valence for a standard that must record confirming and refuting evidence neutrally; the modern glyph is the division sign |
| **Alidade** | In radio astronomy "the alidade" is the rotating azimuth support structure of a large dish; one phoneme from **Aladin**, the CDS sky atlas that already dominates image-plus-overlay work |
| **Stave** | No astronomical connection at all; "score" collides with the numeric grade the schema already carries; `asyml/stave` is an existing annotation-tool framework |

---

## 10. Method, provenance and limitations

### What was run

| Pass | Agents | Shape | Cost |
|---|---|---|---|
| Generation + collision check | 10 | 4 independent naming angles (mark-lineage, astro-acronym, metaphor, architecture) → 25 distinct candidates → 3 web-search collision batches → 3 independent judging lenses (astronomy-community adoption, engineering/spec ergonomics, ten-year longevity) | 674,575 tokens, 210 tool calls, ~29 min |
| Adversarial verification | 4 | Re-checked the two leaders against primary sources with instructions to *refute*; vetted the three names the room itself had floated, which the first pass had been told to skip | 301,901 tokens, 193 tool calls, ~12 min |

The verification pass earned its keep: it **overturned the first pass's top two names** (SkyMark and
PlateMark), **corrected a claim the first pass had gotten wrong in both directions** (the astrometry.net
"skymark" usage — real but far less entrenched than reported, while two worse collisions had been
missed entirely), and **refuted an availability claim** (Indicium's Rust crate was reported gone; it
ships 928k downloads).

### Sources actually queried

ASCL, the arXiv API (metadata and, for Lang et al. 2010, the PDF via pdftotext), the full IVOA document
index, the astropy affiliated-package registry, PyPI JSON and simple APIs, the npm registry, crates.io,
the GitHub REST and code-search APIs, RDAP (PIR and Identity Digital), registry whois, `dig`, and
project source trees (astrometry.net, KStars/Ekos) fetched and grepped directly.

### Limitations — read these before relying on the study

1. **NASA ADS was never queried.** Its API returns HTTP 401 without a token and the SciX web UI returns
   an empty HTTP 202 to a fetcher. Control-verified arXiv and ASCL searches were substituted. Full-text
   literature usage of any candidate is therefore **not** established — only title/abstract and code-registry
   usage.
2. **No trademark clearance was performed for any name.** Justia and Trademarkia return HTTP 403 to
   programmatic access and the USPTO search API refused the request. The trademark facts that do appear
   (Markstone class 33, Markstone Capital's dead mark, Indicium's pending Class 35) came from secondary
   sources. **Counsel must clear the chosen name before public launch** — most urgently if the team ever
   promotes LensMark outward, since it has real commercial users.
3. **Domain and registry states are as of 2026-09-03** and change. `markstone.org` expires 2026-09-07.
4. **Judging is judgement.** The three lenses were scored independently and deliberately not reconciled;
   §3's decision rule is the study's inference from their agreement, not a measured fact.

### Prior art worth reading before writing the spec

- **AVM — Astronomy Visualization Metadata**: IVOA-endorsed, metadata embedded directly in astronomical
  image files (JPEG/TIFF/PNG/GIF), covering title/caption/credit, colour-to-wavelength assignment and
  coordinate projection. The nearest neighbour in concept space; the spec should state plainly what
  AstroMark adds (vector annotation geometry, semantic roles, polarity, a deterministic render contract).
- **DS9 region format** — the community's existing de facto shape-overlay format and the adoption
  precedent the room itself cited.
- Historical: Astronomical Markup Language (AML), Astronomical Instrument Markup Language (AIML),
  Remote Telescope Markup Language — 1999-era XML efforts, all dormant.

---

## 11. Open questions for the follow-up session

1. **Adopt AstroMark, or Markstone, or neither?** The `-Mark` family is what lets LensMark survive
   unchanged; that is an argument of convenience, and the room may weigh it differently.
2. **Does the notation layer need its own name at all,** or is "AstroMark Notation" enough?
3. **Profile naming convention:** `astromark/lens/1.0` (path-style, recommended) vs `astromark.lens/1.0`
   (dotted). Path-style matches the existing `lensmark/1.0` shape and reads better in a `$id` URL.
4. **When does the rename land** relative to the paper? The safest order is: decide the name, register
   the namespaces, then write the spec section, then restem the repo — not the reverse.

This study makes no changes to `apps/LensMark` or to the workshop package. The room's naming discussion
is recorded at `workshops/2026-08-31-LensMark/04-annotation-standard.md` §2.11 as an open item; this
report is the answer to it, pending review.
