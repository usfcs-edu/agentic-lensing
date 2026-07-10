# Subtracting the Known-Lens Catalog from the ClaudeNet v4 DR11-South Candidate Ranking

**Internal data-product note — agentic-lensing program (X. Huang group)**
**Author:** Greg Benson (with an adversarial multi-agent verification workflow) · **Date:** 2026-07-09
**Repo:** `agentic-lensing` @ `806fdc9` · **Script:** `reproductions/claudenet/397_remove_known_lenses.py`

> **Provenance.** The subtraction was performed by a single deterministic script (`397`), then
> verified three ways: (1) every one of the 134,078 sky separations was recomputed with a second,
> independent matcher (`sklearn.neighbors.BallTree(metric="haversine")` vs astropy
> `match_to_catalog_sky`) — they agree to 1.8 × 10⁻¹⁰ arcsec with zero disagreements about which
> catalog entry is nearest; (2) the written CSVs were re-checked against the inputs by a suite of
> 12 post-hoc integrity assertions; (3) the script was put through a four-lens adversarial code
> review (astrometry / dataflow / statistical semantics / reporting), in which each raised finding
> was handed to a separate agent instructed to *refute* it (13 agents, ~356 K tokens, 73 tool calls).
> Of **9 findings raised, 2 survived refutation** — both in the `--flag-only` diagnostic path, both
> fixed, neither affecting the shipped CSV. The astrometry lens raised **zero** findings.

---

## What this document is for

`dr11s_v4_new_candidates_noknown.csv` is a ranked list of **131,337 DESI Legacy Survey
DR11-south galaxies** intended for sharing outside the immediate group. This note records
exactly how it was produced and what it does and does not claim, so nobody has to reverse-engineer
that from the filename.

> **Read this first.** This is a *ranked shortlist for inspection*, not a lens sample. The score
> it is ranked by is **not a calibrated lens probability**, and no visual or model-based vetting
> has been applied. The overwhelming majority of these 131,337 galaxies are **not** strong lenses.
> See [Qualifications](#qualifications).

A copy of this note travels with the data as
`reproductions/claudenet/data/v3/dr11s_v4_new_candidates_noknown.README.md`.

---

## Files

The root `.gitignore` (line 26) excludes every directory named `data`, but these five files are
**force-added exceptions** (`git add -f`) and *are* tracked, so the inputs, the product, and the
audit trail are all pinned in history. The MD5s let you confirm you have the same bytes. The
product is also exactly regenerable from the two inputs — see [Reproduce](#reproduce).

| File | Rows | Size | MD5 |
|---|---:|---:|---|
| `reproductions/claudenet/data/v3/dr11s_v4_new_candidates_noknown.csv` — **the shared product** | 131,337 | 13 M | `33f0da36e6971fc46322cb57aef64a4c` |
| `reproductions/claudenet/data/v3/dr11s_v4_new_candidates_known_removed.csv` — audit trail of what was dropped | 2,741 | 476 K | `559dd7bf9bdc7b858b6301b6965d4c94` |
| `reproductions/claudenet/data/v3/dr11s_v4_new_candidates_noknown_report.json` — machine-readable summary | — | — | — |
| `reproductions/claudenet/data/v3/dr11s_v4_new_candidates.csv` — input candidate list | 134,078 | 13 M | `d294ba0f11b4ca41c4d1c988d308b0b0` |
| `data/klc_clean250629ymh.csv` — input known-lens catalog (KLC) | 12,210 | 504 K | `ed3c670c0894dd78c09c0e914b06538d` |
| `reproductions/claudenet/397_remove_known_lenses.py` — the script | — | — | `b5640bfe0e8108694e9ff026bc63edbe` |

Environment: python 3.12.12, astropy 8.0.1, pandas 2.2.3, numpy 2.3.3.

### Columns

| Column | Meaning |
|---|---|
| `rank` | rank in the full v4 top-150k, by `mean5` (44 … 149,999) |
| `rank_within_new` | rank within the 134,078 "new vs previous sweep" subset (2 … 134,078) |
| `rank_clean` | **contiguous rank 1 … 131,337 in this file**, same ordering |
| `row_id` | parent-catalog source id (`s_<brickid>_<objid>`) |
| `RA`, `DEC` | degrees, J2000 (0.003° … 359.9997°, −89.13° … +35.59°) |
| `mean5` | the ranking score (0.2051 … 0.5852) — **see Qualifications** |
| `footprint` | always `south` |
| `brick` | Legacy Survey brick name (97,328 distinct) |

All three rank columns are monotone in the same direction, so `rank_clean` preserves the original
v4 ordering exactly; the gaps in `rank` / `rank_within_new` are the removed rows.

---

## How it was made

### 1. The v4 ensemble sweep → top 150,000 (`395_combine_resweep.py`, on Perlmutter)

All **53.8 M** DR11-south parent rows are scored by five models, and `mean5` is their plain
arithmetic mean:

- `member_effnet_B`, `member_zoobot_N` — anchors carried over from the original stage-1 ensemble
- `member_effnet_S2_b50_dr11`, `member_effnet_B3_b50_dr11`, `member_resnet46_C_b50_dr11` — three DR11 fine-tuned members

The top 150,000 rows by `mean5` are kept → `survivors_dr11s_v4.parquet`.

### 2. Subtract the previous sweep → 134,078 "new" (`396_export_v4_new_candidates.py`)

Of the 150,000, **15,922** also appear in the earlier union-95k survivor set
(`survivors_dr10_recal.parquet`); the remaining **134,078** are new *relative to that previous
sweep*, and are exported as `dr11s_v4_new_candidates.csv`.

**This is the first of two different meanings of "new," and it is the one that most often gets
misread.** At this stage "new" means *not previously surfaced by our own pipeline* — it does
**not** mean *not a published lens*. The v4 sweep deliberately keeps known lenses in its parent
sample (the 160/163 population choice), so catalog lenses are expected to rank high, and they do:
**43 of the top 100** rows of this file were already-published lenses, decaying monotonically to
0.81 % in the tail — a 21× enrichment that is itself good evidence the v4 ranking carries real
signal. The single highest-ranked "new" candidate (`s_344767_6150`) was a known SuGOHI/DECaLS
lens, matched at 0.50″.

### 3. Subtract the literature → 131,337 (`397_remove_known_lenses.py`, this step)

Every candidate within **5″** of an entry in `klc_clean250629ymh.csv` (12,210 published lenses,
drawn from **39 distinct source catalogs** — Storfer 2023, Huang 2022, Stein21, Petrillo+2019,
SuGOHI 1–8, Jacobs+2019b, Diehl+2017, SLACS, BELLS, … — appearing as 171 distinct `ref` strings,
since a lens found by several surveys carries a combined `ref`) is removed.

```
134,078  input candidates
 −2,741  within 5" of a KLC entry
────────
131,337  survivors
```

Matching uses astropy `match_to_catalog_sky` (true great-circle separation, so RA wraparound and
the poles are handled), at the 5″ radius this repo already uses in `163_crossmatch_known.py`.

**Removal uses the forward nearest neighbour** (candidate → nearest KLC entry). This is exact, not
an approximation: if a candidate's *nearest* KLC entry lies beyond 5″, then no KLC entry is within
5″, so a radius search would drop exactly the same rows.

The 2,741 removed rows decompose as:

- **2,457** distinct KLC entries are some removed row's nearest match
- **+ 284** additional rows are extra Tractor deblends of **256** of those same systems (one lens
  split into several catalog sources) — all removed, correctly
- = **2,741**

Separately, **2,467** KLC entries lie within 5″ of at least one candidate. The 10-entry difference
from 2,457 is *near-duplicate literature rows*: the same physical lens listed twice in the KLC by
two source catalogs a few arcsec apart — e.g. KLC `index` 201 (Stein21) sits 4.11″ from `index`
10386 (Storfer 2023), and the candidate matches the latter at 0.007″ — so only the nearer row is
ever anyone's nearest neighbour.

Matches are tight — median separation **0.13″**, and 2,283 of 2,741 are under 1″ — which is what a
genuine positional association looks like, not chance alignment.

### Reproduce

```bash
cd reproductions/claudenet
python 397_remove_known_lenses.py              # 5", the default; writes all three outputs
python 397_remove_known_lenses.py --radius 3   # sensitivity
python 397_remove_known_lenses.py --flag-only  # keep all 134,078 rows, add is_known/sep_arcsec

# strictly-safer variant, see Qualification 2 (-> 131,336 rows)
python 397_remove_known_lenses.py \
    --extra-known ../huang-2020/data/huang2020_published_catalog.csv
```

Step 1 requires Perlmutter (`survivors_dr11s_v4.parquet`, 53.8 M parent rows). Steps 2–3 run
locally in seconds.

---

## Verification performed

- **Two independent matchers agree.** All 134,078 separations computed with both astropy
  `match_to_catalog_sky` and sklearn `BallTree(metric="haversine")`; agreement to
  1.8 × 10⁻¹⁰ arcsec, zero disagreements on nearest-neighbour identity.
- **Exact partition.** survivors (131,337) + removed (2,741) = input (134,078); the `row_id` sets
  are disjoint and their union is the input set; survivor rows are byte-identical to their source
  rows across all carried columns.
- **Invariant enforced at runtime, not merely checked.** The script asserts that no survivor lies
  within 5″ of any KLC entry (the nearest is at 5.003″) and that every removed row lies within 5″.
- **Audit trail.** Each of the 2,741 removed rows carries `sep_arcsec` and the matched `klc_index`,
  `klc_ra`, `klc_dec`, `klc_ref`, so any single removal can be re-checked by hand.
- **Adversarial review.** Four independent review lenses; every finding re-tested by a separate
  agent instructed to refute it. 9 findings raised, 2 survived — an int64→float64 dtype promotion
  of `klc_index`, and a report-path collision — both confined to the `--flag-only` diagnostic path,
  both fixed and re-verified. Neither touched the shipped CSV.

---

## Qualifications

**1. `mean5` is a ranking score, not P(lens). This list is *unpublished*, not *pure*.**
`mean5` is the arithmetic mean of five model outputs, used to order 53.8 M galaxies and cut the
top 150 k. It is **not calibrated to the survey base rate**, and graded `p_lens` comes only from
the LensJudge vetting stage, **which has not been run on this set**. Removing the KLC removes
things we already know about; it does nothing to purity. For scale: known-lens density falls from
43 % in the top 100 rows to ~1 % by rank 50,000. Treat this as a ranked inspection queue.

**2. The KLC is not a strict superset of every catalog we hold.**
Seven of the 342 lenses in the repo's own `reproductions/huang-2020/data/huang2020_published_catalog.csv`
are absent from `klc_clean250629ymh.csv`. Exactly **one** of those seven survives this subtraction —
`DESI-036.6760-03.6801` (grade C), at `rank_clean` 43,549. The honest description is therefore
"**KLC-subtracted**," not "contains no published lens." The `--extra-known` flag (see
[Reproduce](#reproduce)) removes it, yielding 131,336 rows.

**3. The 20.2 % "known-lens recall" in the JSON report is a floor, not v4's recall.**
2,467 of 12,210 KLC entries fall within 5″ of a candidate in this file. Do **not** read that as
"v4 recovers 20 % of known lenses," for two reasons: (a) the denominator is the whole-sky KLC,
including northern and non-DESI entries this DR11-**south** sweep never looked at; and (b) the
input already excludes the 15,922 rows retained from the previous sweep — precisely where
previously-found known lenses concentrate. A true in-footprint recall needs the sweep parent
manifest, which lives on Perlmutter and was not available locally.

**4. This file is a subset of the top-150k, not the whole of it.**
If you want *"the full v4 top-150k minus known lenses"* rather than *"the new-vs-previous-sweep
subset minus known lenses,"* start from `survivors_dr11s_v4.parquet` on Perlmutter
(`396_export_v4_new_candidates.py --full`) and run step 3 on that. The 15,922 retained rows are
**not** in this CSV.

**5. Deblending cuts both ways.**
A single lens can appear as several `row_id`s. On the removal side this is handled (284 extra rows
dropped across 256 systems). On the survivor side it means the 131,337 rows are **sources, not
systems** — some are multiple components of the same galaxy, so the number of distinct objects is
somewhat lower.

**6. The 5″ radius is a choice, but the result barely depends on it.**
Removals: 2,283 @ 1″ · 2,483 @ 3″ · **2,741 @ 5″** · 3,026 @ 10″ · 3,212 @ 30″. Between 5″ and 30″ —
a 36× increase in search area — only 471 further rows match, so the chance-coincidence background is
low and 5″ sits on a plateau. The full table is in the JSON report.

**7. Footprint.** DR11 **south** only (`footprint == "south"` for every row; DEC −89.1° to +35.6°).
KLC entries outside that footprint cannot match by construction and are irrelevant to the subtraction.

---

*Questions → Greg Benson. Machine-readable counts, the full radius-sensitivity table, and these
caveats are also emitted into `dr11s_v4_new_candidates_noknown_report.json` by the script itself.*
