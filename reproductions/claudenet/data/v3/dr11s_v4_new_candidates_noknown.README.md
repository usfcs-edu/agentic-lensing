# `dr11s_v4_new_candidates_noknown.csv` — provenance and qualifications

**Generated 2026-07-09** by `reproductions/claudenet/397_remove_known_lenses.py`
(repo `agentic-lensing`).

A ranked list of **131,336 DESI Legacy Survey DR11-south galaxies** from the ClaudeNet v4
ensemble sweep, after removing (a) everything the previous sweep already surfaced and
(b) everything within 5″ of a known/published strong lens.

> **Read this first.** This is a *ranked shortlist for inspection*, not a lens sample.
> The score it is ranked by is **not a calibrated lens probability**, and no visual or
> model-based vetting has been applied. The overwhelming majority of these 131,336
> galaxies are **not** strong lenses. See [Qualifications](#qualifications).

---

## Files

| File | Rows | Size | MD5 |
|---|---:|---:|---|
| `dr11s_v4_new_candidates_noknown.csv` — **the shared product** | 131,336 | 13 M | `0fdf9eb85cda52c6e4b8f732414962b8` |
| `dr11s_v4_new_candidates_known_removed.csv` — audit trail of what was dropped | 2,742 | 516 K | `a1de955916df80ad5a7a4733c5f54556` |
| `dr11s_v4_new_candidates_noknown_report.json` — machine-readable summary | — | — | — |
| `dr11s_v4_new_candidates.csv` — input candidate list | 134,078 | 13 M | `d294ba0f11b4ca41c4d1c988d308b0b0` |
| `data/klc_clean250629ymh.csv` — known-lens catalog (KLC) | 12,210 | 504 K | `ed3c670c0894dd78c09c0e914b06538d` |
| `reproductions/huang-2020/data/huang2020_published_catalog.csv` — unioned in | 342 | 16 K | `c675268d004ee2addc09d7d08e69966a` |
| `reproductions/claudenet/v3/external_lens_catalog.csv` — unioned in | 3 | 4 K | `7d8125dcb9f15b5707e87625f7284dba` |
| `reproductions/claudenet/397_remove_known_lenses.py` — the script | — | 20 K | `199d7114fe406bba2c70a3a782f84d40` |

Environment: python 3.12.12, astropy 8.0.1, pandas 2.2.3, numpy 2.3.3.

### Columns

| Column | Meaning |
|---|---|
| `rank` | rank in the full v4 top-150k, by `mean5` (44 … 149,999) |
| `rank_within_new` | rank within the 134,078 "new vs previous sweep" subset (2 … 134,078) |
| `rank_clean` | **contiguous rank 1 … 131,336 in this file**, same ordering |
| `row_id` | parent-catalog source id (`s_<brickid>_<objid>`) |
| `RA`, `DEC` | degrees, J2000 (0.003° … 359.9997°, −89.13° … +35.59°) |
| `mean5` | the ranking score (0.2051 … 0.5852) — **see Qualifications** |
| `footprint` | always `south` |
| `brick` | Legacy Survey brick name (97,327 distinct) |

All three rank columns are monotone in the same direction, so `rank_clean` preserves the
original v4 ordering exactly; the gaps in `rank` / `rank_within_new` are the removed rows.

---

## How it was made

### 1. The v4 ensemble sweep → top 150,000 (`395_combine_resweep.py`, on Perlmutter)

All **53.8 M** DR11-south parent rows are scored by five models, and `mean5` is their
plain arithmetic mean:

- `member_effnet_B`, `member_zoobot_N` — anchors carried over from the original stage-1 ensemble
- `member_effnet_S2_b50_dr11`, `member_effnet_B3_b50_dr11`, `member_resnet46_C_b50_dr11` — three DR11 fine-tuned members

The top 150,000 rows by `mean5` are kept → `survivors_dr11s_v4.parquet`.

### 2. Subtract the previous sweep → 134,078 "new" (`396_export_v4_new_candidates.py`)

Of the 150,000, **15,922** also appear in the earlier union-95k survivor set
(`survivors_dr10_recal.parquet`); the remaining **134,078** are new *relative to that
previous sweep* and are exported as `dr11s_v4_new_candidates.csv`.

**This is the first of two different meanings of "new," and it is the one that most often
gets misread.** At this stage "new" means *not previously surfaced by our own pipeline* — it
does **not** mean *not a published lens*. The v4 sweep deliberately keeps known lenses in its
parent sample (the 160/163 population choice), so catalog lenses are expected to rank high,
and they do: **43 of the top 100** rows of this file were already-published lenses, decaying
monotonically to 0.81 % in the tail. Against the **2.0 %** rate across the whole 134,078-row
list, the top 100 is a **21× enrichment over that base rate** — itself good evidence the v4
ranking carries real signal. (The top-100 rate vs the *tail* rate is a steeper ≈53×; the 21×
figure is the one quoted, and its denominator is the list-wide base rate, not the tail.) The
single highest-ranked "new" candidate (`s_344767_6150`) was a known lens, matched at 0.18″ to
Huang+2020 `DESI-359.8897+02.1399`.

### 3. Subtract the literature → 131,336 (`397_remove_known_lenses.py`, this step)

Every candidate within **5″** of a known lens is removed. "Known" is the **union of every
known-lens catalog this repo ships** — 12,555 rows in total:

| Catalog | Rows | Notes |
|---|---:|---|
| `klc_clean250629ymh.csv` (KLC) | 12,210 | drawn from **39 distinct source catalogs** (Storfer 2023, Huang 2022, Stein21, Petrillo+2019, SuGOHI 1–8, Jacobs+2019b, Diehl+2017, SLACS, BELLS, …), appearing as 171 distinct `ref` strings |
| `huang2020_published_catalog.csv` | 342 | **7 of these are absent from the KLC**; unioning them removes 1 extra candidate |
| `external_lens_catalog.csv` | 3 | contributes **0** removals beyond the above |

```
134,078  input candidates
 −2,742  within 5" of a known lens
────────
131,336  survivors
```

Cross-catalog duplicates are **not** deduped. That is harmless for removal (see below) but
inflates the recall denominator — see Qualification 3.

Matching uses astropy `match_to_catalog_sky` (true great-circle separation, so RA wraparound
and the poles are handled), at the 5″ radius this repo already uses in
`163_crossmatch_known.py`.

**Removal uses the forward nearest neighbour** (candidate → nearest catalog entry). This is
exact, not an approximation: if a candidate's *nearest* entry lies beyond 5″, then no entry is
within 5″, so a radius search would drop the same rows.

The 2,742 removed rows decompose as:

- **2,460** distinct catalog entries are some removed row's nearest match
- **+ 282** additional rows are extra Tractor deblends of **254** of those same systems
  (one lens split into several catalog sources) — all removed, correctly
- = **2,742**

Separately, **2,540** catalog entries lie within 5″ of at least one candidate. The 80-entry
difference from 2,460 is *duplicate literature rows*: the same physical lens listed more than
once, either twice within the KLC by two source catalogs a few arcsec apart (e.g. KLC `index`
201, Stein21, sits 4.11″ from `index` 10386, Storfer 2023) or once in the KLC and again in
huang2020. Only the nearest row is ever anyone's nearest neighbour.

> **A trap when reading the audit CSV.** 24 removed rows carry
> `klc_catalog = huang2020_published_catalog`, but only **one** of them is a *net-new* removal
> (`s_308920_5028`). The other 23 were already being dropped via the KLC and merely have a
> marginally-nearer huang2020 counterpart. The union is a strict superset of the KLC-only
> removal set: +1 row, 0 rows un-removed.

Matches are tight — median separation **0.13″**, and 2,285 of 2,742 are under 1″ — which is
what a genuine positional association looks like, not chance alignment.

### Reproduce

```bash
cd reproductions/claudenet
python 397_remove_known_lenses.py              # full union -> the shipped 131,336-row product
python 397_remove_known_lenses.py --radius 3   # sensitivity
python 397_remove_known_lenses.py --no-default-extra  # KLC only -> 131,337 (one lens leaks)
python 397_remove_known_lenses.py --flag-only  # keep all 134,078 rows, add is_known/sep_arcsec
```

The repo's own catalogs are auto-unioned by default, so a bare invocation regenerates exactly
the shipped CSV. Step 1 requires Perlmutter (`survivors_dr11s_v4.parquet`, 53.8 M parent rows);
steps 2–3 run locally in seconds.

---

## Verification performed

- **Two independent matchers agree.** Every one of the 134,078 separations was computed with
  both astropy `match_to_catalog_sky` and an sklearn `BallTree(metric="haversine")`; they agree
  to 1.8 × 10⁻¹⁰ arcsec, with zero disagreements about which catalog entry is nearest.
- **Exact partition.** survivors (131,336) + removed (2,742) = input (134,078); the `row_id`
  sets are disjoint and their union is the input set; survivor rows are byte-identical to their
  source rows in all carried columns.
- **Invariant enforced in-script.** No survivor lies within 5″ of any catalog entry (the nearest
  is at 5.003″); every removed row lies within 5″. This is asserted at runtime, not just checked.
- **Union is a strict superset.** Adding huang2020 + external to the KLC removes exactly 1
  additional row and un-removes 0.
- **Audit trail.** Each of the 2,742 removed rows carries `sep_arcsec` and the matched
  `klc_index`, `klc_ra`, `klc_dec`, `klc_ref`, `klc_catalog`, so any removal can be re-checked
  by hand.
- The script was put through an adversarial code review (four independent reviewers, each
  finding re-tested by a separate agent instructed to refute it). Two real defects were found
  in the `--flag-only` diagnostic path and fixed; none affected this CSV.

---

## Qualifications

**1. `mean5` is a ranking score, not P(lens). This list is *unpublished*, not *pure*.**
`mean5` is the arithmetic mean of five model outputs, used to order 53.8 M galaxies and cut the
top 150 k. It is **not calibrated to the survey base rate**, and graded `p_lens` comes only from
the LensJudge vetting stage, **which has not been run on this set**. Removing known lenses
removes things we already know about; it does nothing to purity. For scale: the known-lens
density falls from 43 % in the top 100 rows to ~1 % by rank 50,000. Treat this as a ranked
inspection queue.

**2. "Known" means every catalog *this repo holds*, which is not every catalog that exists.**
The union covers the KLC, huang2020 and external_lens_catalog. `_clib.known_lens_catalogs()`
also names `storfer2024_published_catalog.csv`, `inchausti2025_published_catalog.csv` and
`huang2021_published_catalog.csv`, **none of which are staged in this checkout** — they could
not be unioned in. The KLC does carry `Storfer 2023` (1,892 entries) and `Huang 2022` (1,301)
refs, so their content is very likely covered, but this has not been verified entry-by-entry.
A lens published elsewhere, or after the KLC was compiled, can still be in here.

**3. The 20.2 % "known-lens recall" in the JSON report is a floor, not v4's recall.**
2,540 of 12,555 union entries fall within 5″ of a candidate in this file. Do **not** read that
as "v4 recovers 20 % of known lenses," for three reasons: (a) the denominator is the whole-sky
union, including northern and non-DESI entries that this DR11-**south** sweep never looked at;
(b) the union is not deduped, so a lens in both the KLC and huang2020 counts twice in the
denominator; and (c) the input already excludes the 15,922 rows retained from the previous
sweep — which is precisely where previously-found known lenses concentrate. A true in-footprint
recall needs the sweep parent manifest, which lives on Perlmutter.

**4. This file is a subset of the top-150k, not the whole of it.**
If you want *"the full v4 top-150k minus known lenses"* rather than *"the new-vs-previous-sweep
subset minus known lenses,"* you must start from `survivors_dr11s_v4.parquet` on Perlmutter
(`396_export_v4_new_candidates.py --full`) and run step 3 on that. The 15,922 retained rows are
**not** in this CSV.

**5. Deblending cuts both ways.**
A single lens can appear as several `row_id`s. On the removal side this is handled (282 extra
rows dropped on 254 systems). On the survivor side it means the 131,336 rows are **sources, not
systems** — some are multiple components of the same galaxy, so the number of distinct objects
is somewhat lower.

**6. The 5″ radius is a choice, but the result barely depends on it.**
Removals: 2,285 @ 1″ · 2,484 @ 3″ · **2,742 @ 5″** · 3,027 @ 10″ · 3,215 @ 30″. Between 5″ and
30″ — a 36× increase in search area — only 473 further rows match, so the chance-coincidence
background is low and 5″ sits on a plateau. The full table is in the JSON report.

**7. Footprint.** DR11 **south** only (`footprint == "south"` for all rows; DEC −89.1° to
+35.6°). Catalog entries outside that footprint cannot match by construction and are irrelevant
to the subtraction.

---

*Questions → Greg Benson. Script: `reproductions/claudenet/397_remove_known_lenses.py`
(MD5 `199d7114fe406bba2c70a3a782f84d40`). Machine-readable counts, the full radius-sensitivity
table, and these caveats are also in `dr11s_v4_new_candidates_noknown_report.json`.*
