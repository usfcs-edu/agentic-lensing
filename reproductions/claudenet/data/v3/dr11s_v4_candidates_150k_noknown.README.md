# `dr11s_v4_candidates_150k_noknown.csv` — provenance and qualifications

**Generated 2026-07-10** by `reproductions/claudenet/397_remove_known_lenses.py`
(repo `agentic-lensing`).

A ranked list of **145,856 DESI Legacy Survey DR11-south sources** — the **full ClaudeNet
v4 top-150k**, with published lenses removed.

> **Read this first.** This is a *ranked shortlist for inspection*, not a lens sample.
> The score it is ranked by is **not a calibrated lens probability**, and no visual or
> model-based vetting has been applied. The overwhelming majority of these 145,856
> sources are **not** strong lenses. See [Qualifications](#qualifications).

> **Supersedes the earlier 131,336-row cut.** A first version of this list
> (`dr11s_v4_new_candidates_noknown.csv`) kept only the 134,078 candidates *new relative to
> our previous sweep*, which discarded 14,520 real candidates — including v4's **#1-ranked**
> source. This list keeps the whole top-150k and carries an `is_new` flag instead of
> filtering on it; the old view is exactly the `is_new == 1` survivors here.

---

## Files

| File | Rows | Size | MD5 |
|---|---:|---:|---|
| `dr11s_v4_candidates_150k_noknown.csv` — **the shared product** | 145,856 | 15 M | `adfdb4d0d00b51ce08a307ce6c109725` |
| `dr11s_v4_candidates_150k_known_removed.csv` — audit trail of what was dropped | 4,144 | 772 K | `adf19de83882537f3dbe50ef9bd36ddc` |
| `dr11s_v4_candidates_150k_noknown_report.json` — machine-readable summary | — | — | — |
| `dr11s_v4_candidates_150k.csv` — input (full v4 top-150k, `396 --full`) | 150,000 | 15 M | `85b7ec049a3497506ad7a12e576e4b60` |
| `data/klc_clean250629ymh.csv` — known-lens catalog (KLC) | 12,210 | 504 K | `ed3c670c0894dd78c09c0e914b06538d` |
| `reproductions/huang-2020/data/huang2020_published_catalog.csv` — unioned in | 342 | 16 K | `c675268d004ee2addc09d7d08e69966a` |
| `reproductions/claudenet/v3/external_lens_catalog.csv` — unioned in | 3 | 4 K | `7d8125dcb9f15b5707e87625f7284dba` |
| `reproductions/claudenet/397_remove_known_lenses.py` — the script | — | 20 K | `1d021947c9d9a0b74d0b47ba35f7c5fb` |

Environment: python 3.12.12, astropy 8.0.1, pandas 2.2.3, numpy 2.3.3.

### Columns

| Column | Meaning |
|---|---|
| `rank` | rank in the full v4 top-150k, by `mean5` (1 … 150,000) |
| `rank_within_new` | rank within the 134,078 `is_new==1` subset; **blank for `is_new==0` rows** |
| `rank_clean` | **contiguous rank 1 … 145,856 in this file**, same ordering as `rank` |
| `is_new` | 1 = new vs the previous union-95k sweep (134,078 of the 150k); 0 = also surfaced then (15,922) |
| `row_id` | parent-catalog source id (`s_<brickid>_<objid>`) |
| `RA`, `DEC` | degrees, J2000 (0.003° … 359.9997°, −89.13° … +35.59°) |
| `mean5` | the ranking score (0.2051 … 0.7082) — **see Qualifications** |
| `footprint` | always `south` |
| `brick` | Legacy Survey brick name (103,992 distinct) |

`rank_clean` preserves the v4 ordering exactly; the gaps in `rank` are the removed rows.
To recover the earlier 131,336-row list: `is_new == 1`.

---

## How it was made

### 1. The v4 ensemble sweep → top 150,000 (`395_combine_resweep.py`, on Perlmutter)

All **53.8 M** DR11-south parent rows are scored by five models, and `mean5` is their plain
arithmetic mean:

- `member_effnet_B`, `member_zoobot_N` — anchors carried over from the original stage-1 ensemble
- `member_effnet_S2_b50_dr11`, `member_effnet_B3_b50_dr11`, `member_resnet46_C_b50_dr11` — three DR11 fine-tuned members

The top 150,000 rows by `mean5` are kept → `survivors_dr11s_v4.parquet`, exported to
`dr11s_v4_candidates_150k.csv` by `396_export_v4_new_candidates.py --full`. That export tags
each row with `is_new`: **134,078** are new relative to the previous union-95k sweep
(`survivors_dr10_recal.parquet`, held-out recall 0.54/0.32), **15,922** were also surfaced by it.

The `is_new` flag is **metadata, not a filter.** "New" there means *not previously surfaced by
our own — weaker — selector*; it does **not** mean *not a published lens*, and it does not mean
*unexamined* (the 15,922 were never vetted either; the as-run DR11-south vetting used a different
model, `v3blend8`). Since the previous selector had far lower recall, its overlap with v4's top
is heavily weighted to the very top of the ranking, so filtering it out discarded v4's best
candidates. We keep the whole top-150k.

### 2. Subtract the literature → 145,856 (`397_remove_known_lenses.py`, this step)

Every candidate within **5″** of a known lens is removed. "Known" is the **union of all three
known-lens catalogs this repo ships** — 12,555 rows in total:

| Catalog | Rows | Removals attributed |
|---|---:|---:|
| `klc_clean250629ymh.csv` (KLC) | 12,210 | 4,090 |
| `huang2020_published_catalog.csv` | 342 | 51 |
| `external_lens_catalog.csv` | 3 | 3 |

```
150,000  full v4 top-150k
 −4,144  within 5" of a known lens
────────
145,856  survivors
```

Matching uses astropy `match_to_catalog_sky` (true great-circle separation, so RA wraparound and
the poles are handled), at the 5″ radius this repo already uses in `163_crossmatch_known.py`.

**Removal uses the forward nearest neighbour** (candidate → nearest catalog entry). This is exact,
not an approximation: if a candidate's *nearest* entry lies beyond 5″, then no entry is within 5″,
so a radius search would drop exactly the same rows.

The 4,144 removed rows decompose as **3,724** distinct catalog entries that are some removed row's
nearest match, **+ 420** extra Tractor deblends of **379** of those systems (one lens split into
several catalog sources), all removed. Separately, **3,874** catalog entries lie within 5″ of ≥1
candidate; the difference from 3,724 is duplicate literature rows (the same lens listed twice a
few arcsec apart). Matches are tight — median separation **0.13″**, and 3,554 of 4,144 are under
1″ — a genuine positional association, not chance alignment.

### Reproduce

```bash
cd reproductions/claudenet
python 397_remove_known_lenses.py              # full union -> the shipped 145,856-row product
python 397_remove_known_lenses.py --radius 3   # sensitivity
python 397_remove_known_lenses.py --no-default-extra  # KLC only (one huang2020 lens leaks back)
```

The repo's own catalogs are auto-unioned by default, so a bare invocation regenerates exactly the
shipped CSV — verified byte-for-byte from a clean worktree. Step 1 requires Perlmutter
(`survivors_dr11s_v4.parquet`, 53.8 M parent rows); step 2 runs locally in seconds.

---

## Verification performed

- **Two independent matchers agree.** Every one of the 150,000 separations was computed with both
  astropy `match_to_catalog_sky` and an sklearn `BallTree(metric="haversine")`; they agree to
  1.8 × 10⁻¹⁰ arcsec, zero disagreements about which catalog entry is nearest.
- **Exact partition.** survivors (145,856) + removed (4,144) = input (150,000); the `row_id` sets
  are disjoint and their union is the input; survivor rows are byte-identical to their source rows.
- **Invariant enforced in-script.** No survivor lies within 5″ of any catalog entry (nearest is
  5.0001″); every removed row lies within 5″. Asserted at runtime.
- **Superset check.** The earlier 131,336-row list is exactly the `is_new==1` survivors here, and
  every one of its rows is present — switching to the full list loses nothing.
- **Audit trail.** Each of the 4,144 removed rows carries `sep_arcsec` + the matched `klc_index`,
  `klc_ra`, `klc_dec`, `klc_ref`, `klc_catalog`, so any removal can be re-checked by hand.
- The script passed an adversarial code review (four independent reviewers, each finding re-tested
  by a separate agent instructed to refute it); the two real defects found were in the `--flag-only`
  path and fixed, neither affecting this CSV.

---

## Qualifications

**1. `mean5` is a ranking score, not P(lens). This list is *unpublished*, not *pure*.**
`mean5` is the arithmetic mean of five model outputs, used to order 53.8 M galaxies and cut the
top 150 k. It is **not calibrated to the survey base rate**, and graded `p_lens` comes only from
the LensJudge vetting stage, **which has not been run on this set**. Removing known lenses removes
things we already know about; it does nothing to purity. For scale: **81 of v4's top 100** rows are
already-published lenses, falling to ~1 % by rank 20,000. Treat this as a ranked inspection queue.

**2. "Known" means every catalog *this repo holds*, which is not every catalog that exists.**
The union covers the KLC, `huang2020_published_catalog.csv` and `external_lens_catalog.csv`.
`_clib.known_lens_catalogs()` also names `storfer2024_published_catalog.csv`,
`inchausti2025_published_catalog.csv` and `huang2021_published_catalog.csv`, **none of which are
staged in this checkout** — they could not be unioned in. The KLC does carry `Storfer 2023`
(1,892 entries) and `Huang 2022` (1,301) refs, so their content is very likely covered, but this
has not been verified entry-by-entry. A lens published elsewhere, or after the KLC was compiled,
can still be in here.

**3. The "known-lens recall" in the JSON report is a rough number, not v4's recall.**
3,874 of 12,555 union entries fall within 5″ of a candidate in this file (~31 %). Do **not** read
that as "v4 recovers 31 % of known lenses": the denominator is the whole-sky union — including
northern and non-DESI entries this DR11-**south** sweep never looked at — and it is not deduped
(a lens in two catalogs counts twice). A true in-footprint recall needs the sweep parent manifest
and the held-out split; the paper's held-out figures (Inchausti grade-A 0.87, Storfer 0.825) are
the ones to cite. Note this recall is *higher* than the earlier list's (~20 %) precisely because
this list keeps v4's lens-rich top instead of discarding it.

**4. Deblending: these are sources, not systems.**
A single lens can appear as several `row_id`s. On the removal side this is handled (420 extra rows
dropped on 379 systems). On the survivor side it means the 145,856 rows are **sources, not
systems** — some are multiple components of the same galaxy, so the number of distinct objects is
somewhat lower.

**5. The 5″ radius is a choice, but the result barely depends on it.**
Removals: 3,554 @ 1″ · 3,826 @ 3″ · **4,144 @ 5″** · 4,509 @ 10″ · 4,716 @ 30″. Between 5″ and 30″ —
a 36× increase in search area — only 572 further rows match, so the chance-coincidence background
is low and 5″ sits on a plateau. The full table is in the JSON report.

**6. Footprint.** DR11 **south** only (`footprint == "south"` for every row; DEC −89.1° to +35.6°).
Catalog entries outside that footprint cannot match by construction.

---

*Questions → Greg Benson. Script: `reproductions/claudenet/397_remove_known_lenses.py`
(MD5 `1d021947c9d9a0b74d0b47ba35f7c5fb`). Machine-readable counts, the full radius-sensitivity
table, and these caveats are also in `dr11s_v4_candidates_150k_noknown_report.json`.*
