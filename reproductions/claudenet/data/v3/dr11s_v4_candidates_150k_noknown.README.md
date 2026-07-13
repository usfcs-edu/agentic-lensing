# `dr11s_v4_candidates_150k_noknown.csv` — provenance and qualifications

**Generated 2026-07-13** by `reproductions/claudenet/397_remove_known_lenses.py`
(repo `agentic-lensing`).

A ranked list of **145,297 DESI Legacy Survey DR11-south sources** — the **full ClaudeNet
v4 top-150k**, with published lenses removed.

> **Read this first.** This is a *ranked shortlist for inspection*, not a lens sample.
> The score it is ranked by is **not a calibrated lens probability**, and no visual or
> model-based vetting has been applied. The overwhelming majority of these 145,297
> sources are **not** strong lenses. See [Qualifications](#qualifications).

> **This revision (2026-07-13) closes a gap: Inchausti+2025 (DR10) lenses.** The previous
> version removed only lenses in the KLC + Huang+2020 + external catalogs. The KLC turns out
> **not** to contain the group's DR10 search (Inchausti+2025) at all — 0 of its 811 entries —
> so **559 published Inchausti lenses had survived**. This revision unions all four of the
> group's DESI searches (Huang+2020/2021, Storfer+2024, Inchausti+2025) into the removal set,
> dropping those 559 (145,856 → 145,297). Huang+2021 and Storfer+2024 add 0 net removals (they
> are fully in the KLC); they are staged anyway so the guarantee no longer depends on the KLC's
> contents.

---

## Files

| File | Rows | Size | MD5 |
|---|---:|---:|---|
| `dr11s_v4_candidates_150k_noknown.csv` — **the shared product** | 145,297 | 15 M | `5bf6dace4cecb7b95cb841b45aab976f` |
| `dr11s_v4_candidates_150k_known_removed.csv` — audit trail of what was dropped | 4,703 | 872 K | `281dff061651e79144d7bc61314610bc` |
| `dr11s_v4_candidates_150k_noknown_report.json` — machine-readable summary | — | — | — |
| `dr11s_v4_candidates_150k.csv` — input (full v4 top-150k, `396 --full`) | 150,000 | 15 M | `85b7ec049a3497506ad7a12e576e4b60` |
| `data/klc_clean250629ymh.csv` — known-lens catalog (KLC) | 12,210 | 504 K | `ed3c670c0894dd78c09c0e914b06538d` |
| `reproductions/huang-2020/data/huang2020_published_catalog.csv` — DR7 | 342 | 16 K | `c675268d004ee2addc09d7d08e69966a` |
| `reproductions/huang-2021/data/huang2021_published_catalog.csv` — DR8 | 1,312 | 100 K | `efcfb135d73409cccb09ea6b91030440` |
| `reproductions/inchausti-2025/data/storfer2024_published_catalog.csv` — DR9 | 1,895 | 120 K | `7232b982de7c17cbee5eb357590c98c7` |
| `reproductions/inchausti-2025/data/inchausti2025_published_catalog.csv` — DR10 | 811 | 60 K | `97da1f3b4db2ee55230110d4b07a8994` |
| `reproductions/claudenet/v3/external_lens_catalog.csv` — misc. | 3 | 4 K | `7d8125dcb9f15b5707e87625f7284dba` |
| `reproductions/claudenet/397_remove_known_lenses.py` — the script | — | 20 K | `9d311b33134304a0420451c9772c724c` |

Environment: python 3.12.12, astropy 8.0.1, pandas 2.2.3, numpy 2.3.3.

### Columns

| Column | Meaning |
|---|---|
| `rank` | rank in the full v4 top-150k, by `mean5` (1 … 150,000) |
| `rank_within_new` | rank within the 134,078 `is_new==1` subset; **blank for `is_new==0` rows** |
| `rank_clean` | **contiguous rank 1 … 145,297 in this file**, same ordering as `rank` |
| `is_new` | 1 = new vs the previous union-95k sweep (134,078 of the 150k); 0 = also surfaced then (15,922) |
| `row_id` | parent-catalog source id (`s_<brickid>_<objid>`) |
| `RA`, `DEC` | degrees, J2000 (0.003° … 359.9997°, −89.13° … +35.59°) |
| `mean5` | the ranking score (0.2051 … 0.7082) — **see Qualifications** |
| `footprint` | always `south` |
| `brick` | Legacy Survey brick name |

`rank_clean` preserves the v4 ordering exactly; the gaps in `rank` are the removed rows. The
earlier "new candidates" view (new-vs-previous-sweep, published lenses removed) is now `is_new == 1`
here — **131,004** rows, i.e. 332 fewer than the pre-Inchausti-fix 131,336 (those 332 were
Inchausti lenses).

---

## How it was made

### 1. The v4 ensemble sweep → top 150,000 (`395_combine_resweep.py`, on Perlmutter)

All **53.8 M** DR11-south parent rows are scored by five models, and `mean5` is their plain
arithmetic mean: `member_effnet_B`, `member_zoobot_N` (anchors from the original stage-1 ensemble)
plus three DR11 fine-tuned members `member_effnet_S2_b50_dr11`, `member_effnet_B3_b50_dr11`,
`member_resnet46_C_b50_dr11`. The top 150,000 by `mean5` → `survivors_dr11s_v4.parquet`, exported
to `dr11s_v4_candidates_150k.csv` by `396_export_v4_new_candidates.py --full`, which tags each row
with `is_new` (134,078 new relative to the previous union-95k sweep, 15,922 also surfaced then).
`is_new` is **metadata, not a filter** — we keep the whole top-150k.

### 2. Subtract the literature → 145,297 (`397_remove_known_lenses.py`, this step)

Every candidate within **5″** of a known lens is removed. "Known" is the **union of all six
known-lens catalogs this repo ships** — 16,573 rows, covering the group's four DESI searches plus
the KLC compilation and misc. external systems:

| Catalog | Rows | In KLC (5″) | Removed (nearest-attributed) |
|---|---:|---:|---:|
| `klc_clean250629ymh.csv` (KLC) | 12,210 | — | 3,004 |
| `storfer2024_published_catalog.csv` (DR9) | 1,895 | 1,895 (100 %) | 613 |
| `huang2021_published_catalog.csv` (DR8) | 1,312 | 1,312 (100 %) | 475 |
| `inchausti2025_published_catalog.csv` (DR10) | 811 | **0 (0 %)** | 559 |
| `huang2020_published_catalog.csv` (DR7) | 342 | 335 (98 %) | 49 |
| `external_lens_catalog.csv` | 3 | — | 3 |

```
150,000  full v4 top-150k
 −4,703  within 5" of a known lens
────────
145,297  survivors
```

**Attribution ≠ contribution.** The per-catalog "removed" column counts which catalog is a row's
*nearest* match, not which catalog is *responsible* for the removal. Storfer+2024 (613) and
Huang+2021 (475) are fully inside the KLC, so those rows were already removed via the KLC in the
previous revision — they merely re-attribute to the nearer group catalog now. The **only net-new
removals** over the previous 4,144 are the **559 Inchausti+2025** rows, because Inchausti is the
one search absent from the KLC.

Matching uses astropy `match_to_catalog_sky` (true great-circle separation, so RA wraparound and
the poles are handled), forward nearest neighbour, at the 5″ radius from `163_crossmatch_known.py`.
This is exact for removal: if a candidate's nearest catalog entry is beyond 5″, none is within 5″.
The 4,703 removed rows are **4,250** distinct catalog entries + **453** extra Tractor deblends of
**409** systems. Matches are tight — median separation **0.11″**, 4,076 of 4,703 under 1″.

### Reproduce

```bash
cd reproductions/claudenet
python 397_remove_known_lenses.py              # full union -> the shipped 145,297-row product
python 397_remove_known_lenses.py --radius 3   # sensitivity
python 397_remove_known_lenses.py --no-default-extra  # KLC only (huang2020 + 559 inchausti leak back)
```

The repo's own catalogs are auto-unioned by default, so a bare invocation regenerates exactly the
shipped CSV — verified byte-for-byte. Step 1 requires Perlmutter; step 2 runs locally in seconds.

---

## Verification performed

- **Two independent matchers agree** on all 150,000 separations to 1.8 × 10⁻¹⁰ arcsec, zero
  nearest-neighbour disagreements.
- **Exact partition.** survivors (145,297) + removed (4,703) = input (150,000); disjoint `row_id`
  sets whose union is the input; survivor rows byte-identical to their source rows.
- **Invariant enforced at runtime.** No survivor within 5″ of any catalog entry (nearest 5.0001″);
  every removed row within 5″. Verified: 0 of the shipped top-100 match a known lens.
- **All four group searches confirmed subtracted.** 0 of Huang+2020/2021, Storfer+2024,
  Inchausti+2025 survive within 5″ of any shipped row.
- **Bare-invocation reproducibility.** The committed script with no arguments regenerates the
  product and audit CSVs byte-for-byte.
- **Audit trail.** Each removed row carries `sep_arcsec` + matched `klc_index`/`klc_ra`/`klc_dec`/
  `klc_ref`/`klc_catalog`.

---

## Qualifications

**1. `mean5` is a ranking score, not P(lens). This list is *unpublished*, not *pure*.**
Not calibrated to the survey base rate; graded `p_lens` comes only from LensJudge vetting, **not
run on this set**. For scale: **86 of v4's top 100** rows are already-published lenses, against a
3.1 % rate across the full top-150k — a ~27× enrichment, good evidence the ranking carries real
signal. But the large majority of the 145,297 are **not** strong lenses. Treat this as a ranked
inspection queue.

**2. "Known" now covers all four group searches, but still not every catalog that exists.**
The union covers the KLC plus the group's Huang+2020/2021, Storfer+2024 and Inchausti+2025
catalogs and a small external list. A lens published *outside* these — by another group, or after
these catalogs were compiled — can still be in the list. (This qualification previously said the
group searches were "very likely covered via the KLC"; that was wrong for Inchausti+2025 and is
now fixed by staging the catalog directly rather than trusting the KLC.)

**3. The ~38 % "known-lens recall" in the JSON report is a loose number, not v4's recall.**
6,300 of 16,573 union entries fall within 5″ of a candidate. The denominator is the whole-sky
union — northern/non-DESI entries this DR11-**south** sweep never saw — and is **not deduped**, so
a lens in the KLC *and* Storfer+2024 *and* Huang+2021 counts three times (this is why the number
rose with the added catalogs). Cite the paper's held-out figures (Inchausti grade-A 0.87,
Storfer 0.825) for recall.

**4. Sources, not systems.** Deblending means a lens can appear as several `row_id`s; 453 extra
removals on 409 systems handle this on the removal side, but the 145,297 survivors are **sources**,
so the distinct-object count is somewhat lower.

**5. The 5″ radius barely changes the result.** 4,076 @ 1″ · **4,703 @ 5″** · 5,312 @ 30″: only
609 further rows match between 5″ and 30″ (36× the area), so the chance background is low and 5″
sits on a plateau. Full table in the JSON report.

**6. Footprint.** DR11 **south** only (DEC −89.1° to +35.6°). Catalog entries outside it cannot
match by construction.

---

*Questions → Greg Benson. Script: `reproductions/claudenet/397_remove_known_lenses.py`
(MD5 `9d311b33134304a0420451c9772c724c`). Machine-readable counts, the full radius-sensitivity
table, and these caveats are also in `dr11s_v4_candidates_150k_noknown_report.json`.*
