# Golden LITE frame — build summary

`frame.csv` sha 422eacbcdcf3854d · 250 rows · seed 2026 · built by `golden/build_frame.py`

## Per-stratum / substratum counts

| stratum | substratum | n |
|---|---|---|
| T_verified | A | 5 |
| T_verified | B | 5 |
| T_verified | C | 11 |
| T_U | — | 78 |
| K_cowls | pipe_D | 3 |
| K_cowls | pipe_U | 12 |
| K_cowls | unflagged | 16 |
| L_known | pipe_D | 7 |
| L_known | pipe_U | 13 |
| L_known | unflagged | 10 |
| D_refuted | elliptical_nearmiss | 10 |
| D_refuted | merger | 10 |
| D_refuted | other | 10 |
| D_refuted | ring_spiral | 10 |
| U_tail | rank_101_300 | 20 |
| U_tail | rank_301_2024 | 10 |
| N_unflagged | proposal_1063 | 1 |
| N_unflagged | proposal_1727 | 2 |
| N_unflagged | proposal_2561 | 1 |
| N_unflagged | proposal_2662 | 1 |
| N_unflagged | proposal_5398 | 1 |
| N_unflagged | proposal_5594 | 4 |
| N_unflagged | proposal_5890 | 1 |
| N_unflagged | proposal_6434 | 1 |
| N_unflagged | proposal_6480 | 1 |
| N_unflagged | proposal_6675 | 1 |
| N_unflagged | proposal_6882 | 4 |
| N_unflagged | proposal_7763 | 1 |
| N_unflagged | proposal_9263 | 1 |

| stratum | n |
|---|---|
| T_verified | 21 |
| T_U | 78 |
| K_cowls | 31 |
| L_known | 30 |
| D_refuted | 40 |
| U_tail | 30 |
| N_unflagged | 20 |
| **total** | **250** |


## Layout (sw_obs / lw_obs present AND run finite fraction >= 0.55)

| value | n |
|---|---|
| color | 223 |
| gray_lw_only | 18 |
| gray_sw_only | 9 |

## Prior exposure (2 = ranks 1-15 annotated docx; 1 = ranks 16-100 contact sheet)

| value | n |
|---|---|
| 0 | 151 |
| 1 | 85 |
| 2 | 14 |

## Pipeline pass-count grade (`pipe_grade_passcount`; '' = unflagged)

| value | n |
|---|---|
|  | 46 |
| A | 5 |
| B | 5 |
| C | 11 |
| D | 50 |
| U | 133 |

## Literature-known (COWLS, L_known, discovery_status known/field_match, or any row with a literature known_lens_name)

| value | n |
|---|---|
| False | 179 |
| True | 71 |

## DESI pool / bench overlap (2", flag only)

14 of 250 frame rows lie within 2.0" of `outputs/parity_train_pool.csv` ∪ `parity_bench_arm1/2.csv` (72745 positions).

| stratum | split | desi label_source | n |
|---|---|---|---|
| D_refuted | train | random_neg | 1 |
| D_refuted | valsel | random_neg | 1 |
| L_known | train | random_neg | 1 |
| L_known | valsel | random_neg | 3 |
| T_U | train | random_neg | 1 |
| T_U | valsel | random_neg | 1 |
| T_verified | train | random_neg | 2 |
| T_verified | valsel | graded | 1 |
| U_tail | train | random_neg | 3 |

N_unflagged additionally EXCLUDED 256 overlapping unflagged targets before drawing (fixed strata are flagged, not excluded).

## Duplicates handled

- top-100 alias collapse (< 2.0"): J18030108+2309932 → kept `J18030075+2309921` (rank 7); recorded in `alias_ids`.
- 2 multi-member systems at 10.0" (union-find), 4 rows — share a split, excluded from repeats:
  - system 55: `J3807110-4434755` (T_U, rank 35), `J3806901-4434926` (U_tail)
  - system 120: `J5186803-1343778` (T_verified, rank 17), `J5186648-1343587` (T_verified, rank 16)
- L_known: 5 SIMBAD match(es) fell on a COWLS cutout and 2 on top-100 cutouts — both excluded from L_known (the cutout is already in K_cowls / T_*; its literature name is still recorded in `known_lens_name`).

## Fills / shortfalls

- none: every substratum filled from its own pool.
- L_known allocation: fixed {'pipe_D': 7}, stratified fill {'pipe_U': 13, 'unflagged': 10} (largest-remainder, proportional to the U/unflagged pool sizes).
- N_unflagged allocation by proposal (largest-remainder on the flagged proposal mix, capped by eligible unflagged rows): {'1063': 1, '1727': 2, '2561': 1, '2662': 1, '5398': 1, '5594': 4, '5890': 1, '6434': 1, '6480': 1, '6675': 1, '6882': 4, '7763': 1, '9263': 1}

## Source populations observed

- `results.csv`: 5391 rows; flagged 2024; inspected ok 5308; pipeline grades over flagged {'U': 1674, 'D': 328, 'C': 12, 'A': 5, 'B': 5}
- `JWST_top100_master.csv`: 100 rows; verifier grades {'U': 78, 'C': 12, 'A': 5, 'B': 5}; aliases collapsed 1 → 99 top-100 items
- `control_recovery.csv`: 31 COWLS controls, 15 flagged
- `known_lens_recovery.csv`: 239 rows; non-COWLS < 2.0" unique cutouts 55; in top-100 2; on COWLS cutout 5; remaining pool by pipeline bucket {'pipe_U': 24, 'unflagged': 17, 'pipe_D': 7}
- pipeline D: 328 total, 318 after removing K/L ids; by center_galaxy_type {'elliptical': 146, 'ambiguous': 40, 'ring': 35, 's0': 33, 'merger': 26, 'irregular': 16, 'compact': 10, 'spiral': 9, 'star': 2, 'artifact': 1}
- pipeline U: 1674 total; U_tail bands (after K/L removal) {'rank_101_300': 197, 'rank_301_2024': 1374}
- unflagged inspected: 3284; eligible after K/L + DESI-overlap removal 3002; flagged rows span 137 proposals
- DESI catalogue: 72745 positions (pool 69228 + bench 3517)

## Conventions

- `pipe_grade_passcount` is the run's persona PASS-COUNT (A=3/3 … D=0/3, U=never verified); it is not a Huang visual-inspection grade.
- `pipe_inspector_conf` / `pipe_score` are NaN for unflagged rows (the run stores 0 there).
- `rank_top100` is the run's rank only for top-100 rows; U_tail rank bands live in `substratum`.
- `unit_id` order is a seeded shuffle of the assembled frame; `system_id` is numbered by first appearance in that order.
- rng consumption order (seed 2026): L_known fill → D_refuted → U_tail → N_unflagged → unit_id shuffle.
