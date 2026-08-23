# Golden single-grader dataset — running findings

Tracked log of the golden-set program (`lensjudge/golden/`; plan
`~/.claude/plans/i-want-to-explore-golden-seal.md`, approved 2026-08-22). One named expert
(Xiaosheng Huang) grades a blind, shuffled, 250-item JWST NIRCam frame with 40 hidden
byte-identical repeats — score 1–4 (4 = A … 1 = D) + L/M/H confidence — giving the program its
first intra-rater human ceiling, per-rater few-shot exemplars, SFT labels and the
pre-registered align/validate experiment. Package README: `golden/README.md`.
Pre-registration ledger: `golden/REGISTRY.md`.

## 2026-08-23 — Phase 1 built and integrated; frame, stamps and kit are real; no grades yet

**What exists.** Six work packages built in parallel against one shared contract
(`GOLDEN_CONTRACT.md`, scratchpad) and integrated in one pass:

| WP | delivers | tests (no network, no API) |
|---|---|---|
| A | `common/jwst_fetch.py` (vendored `J/scripts/util.py @ 4f81493`, verbatim-guarded) + `golden/build_stamps.py` | `test_golden_jwst_fetch.py` 13/13 |
| B | `golden/build_frame.py` → `frame.csv` (250 rows) + `frame_summary.md` | `test_golden_frame.py` 10/10 |
| C | `schema.py`, `build_kit.py`, `tool/grade_template.html`, `tool/serve.py`, `collect.py` | `test_golden_kit.py` 9/9 |
| D | `stats.py`, `drift_report.py`, `split_halves.py`, `registry.py` | `test_golden_stats.py` 8/8 |
| E | `prompts/jwst_note.md`, `fewshot.py`, `grader_jwst.py`, `build_eval_manifest.py`, `run_golden_eval.py`, `audit_traces.py`, `REGISTRY.md`; surgical edits to `grader_direct.py`, `run_batch.py`, `llm_client.py`, `.gitignore` | `test_golden_model.py` 13/13 |
| F | `build_corpus_golden.py`, `build_desi_agreement_arm.py` → `golden/desi_agreement/` | `test_golden_sft.py` 10/10 |

Whole suite: 13 files, every pre-existing test still green (test_grader_direct 3/3,
test_llm_client 14/14, test_lensbench_gate 6/6, test_openai_tools 8/8, test_tier2_openai
14/14, test_v3 9/9, test_residual_honesty ALL PASS); under `pytest` with the JWST repo hidden
(`LENSJUDGE_JWST_REPO=/nonexistent`, the CI condition) **121 passed**. No golden module
imports `anthropic`/`claude_agent_sdk` at module level (the no-Claude CI job stays valid).

**Real artifacts produced (all deterministic, all `.sha`-pinned):**

- `golden/frame.csv` sha `43d012c5a34fdc59`, 250 rows: T_verified 21 (A5/B5/C11) · T_U 78 ·
  K_cowls 31 · L_known 30 · D_refuted 40 (10/10/10/10) · U_tail 30 · N_unflagged 20 (13
  proposals). Layout 223 colour / 18 gray-LW-only / 9 gray-SW-only. prior_exposure 2:14, 1:85,
  0:151. lit_known 70. desi_pool_overlap 14 (13 `random_neg`, 1 graded, 0 bench). Two
  multi-unit systems (ranks 16/17; rank-35 + a U_tail neighbour at <10″).
- `golden/stamps/` (gitignored, 961 MB): 250/250 candidates, 950 FITS (SW/LW × 10″/20″) +
  251 composites, **0 fetch failures**, 151 new systems in 15.8 min (3 workers, 120 mosaic
  groups). `stamps_manifest.csv` sha `06bb31f2105f1585` (1,201 rows). Finite fraction: median
  1.000, no channel below the 0.55 gate except one LW cutout at 0.000 (below).
  `--check-against J/top100_clean`: **99/99 byte-identical** JPEGs (`stamps_check.csv` sha
  `d073bc51d4584bcc`); the vendored renderer is the run's renderer.
- `golden/kits/jwst_lite_v1/` (gitignored, 22 MB): 250 items, 752×540, one permutation
  (seed 20260822), `manifest_sha ddbf73d4ba99f43e`, kit version 1. Key
  `golden/keys/jwst_lite_v1_key.csv` sha `692d1e1b2bad9586` (tracked, never shipped). Blindness:
  grep of `grade.html` for every candidate_id, alias, unit_id, RA/Dec (3–6 dp), rank,
  known-lens name, COWLS code, render sha and forbidden word → **zero hits**; every served
  JPEG's sha16 equals the key's `render_sha`.
- `golden/desi_agreement/`: `agreement_manifest.csv` sha `47c43d1496011835` (3,055 rows = 726
  Paper II `delSc==0 & pair_ok` [461 C / 165 B / 100 A] + 2,329 consensus grade-D;
  bench_overlap 316; pool_split train 1704 / valsel 310 / gate 62 / none 979);
  `fewshot_manifest_desi.csv` sha `22e958d0854609c0` (15 exemplars 3A/3B/3C/6D);
  `finetune/corpus_golden/sft_desi_agreement.jsonl` 1,703 rows — **all grade-D** (no Paper II
  row has corpus_desi views yet; `--render` is the network path).

**Integration checks that passed.** Collector dry run on 5 synthetic events (+1 re-grade):
`collect.py` → 5 grade rows / 5 labels, last-revision-wins (item 003 → score 2, revision_count
2), flag carried; outputs and the fake events deleted again (`records/` holds only its README).
Downstream seams on those 5 synthetic labels, all into the scratchpad: `split_halves` (3/2,
firewall 0 bad) → `registry sync` (5 units, leak `no`) → `registry assert` OK →
`build_eval_manifest` (5 rows, 11 contract columns + extras) → `fewshot.build_exemplar_blocks`
(2 exemplars, 8 blocks, embargo check OK) → `build_corpus_golden build` (3 train / 2 valsel,
targets unique) → `drift_report` (pilot verdict printed) → `audit_traces --build-lexicon` (24
entries). API smoke (`run_golden_eval.py --smoke 3 --smoke-stratum K_cowls --model sonnet`,
$0.06): parse_ok 3/3, every trace carries a `golden_content_audit` event (1 text block = panel
gloss, 1 image, 0 exemplars, system sha `88a592be1c46086a`), `audit_traces.py` passes with an
empty lexicon and with the synthetic validate lexicon (exit 0). `run_batch --mode jwst` not
exercised beyond import (it needs the eval manifest, which needs labels).

**Pilot observation, not a result (n = 3, zero-shot, no labels involved):** Sonnet graded all
three COWLS literature lenses (u0004 θE 0.79″, u0013, u0015) **D** with p_lens 0.04–0.05 and
confidence 0.72–0.82, citing "no blue arc at 1–5″" and reading the deflector-subtracted panel
as pure over-subtraction artefact. Looking at item 243 (u0004) the arc is not visible at the
v1 stretch and the subtraction panel is the cross-shaped residual Xiaosheng complained about —
consistent with his "subtraction is poor / optimise the stretch" critique and a reason
Campaign 2's v2 render exists. Whether XH sees these lenses at this render is exactly what the
K_cowls stratum measures.

### Deviations from the plan (and why) — all packages, as integrated

Frame (WP-B)
- `lit_known` is True for every `L_known` row too (a SIMBAD-matched lens is literature-known
  by construction), not only COWLS / `discovery_status ∈ {known, field_match}`.
- `known_lens_name/known_type/known_sep_arcsec` are also filled for K_cowls rows and for any
  row with a <2″ non-COWLS SIMBAD match (ranks 1, 48). Additive.
- `pipe_score` is NaN for unflagged rows (the run stores 0.0), mirroring the NaN rule for
  `pipe_inspector_conf`.
- Top-100 alias collapse is generic (any pair <2″ → keep the better rank); on the real data it
  is exactly 14→7.
- `substratum` carries extra provenance (`pipe_D/pipe_U/unflagged`, `rank_101_300`,
  `proposal_NNNN`). `proposal` is a string in the CSV; pandas reads it back as int.
- L_known pool is 48, not the plan's 53 (55 unique − 2 top-100 − 5 on COWLS cutouts). Lite
  quota 30 unaffected.
- **Integration change:** `layout` is derived from observation presence **and** the run's own
  finite fraction (`J/data/manifest.csv` finite_sw/finite_lw ≥ 0.55, the renderer's gate;
  `build_frame.derive_layout`), not presence alone as the contract wording said. Reason: u0163
  (`J21060768-456160`, D_refuted/other) has an LW observation whose cutout lies entirely in an
  F444W `maskbar` coronagraph mask (finite 0.0); the renderer drops it and the composite is
  gray SW-only, which the kit key, the repeat picker (gray excluded), the few-shot eligibility
  (colour only) and the stats subgroup all read from `frame.layout`. Frame re-pinned
  `341591ab5c3800fd → 43d012c5a34fdc59` (one cell). `build_stamps.py` now cross-checks
  `frame.layout` against the rendered composite after every run (`LAYOUT OK … 250`).

Stamps (WP-A)
- Installed `fsspec 2026.7.0` + `aiohttp 3.14.3` into the venv (astropy `use_fsspec` over https).
- FITS headers carry a minimal TAN WCS beyond the 13 contract keys (additive; array stored
  exactly as `cutout()` returns it, row 0 = North, `CDELT2<0`).
- Manifest conventions the contract left open: COMPOSITE rows have `filter`=`F150W+F277W`
  (colour) or the single band (gray), `out_px`=240, full 64-hex `sha256`, `path` relative to
  `reproductions/lensjudge/`; channels with no observation get no row.
- `--frame` mode takes only `candidate_id` (+ra/dec cross-check ≤0.05″) from the frame; footer
  fields come from `J/data/targets.parquet` (byte-faithfulness needs the run's own values).
- `stamps_check.csv` (9 columns) is pinned; byte-identity depends on macOS Arial + PIL 12.2.0.

Kit / collector (WP-C)
- Event file name is `events_<kit_id>_<session_id>.jsonl` (contract) and `serve.py` is
  stdlib-only (ships to the PI's machine). Exported events contain every event in the
  browser store (all sessions); collect de-duplicates exact copies, aborts on conflicts.
- Every manifest version is appended to `keys/<kit>_manifests.jsonl` (tracked): `--add-repeats`
  changes `manifest_sha`, and session-1 events carry the old one.
- `grading_sheet.csv` is sorted by item_id (presentation order would expose inserted repeats).
- Legend is the curated constant `build_kit.LEGEND` (rubric_imaging.md:51-58 re-worded as
  4/3/2/1), not parsed at build time. README.txt tells him "some images may appear more than
  once" without marking which (one line to delete if he should not be told). Score tiles show
  "= A/B/C/D"; Backspace re-grade shows the previous answer.
- `revision_count` = number of commits recorded for the (unit, pass). collect is strict on
  `presentation_index` (raises). Blindness check bans words + all id/coord/sha values
  (integer ranks cannot be grepped without false positives).

Statistics / splits / registry (WP-D)
- `intergrader_stats._stats` is fed `s_lo=min, s_hi=max` (it assumes ordered pairs), so its
  rows are "comparable in kind" to Paper II; pass-ORDER statistics (unsymmetrised QWK,
  drift) are separate rows. `qwk_grader_vs_consensus` dropped (degenerate);
  `exact_agree_within_D` added. Strict 4-column schema kept (`n` as `n_pairs` rows, subgroups
  as prefixed statistic names, the pass-count cross-tab as `xtab_verified[...]` rows with the
  literal "not a QWK headline" suffix).
- Split is two-phase (forced systems → align; strict alternation per stratum×letter cell;
  then move single-unit non-forced ≥3 systems until |Δ(score≥3)| ≤ 1); `forced` is
  system-level. Between-halves 2″ firewall asserted `n_bad==0`; pool/bench overlaps reported,
  never excluded. Drift uses pass-1 rows only for score-vs-position; pilot rule = raw mean,
  300-s-clipped mean reported alongside. All CIs percentile bootstrap (not Jeffreys).
- `registry.sync_from(labels, splits, frame=None, grades=None, path=REGISTRY)` — `leak` needs
  `frame.desi_pool_overlap`; `mark` CLI subcommand added.
- **Integration change:** `stats.py` exits with a clear message when no repeat pairs exist
  yet (the state between session 1 and the repeats) instead of a numpy traceback.
- **Not done:** the `parity/human_baseline_summary.py` hook (block in the WP-D report; file
  outside the integration edit list) and the plan's "dissenter" table from
  `J/results/verifications.csv`.

Model path (WP-E)
- Exemplar header labels A/B → LENS, **C → POSSIBLE LENS**, D → NON-LENS (contract said LENS
  iff score ≥ 3, which would label C NON-LENS); one line in `fewshot.HEADER_LABEL`.
- `grader_direct._example_blocks` header reads `({grade_source} grade {g})` with a NaN-safe
  default "consensus" (behaviour unchanged for existing DESI manifests).
- Banned-string rule is 4-word windows (≥12 chars), not a raw 12-char window (the raw rule
  fires on rubric_imaging_v2 and jwst_note themselves). Audit adds a 5th check (long text
  block must sha-match a known template) and `--check-text` for rubric files.
- `jwst_content` text is item-agnostic (no name, no RA/Dec). `run_batch --rubric` is also
  allowed for `--mode jwst` (E3). E2 on `--split align` drops the exemplar units. "D spread
  over FP families" not implemented (labels carry no substratum). `gp_coverage` exists in
  llm_client but is not written to parquet. `build_eval_manifest` needs `--keys-dir/--kits-dir`.
- **Integration changes:** `run_golden_eval.py --smoke-stratum` (pick smoke rows from one
  stratum, so the plumbing burn is on K_cowls controls); `audit_traces.py` tolerates a missing
  `splits.csv` for pre-split audits (empty align/validate sets — any exemplar is then a
  violation, right for zero-shot) while `--build-lexicon` still requires it; `image_path` in
  the eval manifest is absolute.

SFT / DESI arm (WP-F)
- `corpus_manifest.csv` has two extra columns after the 14 contract ones (`p_lens_target`,
  `image_path`); `split ∈ {train, val, valsel}`. `USER_MSG_JWST` = `"<image>\n" + PANEL_GLOSS`
  (student and Claude read the same user text). Registry exposure is marked BEFORE writing;
  no `label` subcommand (labelling is `run_batch --mode jwst`).
- Agreement arm: SFT mix-in lives in gitignored `finetune/corpus_golden/`; SFT also excludes
  `pool_split ∈ {gate, valsel}`; grade-D rows get confidence H and `contaminant="contaminant"`
  (corpus_desi convention), `grade_source` names Storfer 2024 / Inchausti 2025; `--render` is
  opt-in. `survey_key="ls-dr9"` for Paper II rows (DECaLS DR8 era; `huang2021` is in no
  `SURVEY_LAYER`, so the fetch path would otherwise fall to ls-dr10).
- **Integration change:** SFT `images` paths are absolute (ms-swift resolves nothing).

Environment
- Installed into `~/.venvs/lensjudge`: `fsspec`, `aiohttp` (WP-A), `pytest 9.1.1` (integration,
  to run the suite the way CI does). No change to `requirements.txt` yet.

### Open items (before session 1)

1. PI pre-flight: legend / confidence wording (`build_kit.LEGEND`, `CONFIDENCE`), the scale,
   no docx / contact sheet. Changing wording after this point means `--force` rebuild — only
   before any grading.
2. Decide whether README.txt should say that repeats exist (currently yes, unmarked).
3. Send `kits/jwst_lite_v1/` (22 MB). Never `keys/`.
4. Perlmutter backup (Phase 0) has not been run; record it in `golden/README.md`.
5. After `registry sync`: `registry mark --units u0004 u0013 u0015 --run-tag smoke --kind eval`.
6. Confirm with the user that XH was one of the two Paper II graders (agreement-arm assumption).
7. `fetch.get_cube` cache gotcha for the agreement arm: 73/726 Paper II names are cached as
   DR10 cubes under `cache/cubes/<name>.fits`; a `--render` with `ls-dr9` must bypass the cache.
8. Add the `human_baseline_summary.py` hook (WP-D report) when the first ceiling exists.
9. `REGISTRY.md` rows are pasted by hand from `--print-tuple` before the first validate call.
10. Commit: `golden/` (frame, summaries, stamps/check manifests, keys, records/README,
    desi_agreement, REGISTRY.md, README.md), `common/jwst_fetch.py`, `prompts/jwst_note.md`,
    the six test files, the four surgical edits, this log.

## 2026-08-23 — adversarial review of the integrated build: findings → fixes

Three reviewers (blinding/embargo, contracts/correctness, statistics/design) reported on the
Phase-1 build. Every CONFIRMED finding and every PLAUSIBLE one that verified is fixed below;
the two marked "accepted/noted" are recorded, not fixed. Kit and frame were rebuilt (no
grading had started): `frame.csv` `43d012c5a34fdc59 → 422eacbcdcf3854d` (one cell: u0239
`lit_known`), key `692d1e1b2bad9586 → 0dbd8b662596d757`, manifest `1ab3c2daa733a063`.

### Blinding / embargo
| # | finding | fix |
|---|---|---|
| B1 | 99/250 kit JPEGs byte-identical to `J/top100_clean_scrambled/*.jpg` (whose key he holds) — verified: 99 sha + 99 size matches | `build_kit.JPEG_KW` quality 93→**92**; `assert_no_collision(items, dirs)` in `write_kit_files` (default dirs `top100_clean_scrambled`, `top100_clean`; sha match raises, size coincidence warns); README.txt asks him to delete the old folder + key. Rebuilt: 0 byte / 0 size matches |
| B2 | `001..N` + continuation ids marked every repeat | item ids = seeded draw **without replacement from 001..999** in presentation order (`draw_item_ids`); repeats draw from the unused remainder; retired ids (`--drop-units`) never reused; `presentation_index` stays in the key. Tool resumes by `item_id`, unaffected |
| B3/S1 | `valsel_golden` was the validate half → SFT selection on the test set | `build_corpus_golden`: valsel = 20 % per-letter hash carve of the **align** half; validate rows written nowhere in `corpus_golden/`; `gate` refuses labels naming validate units (`assert_not_validate`, also via golden_labels × splits) |
| B4 | `assert_unexposed` only for arm e2 | runs before **every** `--split validate` run (any arm/model); test covers an SFT-student `e1` run dying on a poisoned ledger |
| B5 | golden manifest through a non-jwst `run_batch` mode leaks ids/coords + DESI pixels | `run_batch.run` raises `SystemExit` when `survey_key`/`catalog == jwst` and `mode != "jwst"` |
| B6/C10 | 16 PI comments verbatim in tracked `audit_traces.py` | strings moved to gitignored `golden/pi_comments.txt` (16 lines, sha16 `15dc9fa585f446a0`); code keeps count + sha only (`load_pi_comments` refuses a mismatching file); `--build-lexicon` requires it unless `--allow-missing-pi-comments`; tests plant synthetic strings and grep the module for the real ones |
| B7 | repeat files carried a later mtime | `freeze_mtimes`: every `items/*.jpg` set to `ITEM_MTIME` (2026-01-01Z) on every write |
| B8 | `--smoke` could pick validate units; `--out` elsewhere dodged the gate | `smoke_frame(splits_csv=…)` excludes validate once splits exist; `scored_outputs` also globs `config.OUT` |
| B9 | E3 rubric never lexicon-checked at run time | `check_system_prompt(sysp, banned, split)`: lexicon hit refuses any run; a validate run with no lexicon is refused |
| B10 | filter names in panel titles fingerprint the programme | **accepted for v1** (pixels are the run's render); drop filter names in the Campaign-2 v2 render |
| B11 | key/labels tracked in the shared remote | **noted** in README rule 4: confirm the grader has no repo read access before pushing, or keep `keys/`, labels, splits, registry out until validate is scored |

### Contracts / correctness
| # | finding | fix |
|---|---|---|
| C1 | `--force` kept the manifest history → old events silently joined to a new key | `--force` deletes key, `.sha`, `_manifests.jsonl`, `_dropped.csv`; every history line now carries the `key_sha` it was written with; `collect.validate_against_kit` refuses when the latest manifest's `key_sha` ≠ the key's pin |
| C2/S7 | pilot fallback had no key-preserving implementation | `build_kit.py --drop-units [--keep U_tail=15,N_unflagged=10]`: retires UNGRADED items of those strata as a new manifest version; graded items kept; survivors = graded ∪ lowest `hash01(unit_id,"pilot_core")`; retired rows → `keys/<kit>_dropped.csv`; `drift_report` verdict names the command |
| C3 | `lit_known=False` for u0239 (SIMBAD LeG 1.31″, U_tail) | `build_frame._attach`: `lit_known |= known_lens_name != ""`; frame re-pinned (lit_known 70→71); repeat-ineligible now |
| C4 | interrupted validate run needed a fake rescore | `.meta.json` is the completion marker: a target parquet without meta resumes; with meta of this tuple → refused; with meta of another tuple → "path collision" (the old default path `preds_golden_{arm}_{split}` collided for sonnet/opus in one arm — default is now `preds_golden_{arm}_{model}_{split}_r{k}`, run tag `golden_{arm}_{model}_{split}_r{k}`, model made path-safe) |
| C5 | `rendered_layouts` read SW/LW from any scale | filters `arcsec == CUT_ARCSEC`; test adds the 10″-fails/20″-passes case |
| C6 | `lensbench_gate` always FAIL on the golden manifest | docs fixed (manifest docstring, README): its verdict does not apply (`source=="random"` needed); `analyze_golden.py` is the gate |
| C7 | two `s_exp` definitions | `gate_aucs` recomputes a missing `s_exp` with `llm_client.logprob_ordinal` (ORDINAL_W); `W_EXP` removed |
| C8 | `run_batch --mode jwst --rubric X` dropped the note | `grader_jwst.grade_candidate` appends the note to any supplied system prompt not already ending with it |
| C9 | grades pinned before labels could raise | labels built in memory first, then both pinned |

### Statistics / design
| # | finding | fix |
|---|---|---|
| S2 | binary endpoint labelled two ways; primary had no implementation | `binary_label` = score ≥ 3 (`binary_label_ge2` extra); endpoint definitions in `REGISTRY.md`; new `golden/analyze_golden.py`: E2−E1 paired ΔAUC (`phase_d_analysis.paired_boot` + `delong_p`) and Δpurity@recall 0.8, pooled + per replicate, absolute AUCs by stratum, QWK vs XH, self-consistency, E3−E1 — **contract deviation** (contract said ≥ 2) |
| S3 | incumbent AUC dropped the pipeline's own misses (204/250) | manifest `p_pipeline = 0` for never-flagged rows (+`pipe_flagged`); `stats --agreement` and `analyze_golden` report `[flagged_only]` and `[all]` |
| S4 | drift conditioned on the score-stratified draw (regression to the mean) | `pivot_pairs` adds `w_ipw` (pass-1 marginal share ÷ repeat share); `ipw_*` rows (drift, exact, within-1, conf, binary) + per-`s1` rows; docstring says which to read; simulation test (exchangeable passes, skewed marginal) shows the signed raw artefact and ~0 after IPW |
| S5 | per-level shortage warning could never fire | `if quota[s] < min_per_level: warn` (names the staged-draw option) |
| S6 | `layout` / `lit_known` subgroups vacuous | `SUBGROUPS = ("prior_exposure", "stratum")` |
| S8 | tuple omitted settings that change the answer | `RunTuple` += `system_sha16`, `thinking`, `effort`; `REGISTRY.md` tables widened; `is_registered` matches all; parquet rows carry them |
| S9a | README said ranks 16/17 are T_U | README: both T_verified C, literature-known |
| S9b | `pivot_pairs` breaks with a second kit | `--kit-id` / `pivot_pairs(kit_id=)`; >1 kit without it raises |
| S9c | events under a superseded manifest accepted silently | `collect` warns with the count of events recorded after the newer version was built |
| S9d | `p_one_grader_reject` misnamed for a self-pair | renamed `p_either_pass_score1` |
| S9e | C's confidence unrecoverable from `p_lens` (jitter > shrink) | **noted**; formula is as specified |

### Tests after the fixes
`test_golden_frame` 10/10 · `test_golden_jwst_fetch` 13/13 · `test_golden_kit` 10/10 (new: drop-units, collision, force, mtime, key-pin refusal) · `test_golden_model` 14/14 (new: PI-comments file contract; gate on meta/collision/thinking; e1-validate unexposed proof; run_batch refusal; note append) · `test_golden_sft` 10/10 · `test_golden_stats` 10/10 (new: IPW simulation, analyze_golden endpoints) — plus the pre-existing suite (see the integrator's run). No network, no API.

### Still open after this pass
- Commit the rebuilt `frame.csv`/`.sha`, `frame_summary.md`, `keys/jwst_lite_v1_*`, `REGISTRY.md`, `analyze_golden.py`, `golden/README.md`; `golden/pi_comments.txt` stays untracked (restore by hand on another machine: 16 lines, sha16 `15dc9fa585f446a0`).
- The plan's naming `preds_golden_{arm}_{split}_r{k}` → `…_{arm}_{model}_{split}_r{k}` (two models per arm). The plan's "binary_label lens if score≥2" → score ≥ 3.
- Re-send the kit (`kits/jwst_lite_v1/`, rebuilt) if a copy was already sent; the old one must be discarded (its key is gone).

## 2026-08-23 — Part 2 built and integrated: evidence-first scheme, truth set on disk, runner gated, Nate drop-in generated; first API smoke

Six work packages (prompts · schemas+aggregation · panel fan-out · truth set · replay/audit/
render/analysis · runner+registry+patch) integrated in one pass. Suite **210 passed** (also with
`LENSJUDGE_JWST_REPO=/nonexistent`); every golden module imports with `anthropic` and
`claude_agent_sdk` blocked. Nothing committed; `J` untouched (`git status` empty).

### What exists now
- **Scheme** `prompts/personas/jwst_v1/{advocate,critic_common,critic_artifact,critic_geometry,critic_morphology,arbitrator}.md` + `panel_gloss.json` (layout-conditional per-role views) + `prompts/jwst_note_v2.md` (role-neutral; no "bluer", no "default C or D"); incumbent briefs `prompts/personas/incumbent/*.md` byte-equal to `verify_workflow.js`. Persona set `e19c402539d141d9`; note `a4ffd0642c09b684`; full system shas advocate `7e035592b694b949` · artifact `73e98dc43514bd4b` · geometry `c1129365bf13d292` · morphology `a226ba63c06e6f59` · arbitrator `e5a741757695f539`. **0 lexicon hits** in 27 model-facing strings/compositions against the 298-entry pi-only lexicon (16 PI comments + 282 holdout ids; `golden/banned_lexicon.txt`, gitignored).
- **Code** `golden/{schemas_panel,aggregate_v2,views,panel,run_truth_eval,sample_truth_negatives,build_truth_manifest,split_truth,render_v2,incumbent_replay,reason_audit,analyze_truth,make_verifier_patch}.py`, seams in `grader_jwst` (`note=`, `schema=`, `extra_views=`, `audit_full_text=`), `imaging/grader_direct` (`schema=`), `audit_traces` (`--pi-only`, `--extra-ids`, `n_extra_views` rule), `registry.seed_from_frame`, `split_halves.firewall(halves=)`, `run_golden_eval` helpers take `section=/cols=/prefix=`. `golden/verifier_patch/` generated (11 files, `git apply --check` tested on a temp copy of J; `aggregate_v2.py` byte-identical, sha `0570cc12137e8fdc`).
- **Zero-API baseline** `incumbent_replay`: reproduces `results.csv` for **350/350**; recovers the **12 `*_ctl*` COWLS ids** graded U; per-persona pass 17/8/12 (artifact/geometry/morphology), κ 0.628/0.486/0.461; COWLS at A/B/C = 0/15 under every counterfactual rule. `outputs/incumbent_replay.csv` sha `d807a5ed86945d08` (deterministic).
- **Truth set** (all pinned): `truth_negatives.csv` `2899110110d6a586` (catalogue union 195,818 positions / 12 files; pool 3,284; pre-purge 2″ 1.86 % · 3″ 2.50 % · 5″ 4.51 %; 400 rows 200/200) · `truth_manifest.csv` **`2560adfa93e00356`** (570 rows: cowls 31 [5/8/13/5], lit_galaxy 26, lit_cluster 30 → **positives 87**; negative 400; stress_D 40, stress_U 29, anomalymatch 10 [A5/B1/C4]; anchors 5; centre_is_deflector 65) · `truth_splits.csv` `032a302c84f3cbe7` (design 288 / holdout 282; positives 45/42; 27 forced; 0 firewall collisions) · `truth_fetch_list.csv` now empty (`51078d55cf265d5d`). Stamps: **428/428 fetched, 0 failures, 42 min**, layout check OK; `stamps_manifest.csv` `410f1ab419648b64`. Verified: every negative > 5″ from the union by an independent re-crossmatch (min 5.11″) and > 10″ from every non-negative row; 570 v1 + 570 v2r served JPEGs all 752×540, shas match bytes, **0 byte collisions** with the kit (`kits_truth/` and `kits_truth_v2r/` both scanned on every kit build).
- **Registry**: `golden_registry.csv` seeded with 250 frame units (blank split); `mark_exposed(kind="eval")` verified (u0123 carries `integration_check_20260823` + the three smoke run tags); unknown units refused.
- **REGISTRY.md**: Truth-eval endpoints (plan text), empty "registered arms" table, five design anchors with ids/units/predictions verified against `JWST_top100_master.csv` and `frame.csv` (all forced to design).

### API smoke (design half, 3 items: COWLS `AAAAAB` · cluster-field negative · stress_D merger; ≈ $0.34)
`run_truth_eval.py --arm {a1,a0(--claim-mode inspector),a2} --split design --limit 3 --ids-file … --out outputs/smoke/…`: parse **3/3 per arm, every role** (15 calls; one paid repair retry); per-role traces carry `golden_content_audit` with `n_extra_views` (geometry/morphology 3 images, others 1); `audit_traces` **0 violations** on all three trace dirs; `reason_audit` and `analyze_truth` run on the parquets. Returned: a0 0/3 → D on all three; a2 S = 0.08 / 0.06 / 0.62 (C/C/B); a1 S = 0.07 / 0.07 / 0.258 (S_arb 0.403), all C — the full stack engaged only on the stress_D (critics merger/companion/residual, a = 0.25/0.5/0.75, arbitrator letter C, `needs_human`). Design observations, not results: the advocate echoed the incumbent on the AAAAAB lens ("four-lobed butterfly … no offset arc", p_ev 0.07–0.08); the full-stack item cost **$0.163** (> the $0.10 cap; critic calls $0.027–0.05 at ~5k in / ~1k out tokens).

### Integration fixes (beyond the six packages)
| # | finding | fix |
|---|---|---|
| I1 | `common/parse._try_balanced` tracked `"` string state from the first character: prose before the JSON with an odd number of `"` (arcsecond marks) desynchronised the brace scan and returned the record's last NESTED object (`measured`) → parse failure → paid repair. Hit on the very first critic call. | string state only inside a brace span; regression test `test_golden_panel.test_parse_record_after_prose_with_arcsec_quotes` (CriticRecord and ImageGrade after prose) |
| I2 | the repair retry in `grader_direct` was a paid call with no trace event (repair rate uncountable) | `direct_repair` event (tokens, cost, parse_ok, text) on both backends |
| I3 | `reason_audit.votes_to_critics` did `json.loads` on the raw model text (prose + JSON) → every critic row dropped; `tag_frame` then crashed on the empty frame | reads with `parse.extract_json_block`; category columns exist on an empty frame; test extended |
| I4 | no path merged `render_v2.py`'s output into the pinned manifest; `render_v2` wrote cwd-relative paths | `build_truth_manifest.attach_v2r` (+ `--render-v2r`, default `kits_truth_v2r/render_v2r.csv`), manifest gains `render_sha_v2r` (contract deviation: one extra column after `render_sha`; the panel already read it), `render_v2.run` resolves `out_dir`; test added |
| I5 | `kits_truth_v2r/` not in the kit collision scan | `build_kit.DEFAULT_COLLISION_DIRS` += `kits_truth_v2r` |
| I6 | `golden/README.md` rule 1 contradicted the user's design-only decision | design-only exception recorded (mechanisms, never wording; 4-gram check; anchors as predictions) |
| I7 | no single run-order document for Part 2 | `golden/TRUTH_EVAL.md` |

### Deviations from the contract / plan (all deliberate, reported by the packages or here)
`refutation_strength` optional at the schema level (required iff an alternative is named); `scale_tension > 0.4` and `no_opinion`+alternative are rejections (validators), not clamps; incumbent claim rides in the USER message (system shas stay item-agnostic; wrapper placeholders read "given with the image below"); gray-layout crops follow `panel_gloss.json` (geometry b,d,e; morphology a,d,e); `audit_full_text=True` for every panel call (item JSON exceeds the 200-char head); arbitrator skipped when a critic failed to parse; `render_version` carries `+ctx20`; `--force-rescore` archives and re-grades; cowls_band = S-code 2·nA+nB+nS; negatives' `half` assigned by the sampler; stress_U = 29 (u0239 is a SIMBAD LeG positive); `image_path` relative to `lensjudge/`; Nate's letters stamped `sonnet_thresholds_uncalibrated` via `resolve_thresholds(fallback_source=)`.

### Open after this pass
- Design iterations (advocate wording first; the AAAAAB miss is the first target), then freeze `t_A`/`t_B` into `thresholds_v2.json › sonnet_api`, **regenerate `golden/verifier_patch/`** (it ships the thresholds file) and register every holdout tuple before the first holdout call.
- Budget: plan full stack at ≈ $0.15/item when critics run (or trim critic output); consider `--cost-cap 0.17` for the design passes so the warning means something.
- `run_truth_eval` resumes a design parquet by name without comparing stored `system_sha16_<role>` to the current tuple — after any prompt edit, start a fresh `--out` (the smoke deliberately lives under `outputs/smoke/`).
- A0 agreement gate (≥ 0.85 vs recorded verdicts on 40–83 frame units) not yet run; a0 smoke reproduced 0/3 on the COWLS control.
- `render_v2` default engine saturates inside ~1″ on bright deflectors (WP-5); `--engine isophote` is a design-only look before the gated arm R is chosen.
- Files to commit (no commit made): everything under `golden/` except the gitignored bulk (`stamps/`, `kits_truth*/`, `banned_lexicon.txt`, `pi_comments.txt`), `prompts/personas/`, `prompts/jwst_note_v2.md`, `common/parse.py`, `imaging/grader_direct.py`, the `tests/test_golden_*.py` files, `GOLDEN_FINDINGS.md`.

## 2026-08-23 — adversarial review of Part 2 (embargo · contracts · fidelity): findings → fixes

Three reviewers reported on the integrated Part-2 build before any design or holdout call.
Every CONFIRMED finding and every PLAUSIBLE one that verified is fixed below (one verified and
documented instead of re-built: F10). Suite **211 passed** (210 + one new test), also with
`LENSJUDGE_JWST_REPO=/nonexistent`; every `test_golden_*.py` runs as a plain script; the
verifier patch was regenerated (`aggregate_v2.py` byte-identical, sha `37be83b4598a36a7`;
`ADD_FILES.patch` passes `git apply --check` on a temp clone of J); 0 lexicon hits in 101
model-facing strings, 80 tracked files, the patch tree and the old smoke traces (`audit_traces`
7 events, 0 violations). Nothing committed; `J` untouched.

Re-pinned: `truth_manifest.csv` `2560adfa93e00356 → 70eea125c426a8b3` (one cell: the COWLS
`einstein_radius = 0` placeholder → NaN). New shas: persona set `5983c47ef6315078`, note
`754655a400f360e6`, gloss `e734968399847d42`, advocate `4d252f5bf2b82e5d` · artifact
`f5ed259652e65ee2` · geometry `a293ddddce11ee4a` · morphology `26bde57ad0478237` · arbitrator
`44542114399ab277`; render_desc v1 `28737c6083dc1978` / v2r `25dd6cc680579747`.

### Embargo / leakage
| # | finding | fix |
|---|---|---|
| E1 / C1 / F1 | **A3 shipped the v2r image with the v1 description**: `panel.py` took `render` only to pick the path; the advocate read "(f) … a CIRCULAR radial-profile model" + the note's butterfly caveat while looking at the signed-χ SW\|LW montage; `render_v2_desc.md` was hashed into the tuple but never sent | `panel_gloss.json` gains `renders["v2r"]` → `composite_{color,gray}_v2r` view sets ((f) = signed-chi elliptical-model residual, "NOT a circular subtraction", `render_desc: true`); `views.view_set/view_text/role_views(render=, render_desc=)` append `render_v2_desc.md` after the VIEW paragraph and REFUSE the v2r image without it; `panel.grade_panel(render_desc=)` (default: the file); the circular caveat (azimuthally-averaged profile, butterfly/bowtie, concentric rings, dipole) moved INTO the v1 composite VIEW texts and OUT of `advocate.md`, `critic_common.md` and the note (all now defer to the VIEW for what model was removed); `render_v2_desc.md` + the composed v2r texts join `model_facing_strings()`; `audit_traces.known_template_shas` registers the composed strings; `test_golden_panel` asserts the v2r advocate/artifact/arbitrator first text block carries the chi VIEW + the description and the crop roles do not |
| E2 | score-once gate filename-scoped: `--out outputs/peek.parquet` or `outputs/smoke/` hid a completed holdout replicate | `scored_outputs` walks `config.OUT` and the target dirs recursively over every `*.meta.json` and matches by TUPLE + split (name-agnostic); on the holdout `--out` must carry the default name and live under `outputs/` (`holdout_out_problem`); meta records `out` |
| E3 | resume mixed tuples in one parquet | `check_resume`: every stored row's `RESUME_COLS` (tuple columns + run_tag) must equal the current run's, else refused (test plants rows with another prompt sha) |
| E4 | lexicon gate vacuous on an empty/stale file ("clean against 0 entries") | `check_lexicon_coverage` on the holdout: every holdout id (+ frame aliases) and all 16 PI comments must be in `banned_lexicon.txt` (test: a pre-Part-2 lexicon is refused) |
| E5 | verbatim PI 4-grams in tracked modules (`render_v2.py`, `build_stamps.py` docstrings; a `test_golden_model` fixture) | paraphrased to mechanism wording; new `test_no_tracked_module_quotes_a_pi_comment` scans every tracked `.py`/`.md` under `golden/`, `prompts/`, `tests/` + `GOLDEN_FINDINGS.md` (80 files, 0 hits); README rule 1 extended |
| E6 | holdout subset peeks allowed (`--limit 20` = the 16 holdout COWLS + 4 lit positives, invisible to the ledger); `analyze_truth` pooled any parquet | `--limit`/`--ids-file` refused on the holdout; `analyze_truth.discover` reads every `.meta.json`, refuses (holdout) a parquet without one, a row count ≠ `meta.n`, `meta.n` ≠ the half's size, a subset, or mixed tuples in a group, and prints each group's tuple (`--allow-incomplete` to override; design warns) |
| E7 / C11 | `--ctx20 --arm attr` registered `jwst_v1+ctx20` while nothing was attached | `--ctx20` accepted for `a1` only |
| E8 | Nate drop-in blinding advisory: "You may crop/zoom with Bash + Python", `<id>_full.jpg` beside the crops, `tau0` in every critic job | JS: explicit BLINDING paragraph ("read NOTHING but the job file and its listed images … do not crop, re-render or fetch"), the crop invitation removed; views in per-kind directories `views/{full,ctx20,geometry,morphology}/<id>.jpg`; `tau0` out of the jobs (in `refuter_index.json`); README states what remains one Read away and that agreement is evidence about the prompts, not a proof of blinding |
| E9 | `calibration_ids.csv` shipped all 31 COWLS incl. 16 holdout-half ids | design-half COWLS only (15; `truth_splits.csv`), the withheld count written into the CSV header comment and the README; the patch tree now has 0 lexicon hits — **contract deviation** (contract said 31) |
| E10 | a0 inspector claim text not lexicon-checked before the call | `attach_claims(lexicon=)`: a hit blanks that item's claim (`claim_blanked`, counted in meta) |
| E11 | `splits_sha16` decorative (items selected by the manifest's `half`) | `check_halves`: manifest `half` must equal the pinned splits for every id, else refused |
| E12 / C12 / F8b | gray-layout morphology views differed (lensjudge (a),(d),(e); 08e (d),(e)) | 08e's panel sets are emitted from `panel_gloss.json` (`make_verifier_patch.panel_sets`); JS VIEW text updated |

### Contracts / correctness
| # | finding | fix |
|---|---|---|
| C2 | a0 `--claim-mode inspector`: unflagged items (220/282 holdout) were told a claim is "given with the image below" | `role_prompts` returns both wrapper variants; `grade_panel(persona_set_noclaim=)` picks by `claim is not None` and REFUSES a wrapper that promises/denies a claim the user message lacks/has; the tuple's `system_sha16s` carries `role:` + `role@noclaim:`; the first-item sha check accepts either variant per role |
| C3 | `covers()` padded the PA endpoints first, collapsing a wrap-form arc (100→80 = 340°) to 20° | pad after `_arc()`: `((start − dpa) % 360, min(360, length + 2·dpa))`; cases 100→80 / 90→80 / 100→60 / 350→340 in `test_golden_aggregate`, `self_test` and the shipped `tests/test_aggregate_v2.py`; `aggregate_v2 2.1.0` |
| C4 | Nate's `letter` applied the arbitrated guards to the unarbitrated S (≠ lensjudge `grade_pred`); `S_arb` NaN without an arbitrator | `09_rank_report_v2.py`: `letter = assign_letter(S, …)` without the arbitrator, new `letter_arb` on `S_arb` with it, `S_arb = score_S_arb(…)` unconditionally (== S when none ran); `regrade_diff.csv` carries `letter_arb` |
| C5 | `sonnet_api` never in Nate's fallback chain, so the freeze changed nothing | `fallback_keys=("sonnet_api", "provisional")`; labels: own key → `<key>_calibrated`, sonnet_api → `sonnet_thresholds_uncalibrated`, provisional → `provisional`; README table of the three states; test exercises both |
| C6 / F8c | `layout_of` ignored the finite gate (8 flagged ids rendered gray were "color"; 08e would crop the slot-(c) subtraction for morphology) | `_present()` gates on `finite_sw`/`finite_lw ≥ 0.55` from `inspections.csv` (the render's own rule) |
| C7 | `rank_score` scaled `pipe_inspector_conf` by 1 (it is 0–100: a U row at 45 outranked every examined row) | every inspector-confidence column is /100 and clamped to [0, 0.999) |
| C8 / F6 | Opus thresholds keyed `opus_api` by the runner and `opus_claude_code` by the analysis; no thresholds in the tuple; provisional letters allowed on the holdout | `aggregate_v2.MODEL_KEYS` (sonnet→sonnet_api, opus→opus_api) used by both; `TruthTuple.thresholds_sha16` (+ REGISTRY tables widened); `analyze_truth` letters P2 from the rows' own `t_A`/`t_B` (`thresholds_from_preds`, refuses disagreeing replicates); holdout refuses `letter_source=provisional` unless `--allow-provisional-thresholds` |
| C9 | record validation diverged (null strength, float criteria, missing `criteria`, duplicate `k`) | `09_rank_report_v2.py` mirrors the pydantic rules: integer criteria, unique `k`, `criteria` default [], "" → null, null strength → 0 only without an alternative, a location box for every named alternative, one ruling per critic |
| C10 | "every verdict id lands in the output" assertion vacuous | asserts `verdict_ids ⊆ {examined ∪ legacy-present}` |

### Fidelity
| # | finding | fix |
|---|---|---|
| F2 | the forbidden-ground monitor counted the scheme's sanctioned channels (a θ_E argument routed into `scale_tension`, a `subtraction_residual` on a feature absent from direct panels, colour beside a structural alternative) | `reason_audit`: `*_forbidden` flags derived from the STRUCTURED record (`votes_to_critics` joins the advocate's `visible_in_direct` per item → `covers_direct`); `forbidden_only` = any forbidden flag and no structural prose; `uses_scale_tension` / `uses_subtraction_residual` reported as separate monitors; incumbent rule unchanged; `analyze_truth` reports the new rows |
| F3 | budget table priced the plan's 244/265 items at the plan's per-item cost | re-derived in `TRUTH_EVAL.md` from 282/288 and the smoke's per-role costs: ≈ $200 + 10 % ≈ $220; after the cut order ≈ $195; `--cost-cap` default 0.17; stress rows stay in (D-rate on stress_D is a registered secondary); R2 must run on the whole holdout (subsets refused) — **a PI budget decision, recorded, not made** |
| F4 | R2 (`a2 --k 3`) collided with the k=1 file name | file names / run tags carry the replicate count: `preds_truth_{arm}_{model}_{split}_k{K}_r{k}` (`out_paths` `{K}` placeholder); `analyze_truth` analyses `a2` and `a2k3` as separate groups (pre-k names read as K=1) |
| F5 | COWLS θ_E = 0.0 placeholder entered Spearman and the θ_E ≤ 1″ stratum | `build_truth_manifest` and `analyze_truth.load_manifest` map θ_E ≤ 0 → NaN; manifest re-pinned |
| F7 | arbitrator-only parse failure kept S finite with `parse_ok=False` | ONE policy: any failed role ⇒ S, S_arb NaN, letters None (docstrings in `schemas_panel`, `panel`, TRUTH_EVAL, REGISTRY agree) |
| F8 | what the holdout certifies ≠ what the drop-in runs (no note, strips not separate images, `context` to the advocate, layout gate) | README section "What differs from the lensjudge truth evaluation" (5 items); the layout gate and the gray morphology set fixed (C6, E12) |
| F9 | P2 as pre-registered ("upper CI ≤ 2.5 % / 7.5 %") needs 0/200 at t_A although t_A is set at ≤ 2/200 | **registry decision, restated in `REGISTRY.md` before any holdout call**: CP 95 % LOWER bound ≤ target (`P2_holds_*`); the plan's wording reported beside it (`P2_upper_ci_ok_*`) |
| F10 | every 1727 `lit_galaxy` lens is in the holdout (0 design / 11 holdout) — undocumented | documented in `TRUTH_EVAL.md` (split kept; re-split treating 1727 lit_galaxy like COWLS named as the remedy) |
| F11 | anchors wording "identical SW-only pixels" false (re-centred 1.17″, 13×35-px shift, own (f)) | `REGISTRY.md` + `analyze_truth.ANCHORS`: "same SW-only field, re-centred 1.17″ apart", with the reason |
| F12 | PA-span direction unspecified to the models but load-bearing in the guard | one sentence each in `advocate.md` and `critic_common.md` (from → to, increasing angle, 350 → 10 through North; critics write their box the same way) |
| F13 | critic briefs said "no subtracted panel reaches you" while the attribution arm sends the full composite | geometry/morphology briefs: "the VIEW description lists the panels you were given … judge an item whose panel is a subtracted one only by what is traceable in direct renderings" |
| F14 | "a named alternative requires a location" not validated | `CriticRecord` validator (+ the drop-in's `validate_critic`); a location-less named critic is a parse failure on both sides |

### Deviations from the contract / plan introduced by this pass
`TruthTuple` gains `thresholds_sha16` (13 fields; REGISTRY headers widened); output names carry
`_k{K}`; `calibration_ids.csv` ships 15 design-half COWLS, not 31; P2 restated (above);
`--cost-cap` 0.17; a0 `system_sha16s` carries six entries under `claim_mode=inspector`;
`aggregate_v2` 2.1.0 (covers padding, U-score scale, `MODEL_KEYS`); the circular-subtraction
caveat lives in `panel_gloss.json`, not the note; any-role parse failure ⇒ S NaN (the plan's
wording, now also for the arbitrator).

### Open after this pass (for the integrator / PI)
- Budget: ≈ $195–220 at the measured per-role costs vs the plan's $150–160 (`TRUTH_EVAL.md`
  table); the levers are R2 at k=2, a 40-item test–retest, and trimming critic output.
- The smoke parquets under `outputs/smoke/` carry the pre-review prompt shas and `_r1` names;
  they are not resumable as any current tuple (by design) — start the design passes clean.
- The holdout's COSMOS-Web `lit_galaxy` stratum is a field the design never sees at galaxy
  scale (F10); read the per-stratum recall rows with that in mind or re-split before registering.
- Nate's blinding remains advisory (README §"What differs"); the lensjudge numbers are the
  certification, his re-run the deployment.
- Register every holdout tuple with `--print-tuple` AFTER the design freeze; the a0 inspector
  row carries both wrapper variants; `--allow-provisional-thresholds` must not be needed.

## 2026-08-23 — Design phase, step 1: A0 agreement gate (Sonnet, 40 design frame units, $0.92)

**Units.** `outputs/design/a0_agreement_ids.txt`: every DESIGN-half frame unit with recorded
incumbent verdicts — 34 rows with `pipe_grade_passcount ∈ {A,B,C,D}` plus the 6 design-half
`*_ctl*` COWLS rows whose three verdicts `incumbent_replay` recovered (published U, replay D);
40 = cowls 7 · lit_galaxy 4 · lit_cluster 6 · stress_D 20 · anchor 3 (layout color 37 /
gray_sw_only 3; recorded letters D 31 / C 4 / B 3 / A 2). The 3 design rows with verdicts that
are not frame units (2 anomalymatch, the rank-14 alias) were left out.

**Run.** `run_truth_eval.py --arm a0 --split design --model sonnet --claim-mode inspector
--ids-file … --out outputs/design/preds_truth_a0_sonnet_design_agreement.parquet` (tuple as in
`TRUTH_EVAL.md`: persona set `f2bb5294e1d89505`, note none, system shas `2e5627171e2ce69c` /
`e5b545f9957120ac` / `a672a7dfc227288e` + the three `@noclaim` wrappers, thresholds
`a40ae6e201a03e65`). Claims attached 40/40, 0 blanked; parse 40/40; 0 over the cost cap;
**$0.919** ($0.023/item); `audit_traces` 120 events, 0 violations; units marked `kind=eval`,
run tag `truth_a0_sonnet_design_k1_r1`.

**Agreement with the recorded verdicts** (`incumbent_replay.csv`, per (id, persona)):

| persona | agree | rate | κ | recorded pass / unc | re-impl pass / unc |
|:--|--:|--:|--:|--:|--:|
| artifact | 33/40 | 0.825 | 0.56 | 9 / 1 | 11 / 0 |
| geometry | 35/40 | 0.875 | 0.50 | 3 / 0 | 6 / 2 |
| morphology | 31/40 | 0.775 | 0.20 | 4 / 0 | 8 / 1 |
| **overall** | **99/120** | **0.825** (CP 95 % 0.745–0.888) | | 16 / 1 | 25 / 3 |

Pass-count letters: exact **28/40 = 0.70**, within one letter 0.875, QWK 0.63 (recorded →
re-impl: A 2→2 A; B 3 → 1 A / 2 D; C 4 → 2 A / 1 B / 1 C; D 31 → 1 B / 5 C / 25 D). By
class: cowls 20/21 votes (letters 6/7), lit_galaxy 11/12, stress_D 50/60 (15/20), lit_cluster
13/18 (3/6), anchors 5/9 (1/3). Strict-34 subset 82/102 = 0.804; the 6 recovered-ctl COWLS
17/18.

**Reading.** The gate (≥ 0.85) is **not met** at the point estimate; the CI includes it and the
Opus-spot-check trigger (< 0.80) is not hit, so no Opus run was made (the rule as given). The
disagreement is one-directional: the re-implementation passes MORE than the recorded run in
every persona (25 vs 16 passes; 5 A / 2 B vs 2 A / 3 B; 6 of the 31 recorded D move to C/B — 5 of
them stress_D). The recorded verdicts came from Opus inside Claude Code with tool access (notes
mention unsharp-mask ridge traces); this arm is Sonnet, one API call, composite only — model and
harness both differ, and the 94 % reference figure is the incumbent's own letter test–retest
(31/33), not a cross-harness number. Consequence for P1: an A0 that is more permissive than the
real incumbent is a STRONGER baseline, so "A1 vs A0" on the holdout is conservative with respect
to the pipeline it replaces. Open for the PI: whether to run the same 40 ids once with
`--model opus` (≈ $1.6 at the runner's per-call constants) to separate model from harness before
the holdout A0 tuple is registered. Cumulative design-phase spend after this step: **$0.92**.

## 2026-08-23 — Design phase, step 2: advocate-only iteration 1 of ≤ 3 (a2, Sonnet, whole design half, $6.12)

**No prompt edit this iteration** (iteration 1 scores the prompt as built). Advocate file
`prompts/personas/jwst_v1/advocate.md` sha16 `54e3bed63b61dec8`; full advocate system prompt
(persona + `jwst_note_v2.md`) `4d252f5bf2b82e5d`; persona set `5983c47ef6315078`; note
`754655a400f360e6`; gloss `e734968399847d42`; thresholds provisional `a40ae6e201a03e65`
(τ0 0.15, t_A 0.80, t_B 0.50). Tuple row as printed by `--print-tuple`:
`| a2 | sonnet | 5983c47ef6315078 | 754655a400f360e6 | advocate:4d252f5bf2b82e5d | jwst_v1 | 28737c6083dc1978 | 032a302c84f3cbe7 | none | off | default | 1 | a40ae6e201a03e65 |`.

**Run.** `run_truth_eval.py --arm a2 --split design --model sonnet --k 1 --concurrency 4 --out
outputs/design/preds_truth_a2_sonnet_design_iter1.parquet` → 288/288 scored, **0 parse
failures**, 0 over the cost cap, **$6.116** ($0.0212/item, max $0.032), run tag
`truth_a2_sonnet_design_k1_r1`, traces `outputs/design/traces_truth_a2_sonnet_design_k1_r1/`
(`audit_traces`: 288 events, **0 violations**). `analyze_truth.py --split design` needs the
canonical `_k1_r1` name, so it was run on a scratch directory of symlinks
(`preds_truth_a2_sonnet_design_k1_r1.{parquet,_votes.parquet,.meta.json}` → the iter1 files);
the numbers below are its output plus a per-stratum breakdown (`scratchpad/analyze_a2_iter.py`).
**For iteration 2 use a fresh sub-directory** (`--out outputs/design/iter2/…`): the trace
directory is named from the run tag, not from `--out`, and would otherwise mix iterations.

### Metrics (design half; positives = cowls + lit_galaxy + lit_cluster, anchors excluded; negatives = the 200 design N1)

| positives | n_pos | n_neg | AUC | recall@5 % FPR (CP 95 %) | thr | recall@10 % FPR | thr |
|:--|--:|--:|--:|:--|--:|:--|--:|
| all (`is_positive`) | 44 | 200 | **0.712** | **0.250** (11/44; 0.13–0.40) | 0.30 | 0.318 (14/44; 0.19–0.48) | 0.20 |
| `centre_is_deflector` | 30 | 200 | 0.716 | 0.233 (7/30; 0.10–0.42) | 0.30 | 0.300 (9/30) | 0.20 |
| cowls | 15 | 200 | 0.606 | 0.133 (2/15; 0.02–0.40) | 0.30 | 0.133 (2/15) | 0.20 |
| lit_galaxy | 13 | 200 | 0.819 | 0.308 (4/13) | 0.30 | 0.462 (6/13) | 0.20 |
| lit_cluster | 16 | 200 | 0.725 | 0.312 (5/16) | 0.30 | 0.375 (6/16) | 0.20 |
| cowls strong / marginal / weak / provenance | 3 / 4 / 6 / 2 | 200 | 0.66 / 0.59 / 0.61 / 0.57 | 0/3 · 1/4 · 1/6 · 0/2 | | same | |
| θ_E ≤ 1″ / 1–2″ (COWLS with a radius) | 8 / 4 | | | 1/8 · 1/4 | | | |
| layout color / gray_lw / gray_sw | 38 / 5 / 1 | | | 10/38 · 1/5 · 0/1 | | 10/38 · 3/5 · 0/1 | |
| field cosmos / cluster / blank | 12 / 25 / 7 | | | 2/12 · 7/25 · 2/7 | | 2/12 · 9/25 · 2/7 | |

`analyze_truth` rows (same run): P2 FPR(A) 0 % [0, 3 %], FPR(A/B) 2 % [0, 4 %] on the
provisional letters (t_A 0.80 / t_B 0.50); P3 positives at A/B 20 % [10, 35 %];
`D_rate_stress_D` 0.05 [0.001, 0.249]; `spearman_S_vs_theta_E` −0.36 (perm p 0.24, n 12 —
not significant, but the sign is the wrong one: the θ_E ≤ 1″ COWLS rows score 1/8).

| class | n | mean p_ev | median | q10 | q90 | < τ0 | < 0.30 | ≥ thr5 (0.30) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| cowls | 15 | 0.125 | 0.05 | 0.04 | 0.30 | 0.80 | 0.87 | 0.13 |
| lit_galaxy | 13 | 0.327 | 0.12 | 0.05 | 0.89 | 0.54 | 0.69 | 0.31 |
| lit_cluster | 16 | 0.292 | 0.095 | 0.04 | 0.88 | 0.56 | 0.69 | 0.31 |
| **negative** | 200 | 0.080 | 0.05 | 0.04 | 0.18 | **0.88** | 0.96 | 0.03 |
| stress_D | 20 | 0.465 | 0.485 | 0.10 | 0.73 | 0.15 | 0.35 | 0.65 |
| stress_U | 14 | 0.212 | 0.16 | 0.04 | 0.45 | 0.50 | 0.71 | 0.14 |
| anomalymatch | 5 | 0.126 | 0.10 | 0.04 | 0.24 | 0.60 | 0.80 | 0.00 |

Negatives: 176/200 below τ0 (the critics would engage on 12 % of N1); letters D 122 / C 75 / B 2
/ A 1 (the single A is a barred two-arm spiral scored 0.82 — the `spiral_arm` case the
morphology critic exists for). stress_D (the incumbent's refuted set, machine-labelled): letter
D on 1/20, p_ev median 0.485 — the advocate finds located evidence on 19/20 of them, so the
D-rate monitor will be decided by the critics, not here. Letters on positives: A 5 / B 4 / C 22
/ D 13.

**The number that matters: p_evidence is a near-deterministic function of the item count.**
Over all 288 records, items = 0 ⇒ p_ev ≤ 0.05 (134/135), 1 item ⇒ ≤ 0.15 (47/49), 2 items ⇒
≤ 0.30 (32/34); 97 records sit at exactly 0.04. On the positives: 13/44 returned `items: []`
(p_ev ≤ 0.05) and a further 20/44 located 1–2 items and then scored them 0.05–0.30, i.e. **45 %
of the positives were located and discounted** (negatives: 61 % empty, 35 % located-and-
discounted). The 5 %-FPR threshold (0.30) falls exactly on the 2-item cap.

### What the advocate wrote (design items; quotes abridged)

Every design COWLS **strong** or **marginal** lens below 0.30 — 6 of 7 (the seventh, θ_E 1.86″,
scored 0.55 with 3 items):

- strong, θ_E 0.70″, p_ev 0.04, `items: []`: "isolated elliptical with extended halo; panel
  (f) shows only the expected four-lobed … subtraction residual … no tangential arc,
  counter-image, or offset extended source is visible in any direct panel".
- marginal, θ_E 1.98″, p_ev 0.04, `items: []`: "smooth, undisturbed elliptical galaxy with no
  curved, offset, or colour-distinct feature at any radius … the nearby edge-on galaxy to the
  SE is a companion projection at ~4″". The stamp shows a thin tangential arc ~2″ north of the
  deflector in (a), (b) and (c) — at r ≈ 2″ it lies entirely outside the 3.5″ zoom row
  (half-width 1.75″), which is where the advocate looked.
- marginal, θ_E 1.17″, p_ev 0.04, `items: []`: "edge_on_disk — … inclined disc galaxy … panel
  (f) shows only the expected … subtraction artefact from the inclined elliptical body".
- marginal, θ_E 0.96″, p_ev 0.08, 1 item: "Bright compact object ~1.5″ to the SE … could be a
  companion galaxy or a lensed image, but shows no tangential elongation or curvature toward
  the deflector" → `nothing_because`: "compact companion or foreground/background galaxy".
- strong, θ_E 1.00″, p_ev 0.10, 2 items: "Elongated companion galaxy ~1.5″ east of the
  deflector; appears bluer … morphology is consistent with a companion galaxy rather than a
  lensed arc — no clear tangential curvature"; item 2 is the subtraction pattern, listed "only
  to flag the dominant subtraction pattern".
- strong, θ_E 0.60″, p_ev 0.12, 1 item at r 1.5″ PA 20–70: "faint elongated feature NE at
  ~1.5″ … possibly tangentially elongated … no convincing tangential curvature at fixed
  radius, no clear counter-image, and the subtracted panel (f) shows only a strong … residual".

The same three moves recur on the other 27 positives below threshold (lit_galaxy/lit_cluster
and the weak/provenance COWLS): (i) a compact or diffuse source at 1–2.5″ is located and then
written off as "companion", "projection", "tidal", "edge-on disc" — 43 % of positive records
contain "companion", 20 % "no (tangential) curvature"; (ii) the subtracted-panel artefact
pattern is cited as the closing argument in `nothing_because`/`notes` — 28/44 positive records
(23 of the 33 below threshold; median p_ev 0.055 with the echo vs 0.19 without; 157/200
negatives carry it too); (iii) the search stops at the zoom row (the θ_E 1.98″ miss; lit_cluster
arcs at 2–3″ scored 0.04–0.07 with "no tangential arc at fixed radius").

The 10 highest negatives (0.28–0.82): a barred two-arm spiral (0.82, "nearly-complete Einstein
ring … bluer"), a blue diffuse halo around an orange deflector with a BCG-like envelope (0.52,
"possibility of a ring galaxy or tidal structure"), a group-scale diffuse halo with two knots at
1.5″ on opposite sides (0.52), and seven 0.28–0.38 records of faint tangential LSB features at
2–3″ with "tidal/merger" hedges. None of the ten is the 4-item echo pattern; these are
morphology/merger alternatives, i.e. the critics' competence (none reaches A/B except the spiral).

Anchors (descriptive, never scored): rank 15 → **A, 0.93**, `scale_class=group`, arc r 3.5″ PA
180→270 + counter-arc NW (prediction A/B: hit); rank 16 → A, 0.98, near-complete ring r 2.8″,
but `scale_class=group` and `deflector_is_centre=true` (prediction cluster / false: letter
"not D" hit, scale-class miss — the 10″ stamp shows the arc at 2.8″ from the tick, so "group" is
what the image supports); ranks 7 / 14 → C 0.15 / C 0.12, both "two compact sources ~1.7–1.8″
apart in a shared envelope … merger or companion" (|Δletter| = 0: hit, at the low end); rank 13
→ A, 0.92 (a bright warm arc r 1.1″ PA 120→250 with a NW counter-feature — the prediction "D
with spiral_arm upheld" is a full-stack prediction and cannot be met by the advocate alone).

### Decision: NOT frozen (iteration 1; nothing to compare against). Five wording changes for iteration 2

Item-agnostic, prose-only (record contract unchanged); none uses the lexicon-banned phrasing,
"circular" or "butterfly" (both are reserved to the VIEW text by `test_golden_prompts`).

1. **Role discipline — locate, do not adjudicate.** The advocate's job is to list and score
   every candidate feature; deciding that a located feature is a companion, a tidal tail or
   an edge-on disc is the critics' job. A suspected alternative may be named in `notes`, but it
   never empties `items`, never removes an item that passed a test, and never lowers
   `p_evidence` below what the located set supports. *Rationale: 45 % of positives were
   located-and-discounted; "companion" appears in 43 % of positive records.*
2. **Silence in a subtracted panel is not evidence.** A subtracted panel that shows only the
   artefact patterns the VIEW names is uninformative, not negative: at galaxy scale the lensed
   images sit at the same radii as the model's residual lobes and the saturated core. Never
   cite the artefact pattern in `nothing_because` or `notes`; judge such cases from the direct
   panels, and say so when the direct panels are saturated inside the relevant radius.
   *Rationale: the echo closes 23/33 sub-threshold positive records; echo median p_ev 0.055
   vs 0.19 without.*
3. **Compact images have no curvature to measure.** Criteria 3 and 5 apply to extended images;
   a compact image of a lensed source at 0.3–2″ from a massive elliptical is a knot, so score
   it on contrast, profile and `counter_image` instead, and search explicitly for its
   counter-image on the opposite side at a similar radius — at the edge of the saturated core
   in (d)/(e), and as a ONE-SIDED residual (not mirrored across the nucleus) in the subtracted
   panel, listed with `visible_in_direct` set honestly. Two knots straddling the nucleus at
   similar radii, or a knot plus an arc, is a lensing configuration (criterion 4) even without
   an arc. *Rationale: "no tangential curvature" dismissed compact 1–2″ sources on 20 % of
   positives; 8/15 design COWLS have θ_E ≤ 1″ (recall 1/8).*
4. **Search the whole 10″ field.** The zoom row covers only r ≤ 1.75″; a feature at 1.75–5″
   appears ONLY in (a)/(b)/(c). Scan the 10″ row at every radius out to ~5″ before writing
   `items: []`, and never conclude "nothing" from the zoom row alone; a thin faint tangential
   arc 1.5–3″ from a bright early-type galaxy is an ordinary galaxy/group-scale configuration.
   *Rationale: the θ_E 1.98″ COWLS miss with a visible arc in the 10″ row; lit_cluster 5/16.*
5. **An anchored p_evidence ladder** (replacing "use the whole range"): ≤ 0.05 only with
   `items: []`; 0.10–0.25 a single item that fails most tests; 0.30–0.60 one or two items that
   pass the offset / fixed-radius / opposite-counter-image tests even if a non-lens
   explanation remains open (that is what the critics are for); 0.70–0.90 an arc whose centre
   of curvature lies on the deflector, or a counter-image; ≥ 0.90 a ring or a multi-image
   configuration. *Rationale: p_ev is a function of the item count with the 2-item cap sitting
   on the 5 %-FPR threshold; 97/288 records at exactly 0.04.*

Expected risk, stated before the run: 1 and 5 will also lift the 35 % of negatives that were
located-and-discounted, so the 5 %-FPR threshold moves up; the iteration is judged on
recall@5 % FPR, not on the raw p_ev shift. The spiral/ring/merger false positives are left to
the critics by design (no anti-FP wording is added to the advocate).

**Spend.** This step $6.12 (sum of `cost_usd` over `preds_truth_a2_sonnet_design_iter1.parquet`).
Cumulative design-phase spend: $0.92 + $6.12 = **$7.04** (cap $95).

## 2026-08-23 — Design phase, step 2: advocate-only iteration 2 of ≤ 3 (a2, Sonnet, whole design half, $6.93) → FREEZE

**Prompt edit (advocate.md only; `jwst_note_v2.md` and the gloss untouched).** The five
item-agnostic changes proposed at the end of iteration 1 were applied to
`prompts/personas/jwst_v1/advocate.md` as five new prose blocks: (1) "Role discipline —
LOCATE, do not adjudicate" (a suspected companion / projection / tidal / spiral / edge-on
explanation goes in `notes`, never empties `items`, never lowers `p_evidence` below the
located set); (2) "Silence in a subtracted panel is not evidence" (a subtracted panel that
shows only the VIEW's artefact patterns is uninformative; never cite the pattern, or the
absence of an arc in a subtracted panel, in `nothing_because`/`notes`; judge from the direct
panels; say when the direct panels are saturated); (3) "Compact images have no curvature to
measure" (criteria 3/5 are for extended images; a knot at 0.3–2″ is scored on contrast /
profile / counter_image, with an explicit opposite-side counter-image search; two straddling
knots or knot + arc = criterion 4); (4) "Search the whole field before you conclude" (the zoom
row is r ≤ 1.75″; scan the 10″ panels to ~5″ before `items: []`); (5) the anchored
`p_evidence` ladder (≤ 0.05 only with `items: []`; 0.10–0.25 one failing item; 0.30–0.60 items
passing the offset / fixed-radius / counter-image tests with a non-lens explanation still open;
0.70–0.90 centred arc or counter-image; ≥ 0.90 ring / multi-image) replacing "use the whole
range". The record contract block is byte-identical (diff-checked); no "bluer", "prefer fail",
"mark it fail", "circular" or "butterfly" (grep); `audit_traces --check-text` on the file: no
lexicon hit (298 entries); `test_golden_prompts.py` 11/11 (80 tracked files + 107 model-facing
strings clean). File sha16 **`54e3bed63b61dec8` → `6a806fbec212eb19`**; advocate system prompt
(persona + note_v2) **`4d252f5bf2b82e5d` → `c41d7f5787bdb472`**; persona set
`5983c47ef6315078` → **`a26d972ecc0b4ee7`**; note `754655a400f360e6`, gloss, thresholds
`a40ae6e201a03e65` unchanged. Tuple row (`--print-tuple`):
`| a2 | sonnet | a26d972ecc0b4ee7 | 754655a400f360e6 | advocate:c41d7f5787bdb472 | jwst_v1 | 28737c6083dc1978 | 032a302c84f3cbe7 | none | off | default | 1 | a40ae6e201a03e65 |`.
The iteration-1 text is kept at `scratchpad/iter2/advocate.iter1.md` (sha `54e3bed63b61dec8`).

**Run.** `run_truth_eval.py --arm a2 --split design --model sonnet --k 1 --concurrency 4 --out
outputs/design/iter2/preds_truth_a2_sonnet_design_iter2.parquet` (fresh sub-directory, so the
trace dir `outputs/design/iter2/traces_truth_a2_sonnet_design_k1_r1/` is this iteration's own).
The first invocation was killed by a 10-minute foreground limit at 175/288 (SIGTERM, 4 calls in
flight); the resume graded the remaining 113 onto the same parquet (`check_resume` accepted the
tuple). 288/288 scored, **0 parse failures**, 0 over the cost cap, **$6.931** by `cost_usd`
($0.0241/item, max $0.034; +14 % per item over iteration 1 for the longer prompt and records).
The traces hold 4 completed responses that the kill dropped before their rows were written
($0.108, re-graded on resume) and 4 requests cut mid-call, so the true API spend is ≈ $7.04–7.14.
`audit_traces`: 296 events (288 + the 8 pre-kill), **0 violations**. `analyze_truth.py --split
design` run through the canonical-name symlink dir `scratchpad/at_iter2/`; its rows agree with
`scratchpad/analyze_a2_iter.py` below.

### Metrics (design half; positives = cowls + lit_galaxy + lit_cluster, anchors excluded; negatives = the 200 design N1). Iteration 1 in parentheses

| positives | n_pos | AUC | recall@5 % FPR (CP 95 %) | thr | recall@10 % FPR | thr |
|:--|--:|--:|:--|--:|:--|--:|
| all (`is_positive`) | 44 | **0.728** (0.712) | **0.295** (13/44; 0.17–0.45) (0.250, 11/44) | 0.35 (0.30) | 0.364 (16/44; 0.22–0.52) (0.318) | 0.28 (0.20) |
| `centre_is_deflector` | 30 | 0.720 (0.716) | 0.333 (10/30; 0.17–0.53) (0.233) | 0.35 | 0.400 (12/30) (0.300) | 0.28 |
| cowls | 15 | 0.588 (0.606) | 0.133 (2/15; 0.02–0.40) (0.133) | | 0.200 (3/15) (0.133) | |
| lit_galaxy | 13 | 0.852 (0.819) | 0.538 (7/13; 0.25–0.81) (0.308) | | 0.615 (8/13) (0.462) | |
| lit_cluster | 16 | 0.758 (0.725) | 0.250 (4/16; 0.07–0.52) (0.312) | | 0.312 (5/16) (0.375) | |
| cowls strong / marginal / weak / provenance | 3 / 4 / 6 / 2 | 0.68 / 0.55 / 0.58 / 0.57 | 0/3 · 1/4 · 1/6 · 0/2 (0/3 · 1/4 · 1/6 · 0/2) | | 0/3 · 1/4 · 2/6 · 0/2 | |
| θ_E ≤ 1″ / 1–2″ | 8 / 4 | | 1/8 · 1/4 (1/8 · 1/4) | | 2/8 · 1/4 | |
| layout color / gray_lw / gray_sw | 38 / 5 / 1 | | 11/38 · 1/5 · 0/1 | | 13/38 · 3/5 · 0/1 | |
| field cosmos / cluster / blank | 12 / 25 / 7 | | 2/12 · 8/25 · 2/7 | | 2/12 · 11/25 · 3/7 | |

Paired on the same 44 + 200 (`scratchpad/iter2/iter2_extra.py`): recall@5 % FPR 0.250 → 0.295,
**Δ = +0.045** (2 positives promoted — both `lit_galaxy` — 1 demoted — `lit_cluster` — 10 kept);
recall@10 % FPR 0.318 → 0.364 (+0.045; 4 promoted / 1 demoted); AUC +0.016; Spearman(iter1,
iter2) on p_ev 0.79. COWLS 2/15 → 2/15 at both operating points. `analyze_truth` rows: P2
FPR(A) 0.5 % [0.01, 2.8], FPR(A/B) 1.0 % [0.1, 3.6] on the provisional letters (t_A 0.80 /
t_B 0.50; iteration 1: 0 % / 2 %); P3 positives at A/B 15.9 % [6.6, 30.1] (20 %);
`D_rate_stress_D` 0.05 (unchanged, letters B 9 / C 9 / A 1 / D 1, median p_ev 0.50);
`spearman_S_vs_theta_E` −0.28 (perm p 0.39, n 12; −0.36 before — still the wrong sign, still
not significant); parse-failure rate 0 [0, 1.3 %]; $0.0241/item.

| class | n | mean p_ev (iter1) | median (iter1) | q10 | q90 | < τ0 | < 0.30 | ≥ thr5 (0.35) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| cowls | 15 | 0.161 (0.125) | 0.150 (0.05) | 0.03 | 0.39 | 0.47 | 0.80 | 0.13 |
| lit_galaxy | 13 | 0.410 (0.327) | 0.350 (0.12) | 0.09 | 0.85 | 0.31 | 0.38 | 0.46 |
| lit_cluster | 16 | 0.277 (0.292) | 0.175 (0.095) | 0.04 | 0.69 | 0.38 | 0.69 | 0.25 |
| **negative** | 200 | 0.103 (0.080) | 0.040 (0.05) | 0.03 | 0.25 | **0.77** (0.88) | 0.92 (0.96) | 0.02 |
| stress_D | 20 | 0.470 (0.465) | 0.500 | 0.20 | 0.73 | 0.05 | 0.20 | 0.70 |
| stress_U | 14 | 0.272 (0.212) | 0.160 | 0.09 | 0.63 | 0.50 | 0.57 | 0.21 |
| anomalymatch | 5 | 0.212 (0.126) | 0.150 | 0.05 | 0.41 | 0.40 | 0.60 | 0.20 |

Negatives: 154/200 below τ0 (176 before — the critics would now engage on 23 % of N1 instead of
12 %, ≈ +$2.4 per full-stack design pass); letters D 108 / C 90 / B 1 / A 1 (the A is the same
barred two-arm spiral, 0.82 in both iterations; the B is the group halo with two opposite knots,
0.52 → 0.55). Negatives ≥ 0.30: 7 → 17; above the 5 %-FPR threshold (0.35) 4, of which 3 were
already above iteration 1's threshold. Letters on positives: A 4 / B 3 / C 25 / D 12.

**Structure.** p_evidence is still a function of the item count: Spearman(p_ev, n_items) 0.93;
0 items ⇒ ≤ 0.05 (123/123); 1 ⇒ median 0.10, max 0.22; 2 ⇒ median 0.15, q90 0.35; 3 ⇒ median
0.35; 4 ⇒ 0.50; 91 records at exactly 0.04. What moved: item counts (positives mean 1.07 → 1.53
on COWLS, negatives 0.65 → 0.88; empty records 13 → 12 on positives, 122 → 108 on negatives) and
the 2-item cap (0.30 → 0.35–0.72), i.e. change 1 and the ladder did what was predicted —
located-and-discounted positives 20 → 16, but negatives located-and-discounted 71 → 75 and the
threshold rose with them. Pattern counts on the 44 positives: the subtraction-artefact echo
32 → 26 (median p_ev with the echo 0.10 vs 0.385 without — the gap widened); "no (tangential)
curvature" 9 → 2 (change 3 worked on the wording); **"companion" 19 → 26** (the adjudication
moved from curvature to companionship). **All 12 positives that returned `items: []` cite the
four-lobed / bowtie artefact in `nothing_because` or `notes`, 9 of them with the word
"expected"** — the explicit "never cite the artefact pattern" sentence was ignored outright,
most plausibly because the VIEW text (which `test_golden_prompts` requires to name the
patterns) is the more salient instruction and is recited back as the closing argument.

### What the advocate wrote (design items; quotes abridged)

Design COWLS strong / marginal below 0.30 — 6 of 7 again (the seventh, θ_E 1.86″, 0.55 / B
unchanged). The five `items: []` COWLS rows of iteration 1 are the same five rows, same 0.03–0.04:

- strong θ_E 0.70″, 0.04, `items: []` (was 0.04): "isolated elliptical … no tangential arc,
  counter-image, or offset residual at any radius; panel (f) shows only the expected four-lobed
  … artefact … with no offset tangential feature traceable in (d) or (e)".
- marginal θ_E 1.98″, 0.04, `items: []` (was 0.04): now scans the 10″ row — "the 10″ row
  (panels a, b, c) reveals a large smooth elliptical deflector with a nearby edge-on disk galaxy
  to the SE (~3″) and a small round companion to the SW; neither shows curvature toward the
  deflector" — and still closes with "(f) shows a classic bowtie subtraction residual … with no
  offset tangential arc residual". The thin arc ~2″ N was not named.
- marginal θ_E 1.17″, 0.03, `items: []`: "inclined disk galaxy … panel (f) shows only the
  expected four-lobed … artifact from applying a … model to an elongated disk".
- marginal θ_E 0.96″, 0.15, 2 items (was 0.08 / 1): "secondary brightness concentration SE …
  possible companion or lensed knot" + "faint galaxy to SE … possible second image candidate";
  `notes`: "likely a merging companion galaxy rather than a lensed image; no tangential arc at
  fixed radius … (f) residuals are consistent with the expected bowtie artifact".
- strong θ_E 1.00″, 0.20, 3 items (was 0.10 / 2): "Elongated companion galaxy ~1.5″ east …
  possibly tangentially elongated" + "faint blue knot … south … ~1″" + an asymmetric halo;
  `counter_image_pos` filled (r 1.0″, PA 180) — and `notes` still ends "may be a companion galaxy
  or merger rather than a lensed arc, as it lacks clear tangential curvature … (f) shows a
  dominant four-lobed … artefact with no obvious offset arc residual".
- strong θ_E 0.60″, 0.20, 3 items (was 0.12 / 1): "compact galaxy or knot ~1.5″ NE" + "compact
  source ~1.5″ SW, roughly opposite the NE compact source, at comparable radius, forming a
  possible two-image configuration" — the ladder says ≥ 0.30 for exactly this, `counter_image`
  scored 3, p_ev 0.20: "no clear tangential arc or curvature is confirmed, and the subtracted
  panel f shows primarily the expected … artefact".

The 31 other positives below 0.35 repeat the same two closings: (i) the artefact recital in 12/12
empty records (four `lit_cluster` rows with `centre_is_deflector = false` are written off as
"edge_on_disk" / "isolated massive elliptical" of the ticked galaxy — the arc belongs to another
deflector in the stamp and the advocate does not look for one; off-centre positives 3/14 at
threshold); (ii) "companion"/"group member"/"tidal" on the located items, with the ladder
ignored — a `lit_galaxy` record with "two bright compact knots straddle the deflector at roughly
similar radii (~1.4″ NE and ~1.6″ SSW) suggesting a possible two-image configuration, but both
could be companion galaxies" scored 0.35. Saturation of the direct panels is mentioned in 2/44
positive records.

**Top-10 negatives** (0.35–0.82; 9 of the 10 were in iteration 1's top ten): the barred spiral
(0.82 / A, "nearly complete partial ring … bluer"), the group halo with two opposite knots at
~1″ plus a diffuse SE arc (0.55 / B), a cluster-field LSB "partial arc system" at 2.2–2.5″ with a
"counterpart structure to the NW" (0.45), a blue sweep at 0.9″ around a warm deflector (0.38),
and six records at exactly 0.35 (the ladder's 0.30–0.60 floor) of faint tangential LSB features
at 2–3″ hedged "tidal / merger / resonance ring / companion" — morphology alternatives for the
critics, as before. None is the echo pattern.

**Anchors (descriptive, never scored):** rank 15 → A, 0.85, `scale_class=group`, arc r 3.5″ SW
+ NE counter-arc + NW segment (prediction A/B: hit); rank 16 → A, 0.97, `group`,
`deflector_is_centre=true` (letter-not-D hit; scale-class / centre miss, as in iteration 1 — the
arc at 3.5″ from the tick is what the 10″ stamp shows); ranks 7 / 14 → C 0.25 / C 0.15, both
"two bright compact nuclei ~1.5″ apart in a shared envelope … merger" (|Δletter| = 0: hit);
rank 13 → A, 0.88 ("a spiral arm origin cannot be fully excluded" — the D-with-`spiral_arm`
prediction is for the full stack).

### Decision: **FREEZE** (pre-registered rule: Δrecall@5 % FPR = +0.045 < 0.05)

The iteration-2 text is the frozen advocate (`advocate.md` **`6a806fbec212eb19`**, system
`c41d7f5787bdb472`, persona set `a26d972ecc0b4ee7`): it is not worse than iteration 1 on any
headline number and better on AUC (+0.016), `centre_is_deflector` recall (7 → 10/30),
`lit_galaxy` (4 → 7/13) and FPR(A/B) (2 → 1 %), at the cost of +14 % per advocate call, a
23 % (from 12 %) critic engagement rate on N1 and one `lit_cluster` demotion. No third
advocate-only pass: the two failure modes that keep COWLS at 2/15 — the artefact recital in
every empty record and the companion adjudication on located knots — did not respond to
advocate-side wording, so another ~$7 of the same lever is not expected to clear the rule.
The design continues to the full stack on this advocate (the critics and arbitrator are the
scheme's answer to the companion / tidal / spiral hedges that make up both the sub-threshold
positives and the top negatives).

Recorded for the full-stack phase and the PI, **not applied** (outside an advocate-only iteration
or outside `advocate.md`): (a) the recital is anchored in the VIEW gloss sentence that names the
artefact patterns — the remedy is in the gloss / note (e.g. stating that the subtracted panel
is uninformative inside the model's lobes) or in a record field that forces a direct-panel
statement, both of which change files the tests pin; (b) four design `lit_cluster` positives are
off-centre arcs the advocate never looks for because every sentence of the brief is about the
ticked galaxy — the "lensing anywhere in the 10″ stamp" label needs either a brief sentence
about other deflectors in the field or a secondary-label-only reading of those rows; (c) the
COWLS θ_E ≤ 1″ images sit inside r ≈ 1″ where the advocate's located items are at 1.5–2.5″ —
no record mentions the saturated core, so a saturation-aware zoom stretch (a render change,
cf. the a3 arm) is the lever there, not wording.

**Spend.** This step $6.93 by `cost_usd` over `outputs/design/iter2/preds_truth_a2_sonnet_design_iter2.parquet`
(≈ $7.04–7.14 actual, see the kill above). Cumulative design-phase spend: $0.92 + $6.12 + $6.93 =
**$13.97** (≈ $14.2 actual) of the $95 cap.

## 2026-08-23 — Design phase, step 3: full-stack design pass + threshold freeze + test–retest + holdout tuples ($24.52 this step)

### (a) a1 full stack, whole design half — one pass, no wording round triggered

**Run.** Tuple (design, `--print-tuple`): `| a1 | sonnet | a26d972ecc0b4ee7 | 754655a400f360e6 |
advocate:c41d7f5787bdb472+artifact:f5ed259652e65ee2+geometry:a293ddddce11ee4a+morphology:26bde57ad0478237+arbitrator:44542114399ab277 |
jwst_v1 | 28737c6083dc1978 | 032a302c84f3cbe7 | none | off | default | 1 | a40ae6e201a03e65 |`
(the frozen advocate; letters provisional — the freeze below post-dates this run by design).
`outputs/design/preds_truth_a1_sonnet_design_pass1.parquet` (+ `_votes.parquet`, `.meta.json`),
traces `outputs/design/traces_truth_a1_sonnet_design_k1_r1/`. **288/288 scored, 0 parse
failures** (repair retries included), 0 items over the $0.17 cap, **$18.735** ($0.0651/item,
median $0.026, max $0.147). `audit_traces` 759 events, **0 violations**. Critics engaged on
118/288 (41%: 27/44 positives, 54/200 negatives, 19/20 stress_D); arbitrator called on 117
(the one engaged item without a named critic went unarbitrated). 76 frame units marked
`kind=eval` (run tag `truth_a1_sonnet_design_k1_r1`). `analyze_truth.py --split design
--baseline a2` (canonical-name symlinks) agrees with the pandas analysis on every number.

**Monitors (pre-registered thresholds) — all PASS, no "fails badly" trigger, so the one
permitted critic/arbitrator wording round was NOT used and no rerun was made:**

| monitor | value | rule | verdict |
|:--|:--|:--|:--|
| no_opinion artifact / geometry / morphology | 0.059 / 0.000 / 0.008 (n 118) | ≤ 0.35 each | PASS |
| forbidden-ground rate (`forbidden_only`, structured) | **0.003** [0.000, 0.018] (n 316 refutations) | < 0.02 | PASS |
| D-rate on design N1 | **0.580** (116/200) [0.508, 0.649] | ≥ 0.50 | PASS |
| parse-failure rate | 0 [0, 0.013] | report | — |
| cost/item | $0.0651 | report | — |

Beside the headline monitor: category shares (a mention, not a sole ground) over_subtraction
0.136 (`_forbidden` 0.085 — artifact 0.280), theta_e 0.019 (`_forbidden` 0.016), colour_only
0.060 (`_forbidden` **0.000**); sanctioned channels `uses_subtraction_residual` 0.066,
`uses_scale_tension` 0.003; `locates_feature` 1.000. The incumbent's same ruler (replay,
1,044 refutations): forbidden_only 0.052, over_subtraction 0.386, colour_only 0.156,
theta_e 0.071 — the new critics argue from structure and location, not the banned priors.

**Metrics (positives = 44 cowls+lit, anchors excluded; negatives = 200 design N1):**

| score | AUC | recall@5%FPR (CP95) | thr | recall@10%FPR | P3 A/B (prov. letters) |
|:--|--:|:--|--:|:--|--:|
| **S (primary)** | 0.644 | **0.250** (11/44; 0.13–0.40) | 0.132 | 0.250 | 0.114 |
| **S_arb** | 0.652 | **0.295** (13/44; 0.17–0.45) | 0.124 | 0.295 | 0.114 |
| p_evidence (a1's advocate) | 0.727 | 0.205 | 0.420 | 0.341 | — |
| a2 iter2 p_ev (same items) | 0.728 | 0.295 | 0.350 | 0.364 | 0.159 |

Paired (analyze_truth): a1 vs a2 ΔAUC −0.084 [−0.188, +0.015], DeLong p 0.108; at 5%FPR 2
promotions / 4 demotions (McNemar 1-sided 0.89). a1arb vs a1: +2 promotions / 0 demotions,
ΔAUC +0.009 [−0.005, +0.026]; S_arb ≥ S on every arbitrated row but one (38 raised, 78 equal,
1 lowered). Per-stratum recall at S's global 5% threshold: cowls 3/15 (strong 0/3, marginal
1/4, weak 2/6), lit_galaxy 3/13, lit_cluster 5/16, θ_E ≤1″ 2/8, off-centre 4/14, gray
layouts 0/6. Spearman(S, θ_E) on COWLS n=12: −0.171 (perm p 0.607; not monotone-negative).
Letters on negatives (provisional): D 116 / C 84, FPR(A) 0%, FPR(A/B) 0%. `D_rate_stress_D`
0.05 (letters C 18 / B 1 / D 1 — the stress panel keeps its items; reported, never gating).
needs_human 17.4% overall (29.5% of positives, 6.5% of negatives).

**What the critics did (the design's question for this pass):**
- They DO discriminate on the items they cover: engaged negatives keep ×0.226 of p_ev on
  average, engaged positives ×0.367. The worst a2 false positive — the barred two-arm spiral
  at p_ev 0.92 that a2 lettered A — is crushed to **S 0.058 → D** by an upheld `spiral_arm`
  critic; only 5 of a1's top-10 negatives were in a2's top-10.
- But the compression squeezes refuted positives BELOW the un-engaged band: a positive at
  p_ev 0.35–0.65 refuted to 0.10–0.27 ranks below never-engaged negatives sitting at
  0.12 (the 2-item advocate cap), while seven partially-covered negatives keep S 0.13–0.46.
  Net: S recall@5%FPR 0.250 vs a2's 0.295, AUC −0.084. The arbitrated product repairs part
  of it (S_arb 0.295 = a2's recall) because overruled critics drop out of the product — the
  rank-16 anchor is the extreme case: S 0.127 (geometry+morphology named edge-on-disk) but
  the arbitrator overrules both from the image ("the feature is visibly curved in all direct
  panels … the tick-centre object is a compact reddish galaxy distinct from the bright blue
  arc") → **S_arb 0.970, letter_arb A**.
- Companion/merger adjudication dominates: `companion_projection` is 27/56/50 of the
  artifact/geometry/morphology alternatives, `merger` 9/27/24. Rulings: upheld 195 / partial
  118 / overruled 30 — the arbitrator overrules critics on 26% of arbitrated positives'
  rulings vs 4% of negatives' (it reads the image, and it is the scheme's best lever).

**Design COWLS (why 3/15):** the five `items: []` advocate records are the same five rows as
iteration 2 (advocate frozen ⇒ p_ev 0.03–0.05, letter D, critics never called). Of the six
engaged COWLS, every one had all three critics name alternatives, 16/18 rulings upheld:
- strong θ_E 0.60″ (p_ev 0.25 → S 0.104, llm D): geometry `companion_projection` upheld —
  "the NE (~PA 45°, ~1.8″) and SE (~PA 145°, ~2.0″) compact knots are separated by ~100° in
  position angle, not the ~180° a two-image lens geometry requires; no source at the SW
  antipode"; morphology adds "no stellar bridge"; artifact calls the NE diffuse envelope
  deep-stretch noise.
- weak θ_E 0.56″ / weak 0.32″ / provenance (p_ev 0.15–0.30 → S 0.048–0.158, llm D on all
  three): items are 1–3.5″ compact sources + one panel-(f)-only residual each; artifact
  `subtraction_residual` upheld on the (f)-only item (admissible: absent from direct panels),
  geometry+morphology `companion_projection` upheld on the rest ("no tangential elongation,
  no counter-image at the conjugate radius").
- weak θ_E 0.64″ (p_ev 0.55 → S 0.162 / S_arb 0.230, llm C, needs_human): the one engaged
  COWLS where items survive — geometry `merger` partial ("a visible tidal bridge"), 2 of 4
  items stand.
- marginal θ_E 1.86″ (p_ev 0.65 → S 0.272 / S_arb 0.330, llm C, needs_human): artifact's
  `diffraction_spike` OVERRULED by the arbitrator ("the feature shows clear curvature …
  inconsistent with a straight diffraction spike"); geometry/morphology `edge_on_disk`
  partial (the 10″ row shows an edge-on neighbour whose tip reaches the item's radius); the
  southern knot survives.
The pattern: located COWLS evidence is compact knots without measurable curvature, and
"companion, ~100° apart, no counter-image" is a competent, image-true refutation that
happens to be wrong about COWLS-faint lenses. This is not a wording failure of the critics
(their arguments point at pixels and use sanctioned grounds); it is the θ_E ≤ 1″ saturation
/ resolution regime recorded at the iteration-2 freeze (render lever, cf. a3).

**Top-10 S negatives:** top-1 S 0.461 (p_ev 0.62, llm B, needs_human — a diffuse LSB arc-like
sweep at 1.5–2″ SE the critics only partially cover; the single negative above t_A), then
0.192–0.120: a faint ring-like halo (llm D), LSB extensions + companions (llm C/D), the
near-complete oval ring matched to the host's own inclined-disc isophotes (llm D), a tidal
plume pair (llm C), an edge-on needle (all three critics: straight, dust lane, no curvature),
and three un-engaged 2-item records at the p_ev 0.12 cap. Eight of ten carry partial
coverage or an un-covered item — coverage, not the echo, is what keeps them high.

### (b) Threshold freeze (t_A / t_B on the design negatives' S) + regenerated drop-in

`t_A = 0.192` (smallest S with design-N1 FPR ≤ 1%: 2/200 at S ≥ t_A; achieved 0.0100),
`t_B = 0.1318` (FPR ≤ 5%: 7/200 = 0.035 — the next ROC step, the 0.12 tie-band, is 15/200).
Written into `golden/thresholds_v2.json › sonnet_api` (tau0 0.15 unchanged);
`thresholds_sha16` **`a40ae6e201a03e65 → 94d31c7b6979e0ca`** (`letter_source
sonnet_api_calibrated`). Implied design letters (assign_letter rules on the stored terms,
anchors excluded): positives A 7 / B 4 / C 20 / D 13 → **recall at A 15.9%, at A/B 25.0%
(11/44 [0.13, 0.40]; the incumbent's P3 reference is 0/31 COWLS)**; by class cowls A1/B2/C5/D7,
lit_galaxy A3/C9/D1, lit_cluster A3/B2/C6/D5; negatives A 1 / B 6 / C 77 / D 116 → **FPR(A)
0.005 [0.000, 0.028], FPR(A/B) 0.035 [0.014, 0.071]** (the A-guard blocks 1 of the 2
negatives above t_A); stress_D A 8 / B 5 / C 6 / D 1 (partial refutation leaves S in the A
band on the PI-refuted panel — reported, not gating, and a flag for the holdout read-out).
`make_verifier_patch.py` regenerated `golden/verifier_patch/` with the frozen file:
`aggregate_v2.py` byte-identical (sha16 `37be83b4598a36a7`), `git apply --check` passes,
Nate's chain now resolves `sonnet_api` → letters stamped `sonnet_thresholds_uncalibrated`
(C5 as designed). Two test files updated for the freeze (behaviour under test unchanged,
recorded here): `tests/test_golden_aggregate.py` — the file-content pin now expects the
frozen sonnet_api block (and its resolution `sonnet_api_calibrated`); `_thr()` now returns
the module-provisional resolution explicitly so the assemble/to_row letter pins stay
threshold-independent; `tests/test_golden_truth_runner.py` — `make_synthetic_repo` resets
the copied `thresholds_v2.json` to an uncalibrated sonnet_api so the rank-report test still
exercises both the provisional and the frozen paths.

### (c) Test–retest (a1, k=2, 40-item design subset, $5.78)

Ids `outputs/design/retest_ids.txt` — deterministic rule, recorded: per positive class
(cowls 7 / lit_galaxy 6 / lit_cluster 7) evenly-spaced quantiles of the pass-1 S order (so
the subset spans the letter bands instead of piling up at S 0.04); negatives = the top-10
pass-1 S (the only band where boundary flips can occur) + 10 evenly spaced over the rest.
Tuple = the a1 design tuple with `k 2` and the frozen `thresholds_sha16 94d31c7b6979e0ca`;
`outputs/design/retest/preds_truth_a1_sonnet_design_retest_r{1,2}.parquet` (+ votes/meta;
traces `traces_truth_a1_sonnet_design_k2_r{1,2}`; audit 118 + 111 events, 0 violations;
80/80 parse OK; letters `sonnet_api_calibrated`).

Letter flip rate (frozen letters): overall 8/40 = **0.20**; per boundary (items with a letter
adjacent to the boundary in either replicate): **A/B 1/11 = 0.091, B/C 3/24 = 0.125, C/D
4/32 = 0.125**. `grade_flip_rate` helper: grade 0.20, lens-call 0.10, mean |ΔS| 0.043.
S stability: ICC(1,1) **0.910**, Pearson 0.916, Spearman 0.739 (the 0.04 tie-band depresses
rank correlation), median |ΔS| 0.020; p_evidence Spearman 0.916. One |ΔS| ≥ 0.3 (a
lit_cluster A→A, S 0.43→0.86 — a critic engaged in one replicate only). By class: cowls 1/7
flips, lit_galaxy 0/6, lit_cluster 3/7, negatives 4/20.
**Pre-registered k rule: flip rate at A/B (0.091) and B/C (0.125) both ≤ 0.25 → NOT
triggered; production stays k=1 (no k=3 on a contested band).**

### (d) Holdout tuples (printed with `--print-tuple`; **NOT registered** — registration is the PI's step) and cost estimates

| arm | tuple row (all: splits 032a302c84f3cbe7, thinking off, effort default) | est. cost (282 items) |
|:--|:--|--:|
| a0 inspector | `a0 · sonnet · f2bb5294e1d89505 · e3b0c44298fc1c14 · artifact:2e5627171e2ce69c+morphology:e5b545f9957120ac+geometry:a672a7dfc227288e+artifact@noclaim:d7683d3099eca7ad+morphology@noclaim:cf92400576d45808+geometry@noclaim:d205721a8e07d03f · jwst_v1 · 28737c6083dc1978 · inspector · k1 · 94d31c7b6979e0ca` | $6.5 ($0.023/item) |
| a1 | `a1 · sonnet · a26d972ecc0b4ee7 · 754655a400f360e6 · advocate:c41d7f5787bdb472+artifact:f5ed259652e65ee2+geometry:a293ddddce11ee4a+morphology:26bde57ad0478237+arbitrator:44542114399ab277 · jwst_v1 · 28737c6083dc1978 · none · k1 · 94d31c7b6979e0ca` | $18.4 ($0.0651) |
| a2 | `a2 · sonnet · a26d972ecc0b4ee7 · 754655a400f360e6 · advocate:c41d7f5787bdb472 · jwst_v1 · 28737c6083dc1978 · none · k1 · 94d31c7b6979e0ca` | $6.8 ($0.0241) |
| R2 = a2 k3 | same as a2 with `k 3` | $20.4 |
| a3 | `a3 · sonnet · a26d972ecc0b4ee7 · 754655a400f360e6 · advocate:c41d7f5787bdb472 · jwst_v2r · 25dd6cc680579747 · none · k1 · 94d31c7b6979e0ca` | $6.8–7.1 (a2 rate + v2r desc) |
| a1 opus | `a1 · opus · a26d972ecc0b4ee7 · 754655a400f360e6 · (a1 shas) · jwst_v1 · 28737c6083dc1978 · none · k1 · a40ae6e201a03e65` | $30.7 (Sonnet × 1.67) |

Holdout subtotal ≈ **$89.6**, +10% overhead ≈ **$98.5** (the gated arm stays cut: design
Δrecall@5%FPR never reached +0.10). Flag for the PI before registering: the **a1-opus tuple
resolves provisional thresholds** (`opus_api` is null; MODEL_KEYS keys the API Opus to its
own entry), so its registration must either state `--allow-provisional-thresholds` in the
note or the PI freezes `opus_api` (e.g. to the sonnet numbers) first — P2 is registered for
the Sonnet letters either way. Second flag: on the design half the primary S UNDER-ranks
the advocate-only a2 (ΔAUC −0.084, recall 0.250 vs 0.295) while S_arb matches a2 and beats
S — the registered holdout read-out (P1 on S primary, S_arb secondary, A3−A2 render effect)
covers exactly this comparison; no design-side change is proposed.

### (e) Suite, ledger, spend

Full suite **212 passed** post-freeze (the two test edits in (b) included);
`test_golden_prompts.py` 11/11 (this entry lexicon-checked: `audit_traces --check-text` 0
hits, 298 entries). No prompt file changed this step (advocate frozen at
`6a806fbec212eb19`; persona set `a26d972ecc0b4ee7`). Registry: 76 units re-marked eval
(pass1) + 18 units × 2 (retest tags `truth_a1_sonnet_design_k2_r{1,2}`). No holdout split
touched, no tuple registered, no golden kit/keys/labels touched, no embargoed source read,
no git commit. **Spend this step $24.52** (a1 pass1 $18.74 + retest $5.78 by parquet
`cost_usd`); cumulative design-phase **$38.48** of the $95 cap ($0.92 A0 + $6.12 iter1 +
$6.93 iter2 + $24.52; ≈ $38.7 actual with the iter2 kill overhead).

## 2026-08-23 — HOLDOUT: all six registered tuples scored once; pre-registered verdict NOT met (P1 fails; P2 and forbidden-ground pass); the ranking gain is real and replicates

**Runs (STEP 2, order as instructed; every tuple from `REGISTRY.md › Truth-eval registered
arms`, scored exactly once, whole holdout half, default `--out`, `--concurrency 4`, Anthropic
API, thinking off; no rescore, no prompt/threshold/aggregator edit, no `--limit`).** 282 items
per arm (42 positives = cowls 16 / lit_galaxy 13 / lit_cluster 13; 200 N1 negatives; stress_D
20 / stress_U 15 / anomalymatch 5). All eight parquets' `.meta.json` tuples verified against
the registered rows, `rescored=False` everywhere; every run completed in one invocation.
`audit_traces` on every trace dir: **0 violations** (a0 846 events with inspector claims
attached 62/282, 0 blanked; a1 722; a2/a3/R2 282 each; opus 378). Frame units marked
`kind=eval`. Costs by parquet `cost_usd`: a0 $5.27 ($0.019/item) · a2 $6.80 ($0.024) · a3
$7.69 ($0.027) · a1 $17.76 ($0.063) · R2 $20.47 ($6.87+$6.79+$6.81) · a1-opus $17.28
($0.061; 24 items over the $0.17 cap, max $0.281, warn-and-count per policy; Sonnet arms 0
over cap). **Holdout spend $75.26** of the $110 phase cap.

**Analysis (STEP 3, zero-API).** `analyze_truth.py --split holdout --model sonnet --baseline
a0` → `outputs/truth_results.csv` (526 rows) + `truth_anchors.csv` (0 rows — all five anchors
are design-half, as registered) + `truth_summary_holdout.md`; re-running reproduces all three
files byte-identically. a1-vs-a2 pairing from a `--baseline a2` pass to scratch; Opus
secondary via `--model opus --allow-incomplete` to scratch. ICC from a scratch script
(registered secondary `analyze_truth` does not emit); everything else below is
`analyze_truth` output.

### P1 — recall of holdout positives at 5 % FPR on holdout N1 (primary; CP 95 % CIs)

| arm | recall@5 %FPR [CI] | thr (achieved FPR) | recall@10 %FPR | AUC |
|:--|:--|:--|:--|--:|
| a0 (baseline) | **9.5 %** (4/42) [2.7, 22.6] | 1/3 (2.5 %) | 9.5 % | 0.535 |
| **a1 S (primary)** | **19.0 %** (8/42) [8.6, 34.1] | 0.125 (3.0 %) | 19.0 % | 0.641 |
| **a1 S_arb (co-registered)** | **21.4 %** (9/42) [10.3, 36.8] | 0.128 (5.0 %) | 21.4 % | 0.641 |
| a2 | 16.7 % (7/42) [7.0, 31.4] | 0.38 (4.5 %) | 33.3 % [19.6, 49.5] | 0.725 |
| R2 = a2 k=3 mean | 23.8 % (10/42) [12.1, 39.5] | 0.307 (5.0 %) | **47.6 %** [32.0, 63.6] | **0.738** |
| a3 (v2r render) | 14.3 % (6/42) [5.4, 28.5] | 0.45 (4.5 %) | 28.6 % | 0.720 |
| a1 opus S / S_arb | 21.4 % [10.3, 36.8] / 19.0 % [8.6, 34.1] | — | same | 0.696 / 0.694 |

A0's full 4-point ROC (registered): (FPR, TPR, thr) = (0, 0, ∞), (0.005, 0, 1.0),
(0.010, 0.024, 2/3), (0.025, 0.095, 1/3), (1, 1, 0).

**Paired (exact McNemar 1-sided on positives; dAUC paired bootstrap 2,000/seed 2026 + DeLong):**

| comparison | prom. | dem. | McNemar p | dAUC [CI] | DeLong p |
|:--|--:|--:|--:|:--|--:|
| a1 vs a0 | 8 | 4 | 0.194 | +0.106 [−0.005, +0.216] | 0.060 |
| a1arb vs a0 | 8 | 3 | 0.113 | +0.107 [−0.005, +0.214] | 0.056 |
| a2 vs a0 | 6 | 3 | 0.254 | +0.190 [+0.110, +0.270] | 7.1e−06 |
| a2k3 vs a0 | 9 | 3 | 0.073 | +0.203 [+0.112, +0.288] | 2.2e−05 |
| a3 vs a0 | 5 | 3 | 0.363 | +0.185 [+0.092, +0.274] | 1.6e−04 |
| a1 vs a2 | 3 | 2 | 0.500 | −0.084 [−0.193, +0.008] | 0.097 |
| a1arb vs a2 | 4 | 2 | 0.344 | −0.084 [−0.189, +0.005] | 0.090 |
| a3 vs a2 (render effect) | 3 | 4 | 0.773 | −0.005 [−0.085, +0.074] | 0.896 |
| a1arb vs a1 | 1 | 0 | 0.500 | +0.000 [−0.015, +0.016] | 0.976 |

**P1 verdict: NOT met.** a1 S recall 0.190 < 0.5 and its CI [8.6, 34.1] overlaps A0's
[2.7, 22.6]; S_arb likewise. a1 beats a0 numerically on every ranking metric, but the
registered bar is recall ≥ 0.5 with CI excluding A0's.

### P2 — design-frozen letters hold their FPR on holdout (registered test: CP 95 % LOWER
bound ≤ 1 % at t_A = 0.192, ≤ 5 % at t_B = 0.1318; a1 Sonnet letters, thresholds from the
rows' own values)

| arm | FPR at t_A [CI] | holds | FPR at t_B [CI] | holds | plan's upper-CI wording (beside) |
|:--|:--|:--|:--|:--|:--|
| **a1 (registered)** | 2.0 % (4/200) [0.55, 5.04] | **PASS** | 2.5 % (5/200) [0.82, 5.74] | **PASS** | t_A fail (5.04 > 2.5), t_B pass (5.74 ≤ 7.5) |
| a1arb | 2.5 % [0.82, 5.74] | pass | 4.0 % [1.74, 7.73] | pass | both fail |
| a2 / a2k3 / a3 (same t on raw p_ev) | 18.5 / 11.0 / 22.0 % | fail | 25.0 / 25.5 / 31.5 % | fail | expected by construction: t_A/t_B were frozen on the full-stack S, not on p_evidence |
| a1 opus (provisional 0.80/0.50) | 0 % [0, 1.83] | pass (trivial) | 0 % [0, 1.83] | pass (trivial) | P2 was registered for the Sonnet letters |

**P2 verdict (registered a1 letters): PASS on both thresholds.** Letter-rule FPRs (A-guard
included): a1 FPR(A) 2.0 %, FPR(A/B) 2.5 %; a1arb 2.0 / 4.0 %; a2 2.0 / 25.0 %; a2k3 1.5 /
26.5 %; a3 2.0 / 31.5 %; a0 0.5 / 1.0 %.

### P3 — fraction of holdout positives at A/B (old scheme: 0/31 COWLS)

a0 2.4 % (1/42) [0.06, 12.6] · **a1 16.7 % (7/42) [7.0, 31.4]** · a1arb 21.4 % [10.3, 36.8]
· a1 opus 2.4 % [0.06, 12.6] (provisional letters; the single A/B is a B — **Opus assigns no
A at all on the holdout**, positives or negatives) · a2 / a2k3 / a3 50.0 / 59.5 / 69.0 %
(letters not calibrated for these arms; their FPR(A/B) runs 25–32 %, see P2).

### Overall pre-registered rule — "better" = P1 recall ≥ 0.5 with CI excluding A0 AND P2
holds AND forbidden-ground < 2 %: **NOT met** (P1 fails; P2 passes; forbidden-ground passes).

### Registered secondaries

**Per-stratum recall@5 %FPR (a1 S | a1arb | opus S):** cowls 3/16 | 3/16 | 3/16 (strong 0/2
all; marginal 2/4 | 2/4 | 1/4; weak 1/7 | 1/7 | 0/7; provenance 0/3 | 0/3 | 2/3); lit_galaxy
**5/13 | 6/13 | 6/13** — the COSMOS-Web 1727 stratum the design never saw at galaxy scale is
the top stratum; lit_cluster 0/13 all; θ_E ≤ 1″ 2/7 | 2/7 | 1/7; θ_E 1–2″ 0/3 | 0/3 | 1/3;
galaxy-scale 8/29 | 9/29 | 9/29 vs cluster-scale 0/13 all; centre 8/35 | 9/35 | 9/35 vs
off-centre 0/7 all; field cosmos 7/27 | 7/27 | 8/27, cluster 1/14 | 2/14 | 1/14.

**Forbidden-ground (structured record, `reason_audit`):** a1 Sonnet `forbidden_only` **0.007**
[0.001, 0.023] (n = 305 refutations) < 2 % **PASS** (per-critic artifact 0.021 / geometry
0.000 / morphology 0.000); opus a1 0.000 [0, 0.061] PASS (n = 59). Sanctioned channels a1:
`uses_subtraction_residual` 0.098, `uses_scale_tension` 0.000; `colour_only_forbidden` 0.000;
`locates_feature` 1.000. Incumbent a0 on the same ruler (n = 817): forbidden_only 0.075,
over_subtraction 0.443, colour_only 0.207 (forbidden 0.021), locates_feature 0.322.

**no_opinion per role (≤ 35 %):** a1 Sonnet 0.045 / 0.009 / 0.009 (artifact / geometry /
morphology; n = 110 engaged) PASS; opus 0.083 / 0.042 / 0.000 (n = 24) PASS.

**Spearman(S, θ_E) on COWLS (n = 10 with radii; must not be monotone-negative):** a1 −0.131
(perm p 0.727), a1arb −0.094 (0.800), a2 +0.111 (0.761), a2k3 +0.332 (0.345), a3 −0.412
(0.236), opus +0.013 (0.982) — none significantly negative: PASS everywhere.

**R2 replicates (a2 k=3, whole holdout):** letter flip rate 0.264 (replicate pairs),
lens-call flip 0.121, mean |ΔS| 0.052; **ICC(1,1) on S = 0.862** (n = 282, k = 3; pairwise
Pearson 0.840–0.902, Spearman 0.820–0.863).

**Stress panels (reported, never gating).** D-rate on stress_D (n = 20): a0 0.75
[0.51, 0.91]; **a1 Sonnet 0.05** [0.001, 0.249]; opus a1 0.35 [0.15, 0.59]; a2/a2k3/a3 0.00
[0, 0.168] — the design-phase flag holds on holdout: partial refutation leaves the PI-refuted
panel out of the D band (Opus is 7× stricter than Sonnet here). Stress_U at A/B (n = 15): a1
20 %, a1arb 33 %, a2/a2k3 80 %, a3 87 %, opus 0 %, a0 7 %. AnomalyMatch at A/B (n = 5): a1
20 %, a2/a2k3 60 %, a3 40 %, opus 0 %, a0 0 %. Escalation (needs_human): a1 15.6 %, opus
2.5 %, a0 6.7 %, advocate-only arms 0 (no channel).

**Parse-failure rate 0 [0, 1.3 %] in every arm** (S = NaN nowhere); cost/item as above.

### Anchors (design-only, never truth; holdout `truth_anchors.csv` empty as registered).
Design-half observations against the written predictions (full-stack a1 / advocate a2):

| rank | written prediction | observed | hit? |
|:--|:--|:--|:--|
| 15 | letter A or B | a1 **B** (S 0.782, diffraction_spike critic partial; letter_llm A); a2 A 0.85 | **HIT** |
| 13 | letter D with `spiral_arm` upheld | a1 **C** (S 0.277; `spiral_arm` named and upheld, but p_ev 0.92 kept S above the D band); a2 A 0.88 | **MISS** (recorded at registration: the critic fired as predicted, the compression did not reach D) |
| 7 / 14 | ‖Δletter‖ ≤ 1 on the same SW-only field | a1 C / C (S 0.005 / 0.033, merger critics upheld); a2 C / C | **HIT** (Δ = 0) |
| 16 | scale_class = cluster, deflector_is_centre = false, letter not D | a1 letter C, S 0.127 → **S_arb 0.97, letter_arb A** (arbitrator overrules edge-on-disk critics from the image); scale_class **group**, centre **true** | letter-not-D **HIT**; scale-class and centre **MISS** (the 10″ stamp shows the arc at ~3″ from the tick — "group" is what the image supports) |

### Honest read-out

1. **The pre-registered composite verdict is NO**: P1's recall ≥ 0.5 bar is far away
   (0.19–0.24 across all arms). Everything else in the rule passes (P2 on the frozen a1
   letters; forbidden-ground 0.7 %).
2. **The design-phase S-under-ranks-a2 finding replicated**: full-stack AUC 0.641 vs
   advocate-only 0.725 (a1 vs a2 dAUC −0.084 [−0.193, +0.008]); a2's gain over a0 is the one
   that is significant (dAUC +0.190, DeLong 7e−06, CI excludes 0) while a1's is not (+0.106,
   p 0.060, CI includes 0). The arbitrator repair helps only marginally here (+0.024 recall,
   dAUC +0.000) — it does NOT recover parity with a2 as it did on design. **Averaging three
   advocate replicates is the best ranker on the holdout** (AUC 0.738, recall@10 %FPR 47.6 %).
3. **The render effect is null**: a3 − a2 dAUC −0.005 [−0.085, +0.074]; a3 COWLS 0/16 at 5 %
   FPR — the v2r signed-χ panel did not unlock the θ_E ≤ 1″ regime.
4. **The frozen letters transfer only for the arm they were calibrated on**: a1 FPR(A) 2.0 %,
   FPR(A/B) 2.5 % on holdout N1, both within the registered bound; the same numbers applied
   to raw p_evidence run 11–32 % — a calibration statement for Nate's drop-in, not a surprise.
5. **The trade-off the secondaries surface**: the a2-family letters put 80 % of stress_U and
   60 % of AnomalyMatch at A/B (un-vetted candidates score high when no critic ever looks),
   while a1 tempers both to 20 % at FPR(A/B) 2.5 % — the full stack is the FPR-controlled
   instrument, the advocate score is the better ranker, and the drop-in ships both (S ranks,
   letters gate).
6. **Opus a1 (provisional letters)** sits between Sonnet a1 and a2 on ranking (AUC 0.696) and
   is much more conservative: no A letters at all, P3 2.4 %, escalation 2.5 %, stress_D
   D-rate 0.35, forbidden grounds 0. 24 items exceeded the per-item cost cap (max $0.281).

### Ledger, suite, spend

Verifier patch: thresholds and aggregator unchanged this phase (`thresholds_v2.json`
untouched since the design freeze; `aggregate_v2.py` sha16 `37be83b4598a36a7`);
`golden/verifier_patch/` regenerates byte-identically from the generator (scripts and tests
verified before the README edit), `git apply --check` passes on a scratch copy of J; the
holdout headline paragraph was added to the patch README via `README_TEMPLATE` +
regeneration (README.md and its embedded copy in `ADD_FILES.patch` are the only changed
bytes). Suite after the write-up: **211 passed, 1 failed** — the failure is
`test_golden_truth_runner.py::test_registry_md_truth_sections_and_anchors`, the stale
empty-table pin (`Truth-eval registered arms == []`) flagged at registration and left
un-edited per the phase's no-edit rule; it asserts the pre-registration state of
`REGISTRY.md` and fails now that six arms are (correctly) registered. Post-phase cleanup,
PI-approved, should re-pin it to the registered rows. Lexicon (`audit_traces --check-text`,
298 entries): 0 hits on this entry, the TRUTH_EVAL status section and the patch README.
No golden kit/keys/labels touched; no embargoed source read; no git commit.

**Spend: holdout phase $75.26** (a0 5.27 + a2 6.80 + a3 7.69 + a1 17.76 + R2 20.47 + opus
17.28) of the $110 phase cap; analysis zero-API. **Part 2 program total $113.74** (design
$38.48 + holdout $75.26; ≈ $114.0 actual with the iter2 kill overhead).
