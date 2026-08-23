# `lensjudge/golden/` — the golden single-grader dataset (Xiaosheng Huang, JWST NIRCam)

Every LensJudge generation so far was scored against labels with no measurable human
ceiling: a two-author DESI consensus (exact agreement 55.4 %, QWK 0.42) or committee letters
with no per-rater identity. This package creates the missing thing — **one named expert's own
grade + confidence on a curated, blind, shuffled JWST frame with hidden byte-identical
repeats** — and wires it into the three consumers that need it:

1. the program's first **intra-rater human ceiling** (E0: pass 1 vs pass 2 on 40 repeats);
2. **in-context exemplars** for Claude graders that say "this is what *one named expert* calls
   an A" (E2), and **SFT labels** for open-weight students (E4/E5);
3. the **pre-registered align / validate** experiment Xiaosheng asked for (E1–E3).

Plan of record: `~/.claude/plans/i-want-to-explore-golden-seal.md` (approved 2026-08-22).
Running log: `lensjudge/GOLDEN_FINDINGS.md`. Pre-registration ledger: `golden/REGISTRY.md`.

Conventions (pinned in every module): the human record is `score_1_4` ∈ {1,2,3,4} +
`confidence_lmh` ∈ {L,M,H} ("how sure are you of this score", *not* P(lens)); letters via
`_util.score_to_letter` (4→A … 1→D, strict — never through `ImageGrade`, whose validator
coerces unknowns to D); `grade_scale="huang_vi_1to4"`. The JWST run's own A/B/C/D/U is a
persona **pass-count** and is always a column named `pipe_grade_passcount`, never `grade`.
Seeds: frame / splits / bootstrap `2026`; kit shuffle + item ids `20260822`. Small CSV/JSON
files under `golden/` are tracked with a `.sha` sidecar (`_util.pin` / `_util.read_pinned`);
bulk artifacts (`stamps/`, `kits/`, `finetune/corpus_golden/`) and the two embargo files
(`banned_lexicon.txt`, `pi_comments.txt` — the PI's 16 verbatim document comments, one per
line, read only by `audit_traces.py --build-lexicon`, pinned by count + sha16 in code) are
gitignored. Nothing here writes into the JWST run repo (`J`, read-only).

## What Xiaosheng sees and does (the protocol)

He receives one folder, `kits/jwst_lite_v1/` (built by `build_kit.py`; never the key):

- `grade.html` — single-file tool, one image per screen, **752×540** (the run's six-panel v1
  composite with the footer strip — id, RA/Dec, magnitude, programme — cropped off at
  y = 540). Header shows only "k / N". Keys: `1–4` score, `L/M/H` confidence, `Enter`/`Space`
  commit, `Backspace` revise the previous item (a new event with `revision+1`), `F` flag
  (rendering problem), `Z` 2× zoom, `E` export, `?` help. **No text field** — the record is
  score + confidence, nothing else. Legend = the rubric's absolute A/B/C/D text re-worded as
  4/3/2/1 under the banner "this is the Huang visual-inspection scale, not the pipeline
  pass-count" (`build_kit.LEGEND` / `CONFIDENCE`; wording to be approved by him before
  session 1). Timer runs from the image's `onload`. Progress lives in the browser's
  localStorage; every 20 commits (and at the end) the page downloads
  `events_<kit>_<session>.jsonl`; "Import events…" resumes on another machine.
- `items/NNN.jpg` — opaque 3-digit ids drawn **without replacement from 001…999** (a seeded
  draw, not a running count: a repeat added later takes an id from the unused remainder and
  is indistinguishable), in ONE seeded permutation of the whole frame (no blocks, so the
  order carries no stratum information). Every served file has one fixed mtime. JPEG quality
  is **92**, deliberately not the 93 of Nate's `top100_clean_scrambled/` set: he holds that
  folder with its own key, and `build_kit` asserts no served file is byte- (or size-)
  identical to anything in it (`assert_no_collision`). README.txt asks him to delete it.
- `grading_sheet.csv` (paper fallback: `item_id,score_1_4,confidence_lmh`), `serve.py`
  (optional stdlib server that also writes every commit to `records/`), `README.txt` (which
  **supersedes** Nate's `top100_clean_scrambled/` README — that set asked for arc position
  and θ_E; this one asks only for score + confidence).

Session 1 is the pilot: the first 30 exposures are timed (`drift_report.py --pilot-n 30`);
if the mean exceeds 75 s the pre-specified fallback is to drop `U_tail`/`N_unflagged` to
15/10 — `build_kit.py --kit-id jwst_lite_v1 --drop-units` retires UNGRADED items of those
strata as a new manifest version (graded items always stay; the survivors are a
deterministic hash core; retired key rows go to `keys/<kit>_dropped.csv`; the key is never
regenerated). After session 1 is ingested, `build_kit.py --add-repeats` inserts **40
byte-identical repeats** (new opaque ids, same JPEG bytes, same mtime) stratified on his
*observed* pass-1 scores, each ≥ 60 exposures after its original and in a later session;
nothing in the kit marks them. He grades in one pass, in the order shown, without the
annotated docx, the contact sheet or any catalogue. Recorded, not pretended away:
`prior_exposure` (2 = ranks 1–15 he annotated; 1 = ranks 16–100 on the contact sheet; 0
otherwise).

Hidden from him throughout: rank, `pipe_*`, literature status, `annotated/` arrows, DESI
strips, his own comments, the key.

## The frame (`frame.csv`, 250 rows, seed 2026)

| stratum | n | what |
|---|---|---|
| `T_verified` | 21 | top-100 with a persona pass-count letter (A5/B5/C11); rank 14 collapsed into rank 7 (1.17″ twin, `alias_ids`); ranks 16/17 (both C, literature-known) kept as two stimuli, same `system_id` (8.78″) |
| `T_U` | 78 | top-100 unverified (`U`) |
| `K_cowls` | 31 | COWLS literature lenses recovered by the pipeline (controls) |
| `L_known` | 30 | other SIMBAD/literature lenses ≤ 2″ (7 pipeline-D fixed + fill) |
| `D_refuted` | 40 | pipeline-D by `center_galaxy_type`: 10 merger / 10 ring+spiral / 10 high-conf elliptical near-miss / 10 other |
| `U_tail` | 30 | ranks 101–300 / 301–2024 |
| `N_unflagged` | 20 | never flagged, 13 proposals, excluding 2″ overlaps with the DESI parity pool + benches |

Layout 223 colour / 18 gray (LW only) / 9 gray (SW only — one of them, u0163, has an LW
observation whose cutout is entirely inside an F444W coronagraph mask, finite 0.0, so the
renderer dropped it; `layout` follows the renderer's gate, not observation presence);
`prior_exposure` 2:14, 1:85, 0:151;
`lit_known` 71 (COWLS, L_known, `discovery_status` known/field_match, or any row carrying a
literature `known_lens_name` — u0239, a SIMBAD-matched `U_tail` item, is literature-known
too); `desi_pool_overlap` 14 (13 `random_neg`, 1 graded — a DESI-trained student
was *taught* these are non-lenses; the registry marks them `leak=desi_train`). Two
multi-unit systems (ranks 16/17; rank-35 `J3807110-4434755` + `U_tail` `J3806901-4434926`)
always share a split half and are never repeats. Full counts: `frame_summary.md`.

## File map

| path | tracked | writes it | what |
|---|---|---|---|
| `_util.py` | yes | — | `pin`/`read_pinned`/`sha_*`/`hash01`/`safe_name`, score↔letter maps, `SEED`, `JWST_REPO` |
| `schema.py` | yes | — | pydantic `GradeEvent` (14 keys, `extra="forbid"`), `GradeRecord`, `GoldenLabel`; `events_from_jsonl`, `read_key` |
| `build_frame.py` → `frame.csv`, `frame_summary.md` | yes | WP-B | the lite frame from `J/results/*.csv` + `J/data/targets.parquet` + parity pool/benches |
| `../common/jwst_fetch.py` | yes | WP-A | vendored `J/scripts/util.py @ 4f81493` (RemoteImage, render_cutout, …) + FITS/JPEG helpers; a test diffs the block against the origin |
| `build_stamps.py` → `stamps/<id>/`, `stamps_manifest.csv`, `stamps_check.csv` | manifest yes, pixels no | WP-A | SW/LW FITS at 10″/320 px (+20″/640 px) and the byte-faithful v1 composite `<id>_v1.jpg`; `--check-against J/top100_clean` |
| `build_kit.py` + `tool/{grade_template.html,serve.py}` → `kits/<kit>/`, `keys/<kit>_key.csv`, `keys/<kit>_manifests.jsonl` (each version + the key pin it was written with), `keys/<kit>_dropped.csv` | kit no, keys yes | WP-C | blind kit + answer key; `--add-repeats`; `--drop-units` (pilot fallback); `assert_blind` greps the shipped HTML for every id/coordinate/sha/forbidden word; `assert_no_collision` vs the sets he already holds; `--force` deletes key + history |
| `collect.py` → `golden_grades.csv`, `golden_labels.csv` | yes | WP-C | events (+ sheet) ⋈ key; last revision wins; one row per (unit, pass) / per unit |
| `records/` | yes (dir) | the tool / serve.py | `events_<kit>_<session>.jsonl` — the raw commits; see `records/README.md` |
| `stats.py` → `outputs/golden_intrarater.csv`, `golden_agreement.csv` | no | WP-D | E0 ceiling (human_ceiling.pair + intergrader_stats + unsymmetrised QWK, drift raw AND inverse-probability-weighted to the pass-1 marginal, per-s1-level rows, subgroups by prior_exposure/stratum); `--agreement` XH vs pipeline (`[flagged_only]` and `[all]` incumbent AUC); `--kit-id`; `--simulate` CI-width design check |
| `analyze_golden.py` → `outputs/golden_results.{csv,json}` | no | WP-D | the registered endpoints (REGISTRY.md): E2−E1 paired ΔAUC at score ≥ 3 + DeLong p, Δpurity@recall 0.8, per replicate and pooled; absolute AUCs by stratum; E1 vs incumbent on flagged / all rows; QWK vs XH; self-consistency |
| `drift_report.py` → `outputs/golden_drift.csv` | no | WP-D | score/timing vs position & session; pilot verdict |
| `split_halves.py` → `splits.csv` (+ `outputs/golden_splits_desi_overlap.csv`) | yes | WP-D | by `system_id`, stratified (stratum × letter), matched score≥3, `prior_exposure=2` forced → align, 2″ firewall `n_bad==0` |
| `registry.py` → `golden_registry.csv` | yes | WP-D | per-unit exposure ledger; `sync`/`show`/`assert`/`mark`; `ExposureError` |
| `REGISTRY.md` | yes | hand-edited | pre-registered tuples for every validate run + rescore log |
| `build_eval_manifest.py` → `outputs/golden_jwst_manifest.csv` | no | WP-E | the exact 11 `build_eval_set` columns (`survey_key=jwst`, `source=golden_huang`, `binary_label` = score ≥ 3) + extras (`p_pipeline` with unflagged = 0, `pipe_flagged`, `binary_label_ge2`); only `run_batch --mode jwst` accepts its rows; `eval/lensbench_gate.py`'s verdict does not apply (it needs `source=random` rows) |
| `fewshot.py` | yes | WP-E | `build_exemplar_blocks(labels, key, n_per_grade, seed, embargo=True, eligible_units)`; refuses if `LENSJUDGE_FEWSHOT_MANIFEST` is set |
| `grader_jwst.py` + `../prompts/jwst_note.md` | yes | WP-E | composite + panel gloss → `grader_direct.grade_candidate(content=…)`; writes a `golden_content_audit` trace event |
| `run_golden_eval.py` → `outputs/preds_golden_{arm}_{model}_{split}_r{k}.parquet` (+`.meta.json` on completion, `traces_golden_*`) | no | WP-E | registry-gated E1/E2/E3 runner (tuple incl. system sha, thinking, effort; `assert_unexposed` before every validate run; system prompt lexicon-checked; interrupted replicates resume); `--smoke N [--smoke-stratum K_cowls]` (validate units excluded once splits exist) |
| `audit_traces.py` → `outputs/golden_audit.json`, `banned_lexicon.txt` | json no, lexicon no | WP-E | post-hoc check of every model context: no validate id / PI comment / grade string, exemplars ⊂ align, `n_images == 1 + n_exemplars`; `--build-lexicon` reads the gitignored `pi_comments.txt` |
| `build_corpus_golden.py` → `../finetune/corpus_golden/` | no | WP-F | ms-swift SFT JSONL from the ALIGN half only (train / 3 % val / 20 % per-letter `valsel` carve; confidence-shrunk soft targets, no dihedral aug); the validate half is written nowhere; `gate` refuses validate units |
| `build_desi_agreement_arm.py` → `desi_agreement/{agreement_manifest,fewshot_manifest_desi}.csv`, `../finetune/corpus_golden/sft_desi_agreement.jsonl` | csv yes | WP-F | DESI "golden-by-agreement" arm: Paper II `delSc==0 & pair_ok` (726) + consensus grade-D |
| `../tests/test_golden_{frame,jwst_fetch,kit,stats,model,sft}.py` | yes | all | plain-script tests, no network, no API |

## Campaign command sequence

All commands from `reproductions/` with `~/.venvs/lensjudge/bin/python` (`py` below).
Steps marked **[PI]** need Xiaosheng; **[$]** spend API money; **[net]** need the network.

```
# 0. safety
#    Perlmutter backup (see "Backup record" below); .gitignore already covers kits/stamps/corpus_golden.

# 1. frame  (deterministic; re-running must reproduce frame.csv.sha 422eacbcdcf3854d)
py lensjudge/golden/build_frame.py

# 2. stamps [net, ~30 min, public S3]  — FITS + v1 composites for all 250; pixel check vs the run
py lensjudge/golden/build_stamps.py --frame lensjudge/golden/frame.csv --workers 3 \
    --check-against /Users/benson/sync/research/jwst-strong-lens-search/top100_clean

# 3. kit  (ONE seeded shuffle + seeded opaque ids; the key is tracked, the kit is not; --force deletes the key
#    AND the manifest history, so it is only ever a pre-campaign operation)
py lensjudge/golden/build_kit.py --frame lensjudge/golden/frame.csv --kit-id jwst_lite_v1 \
    --grader-id XH --seed 20260822 --source-dir lensjudge/golden/stamps

# 4. [PI] pre-flight (<=15 min): legend wording, confidence definition, the scale, no docx/contact sheet.
#    Send lensjudge/golden/kits/jwst_lite_v1/ (zip it). Never send keys/.

# 5. [PI] session 1 (pilot).  He returns events_jwst_lite_v1_<session>.jsonl file(s) -> golden/records/
py lensjudge/golden/collect.py --kit-id jwst_lite_v1 --events "lensjudge/golden/records/events_jwst_lite_v1_*.jsonl"
py lensjudge/golden/drift_report.py --pilot-n 30          # mean >75 s -> pre-specified U_tail/N_unflagged cut:
py lensjudge/golden/build_kit.py --kit-id jwst_lite_v1 --drop-units   # (only if the verdict says so; before step 6)

# 6. repeats (between sessions only; he must reload grade.html afterwards)
py lensjudge/golden/build_kit.py --kit-id jwst_lite_v1 --add-repeats \
    --grades lensjudge/golden/golden_grades.csv --n 40 --min-gap 60
#    -> re-send kits/jwst_lite_v1/ (grade.html + new items/); his progress is kept.

# 7. [PI] remaining sessions (>=24 h after session 1).  Then:
py lensjudge/golden/collect.py --kit-id jwst_lite_v1 --events "lensjudge/golden/records/events_jwst_lite_v1_*.jsonl"

# 8. the ceiling (E0) + drift
py lensjudge/golden/stats.py --out lensjudge/outputs/golden_intrarater.csv
py lensjudge/golden/stats.py --agreement --out lensjudge/outputs/golden_agreement.csv
py lensjudge/golden/drift_report.py --out lensjudge/outputs/golden_drift.csv

# 9. splits + registry + lexicon (needs the gitignored golden/pi_comments.txt; validate runs refuse without the lexicon)
py lensjudge/golden/split_halves.py --seed 2026 --out lensjudge/golden/splits.csv
py -m lensjudge.golden.registry sync
py lensjudge/golden/audit_traces.py --build-lexicon            # -> golden/banned_lexicon.txt (gitignored)

# 10. eval manifest (binary_label = XH score >= 3, the registered endpoint; only --mode jwst accepts its rows)
py lensjudge/golden/build_eval_manifest.py --split all --out lensjudge/outputs/golden_jwst_manifest.csv
py lensjudge/imaging/run_batch.py --mode jwst --manifest lensjudge/outputs/golden_jwst_manifest.csv --limit 3   # plumbing: 3/3 parse

# 11. register BEFORE any validate call: paste the printed row into REGISTRY.md (the row carries the
#     system-prompt sha and the thinking/effort settings; pass the same flags you will run with)
py lensjudge/golden/run_golden_eval.py --arm e1 --model sonnet --print-tuple
py lensjudge/golden/run_golden_eval.py --arm e2 --model sonnet --n-exemplars 3 --print-tuple

# 12. [$] runs: align first (ungated, tuning), then validate (gated, scored once per tuple)
py lensjudge/golden/run_golden_eval.py --arm e1 --model sonnet --split align    --k 3
py lensjudge/golden/run_golden_eval.py --arm e1 --model sonnet --split validate --k 3
py lensjudge/golden/run_golden_eval.py --arm e2 --model sonnet --split validate --n-exemplars 3 --k 3
py lensjudge/golden/run_golden_eval.py --arm e3 --model sonnet --split validate --rubric lensjudge/prompts/rubric_jwst_huang.md --k 3

# 13. audit every trace dir (exit 1 on any violation), then the registered endpoints
py lensjudge/golden/audit_traces.py --traces-dir lensjudge/outputs/traces_golden_e2_sonnet_validate_r1 \
    --check-text lensjudge/prompts/rubric_imaging_v2.md --check-text lensjudge/prompts/jwst_note.md
py lensjudge/golden/analyze_golden.py --split validate \
    --e1 "lensjudge/outputs/preds_golden_e1_sonnet_validate_r*.parquet" \
    --e2 "lensjudge/outputs/preds_golden_e2_sonnet_validate_r*.parquet"   # -> outputs/golden_results.csv

# 14. SFT corpus (align half only: train / val / valsel carve; the validate half is written nowhere;
#     marks exposure kind "sft"). A student is scored on validate ONLY via step 12 with --arm e1 --model <served id>.
py lensjudge/golden/build_corpus_golden.py build --key lensjudge/golden/keys/jwst_lite_v1_key.csv
py lensjudge/golden/build_desi_agreement_arm.py                 # DESI arm (no PI time)
```

Plumbing check before any labels exist (run at integration 2026-08-23, $0.06):
`py lensjudge/golden/run_golden_eval.py --smoke 3 --smoke-stratum K_cowls --model sonnet`
(needs `ANTHROPIC_API_KEY`; tries to mark the 3 units exposed with `run_tag=smoke`, kind
`eval`, which only succeeds once the registry exists). **To-do after step 9's `registry
sync`:** `py -m lensjudge.golden.registry mark --units u0004 u0013 u0015 --run-tag smoke --kind eval`
(the three K_cowls units scored zero-shot at integration; kind `eval` does not block them
from being exemplars, it just records that Sonnet has seen them).

## EMBARGO — what may never enter a model context

The point of a per-rater label set is lost if the model has seen the rater's reasoning or
the answers it is being scored against. These are hard rules, enforced in code where
possible (`fewshot.check_embargo`, `build_kit.assert_blind`, `audit_traces.py`,
`registry.assert_unexposed`) and by hand otherwise:

1. **Xiaosheng's free text is off limits to every model-facing artifact**: the 16 Word
   comments in `agentic-lens-discovery/assets/JWST_top100_annotated.docx`, `J/results/notes/`,
   `J/annotated/`, and the meeting transcript(s). They may be read by people and by
   `audit_traces.py` (as *banned* strings in `banned_lexicon.txt`), never quoted, paraphrased
   or summarised into a prompt, rubric, exemplar header, SFT rationale or trace. The only text
   the few-shot path can emit is the fixed template (`FEWSHOT_LEAD`, the one-line header,
   `[composite]`, `FEWSHOT_TRAIL`); `embargo=True` raises on any deviation.
   *Design-only exception (user decision 2026-08-23, Part 2):* the 16 comments may shape the
   **mechanisms** of the evidence-first scheme (the coverage rule, the forbidden grounds, the
   rank-13 vs rank-15 contrast) — never their wording, never a score. Every persona prompt in
   `prompts/personas/`, `prompts/jwst_note_v2.md`, `golden/render_v2_desc.md` and every
   composed VIEW text is 4-gram-checked against the comments (`audit_traces.py
   --build-lexicon --pi-only`, `tests/test_golden_prompts.py`), and so is every TRACKED
   `.py`/`.md` under `golden/`, `prompts/` and `tests/` — a module docstring or a test fixture
   that quotes a comment is a leak into a file coding agents read (only the gitignored
   `golden/pi_comments.txt` holds the strings). The five PI-derived design anchors are on
   record in `REGISTRY.md` as predictions, never as truth (`golden/TRUTH_EVAL.md`).
2. **Validate-half identities and grades never reach a model**: no validate `candidate_id`,
   alias, coordinate, `unit_id`, score or letter in any prompt; exemplars come only from
   `align` (`fewshot.build_exemplar_blocks(eligible_units=align)`), and
   `registry.assert_unexposed(validate_units, kinds=("fewshot","sft"))` runs before EVERY
   validate call — any arm, any model, so an SFT student scored as `e1` is covered too. The
   system prompt (rubric + note) is run through `banned_lexicon.txt` before any call and a
   validate run without the lexicon is refused. `audit_traces.py` checks every trace afterwards:
   banned-lexicon hit, exemplar sha ∉ align, validate sha used as exemplar, long text block
   not matching a known template, `n_images != 1 + n_exemplars` → exit 1.
3. **Validate is scored once per registered tuple** (`REGISTRY.md`; the tuple includes the
   system-prompt sha and the thinking/effort settings; `run_golden_eval.py --split validate`
   refuses unregistered or already-completed tuples, resumes an interrupted replicate;
   `--force-rescore` leaves a dated row). All tuning — exemplar count, rubric wording,
   thresholds, marginals, SFT init/epochs — is on `align`; the SFT `valsel` set is an align
   carve, never the validate half, and `--smoke` never picks validate units once splits exist.
   The endpoint definitions (binary = score ≥ 3; paired ΔAUC + Δpurity@recall 0.8) are in
   `REGISTRY.md` and implemented once, in `analyze_golden.py`.
4. **The key and the kit are one-way**: `keys/` is tracked but never shipped; the kit carries
   no ids, ranks, coordinates, shas or repeat markers (`assert_blind`: opaque random item ids,
   one mtime, no sequence to read a repeat off), and no served byte string matches a file he
   already holds (`assert_no_collision` vs `top100_clean_scrambled/`, whose key he has). A kit
   is never rebuilt once grading has started: `--force` deletes the key and the manifest
   history, so old events can no longer be collected (loud, never a silent mis-join); the
   pilot fallback is `--drop-units`, which keeps the key. `collect.py` refuses a key whose pin
   differs from the one the latest manifest was written with. **Repo access**: `keys/`,
   `golden_labels.csv`, `splits.csv` and `golden_registry.csv` are tracked in the shared
   remote — confirm the grader has no read access to it before pushing, or keep them out of
   the remote until the validate runs are scored.
5. **`LENSJUDGE_FEWSHOT_MANIFEST` must be unset for golden runs** (`fewshot.assert_env_clean`):
   that path prints "consensus grade" and, for `survey_key="jwst"`, silently serves an
   ls-dr10 DESI cutout as a JWST exemplar.
6. **Grade semantics never mix**: `pipe_grade_passcount` is a persona pass-count; it is
   cross-tabbed against XH (`stats.py --agreement`) but never reported as a QWK headline, and
   no model prompt receives it.
7. **Nothing is written into the JWST run repo** (`_util.JWST_REPO` is read-only by
   convention; `common/jwst_fetch.py` is a vendored copy so no script imports from `J`).

## Backup record (Phase 0 — Perlmutter)

Purge dates: `$SCRATCH/ljv5` ~2026-09-29, `$SCRATCH/ljdesi` ~2026-10-10.
Target: `/global/cfs/cdirs/deepsrch/gdbenson/ljdata_backup_2026-08/` via
`rsync -a` of `$SCRATCH/ljv5/{sft_v5,valsel_v5,bench_v51,hsc_v5_test2,hsc_v5_valsel,hsc_sugohi,cutouts,cutouts_confirm,*.csv}`
and `$SCRATCH/ljdesi/{corpus_desi,parity_bench_arm*.csv,cutouts_parity_*}`, excluding merged
checkpoints (already on CFS).

| date | source | files (scratch) | files (CFS) | bytes | verified by |
|---|---|---|---|---|---|
| 2026-08-22 | `$SCRATCH/ljv5` (excl. `ckpt_*`, `wise_a*`) | 63,022 | 63,022 | 2.8 GB | `find -type f \| wc -l` both sides; `RSYNC_DONE` in `rsync_2026-08-22.log` |
| 2026-08-22 | `$SCRATCH/ljdesi` (excl. `ckpt_*`) | 59,422 | 59,422 | 3.5 GB | same |

Run as `nohup rsync -a --exclude='ckpt_*' --exclude='wise_a*' ljv5 $D/ && rsync -a --exclude='ckpt_*' ljdesi $D/`
on a Perlmutter login node (data only; the 460 GB of merged checkpoints are already on CFS and the
153 GB of WiSE-FT interpolations regenerate from `finetune/wise_ft.py`). Completed 23:42 PDT 2026-08-22.
