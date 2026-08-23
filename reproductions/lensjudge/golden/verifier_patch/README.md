# JWST verifier v2 — evidence-first drop-in for `jwst-strong-lens-search`

Generated 2026-08-23 by `lensjudge/golden/make_verifier_patch.py` from the same prompt
files and aggregator the lensjudge truth evaluation runs (`prompts/personas/jwst_v1/*.md`,
`golden/aggregate_v2.py`). Nothing in your repo is modified: `ADD_FILES.patch` only ADDS the
files listed below (`git apply --check ADD_FILES.patch` from the repo root, then `git apply`).

## Why (what the verdicts on disk show)

All numbers were measured from `results/verdicts/*.jsonl`, `results/verifications.csv`,
`results/results.csv` and the control files on 2026-08-23 (lensjudge `tasks/diag_*.md`).

1. **Recall on the known lenses the verifier examined: 1/24 (4%).** All 15 flagged COWLS
   controls scored 0/3 (12 of them in the `*_ctl*` batches, see §"The 12 ctl verdicts"); end to
   end 0/31 COWLS reached A/B/C; 1/82 catalogued lenses within 2″ did (rank 1, SL2S J021737).
   Two of the rejected COWLS lenses carry 5 and 4 human A-votes (`AAAAAB`, θ_E 1.00″, called a
   "Sérsic bowtie" by all three personas; `AAAAUX`, θ_E 0.60″, "a projected group").
2. **A conjunctive veto of three correlated refuters.** Pass rates 4.9 / 3.4 / 2.3 %
   (artifact / morphology / geometry); geometry never carries a C alone. The personas saw the
   identical JPEG plus the inspector's claim text, so their errors are shared (κ 0.46–0.63;
   under independence 0.01 triple passes are expected, 5 observed). The prompt says "prefer
   fail", every brief ends "mark it fail", and `uncertain` was used 5 times in 1,050 votes.
3. **Wrong hard-coded priors.** "θ_E roughly 0.3–3″; larger implies a group/cluster" drives
   19.6 % (67/342) of geometry fails; every A has θ_E ≤ 1.48″; all four θ_E > 3″ top-100
   objects are capped at B. "A genuine lensed arc is bluer than the deflector" is cited against
   62 % of the real lenses (15/24): a NIRCam SW/LW colour is not a DESI g−r, and dusty z∼2
   sources are red.
4. **The refuters attack a broken residual.** 88 % of the known-lens rejections (21/24)
   invoke the panel-(f) over-subtraction; 347 of 1,008 fail notes overall. Panel (f) is a
   circular radial-profile subtraction that leaves a four-lobed butterfly on every elliptical.
5. **Nothing ranks and the export lost verdicts.** AUC against the known lenses: inspector
   confidence 0.537, `n_pass` 0.505, the composite `score` 0.463, `mag_r` 0.644. U (never
   examined) ranks above D; A and B tie in tier; confidence has ∼15 distinct values. Test–retest
   31/33 but rank 15 = C on one draw, D on the other. Twelve COWLS verdicts are missing from
   `verifications.csv` and were graded U.

## What changes

- **ADVOCATE → three CRITICS → ARBITRATOR → `aggregate_v2`.** An evidence scorer locates every
  item (panel, radius, PA span, `visible_in_direct`), scores five NIRCam-adapted criteria and
  gives `p_evidence`. Critics (artifact / geometry / morphology) see ONLY the numbered items —
  no `p_evidence`, no inspector text — and answer within a stated competence with a NAMED
  alternative, a location box, which items it accounts for, and a graded
  `refutation_strength`; `no_opinion` with a reason is a legitimate answer. An arbitrator
  (image + all texts) rules each critic upheld / partial / overruled. Then, in pure python:
  `S = p_evidence · Π (1 − r_i · a_i)` with `a_i` the fraction of items the critic's box
  GEOMETRICALLY covers; `S_arb` over upheld/partial critics only; letters by FPR thresholds
  plus a D rule (nothing located, or a covering alternative with r ≥ 0.8); `U` strictly below
  every examined item.
- **Forbidden grounds in every critic brief**: the implied Einstein radius (→ `scale_tension`
  with strength ≤ 0.4 only), colour alone, a symmetric panel-(f) residual. No "prefer fail"
  sentence anywhere (`verify_workflow_v2.js` is tested for that).
- **Per-role views** (`08e`): geometry gets (b),(d),(e) + a 20″ context when rendered;
  morphology gets (c),(d),(e) — (a),(d),(e) on single-band "gray" layouts, where slot (c) is
  itself a subtraction; only the advocate and the artifact critic see panel (f). The panel sets
  are emitted from lensjudge's `panel_gloss.json` (the same file the truth evaluation runs),
  the layout is decided the way the render decided it (`sw_obs`/`lw_obs` AND the 0.55 finite
  gate of `05_fetch_cutouts.py`), and each role's views live in their own directory
  (`data/verify/views/{full,ctx20,geometry,morphology}/<id>.jpg`). Footers are removed from
  every view (they carried the id, coordinates and r-mag).
- **No inspector fields in any job** (`08d`): `claim_center`, `claim_quadrant`,
  `claimed_evidence` are gone.
- **`09_rank_report_v2.py`** loads every `verify_*.jsonl` including `*_ctl*`, validates each
  record (malformed → `results/verdicts_rejected.jsonl`, never silently "uncertain"), asserts
  every verdict id lands in the output, ranks by `S` with U below every examined item, and
  writes `results_v2.csv`, `top100_v2.csv`, `verifications_v2.jsonl` (verbatim records),
  `regrade_diff.csv` (v1 grade ↔ v2 letter per id) and `report_v2.md`. `tier`, `center_bonus`
  and `arc_score` no longer enter the ranking.

## Files added

| file | role |
|---|---|
| `scripts/verify_workflow_v2.js` | the workflow; `prompt(job)` dispatches on the job prefix (`advocate_`, `artifact_`, `geometry_`, `morphology_`, `arbitrator_`); briefs embedded verbatim (sha16 per file in the source) |
| `scripts/08d_make_evidence_batches.py` | advocate jobs: `--select top350|all_flagged|ids-file --include-ctl [--known-flagged] [--context]` |
| `scripts/08e_make_refuter_batches.py` | critic jobs + per-role view JPGs (`--tau0`, default from `thresholds_v2.json`) |
| `scripts/08f_make_arbitrator_batches.py` | arbitrator jobs for items with ≥ 1 named alternative |
| `scripts/09_rank_report_v2.py` | validation, aggregation, ranking, report (`--model-key opus_claude_code`) |
| `scripts/aggregate_v2.py` | the shared aggregator (sha16 `37be83b4598a36a7`; byte-identical to lensjudge's) |
| `scripts/thresholds_v2.json` | model-keyed thresholds (`provisional` until calibrated) |
| `scripts/calibrate_thresholds_v2.py`, `scripts/calibration_ids.csv` | the refit script and its id list (200 design-half negatives + 15 design-half COWLS ids) |
| `tests/test_aggregate_v2.py` | synthetic rank-15-like vs rank-13-like records: S ordering, D rule, geometric guard, pass-count table |

## Run order (Claude Code, your subscription)

```
python scripts/08d_make_evidence_batches.py --select top350 --include-ctl        # 362 items -> advocate_e0..e30
# workflow: scripts/verify_workflow_v2.js  ranges [{prefix:"advocate_e", from:0, to:30}]
python scripts/08e_make_refuter_batches.py                                       # tau0 = 0.15
# workflow: artifact_v*, geometry_v*, morphology_v*  (~290 items each, ~25 jobs per role)
python scripts/08f_make_arbitrator_batches.py
# workflow: arbitrator_a*  (~290 items)
python scripts/09_rank_report_v2.py --model-key opus_claude_code
```

Agent-items ≈ 362 (advocate) + ~870 (critics, only items with `p_evidence ≥ τ0`) + ~290
(arbitrator) ≈ **1,520**, i.e. ~130 batches of 12. No per-item timing exists on disk for the
v1 run, so wall time is not quoted here. Via the Anthropic API at Sonnet rates the same set is
≈ 362 × $0.084 ≈ $30 (+$11 with `--context`). Each stage is resumable: the builders skip
(id, role) pairs that already have a verdict.

## Letters: `letter_source`

Letters are set by FPR on a fixed negative set (t_A: ≤ 1 %, t_B: ≤ 5 %), not by a vote
count. `09_rank_report_v2.py --model-key opus_claude_code` resolves the thresholds in this
order and stamps every row with the result:

| `thresholds_v2.json` state | `letter_source` | meaning |
|---|---|---|
| `opus_claude_code` calibrated (after `calibrate_thresholds_v2.py --write`) | `opus_claude_code_calibrated` | set on this model's own output |
| `opus_claude_code: null`, `sonnet_api` frozen (lensjudge design half) | **`sonnet_thresholds_uncalibrated`** | Sonnet-API design-half numbers, not checked on Opus-in-Claude-Code output |
| both null (this file as shipped before the lensjudge design freeze) | `provisional` | the a-priori 0.80 / 0.50 placeholders |

The RANKING (`S`) does not depend on the thresholds; only the A/B/C/D labels do. To
calibrate: run `08d --select ids-file --ids-file scripts/calibration_ids.csv` through the
three stages (200 design-half negatives + 15 design-half COWLS, ~10 h of agent
time), then `python scripts/calibrate_thresholds_v2.py --model-key opus_claude_code --write`
and re-run `09_rank_report_v2.py`. The COWLS rows report recall only; the 16
COWLS ids on the lensjudge HOLDOUT half are deliberately withheld from the list (the holdout
is scored once, after the freeze, and nothing under these prompts touches it first).

## Holdout results (2026-08-23, lensjudge truth evaluation; Sonnet API)

The registered holdout (282 items: 42 catalogued-lens positives, 200 catalogue-purged
negatives) was scored once per arm with exactly these prompt files and this aggregator. The
full stack ranked positives at AUC 0.641 vs 0.535 for a brief-faithful reimplementation of
the incumbent pass-count (recall at 5 % FPR 19 % vs 10 %; paired ΔAUC +0.106, DeLong p 0.06),
and the advocate's raw `p_evidence` was the better ranker still (AUC 0.725 single-call, 0.738
as a mean of three replicates, recall at 10 % FPR 48 %) — but only the full stack controls
letters: the frozen `sonnet_api` thresholds held their design FPR on the holdout negatives
(FPR at t_A 2.0 % [95 % CI 0.5–5.0], at t_B 2.5 % [0.8–5.7]) with 17 % of positives at A/B
vs the incumbent's 0/31 COWLS, while the same thresholds applied to raw advocate output run
19–32 % FPR — so calibrate (section above) before trusting letters on advocate-only runs.
Critics argued from located structure: sole-forbidden-ground rate 0.7 % (incumbent replay
7.5 %), no_opinion ≤ 4.5 % per role, 0 parse failures in any arm. The pre-registered
composite bar (recall ≥ 0.5 at 5 % FPR) was NOT met — these letters are a calibrated
low-FPR instrument and a better ranker than the pass-count, not a solved detector; the
scale-blind regime (small-θ_E and cluster arcs) is where the misses concentrate.

## The 12 ctl verdicts

`results/verifications.csv` has 350 ids; the raw `results/verdicts/verify_*.jsonl` have 362.
The 12 missing ids are COWLS controls verified in `artifact_ctl0/1`, `geometry_ctl0/1`,
`morphology_ctl0/1` (all 0/3). `09_rank_report.py` globs `verify_*.jsonl`, which matches those
files and parses every line — the CSV is simply a stale export from before they were written
(all of them share the single commit `7632b39`); re-running `09_rank_report.py` recovers them.
`09_rank_report_v2.py` makes the regression impossible to miss: it asserts that every verdict
id on disk appears in `results_v2.csv` and lists the 12 in `regrade_diff.csv` with their
legacy pass-count (`legacy_grade_from_jsonl` = D, `legacy_grade_v1` = U).

## Inspector recommendation

The single change that would most raise recall before verification is to make the
inspector emit the advocate's located-evidence record (`items`, `p_evidence`, `scale_class`)
instead of `yes/likely/no + confidence`, and to select for verification on `p_evidence`
(continuous, no 400-way ties) rather than on a self-capped confidence with a cut at 28 that
sits on top of the band where the recovered known lenses landed (15–30). Delete the
inspector's "bluer than the red central galaxy" and "0.5–2.5 arcsec" lines at the same
time: the first fires on 62 % of the real lenses in this data, the second caps the Einstein
radius at the regime where the poster-child candidates live.

## Next step

Run the advocate alone (`08d --select all_flagged`, one stage, no critics) on the **1,674
flagged candidates that were never verified**: ≈ $37 at Sonnet-API rates or ~140 agent
batches. That is the direct test of how much the confidence-350 cut cost, and it turns the
U tail (currently ranked above D) into a scored list.

## What differs from the lensjudge truth evaluation (stated, not hidden)

The prompts and the aggregator are byte-identical; the plumbing around them is not:

1. **System prompt**: lensjudge sends persona + `prompts/jwst_note_v2.md` (resolution regime,
   false-positive glossary, the subtracted-panel caveat, "no tools"); this workflow sends the
   persona + the VIEW/LAYOUT note above. The note is not embedded here because the workflow's
   `prompt()` already carries the job-file instructions in its place.
2. **Images**: lensjudge sends the crops as separate image blocks with a per-panel caption;
   `08e` pastes them into one strip per role (one file per job item).
3. **Context**: the advocate may receive a `context` (20″) image under `--context`; in
   lensjudge only the geometry critic ever sees the context pair (the gated ctx20 arm).
4. **Blinding is advisory.** A Claude Code agent can Read any file. The prompt says to read
   nothing but the job file and its listed images, the critic jobs carry only the advocate's
   items and scale class (no `p_evidence`, no inspector text, no `tau0`), and each role's
   views sit in their own directory — but `results/verdicts/*.jsonl`, `results/inspections.csv`
   and the other views are one Read away, and the item ids encode coordinates. The lensjudge
   side has none of these channels (no tools, item-agnostic text, footer-cropped pixels).
   Treat agreement between the two as evidence about the prompts, not a proof of blinding.
5. **Validation**: `09_rank_report_v2.py` mirrors lensjudge's pydantic records with plain
   dict checks (integer criteria, unique item `k`, a location box for every named alternative,
   `scale_tension ≤ 0.4`, null strength → 0 only without an alternative). A record rejected
   here would be a parse failure there.

## `--known-flagged` is validation-only

`08d --known-flagged` adds the flagged-but-never-verified catalogued lenses
(`results/known_lens_recovery.csv`, sep ≤ 2″) BECAUSE they are known. That is fine for the
evaluation copy of `results_v2.csv` (it shows what the verifier does on lenses the inspector
flagged but the confidence cut dropped) and must never become a production selection rule.


