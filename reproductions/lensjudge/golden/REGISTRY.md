# Golden evaluation registry — the score-once gate

This file is the pre-registration ledger for every model run on the golden set's
**validate** half. `golden/run_golden_eval.py --split validate` refuses to start unless the
tuple it is about to score appears as a row in the *Registered arms* table below, and
refuses again if a completed predictions parquet (one with its `.meta.json`) for that tuple
already exists. The rule exists because the program has watched three "gains" evaporate on
a second look (v2-rubric mimic 0.52→0.71 that did not survive its loop-free replication; the
AION probe 0.84→0.66 across pools; C3 0.623→0.487 valsel→gate): every validate number is
scored ONCE, from a tuple frozen here BEFORE the call, and all tuning (exemplar count, rubric
wording, thresholds, marginals, SFT init and epochs) happens on the **align** half only.

The tuple is `(arm, model, rubric_sha16, n_exemplars, splits_sha16, system_sha16, thinking, effort)`:

- `arm` — `e1` zero-shot, `e2` few-shot (exemplars drawn from align via `golden/fewshot.py`),
  `e3` rubric arm (a different `--rubric`, still zero-shot). An SFT student (E4/E5) is scored
  as `e1` with its served model id.
- `model` — the `--model` alias as passed (`sonnet`, `opus`, or a served open-model id).
- `rubric_sha16` — `sha256[:16]` of the rubric FILE text.
- `n_exemplars` — exemplars per letter (0 for e1/e3).
- `splits_sha16` — the `.sha` sidecar of `golden/splits.csv` at registration time.
- `system_sha16` — `sha256[:16]` of the FULL system prompt the model receives (rubric + the
  JWST note, or the rubric alone under `--no-jwst-note`); this is what the trace audit sees.
- `thinking` / `effort` — the reasoning settings the call runs with (`off` / `adaptive`;
  `default` or the `--effort` level). Two runs that differ only here are two tuples.

Rescoring a tuple (`--force-rescore --rescore-reason "..."`) appends a dated row to the
*Rescores* table and stamps the parquet `rescored=True`; every table that reports a
rescored number must say so. k replicate samples of ONE registered tuple (`--k 3`) are not
rescores — they are the pre-registered sampling design (the Anthropic path runs at the API
default temperature, so replicates are real samples). An interrupted replicate (parquet
without `.meta.json`) is resumed, not rescored.

Add a row by hand, with the date, before the first validate call. `run_golden_eval.py
--print-tuple` prints the row for the flags you are about to use. The system prompt is
run through `golden/banned_lexicon.txt` before any validate call (build it with
`audit_traces.py --build-lexicon`; a validate run without it is refused).

## Endpoints (fixed before the first validate call)

Binary truth for every AUC / purity number is **XH score ≥ 3 (A/B) vs ≤ 2 (C/D)**, the
`binary_label` column of `outputs/golden_jwst_manifest.csv` (`build_eval_manifest.py`). The
rubric's "C = possible" view (score ≥ 2) is carried as `binary_label_ge2` and is secondary
only. `run_batch.summarize` prints the repo's historical A/B/C-vs-D view after every run; that
line is NOT the golden endpoint — `golden/analyze_golden.py` is.

- **Primary**: E2 − E1 paired ΔAUC (`parity/phase_d_analysis.paired_boot`, 2,000 resamples,
  seed 2026, + `delong_p`) on the validate half, per replicate and pooled over k; and
  Δ purity at recall 0.8 (purity = precision among the top-ranked set that recovers 80 % of
  the score ≥ 3 units), paired bootstrap. Score = `p_lens` (Anthropic path) or `s_exp` /
  `p_lens_logprob` (open backend; `s_exp` uses `llm_client.ORDINAL_W` everywhere).
- **Secondary**: absolute AUCs by stratum; raw QWK and calibrated-threshold QWK vs XH
  (thresholds at his align marginals); machine self-consistency QWK across replicates; E1 vs
  the incumbent `p_pipeline` reported BOTH on the pipeline-flagged rows and on all rows
  (unflagged = rank below every flagged item, `p_pipeline = 0`); COWLS / known-lens recall
  per arm; E3 vs E1 by the same ΔAUC.
- Every QWK is read against the ladder {E0 ceiling, 0.42, 0.29, ~0.00}; "human-level letter
  grading" = QWK CI overlaps E0's CI and exact agreement ≥ E0 − 0.10. Falsifier of the
  resolution thesis: E1 AUC ≤ 0.66 with CI excluding 0.80 and no E2/E3 movement. Lite-frame
  numbers are labelled "pilot".

## Registered arms

| registered | arm | model | rubric | rubric_sha16 | n_exemplars | splits_sha16 | system_sha16 | thinking | effort | k | note |
|---|---|---|---|---|---|---|---|---|---|---|---|

## Rescores

| date | arm | model | rubric_sha16 | n_exemplars | splits_sha16 | system_sha16 | thinking | effort | reason |
|---|---|---|---|---|---|---|---|---|---|

## Truth-eval endpoints

Part 2 (`golden/run_truth_eval.py`, truth set `golden/truth_manifest.csv`, halves
`golden/truth_splits.csv`). Copied verbatim from the approved plan (PART 2 › "Evaluation
without a human campaign"), fixed before the first holdout call:

- **Pre-registered endpoints (holdout, once per tuple)**: P1 recall of holdout positives at 5%
  FPR on holdout N1 (A1 vs A0, paired; Clopper–Pearson; exact McNemar on positives — 5 clean
  promotions ⇒ p<0.05; A0's ROC is 4-point, report all points); P2 letters frozen on design hold
  their FPR on holdout (t_A upper CI ≤2.5%, t_B ≤7.5% — **restated below before any holdout
  call**); P3 fraction of holdout positives at A/B
  (old: 0/31 COWLS). Secondary: paired ΔAUC (`phase_d_analysis.paired_boot`, reseed per endpoint;
  δ_min ≈0.09), S_arb vs S, A3−A2 (render effect), per-stratum recall (COWLS S-band, θ_E ≤1/1–2″,
  galaxy vs cluster scale, layout), Spearman(S, θ_E) on COWLS — must not be monotone-negative,
  forbidden-ground rate from `golden/reason_audit.py` (<2% each), no_opinion rate per role (≤35%),
  D-rate on D_refuted (reported, not gating), flip rate/ICC from replicates
  (`lensbench_gate.grade_flip_rate`), parse-failure rate, cost/item, anchors table. "Better" =
  P1 recall ≥0.5 with CI excluding A0's AND P2 holds AND forbidden-ground rate <2%. Limitation
  stated: no truth at θ_E>2″; the 35 fetchable 2–3″ + LeG/LeI rows are a registered LATER
  disjoint draw for the scale rule (~$11).

Positive label for the headline = "lensing anywhere in the 10″ stamp" (`is_positive`);
"centre is the deflector" (`centre_is_deflector`) is the first secondary. Primary score = the
deterministic `S` of `golden/aggregate_v2.py`; `S_arb` is the secondary arm. Parse failure in
ANY called role (arbitrator included) ⇒ `S = NaN` (excluded, rate reported). Anchors (below)
never count toward any endpoint.

**P2 restated (2026-08-23, before any holdout call; adversarial-review finding F9).** The
plan's wording "t_A upper CI ≤ 2.5 %, t_B ≤ 7.5 %" is stricter than the rule it tests: with
200 holdout negatives the Clopper–Pearson 95 % upper bound is 1.83 % at 0/200, 2.75 % at 1/200
and 3.57 % at 2/200, so t_A (set at design FPR ≤ 1 % = ≤ 2/200) would pass only at 0/200, and
t_B (set at ≤ 5 % = 10/200) only at ≤ 7/200 — it would fail on sampling noise alone. The
registered test is therefore: **the holdout FPR is not significantly above its design target —
CP 95 % LOWER bound ≤ 1 % at t_A and ≤ 5 % at t_B** (`analyze_truth` rows `P2_holds_t_A` /
`P2_holds_t_B`; admits ≤ 5/200 at t_A and ≤ 17/200 at t_B). The point estimates, both CI
bounds and the plan's upper-bound wording (`P2_upper_ci_ok_*`) are reported beside it. The
thresholds tested are the ones the rows were lettered with (written on every parquet row and
hashed into the tuple as `thresholds_sha16`), never re-read from a later thresholds file.

## Truth-eval registered arms

The tuple is `(arm, model, persona_set_sha16, note_sha16, system_sha16s, render_version,
render_desc_sha16, splits_sha16, claim_mode, thinking, effort, k, thresholds_sha16)`;
`run_truth_eval.py --print-tuple` prints the row. `--split holdout` refuses unless the tuple
is a row here AND no completed parquet (with `.meta.json`) holds it anywhere under
`outputs/`; `--split design` is ungated. `system_sha16s` is the `+`-joined `role:sha16` of
every role's FULL system prompt (persona text + note) in role order — for `a0` with
`claim_mode=inspector` the three `role@noclaim:` wrappers follow (items the inspector never
flagged get the "not available" wording). `k` is the replicate count (a k=1 arm and its k=3
replicate study are two tuples, written to `..._k1_r1` / `..._k3_r{1,2,3}`).
`thresholds_sha16` hashes the resolved `tau0 / t_A / t_B / letter_source / thresholds_key`
(tau0 gates the critics, t_A/t_B every letter); a holdout run on provisional thresholds is
refused unless `--allow-provisional-thresholds` is stated — say so in the note.

| registered | arm | model | persona_set_sha16 | note_sha16 | system_sha16s | render_version | render_desc_sha16 | splits_sha16 | claim_mode | thinking | effort | k | thresholds_sha16 | note |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 2026-08-23 | a0 | sonnet | f2bb5294e1d89505 | e3b0c44298fc1c14 | artifact:2e5627171e2ce69c+morphology:e5b545f9957120ac+geometry:a672a7dfc227288e+artifact@noclaim:d7683d3099eca7ad+morphology@noclaim:cf92400576d45808+geometry@noclaim:d205721a8e07d03f | jwst_v1 | 28737c6083dc1978 | 032a302c84f3cbe7 | inspector | off | default | 1 | 94d31c7b6979e0ca | incumbent baseline; inspector claims in the USER message (lexicon-checked, a hit blanks the claim) |
| 2026-08-23 | a1 | sonnet | a26d972ecc0b4ee7 | 754655a400f360e6 | advocate:c41d7f5787bdb472+artifact:f5ed259652e65ee2+geometry:a293ddddce11ee4a+morphology:26bde57ad0478237+arbitrator:44542114399ab277 | jwst_v1 | 28737c6083dc1978 | 032a302c84f3cbe7 | none | off | default | 1 | 94d31c7b6979e0ca | primary arm; P1 scored on BOTH S and S_arb, co-registered (see the 2026-08-23 registration note below) |
| 2026-08-23 | a2 | sonnet | a26d972ecc0b4ee7 | 754655a400f360e6 | advocate:c41d7f5787bdb472 | jwst_v1 | 28737c6083dc1978 | 032a302c84f3cbe7 | none | off | default | 1 | 94d31c7b6979e0ca | advocate-only (S = p_evidence) |
| 2026-08-23 | a3 | sonnet | a26d972ecc0b4ee7 | 754655a400f360e6 | advocate:c41d7f5787bdb472 | jwst_v2r | 25dd6cc680579747 | 032a302c84f3cbe7 | none | off | default | 1 | 94d31c7b6979e0ca | v2r render arm; A3 minus A2 = render effect |
| 2026-08-23 | a2 | sonnet | a26d972ecc0b4ee7 | 754655a400f360e6 | advocate:c41d7f5787bdb472 | jwst_v1 | 28737c6083dc1978 | 032a302c84f3cbe7 | none | off | default | 3 | 94d31c7b6979e0ca | R2 replicate study (k=3, whole holdout; flip rate and ICC) |
| 2026-08-23 | a1 | opus | a26d972ecc0b4ee7 | 754655a400f360e6 | advocate:c41d7f5787bdb472+artifact:f5ed259652e65ee2+geometry:a293ddddce11ee4a+morphology:26bde57ad0478237+arbitrator:44542114399ab277 | jwst_v1 | 28737c6083dc1978 | 032a302c84f3cbe7 | none | off | default | 1 | a40ae6e201a03e65 | holdout secondary; opus_api thresholds are null so the run is registered WITH --allow-provisional-thresholds (letters provisional tau0 0.15 / t_A 0.80 / t_B 0.50, letter_source provisional; P2 is registered for the Sonnet letters) |

**Registration note (2026-08-23, before any holdout call).** (i) P1 is scored on BOTH the
primary S and the secondary S_arb, co-registered here before any holdout call; reason,
recorded from the design full-stack pass (step 3a): S under-ranks the advocate-only a2
(paired dAUC -0.084, DeLong p 0.108, not significant; recall@5%FPR 0.250 vs 0.295) while
S_arb recovers parity with a2 (recall 0.295) because overruled critics drop out of the
product. (ii) The gated arm (ctx20 / panel-level chi render) is CUT per the pre-registered
rule: design Delta recall@5%FPR never reached +0.10. (iii) Production k=1 per the
pre-registered retest rule: letter flip rate at the A/B boundary 0.091 and at B/C 0.125,
both <= 0.25, so no k=3 on a contested band; R2 (a2, k=3) is the registered replicate
study. (iv) Known limitation, carried into the holdout read-out: the rank-13 design
anchor's written prediction (letter D with a spiral_arm critic upheld) already MISSED on
the design half - the advocate scored it high and critic coverage of its items was
partial, so S stayed above the D band; the anchors table reports hit or miss either way.

## Truth-eval rescores

| date | arm | model | persona_set_sha16 | note_sha16 | system_sha16s | render_version | render_desc_sha16 | splits_sha16 | claim_mode | thinking | effort | k | thresholds_sha16 | reason |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|

## Design anchors (PI-derived, design-only, never truth)

Five top-100 items whose PI-annotated outcome shaped the scheme's *mechanisms* (forbidden
grounds, coverage rule, the rank-13 vs rank-15 contrast). They are forced into the DESIGN
half, are never scored as truth, never enter any endpoint, and are never shown to a model as
text. Predictions written here BEFORE the first call; hit or miss, both are reported in
`outputs/truth_anchors.csv`. Ids from `J/results/JWST_top100_master.csv`; units from
`golden/frame.csv`. Ranks 7 and 14 are one system (u0042): the same F150W2 field cut at two
catalogue positions 1.17″ apart — the stamps are NOT pixel-identical (a 13×35-px shift and
panel (f) is each stamp's own centred subtraction), which is why the prediction is a
consistency bound, not equality.

| rank | candidate_id | unit_id | layout | pipe_grade_passcount | blind θ_E ″ | written prediction |
|:---|:---|:---|:---|:---|:---|:---|
| 15 | J20954380-1094330 | u0192 | color (F150W2/F322W2) | C | 3.60 | letter A or B |
| 13 | J18805344+1121596 | u0216 | color (F150W/F277W) | C | 0.72 | letter D, with a `spiral_arm` critic upheld |
| 7 | J18030075+2309921 | u0042 | gray_sw_only (F150W2) | B | 4.00 | ‖Δletter(7, 14)‖ ≤ 1 on the same SW-only field, re-centred 1.17″ apart |
| 14 | J18030108+2309932 | u0042 (alias of rank 7) | gray_sw_only (F150W2) | C | 4.60 | ‖Δletter(7, 14)‖ ≤ 1 on the same SW-only field, re-centred 1.17″ apart |
| 16 | J5186648-1343587 | u0153 | color (F150W2/F322W2) | C | 17.60 | `scale_class = cluster`, `deflector_is_centre = false`, letter not D |
