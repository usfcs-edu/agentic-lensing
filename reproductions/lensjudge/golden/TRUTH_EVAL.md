# Truth evaluation of the evidence-first JWST grading scheme (Part 2)

How the replacement for the JWST run's pass-count verifier is scored WITHOUT a human grading
campaign: against truth already on disk (COWLS and literature lenses, catalogue-purged
negatives), on a design / holdout split, with every holdout arm registered before it runs.
Plan of record: `~/.claude/plans/i-want-to-explore-golden-seal.md` › PART 2 (approved
2026-08-23). Ledger: `golden/REGISTRY.md` (Truth-eval sections). Log: `lensjudge/GOLDEN_FINDINGS.md`.
The golden single-grader campaign (`golden/README.md`) is a separate experiment; the two
share stamps, the exposure registry and the embargo, nothing else (splits are independent).

## Purpose

The incumbent verifier (`J/scripts/verify_workflow.js` + `09_rank_report.py`) grades by a
PASS-COUNT over three adversarial personas. Replayed zero-API on its own verdicts
(`golden/incumbent_replay.py`): 23/24 known lenses it examined scored 0/3, all 15 flagged COWLS
controls 0/3, per-persona pass rates 4.9 / 2.3 / 3.4 %, κ 0.63 / 0.49 / 0.46, and re-weighting
the old verdicts under any rule leaves COWLS at 0/15 in A/B/C. The scheme under test replaces
the veto with an ADVOCATE (located evidence, `p_evidence`) → three competence-bounded CRITICS
(named alternative, location box, `accounts_for`, graded `refutation_strength`, `no_opinion`)
→ ARBITRATOR (image + texts; upheld / partial / overruled) → `aggregate_v2` (pure python:
`S = p_ev · Π(1 − r_i·a_i)` with a geometric coverage guard; `S_arb` over upheld/partial
critics; letters by FPR thresholds; U strictly below every examined item). The same prompt
files and aggregator are embedded byte-identically in Nate's drop-in (`golden/verifier_patch/`).

## The set (`golden/truth_manifest.csv`, pinned; `build_truth_manifest.py`)

570 rows, computed from the run's own tables (never hand-summed), deduplicated at 2″
(pinned 2026-08-23: manifest `70eea125c426a8b3` — re-pinned after the review: one COWLS
`einstein_radius = 0` placeholder is now NaN, nothing else changed — negatives
`2899110110d6a586`, splits `032a302c84f3cbe7`, stamps manifest `410f1ab419648b64`; 570/570 v1
JPEGs and 570/570 v2r renders on disk, every served image 752×540, shas verified against the
bytes):

| truth_class | n | what |
|:--|--:|:--|
| cowls | 31 | blind COWLS controls; `cowls_band` strong 5 / marginal 8 / weak 13 / provenance 5 (S = 2·nA + nB + nS) |
| lit_galaxy | 26 | literature lens ≤ 2″ whose catalogue position is the deflector (SIMBAD gLS/gLe/LeQ, COWLS on a non-control, top-100 `known` galaxy-scale) |
| lit_cluster | 30 | literature arc/knot ≤ 2″ (SIMBAD LeG/LeI, top-100 known cluster arcs) |
| negative | 400 | `truth_negatives.csv`: unflagged & ok, purged ≥ 5″ from the 195,818-position catalogue union (12 files), minus frame units / positives (+10″ buffer) and the DESI pool (2″); stratified layout × field_class to the positives' mix; seed 2026 |
| stress_D | 40 | frame `D_refuted` (machine-labelled; reported, never gating) |
| stress_U | 29 | frame `U_tail` (flagged, never verified) |
| anomalymatch | 10 | AnomalyMatch Class A 5 / B 1 / C 4 (human-voted, unconfirmed) |
| anchor | 4 | PI-derived design anchors that are nothing else (rank 16 is `lit_cluster` + `is_anchor`) |

**Positives = 87** (`is_positive` = cowls | lit_*; the primary label is "lensing anywhere in
the 10″ stamp"); `centre_is_deflector` = 65 (first secondary). Pre-purge catalogue matches in
the negative pool: 2″ 1.86 %, 3″ 2.50 %, 5″ 4.51 %; residual contamination after the 5″ purge
(undiscovered lenses) ≈ 0.5–1 % is quoted as a systematic, never relabelled. Anchors never
count toward any endpoint. Images: `image_path` = footer-cropped q92 JPEG (752×540) under
`golden/kits_truth/` from `golden/stamps/<id>/<id>_v1.jpg` (stamps for every truth id live
under `golden/stamps/`, fetched with `build_stamps.py --frame golden/truth_fetch_list.csv`);
`image_path_v2r` / `render_sha_v2r` = the A3 render (`render_v2.py`, `golden/kits_truth_v2r/`),
merged into the manifest by re-running `build_truth_manifest.py` (it reads the pinned
`kits_truth_v2r/render_v2r.csv`). Truth JPEGs are pixel-identical but never byte-identical to a
kit item (`optimize=False`); `build_kit.assert_no_collision` scans `kits_truth/` on every kit build.

## The split (`golden/truth_splits.csv`, pinned; `split_truth.py`)

design 288 / holdout 282 by `system_id` (10″ union-find over all rows); cells truth_class ×
scale (`cowls_band`; galaxy vs cluster for literature); literature positives by WHOLE
proposal; forced to design: every `prior_exposure == 2` frame unit, rank 16 (u0153) + its
system, the rank-14 alias (rank 7's system), every anchor (27 forced rows); negatives 200/200
by field_class; seed 2026; 2″ firewall between halves asserted
(`split_halves.firewall(halves=("design","holdout"))`). Positives 45 / 42 (cowls 15/16,
lit_galaxy 13/13, lit_cluster 17/13); stress_D 20/20. Known lumpiness of the whole-proposal
rule, stated so nobody reads it as a tuning artefact: all of 6882 (11 lit rows) and 5594 (8)
are in design, and **every COSMOS-Web (1727) literature galaxy-scale lens is in the holdout
(0 design / 11 holdout)** — the design half's 13 `lit_galaxy` rows are all cluster-programme
fields (6675 ×5, 5594 ×2, 6882 ×2, 1345, 1837, 6434, 6480), so the advocate wording and
t_A / t_B are tuned on COSMOS-field galaxy-scale positives from COWLS only, and the holdout's
1727 `lit_galaxy` stratum is a field the design never saw at galaxy scale. The split is
kept (re-splitting now would re-draw the pinned halves before any call; the per-stratum
recall rows report it); a later re-split treating 1727 lit_galaxy like COWLS (alternating
within the field) is the remedy if the stratum disagrees with the COWLS rows.

## Arms (`golden/run_truth_eval.py --arm … --split design|holdout`)

| arm | panel mode | what the model sees | score |
|:--|:--|:--|:--|
| a0 | incumbent | the three incumbent briefs byte-equal to `verify_workflow.js` inside `prompts/personas/incumbent/wrapper.md`; `IncumbentVerdict`; `--claim-mode inspector` puts the inspector's claim (J `inspections.csv`, evidence[:400], lexicon-checked — a hit blanks the claim) in the USER message for flagged rows, and the wrapper variant follows the item: "given with the image below" for a flagged item, "not available" otherwise (both variants are in the tuple as `role:` / `role@noclaim:`) | pass-count letter, `p_lens = n_pass/3` |
| a1 | full | advocate (full composite) → critics only if `p_evidence ≥ τ0` (artifact: full composite; geometry: (b),(d),(e) ×2 [+ ctx20 with `--ctx20`]; morphology: (c),(d),(e) ×2 — gray layouts: (a)/(b),(d),(e), never a subtraction panel) → arbitrator only if a critic named an alternative | `S` primary, `S_arb` secondary, letter |
| a2 | advocate_only | one call | `S = p_evidence` |
| a3 | advocate_only on `image_path_v2r` | the jwst_v2r render (panel f → signed-χ SW \| LW) WITH its description: the composite VIEW text is the gloss's `composite_*_v2r` twin ((f) = signed-chi elliptical-model residual, "NOT a circular subtraction") and `golden/render_v2_desc.md` is appended to it — image and description ship as one unit, the panel refuses otherwise | as a2; A3 − A2 = render effect |
| attr | attr | a1 prompts, the SAME full composite to every role (130 design items; `--ctx20` refused: nothing is attached) | isolates the per-role views |

Model `sonnet` = `claude-sonnet-4-6` (primary), `opus` = `claude-opus-4-8` (holdout secondary).
Anthropic path, API default temperature, `thinking=off` recorded. Parse failure in ANY
called role — the arbitrator included — ⇒ `S = NaN`, `S_arb = NaN`, no letter, row kept with
`parse_fail_roles` (one repair retry inside `grader_direct`; one policy, stated in
`schemas_panel.py`). Per-item cost cap `--cost-cap 0.17` (warned + counted; the smoke measured
$0.163 for a full stack). Thresholds from `thresholds_v2.json` keyed by
`aggregate_v2.MODEL_KEYS` (`sonnet → sonnet_api`, `opus → opus_api` — the API Opus, not
Nate's `opus_claude_code`); `null` ⇒ provisional (τ0 0.15, t_A 0.80, t_B 0.50) with
`letter_source=provisional`; `--tau0` is a design-only knob. The resolved thresholds are
hashed into the tuple (`thresholds_sha16`) and written on every row (`tau0`, `t_A`, `t_B`);
`analyze_truth` letters P2 from the rows' own values, never from a later file.

Panel (f) is described ONCE, in the VIEW text of `panel_gloss.json` (the circular-subtraction
caveat — butterfly / bowtie, concentric rings, off-centre dipole — in the v1 composite texts;
the signed-chi description in the `_v2r` twins + `render_v2_desc.md`): the persona .md files
and the v2 note defer to the VIEW for what model was removed, so no role is ever told two
different things about the same panel. The PA-span direction (pa_deg_from → pa_deg_to,
increasing angle, 350 → 10 through North) is stated to the advocate and the critics because
`aggregate_v2.covers` reads it that way.

## The tuple (registered before any holdout call)

`TruthTuple(arm, model, persona_set_sha16, note_sha16, system_sha16s, render_version,
render_desc_sha16, splits_sha16, claim_mode, thinking, effort, k, thresholds_sha16)`;
`--print-tuple` prints the `REGISTRY.md` row. `persona_set_sha16` = sha over every
`.md`/`.json` in the persona dir (incl. `panel_gloss.json`); `system_sha16s` = `+`-joined
`role:sha16` of each FULL system prompt (persona + note, note last; a0 under
`--claim-mode inspector` adds the three `role@noclaim:` wrappers); `render_version` ∈
{jwst_v1, jwst_v1+ctx20, jwst_v2r}; `render_desc_sha16` = sha over the VIEW gloss +
`render_v2_desc.md` (what the composite roles actually read); `thresholds_sha16` = sha of the
resolved τ0 / t_A / t_B / letter_source. Files:
`outputs/preds_truth_{arm}_{model}_{split}_k{K}_r{k}.parquet` (K = the tuple's replicate
count, so a2 k=1 and R2 = a2 k=3 never collide), run tag `truth_{arm}_{model}_{split}_k{K}_r{k}`.

`--split holdout` refuses, before any call: an unregistered tuple; a tuple already scored
ANYWHERE under `outputs/` (the scan is by tuple over every `.meta.json`, not by file name;
`--force-rescore --rescore-reason` archives the replicate as `*.pre_rescore_<stamp>` and logs a
row in "Truth-eval rescores"); a non-default `--out` (default name, under `outputs/`);
`--limit` / `--ids-file` (a subset is not a score-once record); `--tau0`; provisional
thresholds without `--allow-provisional-thresholds`; a lexicon missing any holdout id or any
of the 16 PI comments; a manifest whose `half` disagrees with the pinned splits. A resumed
parquet must carry this tuple on every row. Design is ungated. On the first scored item the
per-role shas the panel actually sent are compared with the tuple's — a mismatch aborts.

Today's rows (`--print-tuple`, 2026-08-23, after the review fixes): persona set
`5983c47ef6315078`, note `754655a400f360e6`, gloss `e734968399847d42`, system shas advocate
`4d252f5bf2b82e5d` · artifact `f5ed259652e65ee2` · geometry `a293ddddce11ee4a` · morphology
`26bde57ad0478237` · arbitrator `44542114399ab277`; render_desc v1 `28737c6083dc1978` / v2r
`25dd6cc680579747`; incumbent (no note) artifact `d7683d3099eca7ad` · morphology
`cf92400576d45808` · geometry `d205721a8e07d03f` (with `--claim-mode inspector`:
`2e5627171e2ce69c` / `e5b545f9957120ac` / `a672a7dfc227288e` + the three `@noclaim` shas above);
splits `032a302c84f3cbe7`; thresholds (provisional) `a40ae6e201a03e65`.

## Endpoints (holdout, once per tuple; verbatim in `REGISTRY.md › Truth-eval endpoints`)

P1 recall of holdout positives at 5 % FPR on holdout N1 (A1 vs A0, paired; Clopper–Pearson;
exact McNemar on positives; A0's ROC is 4-point, all points reported). P2 letters frozen on
design hold their FPR on holdout — **restated in `REGISTRY.md` before any holdout call**: the
CP 95 % LOWER bound at t_A ≤ 1 % and at t_B ≤ 5 % (`P2_holds_*`; the plan's "upper CI ≤ 2.5 % /
7.5 %" needs 0/200 and ≤ 7/200 and is reported beside it as `P2_upper_ci_ok_*`). P3 fraction of
holdout positives at A/B (old: 0/31 COWLS). Secondary: paired ΔAUC (reseeded per endpoint), S_arb vs
S, A3 − A2, per-stratum recall, Spearman(S, θ_E) on COWLS (must not be monotone-negative;
COWLS rows with a 0.0 placeholder radius are NaN, not a point), forbidden-ground rate
(`reason_audit.py`, < 2 % each — derived from the STRUCTURED record: a θ_E argument routed
into `scale_tension`, a `subtraction_residual` on an item absent from every direct panel, and
a colour remark beside a structural alternative are the scheme's sanctioned channels and are
reported as `uses_*` monitors, not as forbidden grounds), no_opinion rate per role (≤ 35 %),
D-rate on D_refuted (reported), flip rate / ICC from replicates, parse-failure rate,
cost/item, the anchors table. "Better" = P1 recall ≥ 0.5 with CI excluding A0's AND P2 holds
AND forbidden-ground rate < 2 %. Implemented once, in `golden/analyze_truth.py`
(`--split holdout --model sonnet` → `outputs/truth_results.csv`, `truth_anchors.csv`,
`truth_summary_holdout.md`). Design anchors and their written predictions: `REGISTRY.md ›
Design anchors` (15 → A/B; 13 → D with `spiral_arm` upheld; |Δletter(7,14)| ≤ 1; 16 →
`scale_class=cluster`, not D).

## Run order

```
cd reproductions; PY=~/.venvs/lensjudge/bin/python
# 0  zero-API baseline + truth set (deterministic; re-running reproduces the .sha files)
$PY lensjudge/golden/incumbent_replay.py                                   # 350/350 reproduced, 12 ctl recovered
$PY lensjudge/golden/sample_truth_negatives.py                             # truth_negatives.csv  2899110110d6a586
$PY lensjudge/golden/build_truth_manifest.py                               # truth_manifest.csv + truth_fetch_list.csv
$PY lensjudge/golden/build_stamps.py --frame lensjudge/golden/truth_fetch_list.csv --workers 3   # [net, ~40 min, public S3]
$PY lensjudge/golden/build_truth_manifest.py                               # fills image_path for every row
$PY lensjudge/golden/split_truth.py                                        # truth_splits.csv + manifest `half`
$PY lensjudge/golden/render_v2.py --manifest lensjudge/golden/truth_manifest.csv --out-dir lensjudge/golden/kits_truth_v2r
$PY lensjudge/golden/build_truth_manifest.py                               # merges image_path_v2r / render_sha_v2r (re-pin)
# 1  embargo + registry
$PY -c "import pandas as pd; s=pd.read_csv('lensjudge/golden/truth_splits.csv',dtype=str); open('lensjudge/outputs/truth_holdout_ids.txt','w').write('\n'.join(s[s.half=='holdout'].candidate_id)+'\n')"
$PY lensjudge/golden/audit_traces.py --build-lexicon --pi-only --extra-ids lensjudge/outputs/truth_holdout_ids.txt --banned lensjudge/golden/banned_lexicon.txt
$PY lensjudge/tests/test_golden_prompts.py                                 # every persona file: 0 lexicon hits
$PY -m lensjudge.golden.registry seed --frame lensjudge/golden/frame.csv   # frame units become registry rows (kind "eval" marks)
# 2  [$] design (ungated): advocate-only iterations (≤3) -> freeze advocate -> full stack (≤2) + attr + test-retest k=3
$PY lensjudge/golden/run_truth_eval.py --arm a2 --split design --model sonnet
$PY lensjudge/golden/run_truth_eval.py --arm a1 --split design --model sonnet
$PY lensjudge/golden/run_truth_eval.py --arm attr --split design --model sonnet --limit 130
$PY lensjudge/golden/run_truth_eval.py --arm a0 --split design --model sonnet --claim-mode inspector --ids-file <agreement units>
$PY lensjudge/golden/audit_traces.py --traces-dir lensjudge/outputs/traces_truth_a1_sonnet_design_k1_r1 --banned lensjudge/golden/banned_lexicon.txt
$PY lensjudge/golden/analyze_truth.py --split design --model sonnet        # monitors: forbidden-ground <2%, no_opinion ≤35%, D-rate on N1 ≥50%, anchors
# 3  freeze: write t_A / t_B into thresholds_v2.json (sonnet_api), regenerate the Nate patch, register every holdout tuple
$PY lensjudge/golden/make_verifier_patch.py
$PY lensjudge/golden/run_truth_eval.py --arm a1 --split holdout --print-tuple   # paste into REGISTRY.md (each arm / model / k; a0 with --claim-mode inspector; a2 --k 3 for R2)
# 4  [$] holdout once per tuple, whole half, default --out (a0 --claim-mode inspector, a1, a2, a3, R2 = a2 --k 3, --model opus a1, the gated arm only if earned)
$PY lensjudge/golden/run_truth_eval.py --arm a1 --split holdout --model sonnet
$PY lensjudge/golden/run_truth_eval.py --arm a2 --split holdout --model sonnet --k 3     # -> preds_truth_a2_sonnet_holdout_k3_r{1,2,3}
$PY lensjudge/golden/analyze_truth.py --split holdout --model sonnet --baseline a0      # refuses a parquet without .meta.json / a subset / mixed tuples
```

## State on 2026-08-23 (integration smoke, design half, ≈ $0.34 of API)

Three design items (COWLS strong `AAAAAB` θ_E 1.0″ / a cluster-field negative / a PI-refuted
merger from stress_D) through a1, a0 (`--claim-mode inspector`) and a2, written under
`outputs/smoke/` so the real design parquets start clean. Every role parsed (15 calls, one
paid repair retry), `audit_traces` 0 violations on all three trace dirs (`n_extra_views`
counted: geometry/morphology 3 images, others 1), `reason_audit` and `analyze_truth` run on
the resulting parquets. Returned (one draw each; design, not truth-scored):

| item | a0 | a2 (S = p_ev) | a1 (S / S_arb, letter) |
|:--|:--|:--|:--|
| COWLS AAAAAB | 0/3 → D | 0.08 C | 0.07 / 0.07 C (2 items, below τ0) |
| negative | 0/3 → D | 0.06 C | 0.07 / 0.07 C (1 item) |
| stress_D merger | 0/3 → D | 0.62 B | 0.258 / 0.403 C (4 items; critics merger/companion/residual, a = 0.25/0.5/0.75, arbitrator letter C) |

Two things the design iterations must look at: the advocate echoed the incumbent on the
AAAAAB lens (p_evidence 0.07–0.08, "four-lobed butterfly … no offset arc") — the recall lever
is the advocate's wording; and the full-stack item cost $0.163 (critic calls ≈ $0.027–0.05:
~5k input tokens incl. three 480×516 views, ~1k output tokens of `measured`/`notes`), hence
the `--cost-cap 0.17` default and the re-derived budget below. The smoke parquets under
`outputs/smoke/` predate the review fixes (older prompt shas, `_r1` file names) and are not
resumable as any current tuple — `check_resume` refuses them by design.

## Status (2026-08-23): Part 2 COMPLETE — holdout scored once, verdicts recorded

Design phase done (A0 agreement gate $0.92; two advocate iterations → freeze at
`advocate.md` `6a806fbec212eb19` / system `c41d7f5787bdb472`; full-stack design pass +
threshold freeze t_A 0.192 / t_B 0.1318 → `thresholds_sha16 94d31c7b6979e0ca`; retest k=2;
design spend $38.48). All six registered holdout tuples then ran EXACTLY once (282 items
each, one invocation each, 0 parse failures, 0 audit violations, `rescored=False`
everywhere; holdout spend $75.26; program total $113.74). Results:
`outputs/truth_results.csv` (526 rows), `truth_summary_holdout.md`, `truth_anchors.csv`
(0 rows — anchors are design-half); full tables in `GOLDEN_FINDINGS.md` (2026-08-23 HOLDOUT
entry). **Pre-registered composite verdict: NOT met** — P1 recall@5 %FPR 0.190 (S) / 0.214
(S_arb), far below the registered 0.5 bar and CI-overlapping A0's 0.095; **P2 PASSES** on
the frozen a1 letters (holdout FPR at t_A 2.0 % [0.55, 5.04], at t_B 2.5 % [0.82, 5.74]);
**forbidden-ground 0.007 PASSES**. Ranking gains are real (a2-vs-a0 dAUC +0.190, DeLong
7e−06; best ranker = mean of 3 advocate replicates, AUC 0.738) but the full-stack S
under-ranks the advocate on holdout too (a1-vs-a2 dAUC −0.084) and the arbitrator repair
did not recover parity as it had on design; render effect (a3−a2) null; frozen letters
transfer only for the arm they were calibrated on. The verifier patch is current (regenerated
byte-identically; `git apply --check` passes; holdout headline paragraph added to its README
via the template). Known open item: the stale
`test_golden_truth_runner.py::test_registry_md_truth_sections_and_anchors` empty-table pin
fails now that the six arms are registered (211/212 suite) — re-pin post-phase with PI
approval.

## Status addendum (2026-08-23, later): Claude-5 advocate arms — opus5 gate MET, sonnet5 control interrupted

Two Claude-5 advocate-only tuples were registered (`opus5 = claude-opus-5`, `sonnet5 =
claude-sonnet-5`; adaptive thinking, xhigh effort, provisional letters, gate + confound
caution pre-registered in `REGISTRY.md`). **a2/opus5 scored once on the whole holdout**
(282/282, 0 parse failures, 0 audit violations, $33.49 = $0.119/item list): recall@5 %FPR
**0.333** [0.196, 0.495] vs the a2 sonnet(4.6) baseline 0.167 — **the pre-registered
full-stack gate (Δ ≥ +0.10) is MET at +0.167** (McNemar 10/3, p(1s) 0.046; dAUC vs
a2-sonnet46 +0.039, null — an operating-point gain, not a global-AUC gain; AUC 0.764, the
best single-draw ranker measured). θ_E ≤ 1″ COWLS 3/7 (was 1/7); lit_cluster still 1/13;
stress_D remains the advocate blind spot (mean S 0.312, highest class; 3 of its 5
provisional-A/B letters are PI-refuted panels). **a2/sonnet5 was stopped by the
pre-registered spend rule** (projection ~$30 > the $20 trigger; the smaller model thinks
longer at xhigh) at 20/282, $2.14 — interrupted, resumable under its registered tuple
(≈ $28 list to complete), so the tier-vs-thinking decomposition is open. Full tables:
`outputs/truth_results_claude5.csv` + `outputs/truth_summary_claude5.md`;
`GOLDEN_FINDINGS.md` 2026-08-23 Claude-5 entry. Whether to fund a1 opus5-xhigh (est.
$54–65, engaged items ≈ $0.39–0.57 ⇒ `--cost-cap ≈ 0.60`), resume sonnet5, or stop is a PI
decision, recorded there, not made here. Phase spend $35.63 of the $70 cap.

## Status addendum (2026-08-24): completion phase — sonnet5 control COMPLETE, a1-opus5 paused mid-run (resumable)

The PI funded the a1 opus5-xhigh arm (gate met; registered with its funding note,
`REGISTRY.md` lines 136 + 168–178) and the sonnet5 completion. **a2/sonnet5 is complete**
(282/282 by resuming the interrupted registered replicate; meta tuple verified,
`rescored=False`, 0 parse failures, 0 audit violations, $33.63 = $0.119/item list):
recall@5 %FPR **21.4 %** [10.3, 36.8], AUC 0.676. The decomposition the control was funded
for: the generation+thinking bundle (sonnet46 → sonnet5) is +0.048 recall / dAUC −0.049
(both null) while the **Opus tier (sonnet5 → opus5) carries +0.119 recall / dAUC +0.088**
(DeLong 0.064) of the +0.167 joint gate delta — directional (neither leg significant
alone), tier-dominant. sonnet5 is also the first arm to move the dead cluster stratum
(lit_cluster 4/13, 3 unseen by opus5) while collapsing on COWLS (1/16); its stress_D
D-rate is 0/20 with 3 of its 5 provisional-B letters PI-refuted panels — the advocate
blind spot is generation- and tier-independent. **a1/opus5-xhigh launched and was paused
by the pre-registered $80 projection rule at 55/282** ($15.52; no `.meta.json`; an
interrupted registered replicate, resumable with the identical command). The crossing was
a composition artifact (manifest orders positives first: all 42 positives in the first 55
rows, early τ0 engagement 34.5 % vs 17.2 % expected on the remainder); the informed
projection is ≈ $61 total — inside the funded $54–65 — with ≈ $45 remaining. The 42-positive
partial (descriptive, no endpoint) shows the critic product under-ranking the advocate
again (13 of a2-opus5's 14 detections fall below the same 0.16 level through the stack;
arbitrator repair not recovering parity so far; stress_D not yet reached). Analysis:
`outputs/truth_results_claude5.csv` (865 rows, v2 driver, deterministic) +
`truth_summary_claude5.md` (rewritten); `GOLDEN_FINDINGS.md` 2026-08-24 entry. Sonnet-frozen
t_A/t_B run hot on Claude-5 advocate scores (t_B 8 % FPR on opus5, 15/51 % on sonnet5) —
the design-half opus5 calibration (~$34 advocate-only) is required before deployment
letters on Nate's opus5 backend. Completion-phase spend $47.01 of the $110 cap; open PI
decisions: resume a1-opus5 (~$45), fund the calibration run, stratify the projection rule.

## Status addendum (2026-08-24, later): a1-opus5-xhigh COMPLETE — the Claude-5 full stack is a veto, not a ranker; 51-row API-incident parse-fail cluster reported

The interrupted a1/opus5-xhigh replicate was resumed to completion under its registered
tuple (identical command; scored names skipped; meta tuple == registered row on all 13
fields, `rescored=False`, audit_traces 520 events 0 violations, 65 frame units kind=eval;
$53.51 total = $0.190/item, inside the funded $54-65 envelope; the resumed tail engaged
17.6% vs the 17.2% informed projection). **One 8.5-minute API window left 51/282 rows
(18.1%) as empty-response $0 parse failures — 49 advocate + 2 critic calls, all negatives,
recorded per the registered policy (S = NaN, rate reported); endpoints use 149/200 N1**
(a registered rescore could top them up, ~= $6 — PI decision). Results
(`outputs/truth_results_claude5.csv` 1116 rows, byte-reproducible;
`truth_summary_claude5.md` rewritten; `GOLDEN_FINDINGS.md` 2026-08-24 (later) entry):
**recall@5%FPR 9.5% (4/42) S and S_arb, AUC 0.468 / 0.484 — the critic product collapses
the a2-opus5 advocate's ranking (paired dAUC −0.301 [−0.424, −0.184], DeLong < 1e−5,
McNemar 1/11 p 0.0063), the first SIGNIFICANT under-ranking**, unrepaired by the
arbitrator, and indistinguishable from a0 as a ranker (dAUC −0.070, p 0.24). 3 of the 4
"detections" are un-engaged advocate 0.14s (one hundredth below tau0 — the tau0 cliff).
What the stack buys: **stress_D D-rate 6/20 = 0.30, mean S 0.312 → 0.085, 0 stress_D at
A/B** (a2-opus5 had 3 in its top 5 letters) — the advocate's stress_D letter inflation is
fixed by the full stack at this tier; needs_human 4/282 = the surviving COSMOS-Web arc
(S 0.40 / S_arb 0.60, letter_arb B) + the 3 highest-S stress panels. Monitors pass
(forbidden_only 0/128; no_opinion ≤ 0.017; Spearman −0.006). Verdict: **ranker =
a2-opus5-xhigh advocate-only; the a1 stack is a demotion/veto + escalation layer; no
deployable Claude-5 letters** — sonnet-frozen thresholds transfer in neither
representation (hot on p_ev, cold on S), so the design-half opus5 calibration (~$34
advocate-only) is the prerequisite for letters on Nate's opus5+adaptive backend, with the
`opus5_api: null` patch regeneration + README caution recommended pending PI approval.
lit_cluster 0/13 (still moved only by sonnet5-xhigh, 4/13). Finishing phase $37.98 of its
$60 cap; funded completion phase $84.99 of $110; Part 2 cumulative $234.36 (all-in
$244.20 with the scrambled-100 one-off). Suite 216 passed.

## Status addendum (2026-08-24, later still): v2-deploy executed — `opus5_api` letters calibrated on design, the 51-row top-up landed (endpoints unmoved), the transfer check selects R2 (D-rule certification only); blind opus5 top-100 in progress

The deployment rule pre-registered in `REGISTRY.md › Deployment rule v2-deploy` was executed
in order. **Item 1:** a2/opus5/adaptive/xhigh on the DESIGN half (288/288 scored and parsed,
$34.79 = $0.121/item; ungated, provisional tuple) → `golden/calibrate_thresholds.py` on the 200
non-anchor design negatives only: **tau0 0.15 / t_A 0.20 (design FPR 1.0 %) / t_B 0.17 (4.0 %)**,
design-positive recall 0.318 / 0.341 reported not fitted; `thresholds_v2.json` sha16
`2446548c6eabbcf1 → f59540c41b302966` (previous file at `outputs/thresholds_v2.pre_opus5.json`),
resolved tuple `a40ae6e201a03e65` (provisional) → `424a8aa9875bacd2` (`opus5_api_calibrated`).
The registered opus5 holdout parquets keep the provisional sha they were scored under; their
calibrated letters are derived, never re-scored. **Item 9:** `run_truth_eval.py --only-nan`
re-scored the 51 NaN rows of the a1/opus5-xhigh holdout parquet (49 advocate transport + 2
artifact parse failures, all negatives) and nothing else — 282/282, 0 NaN, $7.72 metered,
`*.pre_topup_20260824T201714Z` copies kept, the `top-up(only-nan): …` row appended to
"Truth-eval rescores"; parquet total $60.48 ($0.2145/item). Endpoints recomputed on 282/282
(`outputs/truth_results.csv`, 246 rows) against the 231-row values above: **a1 P1 recall@5 %FPR
0.095 (4/42) [0.027, 0.226] → 0.095 (identical), AUC 0.468 → 0.469 (S_arb 0.484 → 0.486), P2
0/149 → 0/200 at the stored provisional letters, P3 0/42 → 0/42, stress_D D-rate 0.30
unchanged, forbidden-only 0/136, parse-failure rate 18.1 % → 0.0 % [0.0, 1.3]**; a2 unchanged
(0.333 [0.196, 0.495], AUC 0.764, P3 1/42). The 51 new negatives land where the 149 were (max
S 0.14); the previous verdict stands verbatim. **Items 6–7** (`golden/transfer_check.py`, zero
API, rebuild parity 0 mismatches on both parquets): letter_rank (a2 parquet) FPR@A 0.0 %
[0.0, 1.8], FPR@A∪B 3.0 % [1.1, 6.4], **P2 PASS**, recall@A∪B **28.6 %** [15.7, 44.6], stress_D
at A∪B 16/20; R1 (a1 parquet) FPR 0/200 at both, recall@A∪B **2.4 %** [0.1, 12.6], stress_D
4/20; R2 (a1 parquet) FPR@A∪B 2.0 % [0.5, 5.0], recall@A∪B 23.8 % [12.1, 39.5], stress_D
10/20. **Selected rule: R2** (0.0238 < 0.5 × 0.1429, the pre-stated fallback). Read honestly:
the calibrated advocate letters transfer (t_B 4.0 % design → 3.0 % holdout) and the A tier is
limited by the A criteria guard, not the threshold (11/42 positives clear t_A, 1 is lettered
A); the arbitrated stack (R1) is still far too aggressive on true lenses — it removes ten of
the a1 parquet's own eleven advocate A/B positives, mostly on `companion_projection` — so the
deployed certification is the D-rule only, which buys about one point of FPR and removes 6 of
the 16 stress_D inflations for about five points of recall (cross-parquet; same-parquet the
D-rule itself costs one positive and removes three stress_D). Confound, stated: letter_rank is
measured on the a2 parquet and R1/R2 on the a1 parquet — independent draws of the same advocate
prompt, agreeing to within one positive at threshold level; the selection rule named this
pairing before the check ran. R2 still leaves 10/20 PI-refuted panels at A∪B; `needs_human`
and the located veto stay on every deployed row. **Item 8:** the blind opus5-xhigh top-100
(`regrade_scrambled.py --model opus5`, `outputs/scrambled100_opus5/`) is IN PROGRESS at the
time of writing — no `.meta.json`, nothing read, results to be appended (then `--reletter`
under `opus5_api`, rule R2). Tooling: `golden/records.py`, `calibrate_thresholds.py`,
`transfer_check.py`, `explain.py`, `annotate.py`, `run_truth_eval --only-nan`,
`regrade_scrambled --model opus5 / --reletter`. Spend this phase $42.51 metered (34.79 +
7.72); Part 2 cumulative $276.87; all-in $286.71 (the running top-100 excluded). Full
tables: `GOLDEN_FINDINGS.md` 2026-08-24 (later still) entry; `outputs/transfer_opus5/`.

## Budget and cut order (re-derived 2026-08-23 from the built halves and the smoke's per-role costs)

The plan priced 244 holdout / 265 design items at $0.057–0.084 per item. The built halves are
**282 holdout / 288 design** (stress rows included: the D-rate on stress_D is a registered
secondary, so they are scored; no `--exclude-stress`), and the integration smoke measured
(`outputs/smoke/*_votes.parquet`) advocate $0.021–0.027, critics $0.027 / $0.032 / $0.050,
arbitrator $0.026, A0 $0.006–0.010 per call. At 45 % τ0 engagement the expected full-stack
item is ≈ $0.024 + 0.45 × ($0.109 + 0.8 × $0.026) ≈ **$0.083** (the engaged item ≈ $0.16, the
`--cost-cap 0.17` default); advocate-only ≈ $0.024; A0 ≈ $0.024 (3 calls).

| step | items × $/item | $ |
|:--|:--|--:|
| advocate-only design iterations ×3 | 3 × 288 × 0.024 | 21 |
| full stack on design ×2 | 2 × 288 × 0.083 | 48 |
| attribution arm, 130 design items | 130 × 0.083 | 11 |
| test–retest, 60 design + 5 anchors × k=3 | 195 × 0.083 | 16 |
| A0: holdout 282 + 40 agreement units | 322 × 0.024 | 8 |
| holdout a1 + a2 + a3 | 282 × (0.083 + 0.024 + 0.024) | 37 |
| R2: a2 k=3 on the WHOLE holdout (subsets are refused on the holdout; the plan's 144-item R2 is not runnable) | 282 × 3 × 0.024 | 20 |
| Opus full stack on holdout (1.67 × Sonnet) | 282 × 0.083 × 1.67 | 39 |
| subtotal | | 200 |
| + 10 % parse/repair overhead | | **≈ 220** |
| gated arm on holdout (ctx20 or R), only if design Δrecall@5 % FPR ≥ +0.10 | 282 × 0.083 | +23 |

**Cut order** (pre-registered): (1) the gated arm if design Δ < 0.10 — expected, already
outside the subtotal; (2) attribution 265 → 130 — already applied; (3) A0 agreement units
83 → 40 — already applied; (4) the second full-stack design pass (−$24). After (4): ≈ $176 +
10 % ≈ **$195**. The plan's "$150–160" is not reachable at the measured per-role costs with
282/288 halves; the further levers, in order, are R2 at k=2 (−$7), test–retest on 40 items
(−$6), and trimming the critics' `measured`/`notes` output (the critic calls are 60 % of a
full-stack item). **Never cut**: holdout a1/a2/a3, A0 holdout, R2, Opus. Nate's 362 run on his
subscription, not this budget. This is a budget decision for the PI, recorded here, not made
here.

## Embargo notes

- Prompts are item-agnostic by construction (no candidate id, coordinate, rank, grade) and every
  persona file, the v2 note, every gloss string, `render_v2_desc.md` and every composed system
  prompt / VIEW text is 4-gram-checked against `golden/banned_lexicon.txt` (PI comments + holdout
  ids; `audit_traces.py --build-lexicon --pi-only`); every TRACKED `.py`/`.md` under `golden/`,
  `prompts/` and `tests/` is checked the same way (`test_golden_prompts`; a coding agent reads
  them); a holdout run without the lexicon, or with a lexicon that lacks any holdout id or PI
  comment, is refused; inspector claim bodies are checked before the call (a hit blanks the
  claim); `audit_traces.py --traces-dir` checks every trace afterwards (`n_images == 1 +
  n_exemplars + n_extra_views`, full-text lexicon check).
- The PI's 16 comments shaped MECHANISMS only (design-only exception, `golden/README.md` rule 1);
  they are never shown to a model, never scored, and the five anchors are predictions, not truth.
- `prompts/jwst_note.md` (v1) carries "bluer" and "default C or D"; the panel passes `note=` explicitly
  (`jwst_note_v2.md`, role-neutral) and the incumbent arm takes no note. Footers (id, RA/Dec,
  r-mag) are cropped from every view; critics see the advocate's items and scale class only.
- Frame units scored here are marked `kind="eval"` in `golden_registry.csv` (a permitted
  zero-shot exposure); non-frame truth rows are not registry rows. A truth-tuned rubric enters the
  golden campaign only on `validate ∖ truth-design`. Nothing model-generated reaches the kit.
- `J` is read-only; the Nate drop-in lives under `golden/verifier_patch/` and is applied by him.
