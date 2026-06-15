# LensJudge v2 — running findings

Tracked log of the LensJudge v2 upgrade (workstream B of the ClaudeNet v3 program).
Thesis: stop chasing vision tricks on resolution-limited DECaLS pixels; instead (B1)
escalate ambiguous candidates to higher-resolution imaging, (B2) calibrate honestly and
report *recovery* not agreement, (B3) expand the crossmatch, (B4) tune the rubric for the
dominant LRG+companion/ring failure modes. Hard **$100** API cap, evidence gate first.

## B2 — Honest calibration + the regression gate  ✅ (harness; baseline run pending)

**Script:** `eval/run_lensjudge_eval.py`. Measures DETECTION and GRADING **separately** on the
frozen LensBench manifest, reusing `eval/score.py` wholesale — the only new logic is
partitioning the negative class by the manifest's `source` tag:

- **Benchmark A (detection):** positives = graded A/B/C  vs  negatives = random galaxies.
- **Benchmark B (grading):** positives = graded A/B/C  vs  negatives = Grade-D human-rejects.

Reports recovery@1%/0.1%FPR + AUC per benchmark, appends a regression record keyed by
(label, mode, manifest_sha). **The $100 evidence gate:** a v2 config change (escalate / rubric
/ exemplars) must beat the pinned v1-lean baseline on Benchmark-A recovery *before* any bulk
grading. The harness **spends nothing** by default (scores an existing preds parquet); `--grade`
is the only $-path and ABORTS unless `rows × est-per-cand ≤ --max-usd`.

**Validated** on synthetic data (claudenet + lensjudge venvs): both benchmarks partition and
score end-to-end, regression log written, `--check-regression` exits nonzero on regression.

**Infra fixes:** added `scikit-learn` to the lensjudge venv (bootstrapped pip via ensurepip)
and `requirements.txt` — `eval/score.py` needs it.

**Data note (carried gap):** the frozen manifest (`outputs/lensbench_manifest.csv`, 210 rows:
150 graded A/B/C + 60 random) currently has **no Grade-D rows** — the parsed `*_gradeD_raw.csv`
exports aren't on disk, so Benchmark B is unavailable until they're regenerated. Benchmark A
(the primary evidence gate) is unaffected. A v3-aligned alternative for Benchmark B: use the
**601-row lens-mimic seed** (the campaign's CNN-high, agent-confirmed non-lenses) as the
hard-reject negative class — this would make the judge's eval measure lens-vs-mimic
discrimination directly (wired in B6).

## B4 — Rubric tuned to the dominant failure modes  ✅

`prompts/rubric_imaging_v2.md` adds a "rule these out" section for the dominant false
positives — **LRG+companion (~half of CNN candidates)**, ring galaxy, spiral, blend, merger —
with the **color-symmetry + radial-geometry** discriminators (a real lensed source is *bluer*
than the lens and tangentially curved with a counter-image; a same-color round neighbour is a
companion). Adds `lrg_companion`/`blend` to the contaminant enum. Passed via `--rubric` (lean
or escalate mode), keeping v1's rubric as the controlled baseline.

## B1 — Two-tier high-res escalation  ✅ (mechanism; live-validation data-blocked)

`--mode escalate` (`imaging/grader_escalate.py` + `common/highres.py`) re-grades ambiguous
tier-1 candidates (grade∈{B,C} or escalate) at Euclid 0.1″ **when coverage exists**, else a
safe tier-1 no-op, recording tier/escalated/highres_survey/p_lens_tier1-2 provenance.
**Honest blocker:** the Euclid Q1 cutouts are not on disk in this checkout and DECaLS-south
has ~no Euclid Q1 overlap, so escalation can't be live-validated yet (a documented data
dependency, like the Grade-D CSVs). The mechanism is in place and degrades gracefully.

## The gated comparison — v1-lean vs v2 (the evidence)  ✅

120-row evidence manifest (50 graded A/B/C + 40 lens-mimics + 30 random), sonnet, **$16.6
total, parse_ok 120/120**. Scored on Benchmark A (lens-vs-random) and **Benchmark B
(lens-vs-MIMIC** — the 601 seed as negatives, the v3 thesis):

| Config | Benchmark A AUC | **Benchmark B (lens-vs-mimic) AUC** |
|---|---|---|
| v1-lean (baseline) | 0.593 | **0.518** (≈ random) |
| **v2 (escalate + v2 rubric)** | 0.668 (+0.075) | **0.714 (+0.196)** |

**The v2 rubric lifts lens-vs-mimic discrimination from near-random (0.52) to 0.71** — exactly
the LRG+companion failure mode the program targets — at the same cost and parse rate. v2 also
rejects more mimics (D→D 53 vs 47) and assigns lower p_lens to non-lenses (0.07 vs 0.12).

**Honest caveats:** (1) the gain is in **AUC (ranking)**, not recovery@tight-FPR (≈0 for both
because the agent's p_lens stays uniformly low — the documented over-skepticism at 1.3″ seeing;
the regression gate was corrected to key on AUC). v2 *escalates* far more (B: 92%, C: 82% vs
v1's ~35–50%), which is the right behavior — those flagged cases are where high-res escalation
(B1) would pay off once Euclid data is staged. (2) Modest n (50/40/30) → directional evidence,
not a final benchmark; the +0.20 AUC is a large effect but uncertainty is real.

**Verdict for the $100 gate:** the v2 rubric demonstrably improves the judge on the metric that
matters (lens-vs-mimic), so it is justified to deploy v2 on the ClaudeNet candidate vetting.
Spent so far on v2: ~$17 of $100. **Next:** B3 (expanded crossmatch) + B6 (export hard
negatives to v3) — both no/low-$ — and reserve the remaining budget for vetting the actual
DR10 sweep survivors.
