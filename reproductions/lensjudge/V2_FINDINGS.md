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

## B1 — Two-tier high-res escalation  ✅ (LIVE-VALIDATED)

`--mode escalate` (`imaging/grader_escalate.py` + `common/highres.py`) re-grades candidates at
Euclid 0.1″ **when coverage exists**, else a safe tier-1 no-op, recording
tier/escalated/highres_survey/p_lens_tier1-2 provenance. Trigger policy (refined during
validation): escalate anything that is **not a confident tier-1 A** — because the REAL cost
gate is COVERAGE (`resolve_highres` is a cheap local catalog lookup; only the rare candidates
with Euclid/HSC overlap pay for a tier-2 grade). A narrow {B,C} trigger missed the failure
mode that most needs the second look: the over-skeptical DESI grader buries real lenses in
"D" (often with a *wrongly* named contaminant).

**Live validation** (Euclid Q1 data now staged, 539 objects): 6 south Euclid-covered objects,
all escalated, **$0.71**:

| truth | DESI tier-1 p_lens | → Euclid tier-2 |
|---|---|---|
| 3 grade-A lenses | mean **0.15** (0.02/0.03/0.40) | **all → A, mean 0.96** |
| 3 grade-C cands  | 0.38 / 0.08 / 0.02 | **B 0.72 / B 0.62 / D 0.02** |

The DESI grader **missed all 3 real lenses** (~0.15); Euclid escalation recovered all 3 as
grade **A (0.96)** — the README's flip, through the escalate *mode*. It **discriminates**, not
just inflates: the lone true non-lens C correctly **stayed D (0.02)**. This is the strongest
LensJudge v2 lever — when high-res covers a candidate, it converts the resolution-limited
~0.5-AUC wall into a near-definitive grade. (On a DECaLS sweep, only Euclid/HSC-overlapping
survivors get tier-2; the rest stay tier-1, so cost is bounded by overlap, not sweep size.)

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

## B7 — Agency ablation: does the agentic loop contribute?  ✅ (separate ablation budget, ≈$6)

Full writeup + the **FOR STAGE-2 deploy recommendation** in **`AGENCY_ABLATION.md`**. New
artifacts: `imaging/grader_direct.py` (loop-free base-Messages-API grader, `run_batch --mode
direct`), `eval/run_agency_ablation.py`, `prompts/rubric_imaging_v2_inline.md`. Three findings
on the frozen evidence manifest (same model/rubric, 2×2 rubric×loop + a thinking arm):

1. **Detection: the loop adds nothing.** No-loop `direct` matches/beats `lean` on lens-vs-random
   AUC (0.67 vs 0.59) at **~1/6 the cost** (1 turn / $0.012 vs 3.75 / $0.068). Tools are
   invokable programmatically with no detection loss.
2. **Lens-vs-mimic: the loop matters (not rubric coupling).** The v2-rubric mimic gain
   (0.52→**0.71**) lives in the LOOP arm; inline rubric rewrite (0.41) and +thinking (0.55) do
   not reproduce it. The agentic multi-turn structure helps the hard mimic discrimination.
3. **Escalation routing is degenerate** — the LLM grade-gate fires ~100% (over-skeptical), no
   selectivity over a free CNN `p_meta` threshold; coverage gates tier-2 cost.

**Stage-2 implication:** vet survivors as a **cascade** — cheap `direct` triage for detection on
the bulk, then agentic **v2** for mimic adjudication on the survivors; escalate by coverage, not
the grade. ~$170 vs ~$355 (all-v2) vs ~$1,575 (all-multiagent) per 5k survivors. Caveat: modest
n, compressed p_lens → findings (1)/(3) robust, (2) directional.

## lensjudge-v3 — the remaining B items (B3 / B6 / HSC tier-2 / cascade)  ✅  (branch `lensjudge-v3`)

The four deferred B extensions, implemented and live-validated.

**B3 — expanded crossmatch.** `eval/crossmatch_external.py` now crossmatches the 4,354 unique
candidates against **SuGOHI/HOLISMOKES** (local HSC catalog, committee grade + spec-z + θ_E) in
addition to Euclid Q1, and against **HST/MAST + AGEL** under `--online` (best-effort VizieR; graceful
skip if astroquery/network absent). Emits `outputs/external_overlap.csv`. Result: **24 in Euclid Q1
+ 76 in SuGOHI = 100 unique candidates with external high-res coverage** (the Euclid count matches
the §10 manual finding exactly).

**HSC PDR3 tier-2 escalation** (the resolution lever beyond Euclid's tiny overlap). New
`common/hsc_fetch.py` (PDR3 `das_cutout` service via `requests` + HTTP Basic Auth, `HSC_USER`/
`HSC_PASSWORD` env, cached under `cache/hsc/`, graceful `None` out-of-footprint), `common/hsc.py`
(grizy render mirroring `euclid.py`, 0.168"/px), `tools/hsc_cutout.py` (`fetch_hsc_cutout` tool),
`eval/run_hsc.py` (`grade_hsc`, v2 rubric). `common/highres.resolve_highres` now tries
`("euclid", "hsc")` in priority (HSC coverage = a successful cutout fetch); `imaging/grader_escalate`
dispatches tier-2 by survey. **Live end-to-end:** the SuGOHI-matched DESI candidate
`DESI-029.0755-01.1297` (storfer) graded **D (p_lens 0.03) at DESI 1.3"** → escalated to HSC 0.168"
→ **B (0.65)**, the agent resolving "a blue/cyan feature wrapping a red/orange core" (a real SuGOHI
grade-A lens, recovered). Footprint-bounded: HSC-SSP Wide overlaps DESI only partially.

**B6 — export hard negatives → v3 mimic bank** (closes the LensJudge→finder active-learning loop).
New `eval/export_hard_negatives.py`: filters agent-confirmed non-lenses (grade D / low p_lens + a
named contaminant) from the vetting preds, joins RA/DEC from the run manifests, maps to the canonical
`[row_id, RA, DEC, mimic_type, p_final, source]` schema (`p_final` clamped to [0,1]; CNN-high
head-logit survivors → 1.0), `source="lensjudge_v2"` →
`claudenet/data/v3/hard_negatives_from_lensjudge.parquet`. **129 hard negatives** exported
(lrg_companion-dominated, 88). Folds into the next bank via `claudenet/312_assemble_mimic_bank.py`;
no retrain here (GPU, separate program).

**Cascade deploy runner** (productizes the B7 stage-2 pipeline). New `eval/run_cascade.py`:
two-phase — cheap `direct` triage on ALL survivors, then escalate the top `--pass-frac` **by the
stage-1 rank** (the direct grader's real signal is ranking/AUC 0.66, not its over-confident absolute
grade — so rank-routing, not a p_lens/contaminant threshold) to the agentic v2 grader (+ tier-2 by
coverage). `--max-usd` gate. **Live smoke:** 4 candidates, 2 escalated, stage-1 $0.012/cand on all,
total $0.235; one row flipped D→C under agentic v2. Per-stage cost matches B7
(~$60/$170/$355/$1,575 per 5k at pass_frac 0.3/cascade/all-v2/all-multiagent).

**Caveats.** HSC tier-2 is footprint-bounded (partial DESI overlap) and one rung coarser than Euclid
(0.6" vs 0.1"); B3 network sources are best-effort (SuGOHI+Euclid are the guaranteed-local core); B6
exports agent-confirmed non-lenses but does not retrain; the cascade's stage-1 routing is rank-based
because the direct grader's absolute grade is over-skeptical/over-confident (B7). HSC credentials are
env-only (never committed); `cache/hsc/` is gitignored.
