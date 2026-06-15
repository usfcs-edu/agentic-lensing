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

**Next:** B4 (rubric + few-shot for LRG+companion/ring, no $) → B1 (Euclid escalation, no $) →
one small `--grade` evidence run comparing v1-lean vs v2 on Benchmark A (the gated spend).
