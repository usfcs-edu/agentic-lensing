# LensJudge v5 — findings log

**Goal.** FULLY Claude-free vetting: open-weight models at runtime AND in training (human catalogs + sims;
open teacher permitted); gates vs expert labels; Claude's v4 numbers = frozen reference bars only.
Plan: `~/.claude/plans/i-want-to-do-melodic-clover.md` (approved 2026-07-03). Research basis: 30-agent
verified sweep (workflow wf_b9e0ba7d) — Qwen3.5 generation, SuGOHI/HOLISMOKES data unlock (1,591+ A/B),
unfrozen-ViT recipe, logprob scoring, vllm-mlx Mac tier.

## Phase A0 — client/serving modernization (DONE, commit 2add797)
- `LENSJUDGE_STRUCTURED` shim: auto-probes server `/version` once → vLLM ≥ 0.24 `structured_outputs:{json}`,
  else legacy `guided_json` (0.24 deprecates it). Forceable `new|legacy`.
- **Grade-token logprob scoring**: `extract_grade_probs` reads P(A/B/C/D) at the `"grade":"X"` position
  (handles `A` vs `"A` tokenizations, top-logprob sums capped at 1); `p_lens_logprob` = P(A)+P(B)
  (uncalibrated); plumbed through direct + agentic loop into parquet columns `gp_A..gp_D`,
  `p_lens_logprob`. `LENSJUDGE_LOGPROBS` default ON.
- Env-gated STRICT tool schemas (`LENSJUDGE_STRICT_TOOLS=1`): strict:true, all-required + nullable
  optionals, additionalProperties:false — off until a live gate shows it beats default.
- `NOTHINK_SERVER=1` serve-script flag → `--default-chat-template-kwargs '{"enable_thinking":false}'`
  (vLLM ≥ 0.24 only; client `LENSJUDGE_NOTHINK` stays as per-request override).
- Tests 14/14 client (+7), 8/8 tools (+1); full suite green.

## Phase A live validation №1 — leakage probe + logprob re-gate (2026-07-03, gpu3, Qwen3-VL-8B)
**Leakage probe (`eval/probe_logprob_leakage.py`): HEALTHY.** Guided-ON vs OFF top-10 at the grade token
are virtually identical (C −0.10 / B −2.35 / D −9.88 / A −16.9 both ways; probs C 0.905 / B 0.095). No
grammar-mask leakage on vLLM 0.23 + guided_json → logprob calibration is trustworthy on this stack.

**Generated vs logprob p_lens (270-row LensBench-VI, human labels, parse 1.00, logprob coverage 1.00):**

| score | detection AUC (lens vs random) | lens-vs-mimic AUC |
|---|---|---|
| generated p_lens | 0.476 (= v4 baseline, exact reproduction) | 0.390 |
| **logprob p_lens** | **0.502** | 0.373 |

**Honest read:** a wash at DESI tier-1 — +2.6 pt detection purely from tie-breaking the collapsed generated
values (0.04/0.09 ties), −1.7 pt mimic. The tier-1 wall is perception, not scoring (as measured all through
v4). The machinery's value is downstream: tier-2 calibration, AUC-based selection granularity, and the
ensemble (which averages exactly these grade-token distributions).

## Next (in flight)
- Phase A ensemble experiment: prompt (rubric v1/v2) × render (lupton/arcsinh) config sweep on the 8B,
  mean-of-grade-distributions ensemble vs best single config (mandatory beat-single-pass gate).
- Then: cascade calibration + Perlmutter DR11 deploy (Phase A), backbone gates (Phase B), data campaign
  (Phase C).
