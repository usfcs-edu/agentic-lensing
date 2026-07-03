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

## Phase A config sweep + ensemble gate (2026-07-03, gpu3, Qwen3-VL-8B DIRECT, 270-row bench)
Logprob scores (`s_ab` = P(A)+P(B); `s_exp` = Σ w·P, w = A1/B0.7/C0.4/D0):

| config | det (s_ab) | mimic (s_ab) | det (s_exp) | mimic (s_exp) |
|---|---|---|---|---|
| c1 v1+lupton | 0.493 | 0.373 | 0.465 | 0.317 |
| c2 v1+arcsinh | **0.549** | 0.413 | 0.506 | 0.377 |
| c3 v2+lupton | 0.462 | 0.622 | 0.432 | 0.666 |
| c4 v2+arcsinh | 0.509 | 0.655 | 0.502 | **0.684** |
| ENSEMBLE (mean of 4) | 0.538 | 0.425 | 0.470 | 0.530 |

1. **Naive equal-weight ensemble FAILS the beat-best-single gate on all 4 metrics** (honest negative —
   averaging in the mimic-blind v1 configs dilutes; the merger paper's recipe requires reliability
   weighting, which with these spreads would just converge to the best config).
2. **Candidate discovery (PENDING CONFIRMATION — winner's curse risk: best of 4 configs × 4 metrics on one
   270-row bench):** `rubric-v2 + arcsinh + expected-grade logprob score` puts the **DIRECT** 8B at
   **lens-vs-mimic 0.684** — above the whole v4 agentic loop (0.562) and the frozen Claude bar (0.520), at
   ~6× less cost than the loop. The v2-rubric (contaminant-aware) effect is large (+0.25–0.31 mimic over
   v1) and consistent across renders; arcsinh adds ~+0.02–0.03 mimic and +0.05 detection.
3. Detection stays ≈0.5 in every config — the tier-1 wall, unchanged (detection is the CNNs' job).

**CONFIRMATION RESULT (fresh disjoint 330-row bench: 150 new lenses / 120 new mimics / 60 new randoms,
0 overlap; parse 1.00, gp coverage 1.00, 0 errors; mimic composition verified matched — same
confident/visual/type mix):** the discovery **COLLAPSED** — c4 mimic 0.655→**0.499**, c3 0.622→**0.474**
(chance). The "v2-rubric effect" was bench-specific overfit + winner's curse (best of 4 configs × 4
metrics on one 270-row bench with 60 mimics). **Firm negative: at tier-1 DESI resolution, NO
prompt/render/scoring/ensemble lever moves the off-the-shelf 8B** — fully consistent with the perception
wall. Methodological rule now standing: **any config selected on a bench must re-verify on a disjoint
draw before it enters the cascade** (v4's single-bench numbers, incl. the agentic-loop mimic 0.562, are
subject to the same rule — re-measure in flight).

## Phase A closed: complete off-the-shelf tier-1 negative (disjoint-verified, 2026-07-03)
The agentic re-measure on the fresh 330-row bench: **mimic 0.516 (logprob) / 0.479 (generated p_lens,
v4-comparable) vs v4's original-bench 0.562**; detection 0.470–0.482; parse 1.00, 3.0 turns (loop
mechanics unchanged). Together with the direct-config collapse:

**FIRM RESULT — no off-the-shelf tier-1 vetting signal exists in the open 8B in ANY configuration**
(direct or agentic × rubric v1/v2 × lupton/arcsinh × generated/logprob/expected-grade scoring ×
equal-weight ensembles): everything lands at 0.47–0.55 on detection AND lens-vs-mimic, on a disjoint
draw. v4's "loop lifts mimic to 0.562" was bench-specific. Note Claude's frozen mimic bar (0.520) was
itself near-chance — the CNN-selected mimic bank is genuinely hard for image-level graders by
construction; nothing about the loop or prompts fixes perception at 0.26″/px.

**Consequences for the plan:** the Phase-A "DR11 swap at parity" premise is gone — there is nothing
off-the-shelf to deploy. The value path shifts to: **Phase B** (do Qwen3.5's stronger fine-perception
towers move the wall at all? 1-day gates), **Phase C/D** (train discrimination in: 13k expert-labeled
HSC mimics + human-soft targets + unfrozen ViT), **Phase E** (AION-1 encoder probe — SSL/CNN encoders
demonstrably CAN separate at tier-1, e.g. ClaudeNet's contaminant-aware finder). The DR11 vetting deploy
is DEFERRED until a backbone or student shows a fresh-bench win. Methodological rule held: every claim
above is disjoint-draw verified.

## Phase B gate №1 — Qwen3.5-9B (2026-07-03, gpu3)
**sm75 VERDICT: POSITIVE.** Qwen3.5-9B (early-fusion, hybrid GDN) loads (17.7 GiB fp16), JIT-compiles its
Triton/FLA GDN kernels on Turing, serves multimodal requests, and honors `chat_template_kwargs
enable_thinking:false` — on vLLM 0.23 + TRITON_ATTN + enforce-eager. First boot needs ~10–20 min of Triton
JIT (warm cache: ready in 120 s). gpu3 stays viable for the v5 small tier. (Ops gotcha re-learned: killing
the serve wrapper orphans `VLLM::EngineCore` holding the VRAM — clean up by owner-checked PID from
`nvidia-smi --query-compute-apps`.)

**Fresh-bench gate (330-row disjoint bench, direct, v1 rubric, logprob scores):**

| model (same bench) | detection AUC | lens-vs-mimic |
|---|---|---|
| Qwen3-VL-8B best direct config (c2) | 0.549 | 0.413 |
| Qwen3-VL-8B agentic v2 | 0.470–0.482 | 0.479–0.516 |
| **Qwen3.5-9B direct v1 (zero-shot)** | **0.621 / 0.624** | 0.495 / 0.489 |

**First fresh-bench evidence that the backbone GENERATION moves tier-1 perception** (+0.07–0.15 at the
same size class, zero-shot; Claude's 0.663 detection was original-bench-only — not directly comparable
under the disjoint rule). Mimic remains at chance for every off-the-shelf model — the CNN-selected mimic
bank stays a *training* problem (Phase D), not a prompting/backbone one. Next: Qwen3.5-27B gate on
Perlmutter (fresh DESI bench + HSC real-lens test) — jobs 55427172 (bf16, hbm80g) + 55434890 (Int4, fast
gpu partition).

**Qwen3.6 question (user asked; live-verified 2026-07-03):** Qwen3.6-27B + 35B-A3B EXIST (Apr 2026,
multimodal); no 9B, no 122B, **no official Int4 yet** (bf16-only → hbm80g or TP2). Architecture is
literally `Qwen3_5ForConditionalGeneration` — 3.6 = refreshed weights on the SAME arch/vision stack
(sweep's flag verified at config level; claimed gains are agentic, our constraint is perception —
empirical question). Decision: **3.6-27B added to the Phase B gate list** (staged to Perlmutter; TP2
2×A100-40 fast-lane slurm ready) — if its weights gate better, the identical arch makes it a zero-cost
swap for the Phase D fine-tune. 3.5 keeps the Int4/sweep + 9B/student/Mac + 122B/397B tiers regardless.

## Phase C groundwork (2026-07-03; zero-GPU, catalogs in gitignored cache, curl-regenerable)
HOLISMOKES tables pulled + profiled: paperVI 467 rows; paperXIII 162+384; **paperXVI 14,152 expert-graded
rows: 598 A/B-like positives (G≥1.5) + 12,880 expert REJECTS (G<1.0) with continuous G scores (0–3,
median 2 graders)** — the rejects are CNN-selected-but-expert-rejected = the ideal HSC hard-negative bank,
and G/3 is a ready-made human-soft p_lens target. Remaining Phase C: SuGOHI DB scrape, GALAXY CRUISE,
IRSA Euclid fetcher, dedup + `build_corpus_v5.py`.
