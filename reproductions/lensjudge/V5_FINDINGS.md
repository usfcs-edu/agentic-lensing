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

**Phase B gate №2 — Qwen3.5-27B-GPTQ-Int4 (Perlmutter A100-40, job 55434890, 19 min):** fresh DESI bench
**detection 0.614/0.611, mimic 0.454/0.420, parse 1.00** — ≈ the 9B (0.621). **The GENERATION helps
(+0.07); 9B→27B SCALE adds nothing at tier-1** (params-don't-clear-the-wall, reconfirmed on the new
generation). The 27B's decisive test is tier-2 (HSC), where capacity should matter. Ops: the job's HSC
leg parsed 0/72 because the gate server had no tool parser and `run_hsc` is agentic — GATE2 switched to
the DIRECT HSC grader (`distill_hsc label`, inline views, no tools; also the better perception gate);
all three gate jobs resubmitted (55440600 bf16 / 55440603 Int4 / 55440604 Qwen3.6-TP2).

## ★ Phase B HEADLINE — Qwen3.5-27B bf16 MATCHES the frozen Claude bar at tier-2, zero-shot (2026-07-03)
HSC real-lens gate (26 confirmed SuGOHI lenses vs 46 true non-lenses; DIRECT inline views; parse 72/72):

| grader | real-lens AUC | recovery | rejection | mean p_lens L/N |
|---|---|---|---|---|
| off-the-shelf old-gen (artifact line) | 0.730 | 8% | 96% | 0.15/0.09 |
| Qwen3.5-27B-GPTQ-Int4 | 0.778 | 38% | 93% | 0.39/0.09 |
| Qwen3.6-27B bf16 (TP2) | 0.780 | 46% | 85% | 0.45/0.16 |
| **Qwen3.5-27B bf16** | **0.823** | **50%** | **91%** | **0.51/0.11** |
| frozen Claude reference | 0.823 | 54% | 89% | 0.46/0.12 |

**An open-weight model matches Claude's real-lens vetting at HSC — zero-shot, offline, Claude-free** (even
the p_lens calibration profile mirrors Claude's). The v4 tier-2 collapse was a model-GENERATION problem.
Caveats: n=72 → the exact AUC tie is luck; the defensible claim is "statistically indistinguishable";
360-row test2 confirmation submitted. Quant matters at tier-2 (Int4 −4.5 pt AUC / −12 pt recovery →
bf16 for tier-2, Int4 for sweeps). Qwen3.6 ≠ better at tier-2; slightly best at tier-1 (det 0.661/0.682,
mimic 0.523/0.529 — first hair above chance, still ~noise).

**Phase-B picks locked (pending test2 confirm):** tier-2 grader = **Qwen3.5-27B bf16** (1×A100-80 or TP2
A100-40; ~20-min job incl. boot for 72+330 gradings); tier-1 = still CNNs for detection (27B det
0.654–0.682 fresh-bench); Phase D fine-tune (41× corpus + unfrozen ViT) is now UPSIDE over Claude parity,
not rescue; 397B teacher gate demoted to optional (27B already at the teacher-quality bar).

**test2 CONFIRMATION (360 rows: 120 typical SuGOHI A/B lenses vs 240 XVI expert rejects; job 55444848,
22 min, 360/360 parse) — the headline gets an essential SCOPE ASTERISK:** Qwen3.5-27B bf16 on the
HSC-NATIVE population = **AUC 0.563, recovery 15% (catalog-A 12% n=26, B 16% n=94), rejection 91%,
mean p_lens L 0.17 / N 0.11.** Interpretation: the 72-gate's 26 lenses are the DESI∩SuGOHI crossmatch —
lenses bright enough to be found at 1.3″ seeing, i.e. the EASIEST sub-population, and that is the only
bench Claude's frozen 0.823/54% was ever measured on. **Zero-shot parity with Claude holds on that
DESI-bright subset; typical HSC-native lenses (fainter, smaller θ_E, vs near-miss rejects) remain
unsolved zero-shot.** Notes: test2's bench is harder by construction (240/240 negatives are
CNN-selected-expert-rejected; some near-miss rejects with G→1.5 may be real lenses; catalog-B positives
are "probable" not certain — label noise lowers the achievable ceiling somewhat, but 0.563 is far from
any ceiling). **Consequence: Phase D's fine-tune on the 3,913-example corpus (drawn from exactly this
population) is the value path for real tier-2 capability; test2 is its primary gate (model-to-model +
fine-tune deltas; no Claude number exists for it under the Claude-free policy, by design).**

**Qwen3.6 question (user asked; live-verified 2026-07-03):** Qwen3.6-27B + 35B-A3B EXIST (Apr 2026,
multimodal); no 9B, no 122B, **no official Int4 yet** (bf16-only → hbm80g or TP2). Architecture is
literally `Qwen3_5ForConditionalGeneration` — 3.6 = refreshed weights on the SAME arch/vision stack
(sweep's flag verified at config level; claimed gains are agentic, our constraint is perception —
empirical question). Decision: **3.6-27B added to the Phase B gate list** (staged to Perlmutter; TP2
2×A100-40 fast-lane slurm ready) — if its weights gate better, the identical arch makes it a zero-cost
swap for the Phase D fine-tune. 3.5 keeps the Int4/sweep + 9B/student/Mac + 122B/397B tiers regardless.

## Phase D run №1 — MODE COLLAPSE, diagnosed + fixed (2026-07-05)
First v5 fine-tune (Qwen3.5-9B, QLoRA + unfrozen ViT, 5,433-example corpus, GDN/FLA recipe validated on
sm75, fp16 nan-rate 3/54 = tolerable): train loss crashed 0.94→0.02 while **valsel AUC fell to chance
(ckpt-75: 0.533 predicting "C/0.45" for 94% of rows; ckpt-150: 0.500 predicting "D/0.0" for 180/180)** —
the student emitted class templates and ignored the images entirely. Run killed at step ~265; the queued
27B A100 job was HELD before submission.

**Root cause (matches the v4 v1-failure at larger scale): CLASS-CONSTANT targets.** My v5 targets used 3
fixed rationale strings + fixed per-grade criteria blocks → ~95% of target tokens were class-deterministic
→ cross-entropy is minimized by emitting the class prior with no vision conditioning. The Euclid PoC
worked precisely because Claude's targets were PER-EXAMPLE UNIQUE — the only way to reduce loss was to
look at the pixels.

**Fix (distill_hsc._v5_target): per-example unique targets from catalog facts** — short fact-bearing
rationales (coords, θ_E where present, committee score, 3 hash-selected phrasings), criteria correlated
with the soft score + deterministic hash jitter, p_lens de-constanted (±0.04 hash jitter on letter-grade
mappings; XVI continuous scores kept exact), varied confidence; plus LR 1e-4→5e-5 and warmup 0.05→0.1.
**Standing rule addition: SFT targets must be per-example unique — verify target diversity before every
training run** (uniqueness check now part of the build).

## Phase D run №2 interim (2026-07-05) — collapse fixed, generalization still missing at 9B/fp16
Run-2 (unique targets, LR 5e-5, warmup 0.1, vit_lr 5e-6): train loss plateaus at ~0.39 (vs run-1's 0.02
memorization crash), grads finite — the template shortcut is structurally gone. But valsel AUC at
ckpt-75/150/225 = 0.481/0.493/0.508 (rising, ≈chance), and ckpt-225 still outputs grade-D for all 180
valsel rows (p_lens spread only 0.01–0.20): **memorize-without-generalize** — 51M LoRA params fit 5,435
train pairs; unseen images fall back to the prior. Two recipe gaps identified: (1) **ZERO augmentation**
in our SFT (CNN lens-finders depend on dihedral augmentation; rotations/flips are label-preserving) —
queued for run-3; (2) possible 9B-capacity/fp16 limits — the **27B bf16 A100 run tests this directly
(job 55537116, submitted with the unique-target jsonls; the stale class-constant jsonls on Perlmutter
were caught and replaced first)**. Phase E's AION-1 frozen-encoder probe launched in parallel on gpu3
(GPU 2): a small head cannot memorize like LoRA — if it generalizes from the same corpus, the corpus is
good and VLM-SFT is the bottleneck; if it also sits at ~0.55, investigate corpus/renders/labels.

## Phase C groundwork (2026-07-03; zero-GPU, catalogs in gitignored cache, curl-regenerable)
HOLISMOKES tables pulled + profiled: paperVI 467 rows; paperXIII 162+384; **paperXVI 14,152 expert-graded
rows: 598 A/B-like positives (G≥1.5) + 12,880 expert REJECTS (G<1.0) with continuous G scores (0–3,
median 2 graders)** — the rejects are CNN-selected-but-expert-rejected = the ideal HSC hard-negative bank,
and G/3 is a ready-made human-soft p_lens target. Remaining Phase C: SuGOHI DB scrape, GALAXY CRUISE,
IRSA Euclid fetcher, dedup + `build_corpus_v5.py`.
