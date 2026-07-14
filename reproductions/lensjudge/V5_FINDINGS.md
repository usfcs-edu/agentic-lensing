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

## Phase D run №2 UPDATE (2026-07-05 evening) — the unique-target recipe LEARNS
Run-2b valsel AUC trajectory: 75: 0.481 · 150: 0.493 · 225: 0.508 · 300: **0.554** · 600: 0.563 ·
750: **0.572 — monotonically rising, no peak yet, at 55% of training.** 0.572 already exceeds the
zero-shot 27B's test2 0.563 on the same population construction — from a 9B trained on one Titan RTX.
The ckpt-225 "all-D" snapshot was pre-signal (full LR from step 136 only). Generalization on the
HSC-native population is emerging from the corpus; the collapse fix (per-example unique targets) is
validated as the enabling change. Next: run to completion → full AUC curve → merge best → test2 + frozen
72-gate; 27B bf16 (queued) should start higher and go further; dihedral augmentation queued for run-3.

## ★★ Phase E experiment №1 — AION-1 frozen-encoder probe: the corpus is GOLD (2026-07-06)
AION-1 ran NATIVELY on our HSC cache (its released HSCImage codec expects exactly our pdr3_wide pixels;
0/6,141 rows skipped; frozen encoder, mean-pooled tokens; gpu3 GPU 2). Linear + MLP probes trained on the
corpus (1,382 lens vs 2,509 expert rejects; softmid excluded), selected on valsel:

| encoder | probe | valsel AUC | test2 AUC | test2 recovery @ rej≈0.91 |
|---|---|---|---|---|
| aion-base | logreg | 0.751 | **0.840** | 55.0% |
| aion-large | logreg | 0.773 | **0.841** | **59.2%** |
| aion-base | MLP-256 | 0.795 | 0.823 | 55.8% |
| *zero-shot Qwen3.5-27B* | — | — | *0.563* | *15%* |
| *9B LoRA run-2b (interim)* | *0.572 valsel, rising* | — | — | — |

**A LINEAR head on frozen domain embeddings beats every VLM number we have on the HSC-native population
by a wide margin.** Consequences: (1) corpus/renders/labels validated — strong, linearly decodable signal
(the VLM memorize-not-generalize is a training/representation problem, not data); (2) the decomposed
architecture is now the LEADING tier-2 scorer candidate: AION+head p_lens (and kNN retrieval) as tools,
the VLM as adjudicator/rationale-writer — exactly the plan's Phase E composition; (3) MLP ≤ linear and
large ≈ base → signal mostly linear in AION space at ~3.9k train rows. Details/deviations in
`outputs/aion_probe/RESULTS.md`; script `eval/aion_probe.py` (extract|probe). Committed c073b9e.
Remaining for continuity: score the SAME probe on the frozen 72-gate (Claude bars 0.823/54%) — in flight.

## Phase E CORRECTIVE — probe does NOT transfer to the frozen gate: pool-covariate contamination (2026-07-06)
The AION probes scored on the frozen 72-gate (identical saved heads, leak-free, 72/72 pixels): **AUC
0.59–0.66, recovery 3.8–19.2% — below BOTH frozen bars** (Claude 0.823/54%; zero-shot 27B 0.823/50%).
Diagnostic (median scores, same head): separation collapses from BOTH sides — gate randoms score HIGHER
than corpus hard negatives (0.32 vs 0.17) and gate lenses LOWER than corpus lenses (0.50 vs 0.60). That is
the signature of the probe partially learning **corpus-construction covariates** (source-pool signatures of
the XVI-reject negative pool / SuGOHI positive pool), not pure lens morphology.

**Revised conclusions (apply to EVERYTHING trained on corpus_v5, incl. the VLM fine-tunes):**
1. corpus_v5 carries genuine signal, but **test2/valsel numbers are upper bounds inflated by shared pool
   construction** — within-corpus transfer ≠ deployment transfer.
2. **New standing rule: CROSS-POOL evaluation is mandatory** — every corpus-trained model gates on the
   frozen 72-gate (different pools by construction) in addition to test2; corpus-internal numbers are
   selection metrics only.
3. **Corpus v5.1 fix (launched):** dilute the negative pool — add random-field HSC negatives (offset
   positions in-footprint) + GALAXY CRUISE mimics (a third, independent pool) to train; rebuild eval sets
   with cross-pool composition. The XVI rejects stay (genuinely hard) but must not be the only negative
   signature.
4. The 9B/27B fine-tune verdicts must now be read primarily on the 72-gate, not valsel/test2.

## ★ Phase D run-2b FINAL — the fine-tuned student TRANSFERS (2026-07-06)
Full valsel curve (19 ckpts): 0.481 → 0.572 (epoch 1) → **0.664 @ ckpt-825** (early epoch 2) → plateau
0.63–0.66 (ckpt-1360 ties at 0.664). Merged ckpt-825, gated honestly:

| model | frozen 72-gate (cross-pool, PRIMARY) | test2 (corpus-internal) |
|---|---|---|
| frozen Claude bar | 0.823 / 54% / 89% rej | — |
| zero-shot Qwen3.5-27B bf16 | 0.823 / 50% / 91% | 0.563 / 15% |
| AION-large probe | 0.662 / 19% (collapsed from 0.84) | 0.841 / 59% |
| **Qwen3.5-9B ckpt-825 (trained, Claude-free)** | **0.743 / 42% / 96%** | 0.591 / 16% |

**Key evidence: the student runs OPPOSITE to the probe across pools** — better on the differently-built
gate (0.743) than its own corpus test split (0.591). That is the signature of learned transferable lens
MORPHOLOGY (gate lenses are DESI-bright/easier → a real-morphology model gains there), not pool
covariates (the probe lost there). The v5 training recipe (unique human-soft targets + unfrozen ViT on
the Claude-free corpus) is validated end-to-end on a 9B/one-Titan budget: zero-shot-9B ≈ chance at tier-2
→ trained 0.743/42%, within 0.08 AUC of the frozen Claude bar with better rejection (96% vs 89%).

**Nexts (ranked):** (1) the 27B fine-tune (same recipe; its zero-shot already ties the Claude bar — queued,
images finally shipping after the silent-rsync root cause); (2) run-3 levers: dihedral augmentation +
corpus v5.1 cross-pool negatives (random-field fetched); (3) student+probe composition (their error
profiles are plausibly complementary); (4) WiSE-FT dial for the conservative bias (rejection 96–97% caps
recovery).

## ★★★ MILESTONE — the Claude-free student MATCHES the frozen Claude bar (2026-07-06)
Scoring the SAME trained 9B (ckpt-825) with the PRE-SPECIFIED A0 logprob method (grade-token
distribution; 72/72 coverage) instead of the generated p_lens:

| scorer (72-gate, honest cross-pool) | AUC | recovery @ rej 0.89 |
|---|---|---|
| student ckpt-825, generated p_lens | 0.743 | 42% (@0.96) |
| **student ckpt-825, LOGPROB score** | **0.858** | **50%** |
| frozen Claude bar | 0.823 | 54% |
| zero-shot 27B bf16 | 0.823 | 50% (@0.91) |
| compositions (rank-mean with AION probe) | 0.80–0.85 | — (don't beat student-lp alone) |

**A fully Claude-free 9B — trained on the human-labeled corpus on ONE Titan RTX, scored with the
pre-registered logprob method — matches (nominally exceeds) the frozen Claude bar on the honest gate:
AUC 0.858 vs 0.823, recovery 50% vs 54% at matched rejection.** Caveats stated plainly: n=72 (26 lenses)
→ AUC s.e. ≈ ±0.06, so "matches within noise" is the defensible claim, and the logprob scorer, while
pre-specified in A0 (not searched here), still deserves confirmation on the larger cross-pool bench that
corpus v5.1 will provide. The +0.115 logprob-over-generated gap confirms the research finding that
generated confidence floats discard signal the token distribution retains. This is plan milestone 2 at
the "matches" level — with the 27B fine-tune, augmentation, and v5.1 still unplayed as upside.

## ★★★★ Phase D 27B FINAL — the Claude-free 27B student EXCEEDS the frozen Claude bar (2026-07-07)
27B bf16 + unfrozen-ViT, same recipe, 6-h A100-80 walltime (~1.1 epoch, ckpts through 750; the 80-min/ckpt
`swift infer` selection path was replaced by merge→vLLM-serve→grade, 43 min for BOTH candidates × 3 sets).
Selection on valsel-180 picked **ckpt-600** (0.636 vs ckpt-750's 0.606 — and mattered: 750 had collapsed to
8% recovery; the AUC-selection procedure caught it).

| model (honest 72-gate) | AUC | recovery @ rej 0.89 |
|---|---|---|
| frozen Claude bar | 0.823 | 54% |
| zero-shot Qwen3.5-27B bf16 | 0.823 | 50% |
| trained 9B ckpt-825 (logprob) | 0.858 | 50% |
| **trained 27B ckpt-600 (generated!)** | **0.861** | **81% (21/26)** |

**Recovery 81% vs Claude's 54% at matched rejection (+27 pts, ~2σ at n=26), AUC nominally above the bar —
on generated scores alone.** test2: 0.625/23% (vs zero-shot 0.563). Cost: one 6-h A100 walltime.

**LOGPROB RE-GATE (current code, 100% coverage, 19-min job):** 72-gate **AUC 0.892 / recovery 77%**
(generated re-scored 0.861 — reproducible); test2 0.648/32%; valsel 0.644. Consolidated final verdict,
robust to scorer choice and consistent across both students (9B-lp 0.858, 27B-gen 0.861, 27B-lp 0.892):

| honest 72-gate | AUC | recovery @ rej 0.89 |
|---|---|---|
| frozen Claude bar | 0.823 | 54% |
| **trained 27B ckpt-600 (logprob)** | **0.892** | **77%** |

**Plan milestone 2 achieved at the EXCEEDS level** (recovery +23–27 pts at matched rejection, ~2σ;
AUC +0.07), subject to the n=72 caveat; corpus v5.1's larger cross-pool bench finalizes the claim.

## Phase C groundwork (2026-07-03; zero-GPU, catalogs in gitignored cache, curl-regenerable)
HOLISMOKES tables pulled + profiled: paperVI 467 rows; paperXIII 162+384; **paperXVI 14,152 expert-graded
rows: 598 A/B-like positives (G≥1.5) + 12,880 expert REJECTS (G<1.0) with continuous G scores (0–3,
median 2 graders)** — the rejects are CNN-selected-but-expert-rejected = the ideal HSC hard-negative bank,
and G/3 is a ready-made human-soft p_lens target. Remaining Phase C: SuGOHI DB scrape, GALAXY CRUISE,
IRSA Euclid fetcher, dedup + `build_corpus_v5.py`.

## Bench v5.1 (n=1,307; 142 lens / 1,165 neg across 6 pools) — first results (2026-07-07)
**9B student ckpt-825: overall LOGPROB AUC 0.884 ± ~0.02, recovery 61% @ 89% rejection** (generated
0.685 — the logprob advantage is +0.20 at scale). Per-pool: recovery desi_xmatch **88%** (23/26 — the
same lenses where frozen Claude recovered 54%) / sugohi_heldout 54%; rejection random_field **99%**,
gc_spiral 99%, gc_ring 95%, gc_tidal 87%, xvi_reject 62% (hardest, honestly), gate_mix 67%. **No
covariate pathology: deployment-like negatives are rejected BEST** (opposite of the probe's failure
signature). 27B student + zero-shot control gradings in flight (Perlmutter job 55649396).

## ★★★★★ BENCH v5.1 FINAL — the claim is certified at high power (2026-07-07)
All three graders on the 1,307-row cross-pool bench (142 lens / 1,165 neg over 6 pools; logprob scoring;
parse 1306/1307 each; AUC s.e. ≈ ±0.02):

| model | AUC | recovery @ 89% rejection |
|---|---|---|
| zero-shot Qwen3.5-27B (control) | 0.813 | 47% |
| 9B student ckpt-825 | 0.884 | 61% |
| **27B student ckpt-600** | **0.892** | **68%** |

- **Training delta vs zero-shot control: +0.079 AUC, +21 pts recovery — decisive at this n.**
- **Per-pool (27B student): rejection random_field 100%, gc_spiral 97%, gc_ring 97%, gc_tidal 92%,
  xvi_reject 58% (near-miss rejects absorb the FP budget — the right place); recovery desi_xmatch 88%
  (the same 26 lenses where frozen Claude recovered 54%), sugohi_heldout 64%.** No covariate pathology.
- The frozen-72-gate number (0.892) reproduces exactly at 18× the sample.
- Note the zero-shot control's 0.813 here vs 0.823 on the old gate — the bench is slightly harder;
  the students' margins over it are the meaningful quantity.

**PROGRAM CONCLUSION (v5 milestone 2, certified): a fully open, fully Claude-free student — trained on
public human-labeled catalogs, one A100 walltime, scored by grade-token logprobs — decisively exceeds
both the zero-shot frontier-open-model control and the frozen Claude reference on real confirmed-lens
vetting at HSC resolution.** Remaining plan: Phase F (SDK-free completeness + no-Claude CI), optional
run-3 levers (augmentation, WiSE-FT, epoch-2), DR11/HSC campaign deployment, PR to main.

## Run-3 lever 1: WiSE-FT dial — smooth but MONOTONIC, pure student stays best (2026-07-09)
alpha-interpolation student600<->base, each graded on bench v5.1 (logprob):

| alpha | AUC | rec@0.89 | xvi_rej | sugohi_rec |
|---|---|---|---|---|
| 0.0 (base) | 0.813 | 47% | 74% | 41% |
| 0.3 | 0.853 | 57% | 71% | 51% |
| 0.5 | 0.869 | 61% | 65% | 55% |
| 0.7 | 0.880 | 66% | 60% | 61% |
| 1.0 (student600) | **0.892** | **68%** | 58% | 64% |

No intermediate point dominates: the trade is ~linear (alpha 0.7 buys +6 pts near-miss rejection for
-2 rec / -1.2 AUC). Consistent with the clean per-pool diagnostics — the fine-tune sacrificed no
covariate-washable robustness for WiSE to recover. **Deployment default stays student600 (alpha=1.0);
alpha~0.7 documented as the knob if a campaign's cost model prioritizes near-miss rejection.** (alpha=0.3 confirmed the
monotone curve: 0.853/57%/71%.) Ops: hbm80g had a 288-job
backlog — the sweep ran via TP2 on 2xA100-40 shared QOS (~instant scheduling, cheaper per-GPU billing);
NERSC shared rule: request 32 cores per GPU.

## Run-3 FINAL — augmented retrain: statistical tie, promoted on robustness (2026-07-10)
27B bf16 + unfrozen ViT on the augmented corpus (12,370 rows = originals + dihedral copies + 1,500
random-field dilution), 2 FULL epochs, 4-GPU DDP, 774 steps, 5.5 h — the first untruncated training of
the program. Valsel curve is a FLAT COLLAPSE-FREE plateau (0.654–0.669 over ckpts 450–774; contrast
run-2's 8%-recovery cliff at 750) — augmentation delivered exactly its stability promise. Best: ckpt-450.

| bench v5.1 (logprob) | AUC | rec@0.89 | desi_rec | sugohi_rec | xvi_rej |
|---|---|---|---|---|---|
| run-2 student600 | 0.892 | 68% | 88% | 64% | 58% |
| **run-3 ckpt-450** | **0.898** | **69%** | **92%** | 64% | 57% |

Within noise (s.e. ±0.02) → honest headline = PARITY. **run-3 ckpt-450 promoted to deployment default**
on priors: dihedral invariance + random-field dilution trained in, flat plateau = low selection risk.
WiSE dial (monotone) + this close run-3: the v5 recipe is at a local optimum; remaining upside is
inference-side (TTA over 8 dihedral views — untested) and data-side (GALAXY CRUISE negatives in
training, Euclid transfer). Ops: full endgame (6× merge+serve+valsel-grade → in-job AUC select → bench
winner) ran autonomously in ONE 1h51m shared-QOS TP2 job.

## Phase F CLOSED — default-Claude-free, SDK retained as engine option (2026-07-10)
Full-tree audit (workflow wf_ea18c2d4: 7 readers over all 95 files + empirical import-blocker tracer +
completeness critic; the critic caught the tracer's own bug — bare ImportError vs the repo's
ModuleNotFoundError guards — and re-traced correctly). True pre-fix state: ONE open-path root break
(grader_direct.py's top-level `from anthropic import AsyncAnthropic`, poisoning distill_hsc/
distill_euclid/augment_sft_v5/build_sft_data/run_cascade(_recall)/run_batch --mode direct), plus the
anthropic-default backend and a "sonnet"-shaped silent model default on the open path.

Changes: (1) LENSJUDGE_BACKEND now DEFAULTS to `openai`; `claude` accepted as alias for the retained
anthropic engine. (2) grader_direct's anthropic client import is lazy with a clear error; claude alias
map applies only on the anthropic branch. (3) config.MODELS is backend-aware — open default model
LENSJUDGE_MODEL (Qwen/Qwen3-VL-8B-Instruct) instead of silently sending "sonnet" to vLLM. (4) Trace.hooks()
and tools.server.build() raise clear SDK-engine errors instead of None-crashes. (5) tools/spectrum gets
the sibling no-op @tool guard (sis_theta_e now SDK-free). (6) All 10 optional-import guards broadened
ModuleNotFoundError -> ImportError (survives broken installs). (7) Retained SDK-only modules
(imaging/{judges,orchestrator}, spectro/grader, eval/run_appendix_*) raise a clear "Claude SDK engine"
ModuleNotFoundError. (8) tests updated + the no-Claude check now blocks BOTH anthropic and
claude_agent_sdk, asserts the openai default and a non-claude default model. (9) NEW
.github/workflows/lensjudge-ci.yml: full test suite in an env with NO Claude packages installed —
any change reintroducing a hard Claude dependency on the default path fails CI.

Verification: all 7 test modules pass (54+ tests incl. the strengthened blocker test); empirical
goal-state trace 25/25 (21 open-path modules import Claude-free with backend unset; 4 SDK-only modules
give the clear error). The Claude engine remains fully selectable: LENSJUDGE_BACKEND=claude.
