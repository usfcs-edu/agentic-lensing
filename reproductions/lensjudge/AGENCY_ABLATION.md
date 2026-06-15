# LensJudge v2 — B7: agency ablation (does the agentic loop contribute?)

**Workstream:** LensJudge v2 B7 (agency ablation). **Builds on:** v2 (reuses the v2 graders,
v2 rubric, frozen `lensbench_evidence.csv`, `eval/score.py`/`run_lensjudge_eval.py`). It does
**not** change any v2 grading behavior — it adds a non-agentic baseline grader + an eval.
**Not a version bump** (see "Naming" below). Run `2026-06-14` on branch `reproductions/claudenet-v3`.

## FOR STAGE-2 (the DR10/DR11 survivor-vetting decision) — read this first

When stage-1 (Perlmutter CNN) hands over survivors, vet them as a **cascade**, not one mode:

1. **Detection triage (bulk):** grade every survivor with the **`direct` (no-loop) grader**
   + the v2-inline rubric. It matches the agentic grader on lens-vs-random AUC (≈0.66) at
   **$0.012/cand, 1 turn** — ~**6× cheaper** than lean ($0.068) and ~26× cheaper than
   multiagent ($0.315), with no detection loss. `run_batch.py --mode direct`.
2. **Mimic adjudication (survivors only):** re-grade the candidates that pass/are-ambiguous
   with the **agentic `v2` grader (mode escalate, v2 rubric)**. This is the **only** config
   that achieves the lens-vs-mimic AUC of **0.71**; every no-loop variant collapses to
   0.40–0.55 (table below). The loop is worth its cost *here* — this is the v3 thesis
   (separating real lenses from CNN LRG+companion mimics).
3. **High-res escalation:** for the rare survivors with Euclid/HSC coverage, escalate to
   tier-2 (the only wall-breaker, README #6). Trigger by **coverage** (or CNN `p_meta`),
   **not** the LLM grade — the grade-gate is degenerate (fires ~100%, see finding 3).
4. **Do NOT use panel/multiagent at sweep scale** — ~4.7× cost, no AUC gain (README #2).

Rough cost on ~5,000 survivors: all-direct ≈ **$60**; cascade (direct on all + v2 on the
~30% that pass) ≈ **$170**; all-agentic-v2 ≈ **$355**; all-multiagent ≈ **$1,575**.

## The question

The agency audit found LensJudge's "agentic" surface is a near-deterministic fixed pipeline:
the model emits a `ToolSearch` call, one `fetch_cutout` (default views), ~38% of the time a
`get_photometry` call, then grades. Orchestration is `asyncio.gather` + a hardcoded vote rule;
the escalate trigger is a Python `if`. **Does the tool-call loop contribute anything over
invoking the tools programmatically?** B7 tests it by ablation, not assertion.

`grader_direct.py` removes the loop entirely: it renders the same views + photometry **in
Python** and makes **one** base Messages-API call with the images inline (no tools, 1 turn).
Comparing it to `lean`/`v2` on the same 120 rows isolates tool-call *planning* from the
multimodal *judgment*. The 2×2 (rubric × loop) plus a thinking arm:

| arm | rubric | loop | Bench-A AUC (lens-vs-random) | Bench-B AUC (lens-vs-mimic) | $/cand | turns |
|---|:--:|:--:|--:|--:|--:|--:|
| lean | v1 | **yes** | 0.593 | 0.518 | 0.068 | 3.75 |
| direct | v1 | no | **0.669** | 0.384 | 0.012 | 1 |
| **v2** | v2 | **yes** | 0.668 | **0.714** | 0.071 | 4.42 |
| direct | v2 (tool-worded) | no | 0.585 | 0.401 | 0.013 | 1 |
| direct | v2-inline | no | 0.659 | 0.407 | 0.012 | 1 |
| direct + thinking | v2-inline | no | 0.531 | 0.548 | 0.030 | 1 |

## Findings (honest)

1. **Detection: the agentic loop contributes nothing.** No-loop `direct` matches/beats `lean`
   on lens-vs-random AUC (0.669 vs 0.593; 0.659 vs 0.668 at v2 rubric) at **~1/6 the cost and
   1 turn vs ~4**. The tools can be invoked programmatically with no detection loss. The
   loop's first action every time is a mechanical `ToolSearch`; it adds latency and tokens,
   not accuracy.
2. **Lens-vs-mimic: the loop matters, and it is not rubric coupling.** The v2-rubric mimic
   gain (0.518→0.714) lives in the **LOOP arm only**. Feeding the *same* judgment content with
   evidence inline (`v2-inline`) does **not** transfer it (0.407). Adding a reasoning
   scratchpad to the single call (thinking) recovers only ~half the gap (0.548) and *hurts*
   detection (0.659→0.531). So the agentic multi-turn structure — the model actively
   inspecting the cutout, then the photometry, reasoning across turns — genuinely helps the
   hard mimic discrimination. This is the calibrated correction to "the loop adds nothing":
   it adds nothing to *detection*, but contributes to *mimic rejection*.
3. **Escalation routing: the LLM-grade gate is degenerate.** The trigger
   (`grade≠A or escalate_to_human`) fires on **~100%** of candidates — the grader is
   over-skeptical at 1.3″, so almost nothing is a confident "A". It carries no routing
   selectivity a free CNN `p_meta` threshold couldn't reproduce; coverage (not the LLM)
   bounds tier-2 cost.

## Caveats
- **Modest n, compressed scores.** Benchmark A is 50 pos / 30 neg; Benchmark B 50 pos /
  40 mimic. The agent's `p_lens` is uniformly low (documented over-skepticism), so Bench-B
  AUCs sit on a compressed distribution — **directional, not definitive** (CIs ≈ ±0.06–0.08).
  The robust qualitative claims are (1) and (3); (2)'s magnitude is softer than its direction.
- **Single seed / one model (sonnet), one manifest.** No bootstrap CIs computed.
- **`grader_direct` always provides photometry** (the programmatic stance); `lean`'s model
  chose to call it only ~38% of the time. This is a deliberate design difference, not a bug.

## Reproduce
```bash
# the no-loop arm (≈$1.4 each):
python lensjudge/imaging/run_batch.py --mode direct --manifest lensjudge/outputs/lensbench_evidence.csv \
    --rubric lensjudge/prompts/rubric_imaging_v2_inline.md --out lensjudge/outputs/preds_direct_v2inline.parquet
# score one arm vs the v1-lean baseline:
python lensjudge/eval/run_lensjudge_eval.py --preds lensjudge/outputs/preds_direct_v2inline.parquet \
    --manifest lensjudge/outputs/lensbench_evidence.csv --label direct-v2inline
# the consolidated 2x2 + routing ($0, scores existing parquets):
python lensjudge/eval/run_agency_ablation.py --report lensjudge/outputs/agency_ablation.md
```
Artifacts: `imaging/grader_direct.py`, `eval/run_agency_ablation.py`,
`prompts/rubric_imaging_v2_inline.md`; results in `outputs/agency_ablation.{md,json}` and
`outputs/eval_direct*.json` (gitignored). Total spend ≈ **$6** of the ablation budget.

## Naming
Run as a **v2 ablation workstream (B7)**, not "LensJudge v3" — "v3" already denotes the
ClaudeNet v3 program (LensJudge v2 is its workstream B), so the label would clash. If stage-2
adopts the cascade (direct triage + agentic mimic adjudication) as the default deploy path,
*that promotion* is the natural v3 milestone.
