# Three LLM-in-Astronomy Papers: A Combined Reading

**Stoppa+2025 (pixels), AstroAlertBench 2026 (benchmark) and ALeRCE text-to-SQL 2026 (interface) differ in one thing that predicts every conclusion each reaches about self-assessment: how much external ground truth the task gives the model to check itself against.**

## The three papers at a glance

| Paper | Task | Data & scale | Headline result | Label source |
|---|---|---|---|---|
| Stoppa+2025, *Nat. Astron.* | Real/bogus on new/ref/diff triplets + rationale | Gemini 1.5-pro-002, 15 exemplars, no training; 2,000 PS1 / ~3,200 MeerLICHT / 2,000 ATLAS | 93% mean accuracy (91.9–94.1%, flat over 0.25″–1.8″ pixels); coherence feedback 93.4% → 96.7% | MeerLICHT expert-labelled; PS1/ATLAS bogus half from pipeline "garbage lists" |
| AstroAlertBench 2026 | 5-class ZTF alerts, 3-stage cascade + self-scored rationales | 1,500 alerts (300/class) × 13 configs | Opus 4.7 think 60.60 ± 1.26%, +9.53 ± 1.80 pp over 2nd-ranked GPT-5.4 high-think; AGN <8% in all 13 runs | ALeRCE stamp-classifier posteriors, narrow confidence band (AGN 0.860–0.899) |
| ALeRCE text-to-SQL 2026 | NL → SQL: schema link, difficulty class, decompose, self-correct | 25 tables / 304 columns; 110 gold pairs (52 test); 13 models × 10 runs | Opus 4.6 sbs best (rank SUM 10); PM_rows 0.97 / 0.44 / 0.59 simple/medium/hard | One expert wrote all 110 gold queries |

## What the three agree on

**Decomposition is the only architectural move all three make.** Stoppa's rubric hard-codes the decision procedure; AstroAlertBench's cascade charges error to a named gate (Stage-2 54.21–81.60%, Stage-3 conditional 32.12–60.78%); ALeRCE's step-by-step pipeline takes six of the top seven ranks. Staging localises failure without removing it: Stage-1 looks "essentially solved" at 77–87% while its majority-class floor is exactly 80.00%.

**Every headline leans on an artefact for at least part of its gold.** The bogus halves of Stoppa's Pan-STARRS and ATLAS sets are the pipelines' own reject lists (MeerLICHT alone is expert-labelled); AstroAlertBench's classes are a CNN's posterior; ALeRCE's queries are one annotator's SQL conventions. Only AstroAlertBench measures a human ceiling on its own gold, on 15 of 1,500 items: five astronomers reproduce the manifest on 30.67 ± 5.32% of 75 trials — in Appendix G, not beside the headline, conceded as "itself a signal". Stoppa's 12-astronomer panel rates prose and never re-classifies; ALeRCE's gold has one author. So 93%, 60.60% and PM 0.44 are capped by construction.

**More effort is not monotone.** Five of six few-shot settings significantly depress Opus 4.6 sbs hard rows below the 0-shot 0.59, bottoming at 0.38 (3-shot Random); none improves them. AstroAlertBench's Qwen3.5-4B *loses* 13.90 ± 1.29 pp when told to think, truncated at a mean 16,623 output tokens/row against Opus 4.7 think's 806.

## Where they disagree, and why it matters

All three instrument the model's self-assessment; they return three incompatible verdicts.

- **Stoppa: trust it.** Self-rated coherence separates correct from incorrect calls across the full MeerLICHT set; re-feeding low-coherence cases as exemplars is worth 3.3 pp (not held out).
- **AstroAlertBench: do not.** Population-level modesty is *inversely* related to accuracy (−31.2 ± 10.0 pp per rubric point); per-alert confidence–correctness r never exceeds 0.2517, collapsing to 0.047 under adaptive thinking. Gemini 2.5 Flash, re-prompted, changed not one classification while raising its own rating by +0.43.
- **ALeRCE: trust it, within its lane.** Self-correction improves or ties 154 of 156 comparisons for $0.09 of a $2.87 run — but fires only on a database execution error, and semantic errors survive it.

The resolving variable is the **oracle**. SQL has a hard one: a bad query throws, and something ran it. Real/bogus has a soft one — the decision is local to the pixels, so the model's own coherence score tracks correctness. Five-class taxonomy has none: the gold is a CNN that five experts reproduce a third of the time. Self-assessment is worth what the task's external check is worth: that, not model tier, orders the three verdicts.

## What this means for the agentic lensing programme

Lens grading sits **below all three**. Nothing throws when an agent calls a merger an arc; the between-team human ceiling is QWK 0.29 [0.12–0.44] and 0.17, and the human grade's own AUC against truth is 0.577 [0.40, 0.76]. AstroAlertBench's verdict, not Stoppa's, is our prior, and our results match it: `uncertain` fired on 5 of 1,050 JWST judgements, and the critic stack collapses the advocate's ranking (AUC 0.764 → 0.468, ΔAUC −0.301) while staying the right FPR layer (D-rule veto, FPR@(A or B) 2.0%).

So stop shopping for better self-critique and **buy oracles**. Two work: spectroscopy (20/20 Hsu-2025 grade-A pairs plausible, 4/4 Foundry-II non-lenses rejected) and resolution (same objects at 1.3″ → 0.1″: mean p_lens 0.14 → 0.75, grade-C 0.05 → 0.90).

Two gaps run the other way. None of the three is agentic in our sense: Stoppa and AstroAlertBench are single-shot prompted classification with no tool use, and ALeRCE executes its SQL but re-prompts only on the database's error text — never reading returned rows, calling a second tool, or designing a pipeline against 6,848 NIRCam observations from one prompt, which is ours to claim. Only ALeRCE prices a run per stage ($0.23 schema / $0.43 difficulty / $1.07 decomposition / $1.05 generation / $0.09 self-correction, of $2.87); AstroAlertBench gives wall-clock and tokens but no dollars; Stoppa, one qualitative "thousands of dollars per night". We price per mode (lean ~$0.06, panel ~$0.27, multiagent ~$0.33), per stage never.

## Recommended actions

1. **Run AstroAlertBench's second-rollout ablation on the evidence-first advocate, *with* the control arm they omit.** Two arms on the `lensjudge/golden/` design split, prior answer attached versus withheld, over the lowest-p_ev rows. Moves: recall@5%FPR off 33.3% (incumbent 9.5%); settles whether their +28.57 pp is self-critique or regression to the mean.
2. **Attach Stoppa's 0–5 self-coherence rating to every advocate call on the `lensjudge/golden/` holdout and score its AUC against correctness.** Moves: a routing signal benchmarked against the advocate's own 0.764, needing no consensus label — the test Stoppa reports as a correlation, never as a ranker.
3. **Re-derive every LensBench-VI and `lensjudge/golden/` headline by bootstrapping over items, not runs.** ALeRCE's ten-run permutation fragility is ours too. Moves: recall@5%FPR 33.3% vs 9.5% from a bare McNemar p 0.046 on ~86 positives to an actual interval.
4. **Meter the evidence-first pipeline per stage in `reproductions/lensjudge/`, as ALeRCE's Table 3 does.** Moves: the ~$244 programme aggregate into advocate / critics / arbitrator dollars — what decides whether a critic stack costing ΔAUC −0.301 earns its keep as a veto layer. (The JWST search's 76.5M tokens over 1,281 subagents is a separate pipeline, wanting its own split in `reproductions/agentic-lens-discovery/`.)
