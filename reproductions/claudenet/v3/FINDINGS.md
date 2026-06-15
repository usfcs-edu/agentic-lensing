# ClaudeNet v3 — running findings

Tracked log of v3 results as phases complete. Data artifacts are regeneratable under
`data/v3/` (gitignored, per the repo's bare-`data` rule); this file + the scripts are the
tracked record. See the plan in the PR description.

## A0 — The mimic metric and the v2 baseline (the motivation)  ✅

**Scripts:** `300_build_mimic_bank.py`, `301_mimic_metric.py`, `302_score_mimic_bank.py`.

**Seed bank** (`data/v3/mimic_bank_seed.parquet`): the **601 status==NEW** rejects of the
DR9 qualification campaign — CNN-high-scoring, dual-agent-confirmed NON-lenses, each with
a typed contaminant. Composition: lrg_companion 278 (46%), merger 85, other 68, blend 64,
noise 34, star_halo 27, unknown 21, ring_galaxy 11, spiral 10, satellite_trail 3; visual
grade D 453 / C 148; 377 are confident mimics (grade D + named type).

**New metric** (`301`): **recovery @ matched-MIMIC-FPR** — identical arithmetic to the v1/v2
headline `_ensemble.recovery_at_fpr`, but the negative class is the lens-mimic bank instead
of random galaxies. Headline φ = 0.05 (a few-hundred-row seed cannot estimate a 1e-3
quantile; φ tightens as A1 grows the bank). Reported with Wilson + paired-bootstrap CIs and
a per-contaminant-type breakdown.

**Harness validated:** integrity check `max|reconstructed p_final − stored p_final| =
0.000e+00` (positive and mimic scores share one isotonic+average scale); random-FPR sanity
reproduces the shipped v2 verdict exactly (Storfer@1e-3 **0.895**, Inchausti@1e-3 **0.961**).

**The motivating result — v2-lean baseline:**

| Positives | recovery @ **random**-FPR(0.01) | recovery @ **mimic**-FPR(0.05) | recovery @ **mimic**-FPR(0.01) |
|---|---|---|---|
| Storfer   | 0.963 | **0.168** (boot95 [0.142,0.222]) | 0.054 |
| Inchausti | 0.996 | **0.307** (boot95 [0.268,0.398]) | 0.126 |

v2-lean recovers 96–100% of held-out lenses against *random galaxies* but only **17–31%**
against *lens-mimics at a looser threshold*. Worst dominant type = `lrg_companion`
(0.167 / 0.307 @ φ=0.05) — matching the campaign's 49% lrg_companion finding. The
mimic-discrimination gap, not architecture or random-galaxy recovery, is the binding
constraint. **This is the number v3 must beat.**
