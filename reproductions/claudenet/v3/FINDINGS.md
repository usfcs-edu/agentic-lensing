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

## C-now — DR10 sweep on v2-lean (the head-to-head vs the published 811)  🟡

Full DR10-south sweep with the shipped **v2-lean** ensemble (DR9-trained), as the baseline the
v3 re-sweep improves on. Pipeline (160→165) reused unchanged.

- **Parent → survivors:** 43,707,140 galaxies (matches Inchausti's ~43M) → **216,987 pass** the
  per-member 1e-4 union threshold → capped to the **150,000** survivor budget.
- **Conformal (165, full-m, random-galaxy null):** **0 selected at α=0.05/0.10/0.25.** Only 139
  survivors beat *all* 354k calibration negatives (conformal-p floor 2.82e-6); the best full-m
  q-value is 0.29. At DR10's m=43.7M (3.5× DR9's 12.3M), the per-group BH needs the floor at
  ~1.6e-7 → a **~6M-negative calibration set** to certify the cleanest 139. DR9 had power at
  m=12.3M with 1M negatives; **the rigorous FDR at DR10 scale needs a proportionally larger
  NegEval** (a Perlmutter job). Reported honestly as a power-floor limitation, not a null result.
- **Head-to-head recall of the published 811** (5″, by *their* grade), 810/811 in our DR10 parent:

  | their grade | n | in our parent | recovered into survivors |
  |---|---|---|---|
  | A | 90 | 89 | **61 (69%)** |
  | B | 104 | 104 | 61 (59%) |
  | C | 617 | 617 | 325 (53%) |

- **Diagnostic on the 28 missed grade-A:** ~17 score high on v2-lean (max member 0.65–1.0, **8 at
  exactly 1.0**) — found by the model but below the strict 1e-4 union threshold; only ~11 score
  genuinely low (<0.25). So **v2-lean finds ~78/89 (88%) of Inchausti's grade-A at high score**;
  the recall gap at the operating point is a **DR9→DR10 threshold-calibration transfer** effect.

### C-now recalibration + bigger-calibration tests — two honest negatives that sharpen v3

Both done all-locally from the existing 43.7M stage-1 scores (no new Perlmutter job;
`370_dr10_recalibrate.py`, 6M DR10-native NegEval sampled from the scored parent):

- **Threshold recalibration to DR10's true 1e-4 FPR makes thresholds TIGHTER, not looser**
  (resnet46_C_hard 0.81→1.00; effnet_S2_hard 0.96→0.98) because DR10 random galaxies have a
  fatter high-score tail than DR9. Survivors drop 217k→63k and **grade-A recall drops 69%→62%**.
  So the recall gap is **not** a threshold artifact — it is a genuine **DR9→DR10 model domain
  shift** (the DR9-trained members under-score a real fraction of DR10 grade-A lenses). The DR9
  thresholds were *lenient* on DR10 (the 162 FPR-overshoot). → motivates **v3 DR10-aware
  retraining**, not a threshold tweak.
- **Growing the conformal calibration to 6M DR10-native negatives STILL selects 0** at α=0.05/
  0.10/0.25 (full-m). So the 0-selection is **not** a calibration-size limit — v2-lean cannot
  separate any DR10 survivor from the hardest random galaxies (which include lens-mimics) enough
  to clear full-m FDR at m=43.7M. **This is the lens-vs-mimic problem at sweep scale** — an
  independent confirmation of the A0 thesis and of why v3's contaminant-aware model + mimic-null
  selection are the fix.

**C-now conclusion:** the v2-lean DR10 baseline is *competitive on recall* (finds ~88% of
Inchausti's grade-A at high score) but **FDR-limited (0 at full-m) and domain-shifted** — exactly
the two gaps v3 closes (contaminant-aware separation + DR10/i-band retraining). The candidate
shortlist for vetting therefore comes from the top NEW survivors by ensemble score (not FDR), as
the DR9 campaign did. Artifacts under `data/v3/sweep_dr10/` (gitignored): `stage2_scores`,
`conformal`{,`_dr10cal`}, `crossmatch`, `survivors_dr10_recal`, `operating_points_dr10.csv`,
`recalibrate_dr10_summary.json`.

### C-vet (DR10) — agentic vetting of the top-30 NEW candidates  ✅ (preliminary)

148,034 NEW survivors → top-300 by p_final (`v3/dr10_candidates_top300_new.csv`). The top-30
(p_final 0.979–0.988) vetted with the LensJudge v2 mimic-aware grader (`--mode escalate`,
`rubric_imaging_v2`, sonnet, **$2.59**, parse_ok 30/30): **2 A, 6 B, 11 C, 11 D**. The D's are
the expected mimics (lrg_companion 8, merger 2, ring 1) — the rubric catches them. The **8 A/B
(2 grade-A, p_lens 0.80)** are in `v3/dr10_vet_AB_candidates.csv`.

This is **better than DR9** (which had 0 A/B among 601 NEW) — the strongest DR10 NEW candidates
include lens-like ones. **Honest caveats:** first-pass *single-grader* grades (the DR9 campaign's
skeptic + dual-grader consensus demoted most first-pass A/B); DESI-resolution-limited; **none had
Euclid coverage** to escalate. So these 8 are **promising follow-up targets, not confirmations** —
full qualification needs the skeptic pass + dual grader + higher-res (Euclid/HSC) escalation.

**C-now bottom line:** v2-lean is a *competitive* DR10 finder (recall, and a non-trivial A/B yield
at the top) but **FDR-uncontrollable at DR10 scale and DR9-domain-shifted** — the v3 contaminant-
aware model + mimic-null selection + DR10/i-band retraining are precisely the fixes, and the
re-sweep (C-v3) is where the qualified-candidate payoff should land.
