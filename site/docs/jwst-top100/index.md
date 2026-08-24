# JWST Top-100: original grades vs the evidence-first judge

<style>
.md-typeset .t100 table:not([class]) th { min-width: 0; }
.md-typeset .t100 table:not([class]) td, .md-typeset .t100 table:not([class]) th { padding: .7em .8em; }
</style>

The 100 top-ranked candidates of the JWST NIRCam strong-lens search, re-graded **blind** by the
evidence-first [LensJudge](../current/lensjudge/index.md) panel — images shuffled, footer stripped,
no candidate id, coordinates, filters, rank or original grade in front of the model — and set
beside the grades the original campaign assigned. Every candidate's original and annotated cutout
and its full advocate → critics → arbitrator record is on the per-rank pages.

[:material-download: Download the comparison table (CSV)](data/comparison.csv){ .md-button .md-button--primary }
[:material-image-multiple: Browse the cutouts, ranks 1–25](ranks-001-025.md){ .md-button }

!!! abstract "The result in one line"
    Under the deployed letter (rule R2), **7** of the 100 scored candidates
    keep their original letter, **90** move up and **3** move down (U, never verified,
    counted below D). The original campaign verified only **22** of them; on those the
    two instruments agree on **7**, the new judge is higher on **12** and lower on
    **3**. At A/B: original **10**, new judge **45** (final letter)
    / **52** (advocate-only rank letter); **26** of the **78** never-verified
    (U) candidates reach A/B under the final letter, and **17** candidates carry the
    arbitrator's *needs human* flag. Model grades are a ranked reading of the pixels, **not** human vetting.

---

## What was graded

The original campaign is an agentic JWST NIRCam strong-lens search whose pipeline flags candidate
deflectors on six-panel cutout composites, has an *inspector* agent score each flag with a confidence
(0–100), and sends the most confident flags to a verification queue. There, three adversarial
*refuter* personas — artifact, morphology, geometry — each try to refute the candidate, and the grade
is the **pass count**: A = 3 passes, B = 2, C = 1, D = 0; U marks a flag that was never verified.
Of the run's top 100 by pipeline rank, **5** are A, **5** are B, **12** are C and **78** are U.

**Blind protocol.** The JWST repository ships the same 100 composites shuffled out of rank order,
renamed 001–100 and with the footer strip (candidate id, coordinates, magnitude) removed, precisely
so a reviewer scores each field on the imaging alone. The judge ran over those files with frozen,
item-agnostic prompts; the only answer-key field read before scoring was the *layout* (two-band
colour vs single-band grey composite: 87 colour / 13 grey),
which selects the per-role views. Ranks, ids and the original grades were joined on afterwards.

## How the new judge grades

The pass-count verifier is replaced by an evidence-first panel in which the grade is *computed from*
the explanation:

1. **Advocate** — lists *located* evidence items (panel, radius, position-angle span), scores five
   NIRCam-adapted criteria (source contrast, low surface brightness, curvature, counter-image, arc
   morphology) and returns **p_evidence**, the probability that the located evidence is lensing.
2. **Three competence-bounded critics** — artifact, geometry, morphology. Each must either abstain or
   *name* an alternative from a fixed vocabulary (merger, companion projection, spiral arm, edge-on
   disk, subtraction residual, PSF wing, diffraction spike, shell/tidal, ring galaxy, …), locate it, say
   which advocate items it accounts for and grade its refutation strength *r*. Forbidden grounds: the
   size of the Einstein radius, colour alone, and the symmetric residual of the circular-subtraction panel.
3. **Arbitrator** — sees the image and every text and rules each critic **upheld**, **partial** or
   **overruled**, writes the rationale paragraph, gives its own free letter and may flag *needs human*.

Scores: **S** = p_evidence × Π(1 − r·a) over every critic (a = the fraction of the evidence the
alternative covers); **S_arb** = the same product over upheld/partial refutations only, with the
arbitrated coverage. Two letters are deployed:

<div class="t100" markdown>

| Letter | From | Meaning |
|:--|:--|:--|
| **Rank letter** | p_evidence, advocate only | The letter whose false-positive rate the calibration controls; the ranking score. |
| **Final letter** (rule R2) | the rank letter, unless the D-rule fires | Equals the advocate's rank letter unless an **upheld** critic's located alternative covers every evidence item at refutation strength r ≥ 0.8 — then **D**. "Demoted by" below is populated only for such D-rule vetoes; S_arb is shown for reference and does not set the letter. |

</div>

**Why this rule.** The deployment rule was pre-registered before any calibrated letter was read
(REGISTRY "Deployment rule v2-deploy", item 6): **R1** is deployed unless, on the already-scored
holdout, R1's recall at A∪B on the holdout positives is below one half of the rank letter's recall at
A∪B — then the pre-stated fallback **R2**.
Transfer check — program holdout result (opus5-xhigh), from `selected_rule.json`: selected **R2** —
“recall_AB(R1; a1 holdout parquet, scored rows) = 0.0238 < 0.5 x recall_AB(letter_rank; a2 holdout parquet) = 0.1429: the pre-stated fallback R2 is deployed” (recall@A∪B rank letter 0.286; R1 0.024; bar 0.5 × rank = 0.143; R2 0.238; 42 holdout positives; letters `opus5_api_calibrated`; t_A 0.20 / t_B 0.17 / τ0 0.15).
This run's letters are assigned under the selected rule **R2**.

Holdout transfer endpoints — program holdout result (opus5-xhigh), from `transfer_check.csv` (95 % Clopper–Pearson
CIs; 200 negatives, 42 positives, 20 stress_D panels):

<div class="t100" markdown>

| Rule | FPR@A | FPR@A∪B | recall@A∪B | stress_D at A∪B |
|:--|--:|--:|--:|--:|
| **Rank letter (advocate only)** | 0.0 % [0.0, 1.8] | 3.0 % [1.1, 6.4] | 28.6 % [15.7, 44.6] | 16/20 (80.0 % [56.3, 94.3]) |
| **R1** | 0.0 % [0.0, 1.8] | 0.0 % [0.0, 1.8] | 2.4 % [0.1, 12.6] | 4/20 (20.0 % [5.7, 43.7]) |
| **R2** — selected | 0.0 % [0.0, 1.8] | 2.0 % [0.5, 5.0] | 23.8 % [12.1, 39.5] | 10/20 (50.0 % [27.2, 72.8]) |

</div>

**Letter rules** (`aggregate_v2.assign_letter`). **A**: S ≥ t_A *and* at least two of {curvature,
counter-image, arc morphology} scored ≥ 6 *and* no included critic with r·a ≥ 0.8. **B**: S ≥ t_B
(and not A). **D**: S < t_B and either the advocate located no evidence and stated a *nothing_because*,
or an upheld critic covers every evidence item at r ≥ 0.8. **C**: otherwise. The rank letter reads
S = p_evidence with no critics. Under R2 the final letter is the rank letter itself; only the D-rule (an upheld critic covering every item at r ≥ 0.8) can override it, to D.
Critics are only called when p_evidence ≥ τ0 = **0.15**; below that the final
letter equals the rank letter.

**Calibration.** t_A = **0.2000** is the smallest advocate score with ≤ 1 % false positives and t_B = **0.1700** the smallest with ≤ 5 % on the 200 clean random negatives of the truth set's design half (fit on p_evidence by `golden/calibrate_thresholds.py`; letters `opus5_api_calibrated`).

**Critics engaged.** In this run the critics were called on **70** of the 100 candidates
(p_evidence ≥ τ0) and the arbitrator ruled on **69**; the stack vetoed to D **14**.

**This run.** Model `opus5` (claude-opus-5) on the `anthropic` backend, thinking `adaptive`,
effort `xhigh`, k = 1 (arm `a1`, full stack), scored 2026-08-24, letters re-assigned 2026-08-24; thresholds key `opus5_api` (`424a8aa9875bacd2`); API cost $46.25.

## How to read the annotated cutouts

Each candidate page shows the composite the judge saw and the same composite with the panel's
stored records painted on it. Panels in the two-band **colour** layout: **a** normal 10″, **b** deep 10″,
**c** two-band colour 10″, **d** deep 3.5″ zoom, **e** colour 3.5″ zoom, **f** circular-subtraction
residual 3.5″. In the single-band **grey** layouts there is no colour information: **c** is the 10″
circular-subtraction residual and **e** the normal-stretch 3.5″ zoom. North up, East left; the ticked
galaxy is the flagged deflector at every panel's centre.

- **Cyan arcs** (k1, k2, …) are the advocate's located evidence items, drawn at their stated radius and
  position-angle span in the cited panel, in panel **a**, and in the 3.5″ zoom **d** when they fit
  (r ≤ 1.7″). A zero-length span is a small circle, a full 360° span a ring, a counter-image a cyan cross.
- **Dashed sectors** are the critics' location boxes (radius range × PA range), coloured by the
  arbitrator's ruling: <span style="color:#ff5050">**red = upheld**</span>,
  <span style="color:#ffaa28">**orange = partial**</span>, <span style="color:#aaaaaa">**grey = overruled**</span>,
  <span style="color:#c8b400">**yellow = no ruling**</span> — no arbitrator ruling for that critic (the
  arbitrator was absent or gave no ruling on it). The label is the role (Art / Geo / Mor) and the
  named alternative; abstaining critics draw nothing.
- The **legend strip** below the composite gives the rank → final letter, the veto if any, p_ev / S / S_arb,
  the arbitrator's letter and *needs_human* flag, scale class and layout; then one cyan line per
  evidence item (`*` = the arbitrator kept it; `[not drawn]` / `[r > panel X]` when it could not be
  placed) and one line per critic in its ruling colour with its refutation strength r.

## Agreement

Original pass-count grade (rows) against the new judge's **final letter** (rule R2):

<div class="t100" markdown>

| Original \ final | A | B | C | D | Total |
|:--|--:|--:|--:|--:|--:|
| **A** (5) | 4 | 1 | 0 | 0 | 5 |
| **B** (5) | 3 | 2 | 0 | 0 | 5 |
| **C** (12) | 6 | 3 | 1 | 2 | 12 |
| **U** (78) | 0 | 26 | 38 | 14 | 78 |
| **Total** | 13 | 32 | 39 | 16 | 100 |

</div>

…and against the advocate-only **rank letter** (before the critic stack):

<div class="t100" markdown>

| Original \ rank letter | A | B | C | D | Total |
|:--|--:|--:|--:|--:|--:|
| **A** (5) | 4 | 1 | 0 | 0 | 5 |
| **B** (5) | 3 | 2 | 0 | 0 | 5 |
| **C** (12) | 6 | 5 | 1 | 0 | 12 |
| **U** (78) | 0 | 31 | 45 | 2 | 78 |
| **Total** | 13 | 39 | 46 | 2 | 100 |

</div>

The 14 D-rule vetoes name these upheld, full-coverage alternatives (a candidate can
carry more than one): merger (6), spiral_arm (4), edge_on_disk (3), companion_projection (1).

## All 100

Sorted by original pipeline rank. Click a candidate for its cutouts and the full record. Design
anchors are marked †[^anchor]; ⚑ marks the arbitrator's *needs human* flag. p_ev is the
advocate's p_evidence (the ranking score and, under R2, the score the letter is read from); S_arb is
the arbitrated score, shown for reference.

<div class="t100" markdown>

| Rank | Candidate | Original | Ours (final) | Rank letter | p_ev | S_arb | Demoted by |
|--:|:--|:--:|:--:|:--:|--:|--:|:--|
| 1 | [J3440482-522486](ranks-001-025.md#rank-1) | A | **A** | A | 0.95 | 0.950 | — |
| 2 | [J15199556+2122210](ranks-001-025.md#rank-2) | A | **A** | A | 0.78 | 0.780 | — |
| 3 | [J34707505-219476](ranks-001-025.md#rank-3) | A | **A** ⚑ | A | 0.76 | 0.671 | — |
| 4 | [J30791374-4058431](ranks-001-025.md#rank-4) | A | **A** ⚑ | A | 0.70 | 0.469 | — |
| 5 | [J34084513-960342](ranks-001-025.md#rank-5) | B | **A** ⚑ | A | 0.88 | 0.611 | — |
| 6 | [J18937456+6221686](ranks-001-025.md#rank-6) | B | **A** ⚑ | A | 0.45 | 0.316 | — |
| 7 | [J18030075+2309921 †](ranks-001-025.md#rank-7) | B | **A** ⚑ | A | 0.45 | 0.197 | — |
| 8 | [J16644236-1024898](ranks-001-025.md#rank-8) | B | **B** ⚑ | B | 0.28 | 0.093 | — |
| 9 | [J19789495-132348](ranks-001-025.md#rank-9) | B | **B** ⚑ | B | 0.35 | 0.245 | — |
| 10 | [J23069956+2559453](ranks-001-025.md#rank-10) | A | **B** ⚑ | B | 0.35 | 0.190 | — |
| 11 | [J8999381-2012203](ranks-001-025.md#rank-11) | C | **A** ⚑ | A | 0.80 | 0.653 | — |
| 12 | [J2451534-2191925](ranks-001-025.md#rank-12) | C | **A** ⚑ | A | 0.60 | 0.480 | — |
| 13 | [J18805344+1121596 †](ranks-001-025.md#rank-13) | C | **A** ⚑ | A | 0.58 | 0.295 | — |
| 14 | [J18030108+2309932 †](ranks-001-025.md#rank-14) | C | **A** ⚑ | A | 0.45 | 0.197 | — |
| 15 | [J20954380-1094330 †](ranks-001-025.md#rank-15) | C | **A** ⚑ | A | 0.45 | 0.315 | — |
| 16 | [J5186648-1343587 †](ranks-001-025.md#rank-16) | C | **D** | B | 0.20 | 0.002 | edge_on_disk (Geo, Mor) |
| 17 | [J5186803-1343778](ranks-001-025.md#rank-17) | C | **D** | B | 0.24 | 0.019 | spiral_arm (Mor) |
| 18 | [J4006002-161474](ranks-001-025.md#rank-18) | C | **A** ⚑ | A | 0.45 | 0.150 | — |
| 19 | [J33372744-1401097](ranks-001-025.md#rank-19) | C | **B** | B | 0.25 | 0.120 | — |
| 20 | [J710586-7562357](ranks-001-025.md#rank-20) | C | **C** | C | 0.12 | 0.120 | — |
| 21 | [J6403415-2406677](ranks-001-025.md#rank-21) | C | **B** | B | 0.27 | 0.131 | — |
| 22 | [J5414235-4704895](ranks-001-025.md#rank-22) | C | **B** ⚑ | B | 0.50 | 0.272 | — |
| 23 | [J18818168+1282960](ranks-001-025.md#rank-23) | U | **C** | C | 0.14 | 0.140 | — |
| 24 | [J6439169-1191885](ranks-001-025.md#rank-24) | U | **D** | C | 0.15 | 0.009 | merger (Mor) |
| 25 | [J3996679-159719](ranks-001-025.md#rank-25) | U | **B** | B | 0.25 | 0.031 | — |
| 26 | [J357850-3044809](ranks-026-050.md#rank-26) | U | **B** | B | 0.27 | 0.059 | — |
| 27 | [J15032484+248405](ranks-026-050.md#rank-27) | U | **B** | B | 0.20 | 0.011 | — |
| 28 | [J26831233+6515810](ranks-026-050.md#rank-28) | U | **D** | B | 0.18 | 0.009 | companion_projection (Geo) |
| 29 | [J20148636+4809579](ranks-026-050.md#rank-29) | U | **C** | C | 0.15 | 0.025 | — |
| 30 | [J9006236-2015944](ranks-026-050.md#rank-30) | U | **C** | C | 0.15 | 0.068 | — |
| 31 | [J26876212+6510625](ranks-026-050.md#rank-31) | U | **C** | C | 0.16 | 0.050 | — |
| 32 | [J804181+1819597](ranks-026-050.md#rank-32) | U | **B** | B | 0.25 | 0.094 | — |
| 33 | [J32232919-763201](ranks-026-050.md#rank-33) | U | **D** | B | 0.28 | 0.022 | spiral_arm (Mor) |
| 34 | [J17771210-2809562](ranks-026-050.md#rank-34) | U | **B** | B | 0.25 | 0.042 | — |
| 35 | [J3807110-4434755](ranks-026-050.md#rank-35) | U | **C** | C | 0.15 | 0.009 | — |
| 36 | [J15055009+250185](ranks-026-050.md#rank-36) | U | **B** | B | 0.32 | 0.038 | — |
| 37 | [J15009907+202286](ranks-026-050.md#rank-37) | U | **C** | C | 0.15 | 0.020 | — |
| 38 | [J15010748+230270](ranks-026-050.md#rank-38) | U | **C** | C | 0.15 | 0.031 | — |
| 39 | [J16305610+3001292](ranks-026-050.md#rank-39) | U | **B** ⚑ | B | 0.38 | 0.149 | — |
| 40 | [J15029814+254087](ranks-026-050.md#rank-40) | U | **B** | B | 0.20 | 0.021 | — |
| 41 | [J34708849-217825](ranks-026-050.md#rank-41) | U | **B** | B | 0.17 | 0.024 | — |
| 42 | [J12411163+1923209](ranks-026-050.md#rank-42) | U | **C** | C | 0.12 | 0.120 | — |
| 43 | [J15049363+206401](ranks-026-050.md#rank-43) | U | **C** | C | 0.12 | 0.120 | — |
| 44 | [J26807645+6520932](ranks-026-050.md#rank-44) | U | **B** | B | 0.17 | 0.038 | — |
| 45 | [J8570140-2144889](ranks-026-050.md#rank-45) | U | **C** | C | 0.14 | 0.140 | — |
| 46 | [J9002960-2014057](ranks-026-050.md#rank-46) | U | **B** | B | 0.18 | 0.025 | — |
| 47 | [J15069997+222588](ranks-026-050.md#rank-47) | U | **C** | C | 0.12 | 0.120 | — |
| 48 | [J4312165-2100739](ranks-026-050.md#rank-48) | U | **D** | B | 0.20 | 0.005 | merger (Geo, Mor) |
| 49 | [J5282421-774282](ranks-026-050.md#rank-49) | U | **D** | C | 0.15 | 0.009 | merger (Mor) |
| 50 | [J20945435-1082161](ranks-026-050.md#rank-50) | U | **D** | C | 0.15 | 0.014 | merger (Mor) |
| 51 | [J4310830-518328](ranks-051-075.md#rank-51) | U | **C** | C | 0.14 | 0.140 | — |
| 52 | [J14920654+6969086](ranks-051-075.md#rank-52) | U | **C** | C | 0.10 | 0.100 | — |
| 53 | [J15067319+221999](ranks-051-075.md#rank-53) | U | **C** | C | 0.12 | 0.120 | — |
| 54 | [J4784995-5841089](ranks-051-075.md#rank-54) | U | **C** | C | 0.12 | 0.120 | — |
| 55 | [J4205283+264369](ranks-051-075.md#rank-55) | U | **C** | C | 0.12 | 0.120 | — |
| 56 | [J14977486+224978](ranks-051-075.md#rank-56) | U | **B** | B | 0.25 | 0.069 | — |
| 57 | [J21006317+1456150](ranks-051-075.md#rank-57) | U | **B** | B | 0.18 | 0.070 | — |
| 58 | [J2235243-7358297](ranks-051-075.md#rank-58) | U | **C** | C | 0.12 | 0.120 | — |
| 59 | [J3419795-502846](ranks-051-075.md#rank-59) | U | **B** | B | 0.28 | 0.068 | — |
| 60 | [J20960149-304690](ranks-051-075.md#rank-60) | U | **B** | B | 0.18 | 0.011 | — |
| 61 | [J8149689-2410706](ranks-051-075.md#rank-61) | U | **C** | C | 0.12 | 0.120 | — |
| 62 | [J15261689-1266966](ranks-051-075.md#rank-62) | U | **C** | C | 0.12 | 0.120 | — |
| 63 | [J15067092+210603](ranks-051-075.md#rank-63) | U | **C** | C | 0.14 | 0.140 | — |
| 64 | [J20865088+7724666](ranks-051-075.md#rank-64) | U | **B** | B | 0.30 | 0.036 | — |
| 65 | [J4429240-2347200](ranks-051-075.md#rank-65) | U | **D** | C | 0.16 | 0.019 | merger (Mor) |
| 66 | [J18850875+1133833](ranks-051-075.md#rank-66) | U | **C** | C | 0.15 | 0.018 | — |
| 67 | [J20948181-1096952](ranks-051-075.md#rank-67) | U | **D** | D | 0.04 | 0.040 | — |
| 68 | [J17772978-2808033](ranks-051-075.md#rank-68) | U | **B** ⚑ | B | 0.30 | 0.180 | — |
| 69 | [J18931633+6219972](ranks-051-075.md#rank-69) | U | **B** | B | 0.18 | 0.057 | — |
| 70 | [J17174762+4246955](ranks-051-075.md#rank-70) | U | **C** | C | 0.12 | 0.120 | — |
| 71 | [J12391133+1822895](ranks-051-075.md#rank-71) | U | **C** | C | 0.12 | 0.120 | — |
| 72 | [J21007480+1455269](ranks-051-075.md#rank-72) | U | **D** | B | 0.22 | 0.023 | merger (Mor) |
| 73 | [J8575205-2143155](ranks-051-075.md#rank-73) | U | **C** | C | 0.12 | 0.120 | — |
| 74 | [J16193787+1252943](ranks-051-075.md#rank-74) | U | **C** | C | 0.15 | 0.028 | — |
| 75 | [J35084320-3043955](ranks-051-075.md#rank-75) | U | **B** | B | 0.20 | 0.017 | — |
| 76 | [J8732838-6207334](ranks-076-100.md#rank-76) | U | **D** | B | 0.17 | 0.008 | edge_on_disk (Mor) |
| 77 | [J9192602-4479623](ranks-076-100.md#rank-77) | U | **B** | B | 0.18 | 0.022 | — |
| 78 | [J18895415+499333](ranks-076-100.md#rank-78) | U | **C** | C | 0.10 | 0.100 | — |
| 79 | [J16456132+2953030](ranks-076-100.md#rank-79) | U | **C** | C | 0.12 | 0.120 | — |
| 80 | [J15424410+3903685](ranks-076-100.md#rank-80) | U | **C** | C | 0.12 | 0.120 | — |
| 81 | [J31352710-005281](ranks-076-100.md#rank-81) | U | **C** | C | 0.10 | 0.100 | — |
| 82 | [J21483295+5282544](ranks-076-100.md#rank-82) | U | **B** | B | 0.18 | 0.036 | — |
| 83 | [J3470183+013617](ranks-076-100.md#rank-83) | U | **C** | C | 0.15 | 0.028 | — |
| 84 | [J15057245+223798](ranks-076-100.md#rank-84) | U | **D** | D | 0.04 | 0.040 | — |
| 85 | [J20861957+7726170](ranks-076-100.md#rank-85) | U | **D** | C | 0.15 | 0.006 | edge_on_disk (Geo, Mor) |
| 86 | [J21518078+5278672](ranks-076-100.md#rank-86) | U | **C** | C | 0.10 | 0.100 | — |
| 87 | [J26816024+6505016](ranks-076-100.md#rank-87) | U | **D** | C | 0.15 | 0.012 | spiral_arm (Mor) |
| 88 | [J32840780+1769774](ranks-076-100.md#rank-88) | U | **B** | B | 0.20 | 0.068 | — |
| 89 | [J9003772-2015513](ranks-076-100.md#rank-89) | U | **C** | C | 0.15 | 0.021 | — |
| 90 | [J1637643+1340558](ranks-076-100.md#rank-90) | U | **D** | C | 0.15 | 0.019 | spiral_arm (Mor) |
| 91 | [J14995729+243834](ranks-076-100.md#rank-91) | U | **C** | C | 0.12 | 0.120 | — |
| 92 | [J2104758+067717](ranks-076-100.md#rank-92) | U | **B** | B | 0.30 | 0.063 | — |
| 93 | [J11084396-7342226](ranks-076-100.md#rank-93) | U | **C** | C | 0.10 | 0.100 | — |
| 94 | [J4637336-3182658](ranks-076-100.md#rank-94) | U | **C** | C | 0.14 | 0.140 | — |
| 95 | [J34135498+2813722](ranks-076-100.md#rank-95) | U | **C** | C | 0.14 | 0.140 | — |
| 96 | [J33389992-3276033](ranks-076-100.md#rank-96) | U | **C** | C | 0.15 | 0.008 | — |
| 97 | [J18939200+6216064](ranks-076-100.md#rank-97) | U | **B** | B | 0.20 | 0.033 | — |
| 98 | [J1747804-3079539](ranks-076-100.md#rank-98) | U | **B** | B | 0.30 | 0.041 | — |
| 99 | [J35128272-4120137](ranks-076-100.md#rank-99) | U | **C** | C | 0.10 | 0.100 | — |
| 100 | [J6928514+070232](ranks-076-100.md#rank-100) | U | **B** | B | 0.30 | 0.054 | — |

</div>

[^anchor]: Ranks 7, 13, 14, 15, 16 are the scheme's **design anchors**: their
    outcomes known during design shaped the scheme's mechanisms (forbidden grounds, the coverage
    rule). They are not an independent test of the judge, whatever letter they receive here.

## Ranked by the new judge

The top 25 by the advocate's **p_evidence** (ties broken by S). This is the ranking the program
deploys — program holdout result (opus5-xhigh): the advocate ranks far better than the full critic product, whose job
is to certify and demote, not to order.

<div class="t100" markdown>

| # | Orig. rank | Candidate | Original | Ours (final) | Rank letter | p_ev | S_arb | Demoted by |
|--:|--:|:--|:--:|:--:|:--:|--:|--:|:--|
| 1 | 1 | [J3440482-522486](ranks-001-025.md#rank-1) | A | **A** | A | 0.95 | 0.950 | — |
| 2 | 5 | [J34084513-960342](ranks-001-025.md#rank-5) | B | **A** ⚑ | A | 0.88 | 0.611 | — |
| 3 | 11 | [J8999381-2012203](ranks-001-025.md#rank-11) | C | **A** ⚑ | A | 0.80 | 0.653 | — |
| 4 | 2 | [J15199556+2122210](ranks-001-025.md#rank-2) | A | **A** | A | 0.78 | 0.780 | — |
| 5 | 3 | [J34707505-219476](ranks-001-025.md#rank-3) | A | **A** ⚑ | A | 0.76 | 0.671 | — |
| 6 | 4 | [J30791374-4058431](ranks-001-025.md#rank-4) | A | **A** ⚑ | A | 0.70 | 0.469 | — |
| 7 | 12 | [J2451534-2191925](ranks-001-025.md#rank-12) | C | **A** ⚑ | A | 0.60 | 0.480 | — |
| 8 | 13 | [J18805344+1121596 †](ranks-001-025.md#rank-13) | C | **A** ⚑ | A | 0.58 | 0.295 | — |
| 9 | 22 | [J5414235-4704895](ranks-001-025.md#rank-22) | C | **B** ⚑ | B | 0.50 | 0.272 | — |
| 10 | 6 | [J18937456+6221686](ranks-001-025.md#rank-6) | B | **A** ⚑ | A | 0.45 | 0.316 | — |
| 11 | 7 | [J18030075+2309921 †](ranks-001-025.md#rank-7) | B | **A** ⚑ | A | 0.45 | 0.197 | — |
| 12 | 15 | [J20954380-1094330 †](ranks-001-025.md#rank-15) | C | **A** ⚑ | A | 0.45 | 0.315 | — |
| 13 | 14 | [J18030108+2309932 †](ranks-001-025.md#rank-14) | C | **A** ⚑ | A | 0.45 | 0.197 | — |
| 14 | 18 | [J4006002-161474](ranks-001-025.md#rank-18) | C | **A** ⚑ | A | 0.45 | 0.150 | — |
| 15 | 39 | [J16305610+3001292](ranks-026-050.md#rank-39) | U | **B** ⚑ | B | 0.38 | 0.149 | — |
| 16 | 9 | [J19789495-132348](ranks-001-025.md#rank-9) | B | **B** ⚑ | B | 0.35 | 0.245 | — |
| 17 | 10 | [J23069956+2559453](ranks-001-025.md#rank-10) | A | **B** ⚑ | B | 0.35 | 0.190 | — |
| 18 | 36 | [J15055009+250185](ranks-026-050.md#rank-36) | U | **B** | B | 0.32 | 0.038 | — |
| 19 | 68 | [J17772978-2808033](ranks-051-075.md#rank-68) | U | **B** ⚑ | B | 0.30 | 0.180 | — |
| 20 | 100 | [J6928514+070232](ranks-076-100.md#rank-100) | U | **B** | B | 0.30 | 0.054 | — |
| 21 | 98 | [J1747804-3079539](ranks-076-100.md#rank-98) | U | **B** | B | 0.30 | 0.041 | — |
| 22 | 64 | [J20865088+7724666](ranks-051-075.md#rank-64) | U | **B** | B | 0.30 | 0.036 | — |
| 23 | 92 | [J2104758+067717](ranks-076-100.md#rank-92) | U | **B** | B | 0.30 | 0.063 | — |
| 24 | 8 | [J16644236-1024898](ranks-001-025.md#rank-8) | B | **B** ⚑ | B | 0.28 | 0.093 | — |
| 25 | 59 | [J3419795-502846](ranks-051-075.md#rank-59) | U | **B** | B | 0.28 | 0.068 | — |

</div>

## Caveats

- **Model grades are not human vetting.** Every letter here is a language model's reading of six
  panels. The letters are calibrated for false-positive rate on random negatives, not validated
  against expert inspection of these 100 fields; a human grader's blind pass on the same kit is the
  test that matters.
- **Blind to metadata.** The judge saw pixels only — no filters, redshifts, coordinates, Einstein-radius
  estimate, inspector confidence or original grade. A cluster-scale system with its arcs outside the
  10″ field, or a system whose case rests on the footer, is at a disadvantage by construction.
- **Design anchors.** Ranks 7, 13, 14, 15, 16 (†) shaped the scheme's
  mechanisms during design; their letters are consistency checks, not evidence.
- **Recall on hard lenses is limited** — program holdout result (opus5-xhigh). t_A / t_B were fit on clean random
  negatives, so the letters control false positives, not completeness. On the truth holdout (COWLS
  and literature lenses vs catalogue-purged negatives) the opus5-xhigh advocate reaches recall
  0.333 at 5 % FPR with AUC 0.764; the full critic stack ranks at AUC
  0.468 on 149 scored negatives — it under-ranks the advocate — but it is the
  layer that demotes refuted panels: its D-rate on the stress_D panels was 6/20 = 0.30
  against 0/20 for every advocate-only arm (provisional letters).
  After calibration the transfer check gives — rank letter: FPR@A∪B 3.0 %, recall@A∪B 28.6 %, stress_D at A∪B 16/20; R1: FPR@A∪B 0.0 %, recall@A∪B 2.4 %, stress_D at A∪B 4/20; R2: FPR@A∪B 2.0 %, recall@A∪B 23.8 %, stress_D at A∪B 10/20 (table under
  "Why this rule").
  Hence *advocate ranks, critic stack certifies*.
  This run uses that opus5-xhigh configuration (thinking `adaptive`, effort
  `xhigh`).
  For scale: the original pass-count verifier gave 0 of 3 passes to 23 of the 24 known lenses it
  examined, and none of the 31 COWLS lenses in the truth set came out of it at A/B.
- **needs_human flags.** The arbitrator flagged **17** candidates (17 of
  them at A/B) as contested — the escalation set a human should look at first.
- **One draw.** k = 1 replicate per item; a re-run moves individual scores.

---

Pages: [Ranks 1–25](ranks-001-025.md) · [Ranks 26–50](ranks-026-050.md) · [Ranks 51–75](ranks-051-075.md) · [Ranks 76–100](ranks-076-100.md).
