# Engagement memo — claude-giga-lens-linus campaign (DRAFT for Greg Benson to review & send)

To: Xiaosheng Huang & the Strong Lens team (Linus, Sean, …)
From: Greg Benson
Date: 2026-07-15 (draft — send after review)
Re: A bridge campaign building on your next-gen GIGALens; coordination points, gifts, and a small RFC

---

## 1. What we're doing

We just closed a lens-modeling campaign (claude-giga-lens, report available) that built and
validated a **drizzle correlated-noise likelihood** (convolutional whitening + amplitude
marginalization with the Occam term) and ran the first **sampler benchmark on real lens
posteriors** (the working recipe on hard real posteriors is tempered SMC with per-basin
evidence). On the real HST lens DESI-165.4754−06.0423 the correlated likelihood proved
*necessary but not sufficient*: it kills the binned-product bimodality (a 191-nat evidence flip —
the steep basin is a noise-covariance artifact) but over-corrects the slope, leaving a
scale-dependent bracket we're still hunting.

We're now starting a follow-on campaign that builds directly on your next-gen work
(`gigalens @ linusu-dev-merge` + GIGALens-Code), pinned at `80916d2` (2026-07-15), vendored and
unpatched. Everything happens in our repo; nothing touches yours. The campaign runs
self-contained — this memo is coordination, not a request for anything.

## 2. What we will NOT touch (and where we'd like a quick "go/no-go")

We read your lab notebooks and plans carefully and want to stay out of your in-flight work:

- **Your sampling-methods paper** (MCLMC/MAMS/LAPS, old API): we will publish **no result
  readable as a sampler-efficiency comparison on unimodal targets**. Every sampler number we
  produce attaches to multimodality, evidence (logZ), cold-start, or noise-model questions —
  the areas your laps-spec explicitly marks out of scope / future work.
- **Vela source-systematics**: untouched. We prototype an evidence-scored (Bayes-factor)
  variant on 4 systems internally and offer it as an add-on column if useful (§4).
- **One flag we want to raise proactively:** we plan to run benchmark cells on the **carousel**
  system (as the hardest known real target) including a *minimal* flow-preconditioned-MAMS arm —
  i.e., a small version of your approved plan — because our SMC bet needs an honest comparison
  point. All carousel/LAPS-adjacent results come to you FIRST, will not appear in any external
  artifact without your sign-off, and if you'd rather we drop or defer those cells, say the word
  and we will. This is the one place we're knowingly near your lane.

Everything derived from your unpublished repos is publication-gated on your explicit sign-off.
Our results on your substrate are labeled **UNCERTIFIED (external)** in your vocabulary — we
adopt your artifact formats (pre-registered design checkpoints with derived thresholds, claims
registers, plots-before-metrics) so your grader can certify or reject without rerunning.

## 3. RFC: correlated noise as a scene-API `LikelihoodTerm` (comment welcome, not blocking)

We're implementing correlated noise against your documented `Dataset`/`LikelihoodTerm` seam,
no changes to your code:

- `CorrelatedImageData(ImageData)`: takes a frozen whitener bundle (FFT-designed convolutional
  whitening taps + heteroscedastic D^{-1/2} + eroded keep-mask + operator-norm certificate
  e_op ≤ 0.02), validates raise-never-default; `event_size` = kept whitened dof.
- `CorrelatedImageLikelihoodTerm`: ONE `lstsq_simulate(return_stacked=True)` render per call
  (your single-forward-eval contract); whitens residual + design columns; generalized-ridge
  amplitude marginalization **with the −½ log det A Occam term** (your current lstsq drops it —
  it matters for any Bayes factor downstream, including substructure); reports whitened χ²
  via `reports_chi2=True`; masks via `where()` per your miscompile note.
- Diagonal-limit gate: with a delta kernel it must equal your stock `ImageLikelihoodTerm` to
  ≤1e-10 — which also makes it a cross-check instrument for the scene API itself.

If you have opinions on where this should eventually live (`gigalens/jax/correlated_noise.py`?)
or on the whitener-bundle format, we'll shape to match. Silence is fine; we proceed regardless
and hand you a PR-ready diff at the end.

## 4. Gifts (no strings; adopt, adapt, or ignore)

1. **Scene-API certification slice**: bit-level parity harness old-validated-stack ↔ scene API
   (forward image, design matrix, loglik, gradients) for the EPL+shear+Sérsic/shapelet class —
   the first external validation evidence for the new API (your `tests/validation` README marks
   it unrun). You get the harness + report JSON.
2. **SBC harness** adapted to `hundred_systems_GL2` (formal rank/coverage calibration — Talts et
   al., not invented numerics). We deliver rank histograms as *data*; interpretation is yours.
   Full disclosure: our own pipeline's SBC has an open γ-rank failure we're co-investigating in
   the same harness — this is a shared instrument, not a judgment.
3. **`SMCStage` adapter spec** for your `pipeline.py` + a tempered-SMC-with-MAMS-mutations
   implementation (prior-seeded, produces logZ + per-basin evidence — the annealing/evidence
   wrapper your laps-spec names as missing). Take it or leave it.
4. **HessianSurrogateStage restore** (from b82397c, per your own note) — we need a scene-API
   Laplace stage ourselves; happy to hand over the diff.
5. **Fermat Δt sensitivity number**: how much does the noise-model choice (diagonal vs
   correlated) move Fermat-potential differences on a real drizzled lens? One number, relevant
   to any future TD/H0 ambitions.

## 5. Authorship & credit posture

Any paper whose results run through your substrate or profile library: co-authorship offered to
the relevant authors (you tell us who). Nothing from your unpublished work is cited or
characterized externally without sign-off. Our two extraction-ready papers from the previous
campaign (correlated-noise likelihood; sampler benchmark) predate this work and stay independent.

## 6. What we'd love (entirely optional)

- A go/no-go on the carousel cells (§2).
- A pointer if `design/phase1-model-data-api.md` (referenced as spec-of-record in the docs but
  absent from the checkout we have) lives somewhere we can read.
- Anything you'd like measured while we're set up: we'll have prior-seeded SMC-with-evidence
  running on scene-API targets and can add cells cheaply.

— Greg

---
*Prepared 2026-07-15 as part of the claude-giga-lens-linus campaign (P0). Campaign plan:
`reproductions/claude-giga-lens-linus/plans/PLAN.md`. Contact for technical details: the
campaign ledger CAMPAIGN.md carries every gate and number.*

---

## Addendum — 2026-07-21 (experimental program complete; 53.5 of 100 A100-h)

What's new since the memo was drafted. Full claims register (your lab-notebook format, every
verdict `proposed (UNCERTIFIED — external)`): `papers/handoff/CLAIMS.md`.

1. **The scene-API certification gift is delivered.** Forward image / design columns match our
   validated old stack to ≤6e-15 rel, gradients to 1.5e-11, on both phoenix and Perlmutter —
   the first external certification evidence for the scene API (EPL+shear+Sérsic/shapelet
   class), given three documented convention reconciliations. One heads-up: the two stacks use
   different Sérsic b_n approximants (~2e-3 model-level).

2. **The correlated-noise port works cross-stack at posterior level.** The scene-API
   `CorrelatedImageData` refit reproduces our money number γ=1.103 to Δ0.0027 and its logZ to
   0.11 nats across stacks and machines. The Occam −½logdet A term is in (your lstsq drops it).

3. **Carousel — thank you for the real MUSE cutouts.** We ran the descoped real-data cell:
   prior-seeded SMC spent 11.3 A100-h reaching only λ=0.15 (sampler healthy, killed by cost),
   and the budget-matched warm MAMS baseline never finished burn — **neither vehicle converges
   at campaign-affordable budgets on this target class**, which corroborates your min-ESS
   12/16000 experience rather than beating it. Everything is in your hands first, as promised.
   One limitation: your carousel posterior arrays are gitignored/absent from our mirror, so our
   comparison rows are summary-stats-only — **if you can transfer posterior draws, we can build
   a draw-level comparison.**

4. **SBC (gift #2, delivered as ranks-not-verdicts) + one validity question.** At reduced
   budgets on our certified port, the pipeline class shows a severe one-sided lens-light rank
   miscalibration (|z| up to 5.8) and 2/32 healthy fits — with our own glass-house E1c
   precedent (sampler-induced rank failure that vanished on healthy reruns) argued against
   ourselves in the doubt report. A question, not an accusation: the frozen
   `100SystemsStandard80px` set appears (per your own t13_resim STEP 0) to be generated from a
   narrower prior than today's modeling prior, which would fail SBC by construction — is that
   intended? We ran on prior-matched regenerated mocks instead.

5. **A warning relevant to your sampler work:** unadjusted-MCLMC mutations inside tempered SMC
   inflated a minor-mode's evidence **×56** [22–146] while leaving the within-basin location
   untouched (Δγ 0.0016) — resampling does not launder the bias. MAMS (adjusted) is clean.
   Internal-channel material; sign-off rules apply as before.

6. **Also in hand:** stationarity of our stationary-kernel noise class is REJECTED on the real
   field (calibrated p=0.010 — the standing caveat on our own 1.103); the noise-model choice
   moves Fermat Δφ by 60–90% (illustrative); and a B5 gate failure that indicts **our old
   campaign's** frozen evidence reference's coverage — our lesson, shared in case you freeze
   MCMC-derived references as gates too.

7. **Honesty items:** the S7 flow-MAMS arm and S4/S5 LAPS arms were NOT run (wave-1 costs ate
   the budget; bright line kept); the HessianSurrogateStage restore (gift #4) was never
   started — strike it or tell us it's still wanted; the anchor arbitration on v2d closed
   PARTIAL (four vehicles all reproduce the known target-intrinsic difficulty; no stack
   discrepancy found).

— Greg
