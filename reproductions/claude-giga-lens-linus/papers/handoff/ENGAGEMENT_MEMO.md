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
