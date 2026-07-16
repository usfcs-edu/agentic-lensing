# X2 — First formal SBC of the GIGALens pipeline class (ranks-as-data gift)

**Last updated:** 2026-07-16

Cross-pollination deliverable (PLAN X2; design checkpoint + pre-registered gates in
`research/checkpoints_x2.md`, written before any run). This document follows the
claims-register conventions of GIGALens-Code `docs/logs/lab-notebook-TEMPLATE.md`.
**All verdicts are `proposed (UNCERTIFIED — external)`; the ranks are the
deliverable; interpretation and certification belong to the GIGALens team.**
Producer: claude-giga-lens-linus campaign (Front D). Grader: _pending (theirs)_.

---

## Current state

Complete single-arm SBC run: N=32 prior-matched mock systems of the
hundred_systems_GL2 config class, pipeline = vendored scene-API
MAP->SVI->HMC (forward mode, diagonal plug-in likelihood), 22 params/system,
ranks vs 127 thinned pooled HMC draws. Artifacts: `data/x2_sbc.json`,
`figs/x2_rank_hist_grid.png`, `figs/x2_rank_hist_glasshouse_gamma.png`,
per-system `data/x2_sbc_runs/sys*.npz`, harness `30_sbc_gift.py`.
Cost: 28.3 A16-GPU-h (batch) + 1.0 (smoke) of the 36 committed.

## Validity precondition (why this is OUR mock set, not their frozen npz)

SBC rank uniformity holds iff truth-draws ~ fitting prior. For the frozen
`100SystemsStandard80px.npz`: **truth-draws != fitting prior** — their own
provenance (`experiments/why_hard_to_sample/t13_resim.py` STEP 0: "GENERATION
prior ... NARROWER than today's MODELING prior", cell 13 of
attic/Linus-FourSim.ipynb, PRNGKey(0)) and their registry
(`gigalens_research/simtests/experiments/gl2_sersic.py`: `gl2_simulation_prior`
[Eq. 8 Simulation column] vs `gl2_inference_prior` == `make_default_prior`
[Prior column]). Running SBC on the frozen set would fail by construction, not by
pipeline defect. (The npz + params yaml are also absent from the local mirror.)
**Arm run = prior-matched regeneration**: truth z_i ~ the FITTING prior
(make_default_prior re-expressed as scene Components, P0 parity-audited), render
SceneSimulator.simulate (0.065"/80px/ss2, vendored assets/psf.npy), noise
np.random.seed(3000+i) + lenstronomy add_poisson(exp_time=100) +
add_background(sigma_bkd=0.2) — systems 0..7 bit-identical to the certified
`hs2_sys{0..7}` zoo targets (PRNGKey(1000+i)).

Per-system render+noise consistency: chi2/pixel at truth median 1.001
(range 0.967–1.036, all 32) — the generative and likelihood noise scales agree.

## What was run (settings actually executed)

- Pipeline (vendored `gigalens.jax.inference.ModellingSequence`, no
  gigalens_research import; optimizer/stage values copied with attribution from
  their `inference_utils/pipeline.py` defaults): MAP adabelief(1e-2, b1=.95,
  b2=.99, nesterov), 350 steps x **128** samples (their default 500 OOMs the
  15.3-GB A16: 23.8 GiB asked — ledgered Deviation 2); SVI full-rank
  MultivariateNormalTriL, adabelief(1e-4), 1500 steps x **128** n_vi (remat twin);
  HMC PreconditionedHMC + trajectory-length + step-size adaptation, 32 chains x
  (300 burnin + 750 kept), init_eps 0.3, init_l 3, max_leapfrog 30. f64.
  Stage seeds vary per system (their campaign fixes seed=0 — documented deviation;
  SBC needs independent pipeline randomness).
- Their campaign.yaml REFERENCE settings (MAP 1000x2000, SVI 5000x1000, HMC
  64x2000) are INFEASIBLE-ON-A16 (memory) — this readout is therefore of the
  pipeline CLASS at a reduced budget, on our certified port, and NOT a statement
  about their production runs at reference settings.
- N=32, not the assigned >=64: the pre-stated budget rule (checkpoint) selected
  the largest multiple of 8 fitting 36 A16-h at the measured 3508 s/system.
  UNDER-DELIVERY stated plainly; the harness + generation recipe make an N>=64
  rerun turnkey on faster-FP64 hardware. n=32 caveat: 8-bin chi2 has expectation
  4/bin; every p below carries it.
- Runtime monkeypatch (vendor tree UNPATCHED): `_shard_map` re-bound with
  `check_vma=False` — jax 0.6.2's varying-mode replication CHECKER rejects the
  FFT-conv cotangent in MAP/SVI backward passes; computation unchanged
  (single-device mesh). Ledgered as Deviation 1.
- Rank machinery copied verbatim with attribution from the OLD campaign's
  `cgl/e1.py` (thin_indices / sbc_rank / rank_uniformity_chi2; n_use=127, 8 bins).
  Ranks computed in flat-z; runtime guard PASSED (all 22 bijectors increasing,
  z-ranks == constrained-space ranks on 2048 thinned draws, 0 mismatches).

## Claims register

### X2-C1 — The pipeline class at this budget is SEVERELY rank-miscalibrated in the lens-light photometric block
- **Status:** `proposed (UNCERTIFIED — external)`
- **Criterion (pre-registered):** severe = |rank-location z| > 5 anywhere (this was
  the checkpoint FALSIFIER for "mild at worst"; it TRIPPED).
- **Numbers (N=32, all fits):** LL.Ie z=−5.27 (chi2 p=7.6e-11; 18/32 truths in the
  BOTTOM rank bin ⇒ posterior systematically OVER-estimates Ie), LL.R_sersic
  z=+5.76 (p=1.5e-10; 18/32 in the TOP bin ⇒ UNDER-estimates R_sersic),
  LL.n_sersic z=+4.75 (p=8.1e-7). One-sided pile-ups with coherent signs — the
  Ie/R_sersic/n degeneracy slides one way, which plain under-mixing does not
  naturally produce (under-mixing gives symmetric U-shapes).
- **Evidence:** figs/x2_rank_hist_grid.png (row 2), data/x2_sbc.json.
- **Doubt report (producer, mandatory):** (1) 30/32 fits fail our health rule
  (max split-Rhat ≤ 1.05 ∧ min ESS ≥ 200; median max-Rhat 1.76, worst 584) — the
  E1c precedent (below) shows sampler pathology can fake rank failures; however
  the LL-block signature is one-sided, not U-shaped, and survives sign-coherently
  across all 32 systems. (2) The reduced MAP/SVI budget may under-precondition
  HMC specifically along the LL photometric ridge. (3) The plug-in error map
  (bg² + observed/exp_time) mis-weights the bright lens-light-dominated pixels —
  a genuine likelihood-approximation channel that would bias exactly this block.
  The n_healthy=2 secondary readout is uninformative (reported, not used); these
  three channels are NOT separable from this run alone.

### X2-C2 — Mass block: no large location bias, but rank uniformity fails via U-shapes
- **Status:** `proposed (UNCERTIFIED — external)`
- **Numbers:** worst mass |z| = 2.25 (EPL.center_x); all 8 mass params |z| < 2.3.
  But 6/8 fail the chi2 p>0.01 gate (EPL.gamma p=1.4e-8: 12+13 of 32 in the two
  extreme bins; theta_E p=7.6e-4, e1 p=9.5e-5, e2 p=4.1e-4, SHR.gamma2 p=2.1e-3,
  center_x p=2.5e-3). The pre-registered "mass calibrated" prediction FAILED at
  the uniformity level while holding at the location level.
- **Doubt report:** U-shapes (truth in both tails too often) are the classic
  under-dispersion/stuck-chain signature and 30/32 fits are unhealthy — the most
  economical explanation is under-mixing at this budget, exactly the OLD E1c
  failure mode (glass-house below). A genuine under-dispersed-posterior
  explanation cannot be excluded without healthy replicates.

### X2-C3 — Source block: mildest of the three
- **Status:** `proposed (UNCERTIFIED — external)`
- **Numbers:** all 7 source params |z| ≤ 1.8 (worst SRC.n_sersic +1.79, p=0.085);
  photometric trio Ie/R_sersic/n_sersic all pass p>0.01; centers + e1 fail
  uniformity via U-shapes (center_x p=4.1e-5, center_y p=7.7e-5, e1 p=2.1e-3).
  The checkpoint's source-adjacent |z|~1–3 prediction is CONSISTENT at the
  location level; the severe miscalibration landed in the LENS-light photometric
  block instead.

### X2-C4 — Convergence-health datum: 2/32 fits healthy at this budget
- **Status:** `proposed (UNCERTIFIED — external)`
- **Numbers:** healthy (Rhat ≤ 1.05 ∧ ESS ≥ 200 over all 22 dims): 2/32. Median
  max-Rhat 1.76; 10/32 have max-Rhat > 3; per-system rows in x2_sbc.json.
- **Note for their benchmark table:** this is on OUR reduced-budget A16 rung —
  NOT their published campaign settings; it strengthens the campaign-wide thesis
  that per-target tuning/budget dominates pipeline reliability on this class.

## Glass-house disclosure (our own house, same test)

Our OLD campaign's E1c SBC (44 drizzle-mock fits, correlated-likelihood
MAP->SVI->ChEES-HMC) **FAILED the same gate**: gamma rank p = 6.5e-5 (16/44 in
the first bin), cov68 = 0.34 (`../claude-giga-lens/data/e1_report.json`).
The amended healthy-only re-readout (n=13, deep two-stage reruns) flipped every
param to PASS (gamma p=0.53) — i.e. OUR gamma pathology was sampler-induced.
Side-by-side gamma histograms: `figs/x2_rank_hist_glasshouse_gamma.png` (this
work: U-shaped 12+13/32 in extreme bins; E1c: bottom-heavy 16/44). The same
amendment path (deep/re-preconditioned reruns, then healthy-only re-readout) is
the natural next step here and is exactly what n_healthy=2 currently blocks.

## Physicality-layer accounting (arms note)

The assigned {physicality ON, OFF} pair is VACUOUS on this substrate — verified
from vendored code before the run: `gigalens/physicality.py` validates at
LensModel construction and diagnoses posteriors; it never enters
log_prob/log_prior, so there is no density truncation to toggle. Single arm run.
As data instead: (a) construction-time findings on the GL2 paper prior itself —
EPL `e1²+e2² < 1` prior mass 5.3e-6 (their own hard-domain warning fires on their
default prior; threshold 1e-6) and shear `|γ| ≤ 0.2` plausibility mass 0.136;
(b) posterior diagnosis (sys00, 512 draws): clean — 0 hard violations, 14 checks.

## Prediction vs observation (checkpoint closure)

| Pre-registered | Observed |
|---|---|
| source-adjacent \|z\| ~ 1–3, mass \|z\| < 2 & p > 0.01 | source \|z\| ≤ 1.8 (ok); mass \|z\| ≤ 2.25 but 6/8 uniformity FAIL (U-shapes) |
| falsifier: \|z\| > 5 anywhere | **TRIPPED** — LL.Ie −5.27 / LL.R_sersic +5.76 (lens-light, not source) |
| health reported, primary = all fits | 2/32 healthy; primary readout stands; healthy-only n=2 uninformative |

The assigned lstsq/ridge mechanism was not testable (this config is forward-mode,
amplitudes sampled — scope correction in the checkpoint); the amplitude-adjacent
block nonetheless carries the severe signal.

## Follow-ups (offers, not actions)

1. N>=64 at their REFERENCE settings on A100/L4 (the decisive run; harness is
   turnkey: `30_sbc_gift.py run` + `harvest`).
2. E1c-style amendment: deep/staged re-runs of the 30 unhealthy fits, then
   healthy-only re-readout — separates under-mixing from posterior
   mis-approximation on the LL block.
3. If the LL-block bias survives healthy replicates: test the plug-in error-map
   channel (iterate err from model rather than observed image) — a one-line
   ProbModel variant.

## Artifacts

- `data/x2_sbc.json` (full per-param table, health roster, gates, provenance)
- `figs/x2_rank_hist_grid.png`, `figs/x2_rank_hist_glasshouse_gamma.png`
- `data/x2_sbc_runs/sys{00..31}.npz` + worker logs; `data/x2_sbc_runs/settings_decision.json`
- Harness: `30_sbc_gift.py`; design record: `research/checkpoints_x2.md`
- Cost: 28.3 (batch) + 1.0 (smoke) = 29.3 A16-GPU-h of 36 committed; 0 A100-h.
