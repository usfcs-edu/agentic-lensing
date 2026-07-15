# claude-giga-lens-linus — Campaign Plan
## Bridging the claude-giga-lens program onto the team's next-gen GIGALens (scene API): microcanonical SMC evidence layer, correlated noise as a LikelihoodTerm, and the profile-class fork

Branch: `claude-giga-lens-linus` (new, off `main`) · Work dir: `reproductions/claude-giga-lens-linus/`

---

## 1. Context

Two lineages converge. **Ours:** claude-giga-lens (complete, 2026-07-14) delivered a validated
drizzle correlated-noise likelihood (necessary-but-not-sufficient on the real lens: 191-nat basin
flip, but γ over-corrects to 1.103±0.008 vs the 1.433 native anchor — the "bracket" mystery), the
first lens-posterior sampler benchmark (per-basin tempered-SMC evidence is what works on real
saddle/bimodal posteriors), and a tiered follow-on program (NEXT_DIRECTIONS.md). **Theirs:**
Xiaosheng's group is building next-gen GIGALens in two coupled repos — the library
`gigalens-linus @ linusu-dev-merge` (JAX-only "scene API" rewrite: unified multi-plane/multi-band
LensModel/ProbModel, flat-z bijector, physicality layer, differentiable cosmology, point-source/
time-delay data, MCLMC/MAMS/NUTS/NeuTra experimental samplers, adaptive supersampling, 18 mass
profiles incl. dPIE/BPL; validation suite UNCERTIFIED) and the research repo `GIGALens-Code`
(staged-inference pipeline, LAPS sampler, heavy pre-registration/proposer≠grader rigor culture,
sampling-methods paper in progress on the old API).

**The fit (exploration-verified):** their named gaps are exactly our assets — no correlated-noise
likelihood, no SMC/evidence/logZ (their own laps-spec: multimodality "would need an
annealing/SMC/tempering wrapper"), no SBC, no PSF marginalization, no ridge/Occam in lstsq. Their
named open problems (LAPS cold-start fails via broad-basin scale-lock; carousel min-ESS 12/16000;
~5% secondary mode that reverse-KL flows miss) are adjacent to what our SMC machinery does. Their
profile library (dPIE, BPL) enables a genuinely NEW attack on our bracket mystery: **is the EPL
single-power-law assumption itself the residual systematic?** (the fifth suspect NEXT_DIRECTIONS
never listed).

**Goal:** improve the state of the art in strong lens modeling; deliverables of publishable
quality; upstream-ready gifts for the team.

## 2. User decisions (LOCKED)

- Branch `claude-giga-lens-linus`; work in `reproductions/claude-giga-lens-linus/`.
- **Team contact:** prepare a one-page engagement memo + interface RFC as week-1 deliverables for
  BENSON to send; campaign proceeds fully self-contained in parallel (no waiting on their
  feedback; adoption at their discretion). All results derived from their unpublished repos are
  publication-gated on their sign-off (hard rule §8).
- **Carousel: INCLUDE FULLY** — the B1 benchmark cell runs on their hardest ESS-limited system,
  including a minimal flow-preconditioned-MAMS comparison arm (S7, the minimal version of their
  approved plan). Results shared with the team (via memo channel) before any external use.
- **Budget: new allocation, cap 100 A100-h** (commit ~80, HARD STOP 100; shared-QOS single-GPU
  jobs on `cosmo_g`; ledger row appended BEFORE reading results — house rule).
- Implementation uses **Workflow orchestration** (ultracode): each phase = scout inline → Workflow
  fan-out (build/experiment cells) → adversarial verify → ledger. Autonomous operation with the
  dead-man watchdog (built in P0 — the 4-day P1c stall must not recur).

## 3. Substrate & environment

- **Vendor `gigalens-linus` @ `80916d24f3e616edecf9fb66b041c716fa111c29`** (branch linusu-dev-merge,
  verified clean) into `vendor/gigalens-linus/` via `git archive` (no `.git`), UNPATCHED,
  `VENDORED_REF.txt` = full SHA. Install `pip install -e vendor/gigalens-linus --no-deps` (the
  `tensorflow==2.19.0` dep is inert — verified never imported from `src/gigalens/`).
  **Re-pin procedure** (moving-branch discipline, ledgered): motivation → re-archive → diff-review
  of the 3 subclassed surfaces (`scene_prob_model.py`, `scene_simulator.py`, `scene.py`) → full
  gate battery re-run green → ledger row.
- **New venv `/raid/benson/.venvs/cgl2`** (never touch `gigalens` blessed or `cgl`): python 3.13,
  **jax/jaxlib 0.6.2** (their declared pin; only version validated on aarch64 phoenix with our XLA
  workaround flags; only intersection ever validated for the old stack — cross-stack parity needs
  both stacks in ONE process), blackjax==1.3 (`adjusted_mclmc` verified present),
  tfp==0.25.0, numpy 2.4.6 (deviation from their 2.1.3 recorded; covered by gates), NO tensorflow.
  Pins seeded from claude-giga-lens `constraints.txt`; all installs `-c constraints.txt` (the chex
  lesson). Gate F8 runs the parity harness once under a NERSC jax-0.10-nightly-like env,
  informational ≤1 A100-h.
- **Old validated stack imported by path**: `../claude-giga-lens/vendor/gigalens-sean` (@58ec9a7),
  `../claude-giga-lens/cgl` (unmodified), foundry-i data products, whitener bundles.
- Perlmutter: extend `requirements-perlmutter.txt`; DEPLOY LESSON as policy (rsync + md5-audit of
  all campaign `.py` before every production run; canary job first).

## 4. Directory layout & shim package

```
reproductions/claude-giga-lens-linus/
├── CAMPAIGN.md                  # ledger: locked decisions, A100-h table, gate rows, stage log
├── README.md                    # §Verification = ALL gates below, FROZEN at P0; copy-vs-import rules
├── pyproject.toml  constraints.txt  requirements-{aarch64,perlmutter}.txt
├── vendor/gigalens-linus/       # @80916d2, UNPATCHED, VENDORED_REF.txt
├── cgl2/
│   ├── paths.py                 # vendor bootstrap + EXPECTED_VENDOR_REF; import-by-path constants
│   ├── guards.py                # carried: require_x64/gpu/single_device, assert_psf_sampling,
│   │                            #   floor_svi_covariance, model-subtracted-sky
│   │                            # NEW: require_vendor_ref, require_jax_pin("0.6.2"),
│   │                            #   assert_scene_config_certified (refuses uncertified
│   │                            #   profile/ss/PSF combos unless CGL2_UNCERTIFIED_OK=1),
│   │                            #   assert_whitener_bundle (e_op≤0.02 + hash),
│   │                            #   assert_flatz_roundtrip (z_param_names-keyed only),
│   │                            #   physicality_check_or_raise, require_where_mask_semantics
│   ├── param_map.py             # old 46-dim z ↔ scene params dict (via z_param_names, NEVER order)
│   ├── scene_build.py           # LensModel/ImageData factories for the certified config class
│   ├── whiten.py  marg.py  noise.py   # COPIED with attribution from ../claude-giga-lens/cgl/
│   ├── correlated.py            # CorrelatedImageData + CorrelatedImageLikelihoodTerm (upstream-shaped)
│   ├── evidence.py              # per-basin evidence bookkeeping (ported 24_basin pattern)
│   └── samplers/
│       ├── common.py            # copied w/ attribution; mutation-kernel generalization
│       └── smc_micro.py         # NEW: tempered SMC with MAMS (primary) / MCLMC (diagnostic) mutations
├── 00_env_check.py  00_run_tests.sh  01_parity_scene.py  02_port_whiteners.py
├── 03_correlated_term_validation.py  04_smc_micro_validation.py  05+_science stages
├── tests/  slurm/  data/(gitignored)  figs/  research/  papers/ (incl. papers/handoff/)
```

**Correlated-noise port design** (`cgl2/correlated.py`): subclass the documented `Dataset`/
`LikelihoodTerm` seam (scene_prob_model.py:46–100). `CorrelatedImageData.__init__` takes a frozen
whitener bundle (h_taps, sqrt_d_inv, eroded keep-mask, e_op, kernel hash), validates
raise-never-default (their M3 pattern); `event_size` = kept whitened dof. The term takes ONE
`lstsq_simulate(..., return_stacked=True)` basis render (honors their single-forward-eval
contract; reuses their fused conv-pool + remat_basis verbatim), whitens residual + design columns
(jax.checkpoint on the whitening conv — the ≤200 MB/particle fix), ridge-marginalizes with the
Occam −½logdet A term (their lstsq lacks it; it's on foundry-i's owed-upstream-fixes list),
reports whitened χ² through `reports_chi2=True` (satisfies ProbModel's chi2-channel requirement).
Masked ops via `jnp.where`, never multiply (their jaxlib miscompile note + our convergent lesson).

**MC-SMC design** (`cgl2/samplers/smc_micro.py`): generalize our validated
`adaptive_tempered_smc` driver (`ProbModel.log_prior(z)`/`log_like(z)` is exactly blackjax's
(logprior, loglik) split — verified) with mutation kernel ∈ {hmc (proven baseline), **mams
(primary — Metropolis-adjusted ⇒ exact π_λ invariance ⇒ unbiased evidence)**, mclmc
(diagnostic-only behind a pre-registered bias gate)}. Per-λ tuning frozen at P0: mass matrix from
weighted particle covariance (their window_adaptation regularizer as reference, copied not
imported — their experimental/ modules import private blackjax internals + the old simulator);
step size via short dual-averaging pilot to MAMS acceptance ~0.9; L from their heuristic;
systematic resampling at target_ess 0.7. Do NOT import their experimental kernels directly.

## 5. Certification gates (FROZEN at P0, README §Verification)

Cross-stack parity `01_parity_scene.py` — old validated 58ec9a7 stack vs scene API, same foundry-i
v2d/v3b inputs, z_ref + 3 seeded perturbations, **compared in constrained space** through
`param_map.py`:

| Gate | Statement | Threshold |
|---|---|---|
| F1 | forward image, old vs SceneSimulator.simulate | ≤1e-12 rel |
| F2 | design-matrix columns, old `_design_ret` vs `lstsq_simulate(return_stacked=True)` | ≤1e-12 rel |
| F3 | diagonal masked loglik + χ², old vs ImageLikelihoodTerm | ≤1e-8 |
| F4 | grad of loglik wrt constrained params (chain rule through each stack's bijector) | ≤1e-8 rel-L2 |
| F5 | delta-kernel CorrelatedImageLikelihoodTerm ≡ stock ImageLikelihoodTerm | ≤1e-10 |
| F6 | Occam −½logdet A vs numpy slogdet | ≤1e-10 |
| F7 | unconstrained(constrained(z))==z round-trip + z_param_names audit | exact (info) |
| F8 | same harness under NERSC jax-0.10 env, 1 shared-QOS cell | report-only |

F1–F4 = the first certification of the scene API forward model for the EPL+shear+Sérsic/shapelet
config class (their validation is an unrun skeleton — this is a genuine gift). Plus lenstronomy
oracle subset for our profile class (informational). MC-SMC correctness gates (P0): adapter logp
parity ≤1e-8 at 64 draws; T0 analytic targets — mix2 |logZ−analytic|≤3σ_boot AND minor-mode weight
|ŵ−0.2|≤0.053 (3× binomial, N=512); funnel10 logZ; illcond46 worst-param |z|<3; MCLMC-vs-MAMS
ΔlogZ bias screen (>3σ ⇒ MCLMC demoted to cost-frontier-only).

## 6. Science program

### P0 (wk 1; ≤1 A100-h) — scaffold + adapters + MC-SMC v0
**FIRST ACTIONS (before any other work, per user):** (1) commit THIS plan verbatim into the
campaign tree at `reproductions/claude-giga-lens-linus/plans/PLAN.md`; (2) draft the engagement
memo + correlated-noise interface RFC at `reproductions/claude-giga-lens-linus/papers/handoff/
ENGAGEMENT_MEMO.md` for Benson to review and send (do-not-touch list, RFC, gift list,
co-authorship posture).
Then: venv, vendor, cgl2 skeleton, guards, parity harness, tests; dead-man watchdog (half-day);
zoo target adapters wrapping their `ProbModel.log_prob(z)`: `carousel33`, `dspl20_orig`,
`dspl20_ratio`, `hs2_sys{0..7}` (hundred_systems), + our T2/T3 by path. MC-SMC v0 + B0 correctness
gates.
Exit: plan + memo committed; F1–F7 + B0 pass.

### P1 (wk 1–2, ∥ P0; ≤10 A100-h) — certification slice on the OLD validated stack
Only the NEXT_DIRECTIONS items that are load-bearing prerequisites here:
- T0.2: jax.checkpoint the whitened conv (parity ≤1e-12; ≤200 MB/particle) + 2 SMC seeds/basin.
  Gates: σ_seed(γ)≤0.008, σ_seed(ΔlogZ)<5 nats — these parameterize every downstream gate.
- T0.3 companion-mask discriminator (~2 h). T0.4 free CPU checks (stationarity, λ-arm, real-space
  head-to-head).
- E3 kernel scan DEFERRED with written justification: X1 Bayes factors are fixed-whitener (log|C|
  cancels); E3 remains the standing caveat on absolute-γ claims (contingency-fundable).
Kill: σ_seed(γ)>0.024 ⇒ 1.103 uncertified ⇒ X1 re-scoped to mocks.

### P2 (wk 2–4; ≤24 A100-h) — the MC-SMC benchmark (central bet)
Arms: S1 MAMS-SMC (bet) · S2 MCLMC-SMC · S3 HMC-SMC (isolates the kernel question) · S4/S5
LAPS warm/cold (internal reference only — see bright lines) · S6a MCLMC-alone · S6b MAMS-alone-warm
(MAP cost billed) · **S7 minimal flow-preconditioned MAMS on carousel (~4 h; the minimal version of
their approved plan — per user decision, run fully; results to the team first)**.
Cells (each with their design-checkpoint format: hypothesis + predicted direction/magnitude +
falsifier + derived threshold, logged BEFORE runs):
- **B1 carousel33 (8 h)**: self-consistency (2 seeds, worst-param diff <0.27σ at ESS≥128); logZ
  repeatability ≤2 nats; efficiency WIN = ESS/10⁶ grads ≥2× MAMS-alone, PARITY [0.7,2), LOSS <0.7;
  **cold-start gate**: prior-seeded S1 width-ratio vs warm reference ∈[0.7,1.4] and |z|<3.
  Pre-registered honest prediction: parity-to-3×; falsifier: unique particles <N/4 at λ≳0.8
  (rotating-ridge geometry defeating affine per-λ preconditioning). conv=float64 (their ~0.3-nat
  float32 noise floor pollutes MH acceptance).
- **B2 dspl20_orig (2 h)**: prior-seeded S1 in the ORIGINAL pathological coords recovers their
  pre-registered arm mass |m̂−0.103|≤0.045; control dspl20_ratio reproduces Run A. Headline if it
  passes: the sampler fixes what previously required a bespoke exact reparameterization.
- **B3 hundred_systems-8 (phoenix, ≤2 h fallback)**: accuracy z<3 vs their MCLMC reference;
  two-sided cost prediction (SMC loses 1–5× evals-to-ESS, may win wall-clock via particle
  vectorization). The honest easy-target row.
- **B4 T2 foundry_marg46 (3 h)**: does per-λ ensemble preconditioning replace the two-stage
  recipe on cond-1e14? z<3 vs P2c reference; ESS≥0.3N; N=256 f64.
- **B5 T3 foundry_v3b74 (6 h)**: the multimodality certificate — both basins at λ=1; minor-basin
  occupancy within binomial of the P2c reference; basin ΔlogZ within 3σ(bootstrap⊕σ_seed); B5-G3
  does SMC resampling launder unadjusted-MCLMC bias (prediction: 1.5–3× minor-mode distortion).
Pre-written honest-risk statement: flow-MAMS plausibly beats S1 on raw ESS/eval on unimodal curved
targets; the deliverable is the **decision matrix** ({cold-start, evidence, multimodality} →
MC-SMC; {warm unimodal max-efficiency} → flow-MAMS) + the flagged synthesis (flow-preconditioned
mutation kernels inside SMC) as joint follow-up. Nobody has published any of these on lens
posteriors.
Kill: S1 fails B5 both-basins AND B1 diversity ⇒ bet dead ⇒ negative-result + HMC-SMC evidence
layer (still their missing logZ machinery).

### P3 (impl wk 2–3 ∥ P2; runs wk 4; ≤17 A100-h) — the likelihood payload on the scene API
- **L0 port + parity + ANCHOR ARBITRATION (~5 h)**: CorrelatedImageData port; gates F5 + old-vs-new
  whitened logp/grad ≤1e-8 at 64 draws; **two-stack anchor check** (red-team wildcard): refit v2d
  native-diagonal END-TO-END on the scene API — does 1.433 reproduce at posterior level? (falsify
  the premise before explaining it); **L0-G2**: v3b-low EPL refit on new substrate reproduces
  γ=1.103 within 2√(σ²_stat+σ²_seed) + low-basin logZ sign. L0-G2 licenses all X1 real-lens claims.
- **L1 full SBC (~6 h, budget-100 upgrade)**: 16 binned drizzle mocks, γ from prior, production
  SMC config @96 particles; rank z-scores + coverage (n=16 stated honestly).
- **L2 supersampling decomposition (~3+2 h)**: exploration finding — their adaptive supersampling
  bins FIRST and convolves with the native kernel ("cannot add back sub-pixel PSF structure") so
  it does NOT subsume our ss2 test; this becomes the experiment: arm (a) old-stack uniform ss2
  native-diagonal with ePSF resampled to 0.064125″ (the trap; parity re-run first) = quadrature +
  sub-pixel PSF; arm (c) scene-API adaptive-ss = quadrature only; (a)−(c) isolates the
  sub-pixel-PSF term; arm (b) binned-corr ss2 only if (a) moves the anchor ≥3σ_stat (≈0.10 in γ).
Stays on old stack: everything feeding certified numbers (P1, arms a/b, 1.103/1.433 provenance).
Migrates: source ladders (their caching + EllipticalShapelets), X1 (needs dPIE/BPL), arm (c),
multi-plane/multi-band, MC-SMC on their native targets.
Kill: L0 parity unfixable ⇒ payload stays old-stack; X1 falls back to porting BPL deflection into
the old array-only likelihood, or to mocks.

### P4 (wk 4–6; ≤10 A100-h; blocked on P1 + L0-G2) — X1: the profile-class fork (sleeper hit)
Pre-registered hypothesis: EPL single-power-law rigidity is the residual systematic — whitening
reweights spatial frequencies ⇒ products weight different effective radial ranges ⇒ a single-slope
fit to a curved κ(r) returns different local slopes per product (the observed 1.816/1.433/1.103
bracket).
- **X1-G0 entry gate (FREE, before any GPU)**: Fisher-weighted effective radius per product from
  on-disk posteriors; hypothesis requires an ordering that can produce the sign pattern under some
  monotone κ-curvature — no ordering ⇒ hypothesis structurally dead at zero cost. Plus MAP+Laplace
  2×2 {EPL,BPL}×{n_max 6,8} interaction screen (design-only; MAP γ verdicts inadmissible).
- Runs: per-basin 128-particle SMC (dogfood S1 if P2 certified it, else HMC-SMC) with checkpointed
  whitened likelihood: BPL binned-corr (2 basins ~3.5 h) + BPL native-diag (~2 h) + dPIE
  binned-corr (~2 h); EPL controls exist (L0-G2 + anchor).
- Gates: **X1-G1** ΔlogZ(BPL−EPL) fixed-whitener; decisive >3√2·σ_seed (~15 nats provisional,
  finalized from P1). **X1-G2 (THE test)**: cross-product spread of local log-slope γ_loc(θ_E) —
  the quantity lensing actually constrains, solving EPL-vs-BPL γ incomparability — shrinks to
  ≤0.11 (≥3× reduction of the 0.330 spread); falsifier ≥0.22 ⇒ profile hypothesis dead,
  source/PSF reinstated. **X1-G3**: r_break/r_core posteriors interior, not prior-railed (their
  physicality layer; railed ⇒ freedom is a nuisance sponge, claim demoted).

### Cross-pollination track (phoenix-only, ∥ everything, 0 A100-h)
- **X2 — first formal SBC of the GIGALens pipeline class** on hundred_systems (N≥64, L4/A16;
  physicality ON/OFF arms — their layer truncates prior mass, improper accounting shows in ranks).
  GLASS-HOUSE RULE: disclose and co-investigate our own open E1c γ-rank FAIL (p=6.5e-5) in the
  same harness. Deliverable = ranks-as-data + harness (their OT-1 distrusts invented numerics;
  SBC is Talts-et-al literature) — interpretation and verdict are THEIRS.
- **X3 — evidence-scored Vela ladder prototype**: 4 Vela systems × 5 source rungs, diagonal SMC →
  logZ; concordance gate vs their bias-ranking. Internal prototype + memo offer (never
  unilaterally applied to their staged 780-run campaign).
- **Fermat Δt teaser (<1 GPU-h)**: push the two existing DESI-165 posteriors through the
  differentiable deflection → fractional Δt shift between noise models (TDCOSMO has no
  coadd-covariance error-budget line item). One number, fail-safe.
- **Hessian-stage byproduct**: restore their stubbed HessianSurrogateStage from b82397c because WE
  need a scene-API Laplace stage for definiteness triage; offer as byproduct PR (1 day).

### Stretch (contingency pool ~20 h, priority order)
1. Flow-within-SMC synthesis (flow-preconditioned mutation kernels — merges their approved plan
   with our NEXT_DIRECTIONS item; ~6 h) — only if S7 + B1 both land cleanly.
2. PSF-marginalization MVP (T2.2, ~8–12-dim ePSF PCA basis) on the OLD stack (~8 h).
3. Second real system DESI J238.5690+04.7276 (GIGA-Lens 2.0's system) through the ported
   correlated likelihood (~6 h).

### P5 (wk 6–8; 0 h) — report + handoffs
`papers/main.tex` (publishable slices: microcanonical-SMC decision matrix; profile-class result
either way; calibration certificate; anchor arbitration). `papers/handoff/`: CLAIMS.md in their
claims-register format (every verdict `proposed (UNCERTIFIED external)`), SMCStage adapter spec,
CorrelatedImageData + ridge/Occam-lstsq fix (closes their owed foundry-i item), SBC harness
adapter, DSPL note, PR-ready diffs vs 80916d2. REPRODUCTIONS.md row + NEXT_DIRECTIONS delta.

## 7. Budget (cap 100 A100-h, new allocation)

| Phase | Cap | Cumulative commit |
|---|---|---|
| P0 scaffold+smoke | 1 | 1 |
| P1 certification slice | 10 | 11 |
| P2 MC-SMC benchmark (incl. carousel 8 + S7 4) | 24 | 35 |
| P3 payload (L0 5, L1 6, L2 5) | 17 | 52 |
| P4 profile fork | 10 | 62 |
| Stretch pool | 20 | 82 |
| **HARD STOP** | | **100** |

All Perlmutter jobs: single-GPU shared-QOS on cosmo_g; ledger-before-results; md5 deploy audit;
watchdog armed. Phoenix (A16/L4) carries B3, X2, X3, mock generation, flow training.

## 8. Bright lines (collision rules — verbatim into README)

1. **No published result readable as an MCLMC-vs-HMC efficiency comparison on a unimodal target**
   (their in-progress paper's territory). Every sampler number attaches to a multimodality,
   evidence, cold-start, or noise-model question.
2. **Nothing derived from gigalens-linus or GIGALens-Code** (code, numbers, characterizations of
   LAPS/carousel/Vela behavior) **appears in any external artifact without the group's sign-off.**
   Internal use + memo channel is fine (user decision). LAPS arms S4/S5 are internal reference only.
3. Vela staged campaign untouched; X3 is a prototype + offer.
4. Reserve "validated" for the certified old stack; scene-API results are "reproduced/measured,
   UNCERTIFIED (external)" in their vocabulary.
5. Out of scope: full joint PSF+pixelated-source+mass campaign; TD-lens refits; porting Voronoi
   sources; new sampler development beyond tempered-SMC variants; flt-level arbiter (parked).
6. Co-authorship offered on any paper whose results run through their substrate; papers A/B from
   the completed campaign remain independent.

## 9. Top risks

| Risk | Mitigation |
|---|---|
| Upstream branch drifts mid-campaign | pinned UNPATCHED vendor + re-pin procedure + gate re-run |
| Scene physics wrong for our config class (suite UNCERTIFIED) | F1–F6 vs validated 58ec9a7 stack; certified-config guard fence |
| param_map convention bug (flat-z ordering, PA/ellipticity) | constrained-space parity catches by construction; z_param_names-only keying; F7 |
| MCLMC-mutation evidence bias | MAMS primary (adjusted ⇒ invariant); MCLMC diagnostic-only behind bias gate |
| Correlated term double-render / SMC memory | single return_stacked render; jax.checkpoint + remat_basis; ≤200 MB/particle gate |
| jax 0.6.2 vs their 0.10 nightlies | F8 informational cell; require_jax_pin; scoped cgl2-nightly fallback as ledgered decision |
| Carousel cell reads as scooping their flow-MAMS plan | user-approved; S7 runs THEIR plan minimally; results to team first; sign-off gate on publication |
| 1.103/1.433 premise itself artifactual | L0 anchor arbitration runs EARLY (falsify the premise before explaining it) |

## 10. Verification (end-to-end)

- `00_run_tests.sh`: unit tests (param_map round-trips, correlated-term reduction, marg vs numpy,
  guards) + GPU parity → `data/parity_report_scene.json` all-pass.
- `04_smc_micro_validation.py`: B0 analytic-evidence + zoo-target gates vs HMC-SMC baseline.
- `03_correlated_term_validation.py`: delta-kernel identity + small-grid dense-Cholesky exact ref.
- Per-cell benchmark JSONs + λ-schedule/ESS traces; `figs/` before metrics (their rule).
- Every headline number → script + artifact + commit in CAMPAIGN.md; report builds via
  `make -C papers pdf`; REPRODUCTIONS.md row.

## 11. Critical files (reuse/reference)

- `../claude-giga-lens/cgl/samplers/bj_smc.py` — SMC skeleton for mutation-kernel generalization
- `../claude-giga-lens/cgl/{whiten,marg,noise}.py` — port sources (copy w/ attribution)
- `../claude-giga-lens/01_parity_harness.py` — gate-battery template
- `../claude-giga-lens/research/NEXT_DIRECTIONS.md` — Tier-0 prerequisites + critic warnings bind
- `/raid/benson/lensing-repos/gigalens-linus/src/gigalens/jax/{scene_prob_model,scene_simulator,scene}.py` — the three subclassed surfaces
- `/raid/benson/lensing-repos/gigalens-linus/src/gigalens/jax/experimental/{mams,mclmc,window_adaptation}.py` — kernel reference implementations (copy logic, never import)
- `/raid/benson/lensing-repos/GIGALens-Code/docs/logs/lab-notebook-TEMPLATE.md` — claims-register format for handoffs
- `/raid/benson/lensing-repos/GIGALens-Code/experiments/{hundred_systems_GL2,sim_carousel,sample_cosmology}/` — benchmark target builders
- foundry-i data products + `../claude-giga-lens/data/` whitener bundles — import by path
