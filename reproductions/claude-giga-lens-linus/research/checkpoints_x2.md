# X2 design checkpoints (Front D — SBC gift)

Per the 2026-07-15 multi-front mobilization entry in CAMPAIGN.md, this file holds the
X2 front's pre-run design checkpoints (same their-format content as a CAMPAIGN.md
checkpoint; folded into the ledger at harvest). Written BEFORE any GPU run.

---

## X2 DESIGN CHECKPOINT (pre-registered, written 2026-07-15 BEFORE the SBC runs)

### Hypothesis

The GIGALens pipeline class (multi-start MAP -> full-rank Gaussian SVI -> SVI-cov
preconditioned HMC, diagonal Gaussian plug-in likelihood with
err^2 = background_rms^2 + clip(observed,0)/exp_time) has never been formally
calibrated (no SBC exists in either of their repos; their hundred-systems metrics are
z-scores/percent-error against truth, not rank calibration). Hypothesis as assigned:
amplitude-bearing pipelines without a ridge/Occam accounting show mild under-coverage
on source-adjacent parameters. SCOPE CORRECTION (measured from their code before any
run, stated here rather than silently rewritten): the hundred-systems inference class
is FORWARD-mode — `epl_shear_sersic_sersic` builds every Sersic with
`use_lstsq=False`, so the amplitudes (lens-light Ie, source Ie) are SAMPLED, not
lstsq-marginalized; the assigned lstsq/ridge mechanism is therefore not present in
this config. The testable transcription of the hypothesis for this class: the
source-adjacent block (source Ie/R_sersic/n_sersic + lens-light Ie — the parameters
that trade against each other through the plug-in observed-image error map and the
Poisson-vs-Gaussian approximation) shows MILD miscalibration; the mass block
(theta_E, gamma, e1, e2, gamma1, gamma2) is calibrated.

### Prediction (direction/magnitude)

- Source-adjacent params: |rank-location z| ~ 1–3 (z = (mean(rank) - n_use/2) /
  sqrt(Var_unif/N), Var_unif = ((n_use+1)^2 - 1)/12, n_use = 127, N = 64).
- Mass params: |z| < 2 and rank-uniformity chi^2 p > 0.01 (8 bins — the E1c gate
  convention).
- No prediction on the SIGN per source param (the plug-in error map overweights
  bright pixels but the Gaussian tail underweights Poisson skew; direction is the
  measurement).

### Falsifier

- SEVERE miscalibration anywhere — |rank-location z| > 5 on any of the 22 params
  (primary readout, all fits) — falsifies "the pipeline class is mildly miscalibrated
  at worst" and (if concentrated in mass params) would flag the pipeline class as
  unsafe for the population studies it is being used for.
- The physicality-layer arm CANNOT shift ranks on this substrate — VERIFIED FROM CODE
  BEFORE THE RUN (see Arms). If a rank shift were observed between arms it would be a
  bug, not physics.

### Derived thresholds (frozen)

- Per-param rank-uniformity gate: chi^2 p > 0.01 (E1c convention: n_use=127, 8 bins,
  df=7). At N=64, expected 8 per bin.
- Rank-location gate: |z| < 3 calibrated; 3–5 = mild-miscalibration zone (report with
  sign, per param class); > 5 = severe (falsifier).
- Health (per fit, the hundred-systems metric convention): max split-Rhat over the
  22 z-dims <= 1.05 AND min ESS >= 200 (tfp potential_scale_reduction /
  effective_sample_size on the (draws, chains, dim) HMC array). PRIMARY readout =
  ALL N fits (the pipeline class as actually run — the E1c lesson is that dropping
  sick fits changes the verdict; both must be shown). SECONDARY readout = healthy-only
  (the E1c-amended convention), reported side-by-side with n stated.
- No coverage-interval gates beyond the rank gates (ranks ARE the deliverable;
  interpretation and verdict belong to the team — X2 charter).

### VALIDITY PRECONDITION (step 1 of the brief) — VERDICT: FROZEN SET INVALID FOR SBC; PIVOT ARM EXERCISED

Evidence (all read before this checkpoint, none of it a run):

1. `GIGALens-Code/experiments/hundred_systems_GL2/campaign.yaml`: dataset generator
   `gl2_existing` adapts the frozen `100SystemsStandard80px.npz` +
   `100SystemsStandardParams.yaml` in place ("no re-simulation"); inference builder
   `epl_shear_sersic_sersic`, pipeline `map_svi_hmc`.
2. `GIGALens-Code/src/gigalens_research/simtests/experiments/gl2_sersic.py` documents
   TWO priors: `gl2_simulation_prior()` — "the tighter draw distribution used in
   attic/Linus-FourSim.ipynb (the Simulation column of paper Eq. 8)" — vs
   `gl2_inference_prior()` == `make_default_prior()` (the Prior column; the fitting
   prior of hundredsystems.py and of the campaign builder).
3. `GIGALens-Code/experiments/why_hard_to_sample/t13_resim.py` (their own provenance
   reconstruction of the SAME frozen npz): "TRUTH params: cell 13 defines the
   GENERATION prior (NARROWER than today's MODELING prior)"; cell 14 samples the 100
   systems with PRNGKey(0). Parameter-by-parameter deltas documented there and in
   gl2_sersic.py: theta_E LogNormal sigma 0.25 vs 0.4; gamma TruncNormal sigma 0.25
   vs 0.5; EPL e1/e2 TruncatedNormal(0,0.2,±0.5) vs Normal(0,0.2) untruncated; centers
   0.03 vs 0.06; shear 0.05 vs 0.1; lens-light R 0.15 vs 0.25, n U(2,6) vs U(0.5,8),
   e 0.05 vs 0.1, centers 0.01 vs 0.02, Ie sigma 0.3 vs 0.5; source n U(0.5,4) vs
   U(0.5,8), centers 0.25 vs 0.5, Ie sigma 0.5 vs 0.9.
4. `cgl2/zoo.py::build_hs2` provenance: the frozen npz + params yaml are ABSENT from
   the local GIGALens-Code mirror (BLOCKED; they live on Perlmutter under linusu's
   home) — independently confirmed by `find` on the mirror this session.

Conclusion: truth-draws != fitting prior for the frozen 100-system set, so SBC ranks
on it would be non-uniform BY CONSTRUCTION (rank uniformity holds iff truth ~ fit
prior — the same rule the OLD campaign's e1.py docstring pre-registered, decision 1).
The frozen set is additionally physically unavailable on phoenix. ARM RUN =
PRIOR-MATCHED REGENERATION (the brief's pivot arm): our own mock set on the scene
API, same config class, truth drawn from the FITTING prior (make_default_prior
re-expressed verbatim as scene Components — `cgl2/zoo.py::_hs2_prior_components`,
already parity-audited against their setup.py at P0).

### Design (the run)

- N = 64 systems, indices i = 0..63. Per system (extending the build_hs2 recipe,
  replicated into 30_sbc_gift.py with attribution — zoo.py untouched, file-ownership
  rule): truth z_i = bijector.inverse(prior.sample(seed=PRNGKey(1000+i)));
  render = SceneSimulator.simulate (forward mode, delta_pix 0.065, num_pix 80, ss 2,
  vendored assets/psf.npy PSF); noise np.random.seed(3000+i) + lenstronomy
  add_poisson(exp_time=100) + add_background(sigma_bkd=0.2) — i = 0..7 coincide
  bit-for-bit with the certified hs2_sys{0..7} adapters (B0 logp-parity-gated).
  22 named params/system; z_param_names-keyed throughout (never order).
- Pipeline per system (vendored scene ModellingSequence on OUR port — no
  gigalens_research import): MAP (adabelief 1e-2 b1=0.95 b2=0.99 [+nesterov if
  supported], output best_step — their _default_map_optimizer) -> SVI full-rank
  MultivariateNormalTriL (adabelief 1e-4 b1=0.95 b2=0.99, init_scales 1e-3) -> HMC
  (PreconditionedHMC, SVI-cov momentum, GradientBasedTrajectoryLengthAdaptation +
  DualAveragingStepSizeAdaptation, init_eps 0.3, init_l 3, max_leapfrog 30). f64
  (SimulatorConfig likelihood_precision default = float64; GIGALENS_X64=1).
  Stage seeds: MAP seed=i, SVI seed=10000+i, HMC seed=20000+i (varied per system —
  SBC requires independent pipeline randomness across systems; their campaign held
  seed=0 fixed, a deviation we document rather than copy, since a fixed seed would
  correlate rank noise across systems).
- SETTINGS LADDER (pre-stated; decided by a timing smoke on sys0 BEFORE the batch,
  smoke discarded from the readout — its system index 0 is refit in the batch):
  * REFERENCE (their campaign.yaml): MAP 1000 steps x 2000 samples; SVI 5000 x 1000;
    HMC 64 chains x (500 burnin + 1500 results).
  * REDUCED (their stage DEFAULTS, the documented pipeline-class defaults): MAP 350 x
    500; SVI 1500 x 250; HMC 32 chains x (300 burnin + 750 results).
  * RULE: run REFERENCE if projected 64-system wall total <= 36 A16-h on 4 GPUs
    (16 systems/GPU sequential); else REDUCED if its projection <= 36 A16-h; else
    reduce N to the largest multiple of 8 fitting 36 A16-h at REDUCED settings and
    SAY SO. Projection = smoke wall x 64 x 1.15 (compile amortization margin).
- Rank statistic (copied with attribution from OLD cgl/e1.py: thin_indices, sbc_rank,
  rank_uniformity_chi2 — the E1c machinery, unchanged): per system & param, rank of
  z_truth among n_use=127 evenly-thinned pooled HMC draws (pooled over chains after
  the canonical (chains, draws, dim) reshape, C-order flatten draws-major). Ranks are
  computed in flat-z space: every hs2 param is a scalar prior with a strictly
  monotone event-space bijector, so per-coordinate ranks are IDENTICAL in constrained
  space (asserted for one system at runtime as a guard).
- Arms: the brief's {physicality ON, OFF} pair is VACUOUS on this substrate — read
  from the vendored code before the run: gigalens/physicality.py is a
  construction-time validator (raise on hard fixed-value violations, warn on prior
  mass outside domains) + a posterior DIAGNOSIS (`validate_posterior_samples`,
  "never raises"); it is invoked only from scene.py LensModel.__init__ (lines
  477–486) and NEVER enters ProbModel.log_prob/log_prior — there is no density
  truncation to toggle. Therefore ONE arm is run (the layer active at construction,
  as always), and the prior-truncation accounting is delivered as DATA instead:
  (a) construction-time physicality warnings captured per model; (b) their
  validate_posterior_samples fractions on the pooled posterior draws; (c) prior mass
  outside hard domains (their report). If the team's intended "physicality ON" means
  a future density-level truncation, this SBC is the OFF baseline for it.
- Compute: A16s 4–7 ONLY (CUDA_DEVICE_ORDER=PCI_BUS_ID; one process per GPU via
  CUDA_VISIBLE_DEVICES pinning, 16 systems each, sequential). Budget ~40 A16-h
  (commit 36). No Perlmutter, 0 A100-h.
- Glass-house arm (CPU, free): load OLD `data/e1_report.json` e1c block (n=44,
  gamma rank p=6.5e-5 FAIL, 16/44 first bin, cov68 0.34) + the E1c-AMENDED
  healthy-only n=13 row (OLD CAMPAIGN.md) and present BOTH calibrations side-by-side
  in the deliverable, same histogram format.
- Deliverables: research/x2_sbc_gift.md (their claims-register format; every verdict
  `proposed (UNCERTIFIED external)`; interpretation explicitly left to the team) +
  figs/x2_rank_hist_*.png (plots BEFORE metrics) + data/x2_sbc.json +
  per-system npz under data/x2_sbc_runs/ (gitignored data dir).

### Documented deviation (found at smoke, BEFORE any batch run; no numerics change)

Under jax 0.6.2 the vendored MAP/SVI `shard_map` wrappers fail in the backward pass
("cotangent type does not match function output ... {V:device}"): the varying-mode
replication CHECK (`check_vma`) rejects the FFT-convolution cotangent (complex128
rfft buffers) — the vendored shims target jax 0.10 pcast semantics. Runtime
monkeypatch in 30_sbc_gift.py::_patch_shard_map_check re-binds
`gigalens.jax.inference._shard_map` with `check_vma=False` (disables only the static
replication checker; computation/sharding/RNG/math unchanged; single-device mesh
here anyway). The vendor tree stays UNPATCHED (D1). Verified end-to-end on a
micro-run (MAP 20 steps -> chi2 8.06 decreasing; SVI + HMC complete; HMC output
canonicalized from the measured (draws, dev, chains, dim) layout).

### Deviation 2 (A16 memory constraint, found at smoke BEFORE the batch; ledgered here)

REDUCED-rung MAP (350 steps x 500 samples, f64) OOMs on the 15.3-GB A16: XLA asks
23.8 GB even with the MAP-scoped remat. The REFERENCE rung (2000 samples) is
therefore INFEASIBLE-ON-A16 outright (~4x more). Revised rung actually run,
"REDUCED-A16": map_samples 500 -> 128 and n_vi 250 -> 128 (largest power of two
projected to fit with XLA_PYTHON_CLIENT_MEM_FRACTION=0.90); step counts, optimizers,
HMC config UNCHANGED (32 chains x (300 burnin + 750 results)). SVI runs on the
remat twin (`pm.with_map_remat()` — mathematically exact recomputation, the same
mechanism MAP already self-applies). Rationale: MAP breadth / n_vi affect only
warm-start + preconditioner quality, never the stationary posterior the ranks test;
degraded warm starts show up in the health metrics (Rhat/ESS), which are reported
per system and gate the secondary readout. This is a hardware-forced deviation from
the frozen ladder, recorded before any batch system ran.

### Rule execution record (smoke readout, 2026-07-15 23:17 PT — BEFORE the batch)

Smoke sys0 at reduced_a16: wall 3507.6 s/system (map 244.6 + svi 1003.7 + hmc 2252.1;
HMC dominates on the A16's weak FP64), chisq_truth 0.988 (render+noise consistency),
MAP final chi2 1.019, monotone-rank guard PASSED (all 22 bijectors increasing;
z-space ranks == constrained-space ranks on 2048 thinned draws, 0 mismatches);
sys0 max_rhat 1.535 (health metric live, as designed). Projection at N=64 = 71.7
A16-h > 36 budget ⇒ the pre-stated rule selects **N=32 at reduced_a16**
(35.9 A16-h projected, largest multiple of 8 fitting the commit). This UNDER-DELIVERS
the assigned N>=64 — stated plainly; the n=32 caveat (bin expectation 4) will be
quoted with every chi^2; the harness + prior-matched regeneration recipe make the
N>=64 extension a turnkey re-run on faster-FP64 hardware (an L4/A100 pass is the
obvious follow-on). Decision artifact: data/x2_sbc_runs/settings_decision.json.

### READOUT (2026-07-16, observed vs predicted — checkpoint CLOSED)

Batch complete: 32/32 systems, 0 errors, 28.3 A16-h (batch) + 1.0 (smoke) actual.
Observed vs predicted: source block \|z\| <= 1.8 (prediction held at location
level); mass block \|z\| <= 2.25 BUT 6/8 fail chi2 uniformity via U-shaped
histograms (prediction failed at uniformity level); **FALSIFIER TRIPPED** —
severe one-sided miscalibration in the LENS-LIGHT photometric block (LL.Ie
z=-5.27 p=7.6e-11, LL.R_sersic z=+5.76 p=1.5e-10, LL.n_sersic z=+4.75).
Health: 2/32 fits pass Rhat/ESS (median max-Rhat 1.76) — under-mixing is a live
confound for the U-shapes (E1c precedent); the one-sided sign-coherent LL trio is
the part not naturally explained by mixing. Guard PASSED; chisq_truth median
1.001 on all 32. Full readout + doubt reports: research/x2_sbc_gift.md;
data/x2_sbc.json; figs inspected before gate math. Interpretation deferred to the
team per the X2 charter.

### Honesty pre-commitments

- Ranks-as-data: no verdict language beyond the pre-registered gate arithmetic;
  the team owns interpretation (X2 charter, PLAN cross-pollination track).
- Failed health on many systems is REPORTED, not silently dropped (E1c lesson);
  primary readout is all-fits.
- n=64 caveat quoted with every chi^2 p (bin expectation 8).
- Any deviation from this checkpoint (settings ladder rung actually used, N actually
  run, crashed systems) is ledgered in the deliverable verbatim.
