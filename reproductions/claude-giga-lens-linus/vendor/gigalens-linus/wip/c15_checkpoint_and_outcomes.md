# Design checkpoint draft (2026-07-09) — fused conv+pool (spectral fold) + scatter-free
# shapelet recurrence. To merge into GIGALens-Code docs/logs/compute-profiling.md when
# $HOME quota frees. User approved implement+validate+speed-test of C-14 targets 1-2
# in-session (2026-07-09).

Classification: two deterministic-identity claims + stochastic timing claims.
Code: pscratch clone $PSCRATCH/claude_perf/gigalens-fold, branch conv-pool-fold off
lstsq-fast @ bfaa59b (HOME at quota -> clone lives on scratch; origin push deferred).
Gates: wip/validate_fold_stack.py there. NOTE pscratch is purged ~180d — migrate
artifacts to home/CFS once space frees.

## Change A — fused PSF-conv + average-pool via spectral fold
Cause hypothesis: C-14 traces put the FFT-conv pipeline at ~42% of the vela-cell
gradient, half of it the LARGE inverse transform + overhead passes (scal, pre/post,
pad/crop) + the separate pooling pass. Since average-pool = box-conv + stride, fold
the box into the kernel spectrum and evaluate the inverse transform DIRECTLY on the
detector grid via the stride-ss spectral fold (B[f'] = (1/ss) sum_r A[f'+rM] phase);
the c2r shrinks by ss^2 and the pool/crop passes disappear.
- Equivalence prediction: f64 chi2/grad rel ~1e-13 (same linear map, fp
  reassociation); falsifier rel > 1e-10 (chi2) / grad L2 > 1e-10 => NOT shippable.
  Production conv f32: pooling moves INSIDE f32 (was f64 post-cast) => predict ~1e-6,
  falsifier chi2 rel > 2e-5 or any VJP NaN.
- Speed prediction: (200,ss4,30) grad -15..25%; falsifier < 5% => revert (the fused
  path is more complex than what it replaces; below 5% it is not worth owning).
- Blind spots: (i) fold gather creates a (My,Mx/2+1,ss,ss) complex intermediate ~ the
  half-spectrum size — could offset wins at ss=2; (ii) Hermitian-gather correctness at
  Nyquist/DC bins — covered by the exact f64 gate + a CPU brute-force unit test over
  odd/even kernel sizes and ss in {2,4} BEFORE the GPU run; (iii) fused path only
  active when kernel is present AND ss>1 (else legacy path).

## Change B — scatter-free shapelet recurrence (stack instead of .at[n].set)
Cause hypothesis: C-14 traces attribute 13-18% (two input_scatter_fusion kernels,
scaling as depth and ss^2) to the recurrence buffer writes + VJP. A Python-list +
jnp.stack lowers to concatenate (+ slice VJP) instead of scatter.
- Equivalence prediction: identical values (same ops, same per-element order);
  falsifier chi2 rel > 1e-12 or grad L2 > 1e-12 => NOT shippable.
- Speed prediction: (200,ss4,30) grad -5..15%; falsifier < 2% => revert (keep the
  buffer form; would mean the scatter attribution was wrong or stack lowers to the
  same copies — record either way).

## Combined
- EC sanity: new-vs-old grad L2 <= 1e-8 at f64.
- Speed matrix: 2x2 {fold,unfold}x{stack,buffer} at (200,ss4,30); combined new-vs-old
  at (200,ss2,15) and (200,ss2,30). Median-of-30 + min, contention rule as before.
- Cost: ~25 GPU-min login A100. Timing at production conv f32.
- Mandatory companion: pytest tests/validation on the final tree (golden anchor;
  known pre-existing LOG_PRIOR failure C-13 expected — any NEW failure blocks).

## Amendments (pre-launch, at grading — rigor-grader NEEDS-MORE, all applied)
1. Brute-force test committed as wip/test_fold_bruteforce.py with its 2e-15
   output recorded (was heredoc-only — record integrity).
2. Multi-kernel branches of _convolve_pool_components REMOVED and replaced by a
   loud NotImplementedError (unreachable from SceneSimulator, no gate covered
   them, first draft had a latent shape divergence in the nk>1 simulate-shaped
   case). Single-shared-PSF only; depths arg is a no-op under one kernel, as in
   the legacy case-1 branch. _next_fast_len_multiple guards non-7-smooth ss.
3. Pre-registered anchor reading: the frozen golden was produced on the UNFUSED
   pipeline and tolerances.py §6b's "SAME PSF pipeline" premise is now stale;
   predicted fused-vs-golden diff ~1e-13 << ANCHOR_IMAGE_RTOL 1e-10 => pass.
   Anchor failure = blocking NEW failure. On ship, update the tolerances.py
   comment.
4. Marginal-gain definitions: fold gain = (unfold+stack − fold+stack)/
   (unfold+stack); stack gain = (fold+buffer − fold+stack)/(fold+buffer); both
   at (200,ss4,30), production f32, median-of-30.
5. Magnitude-miss readings: EA32 in (5e-6, 2e-5] or EB in (1e-14, 1e-12] =
   gate PASSES but the magnitude prediction MISSED => record a mechanism note
   before ship (not a silent pass).
6. Blind spot added: the fold gather's VJP introduces a scatter-add into the
   backward — a timing risk (speed gate catches it), pre-registered here.
## OUTCOMES (2026-07-09, run 2 after amendment 7; login A100, jax 0.10.0.dev20260709)
- Equivalence: ALL PASS. (200,ss2,15): EA 2.0e-15 chi2 / 7.4e-15 gradL2; EB 1.3e-16 /
  2.0e-15; EC 7.4e-15; EA32 8.4e-07. (120,ss4,30): EA 6.7e-15 / 1.25e-12; EB 5.9e-15 /
  3.5e-13; EC 7.0e-13; EA32 1.1e-06. No NaNs. Magnitude-miss reading (amendment 5)
  FIRES on EB at the ss4 cell (3.5e-13 in (1e-14,1e-12]): values of phi are identical
  but the concat-vs-scatter lowering changes downstream FUSION BOUNDARIES, so gradient
  accumulation order reassociates at fp level — mechanism note recorded, gate passed.
  EA32 1.1e-6 < 5e-6: no miss.
- Speed at (200,ss4,30), production f32, marginals per amendment 4:
  fold gain (unfold+stack -> fold+stack): (103.02-94.05)/103.02 = 8.7% — falsifier
  (<5%) NOT fired, but the 15-25% prediction MISSED LOW: mechanism = the spectral-fold
  gather+phase pass and its VJP scatter-add (blind spot 6) eat part of the c2r saving.
  stack gain (fold+buffer -> fold+stack): (99.69-94.05)/99.69 = 5.7% — in the
  predicted 5-15% band. Combined new-vs-old at the vela cell: 108.42 -> 94.05 ms
  (-13.3%); peak 7016 -> 6818 MB.
- ss=2 cells (combined new-vs-old): time ~0..-3% (min-based; anchor MEDIAN contention
  48% — pre-registered rule: medians untrusted, mins used) but peak memory REGRESSED
  ~20% (532->652, 1677->2023 MB): the fold gather intermediate outweighs the
  small-stride savings. DEPLOYMENT RULE (evidence-based, both configs measured and
  exact): fuse only for supersample >= 3 (_FUSE_CONV_POOL_MIN_SS); stack ships
  everywhere (it also cut vela peak 7125->6818 MB). Final-config re-verification of
  the ss2 cells + pytest recorded below.
- pytest tests/validation on the fused tree: 60 passed, only the pre-existing C-13
  LOG_PRIOR failure; the frozen golden anchor (fused-vs-golden) PASSED as
  pre-registered in amendment 3. Re-run on the FINAL tree (ss>=3 threshold):
  identical result (60 passed + C-13 only).
- FINAL-CONFIG timings (fold auto-off at ss2, stack everywhere; min-based where
  contended): (200,ss2,15) 8.26 ms/535 MB vs old 8.24(min)/532 MB — time parity,
  memory parity (the +23% fold regression is GONE); (200,ss2,30) 27.19/1598 vs
  28.58/1677 — -5% time, -5% memory (stack alone); (200,ss4,30) 93.93/6818 vs
  108.42/7016 — -13.3% time, -3% memory (fold+stack). Cumulative vs the
  pre-rfft2 baseline at the vela cell: 148.0 -> 93.9 ms = -36.6%.

7. (post first launch, pre results) The f64 EA reference at (200,ss4,30) OOMs
   the 40GB A100 (complex128 spectra of the depth-496 stack in the UNFUSED
   reference path); the f64 identity cells move to (120,ss4,30) — same ss=4
   fold, same depth-496 conditioning, size-independent identity. (200,ss4,30)
   remains covered at production f32 by the speed cells and by pytest goldens
   (which PASSED on the fused tree before this amendment: 60 passed, only the
   pre-existing C-13 failure). Anchor-cell equivalence had already PASSED
   (EA 1.6e-15, EB 2.4e-16, EC 7.7e-15, EA32 8.4e-7, no NaN) before the crash;
   this amendment changes feasibility, not thresholds.
