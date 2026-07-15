"""Derived numerical tolerances for the validation suite.

Method-discipline §3: *a threshold you cannot derive is not a test.* Every constant
here carries the reasoning for why **that** number, in **those** units, is the line
between "numerical noise" and "a bug." If you change one, change its derivation too.

We deliberately do NOT default to the existing ``tests/test_profiles.py`` float32
band (rtol=1e-5/atol=1e-4). The framework is float64-first, so analytic parity is
held near machine precision; the float32 band is used only for explicit float32-mode
tests where it is the genuine noise floor.
"""

# --- Analytic profile parity (mass deriv/hessian/convergence, light) -------------
# float64 machine eps ~2.2e-16. Mass/light profiles are O(10)-operation closed forms;
# lenstronomy evaluates the *same* closed forms in float64 but with different
# operation order/branching. Near profile centers and critical curves, cancellation
# costs a few digits. Holding ~1e-11 relative (with a small absolute floor for
# values near zero) is ~5 orders above eps — comfortably "numerics," while still
# catching any genuine formula/normalization disagreement.
PROFILE_F64_RTOL = 1e-11
PROFILE_F64_ATOL = 1e-12

# float32 eps ~1.2e-7; accumulation over the same closed forms ~1e-5 relative. This
# reproduces the existing suite's band and is used ONLY for float32-mode parity.
PROFILE_F32_RTOL = 1e-5
PROFILE_F32_ATOL = 1e-4

# --- Cosmology distances ----------------------------------------------------------
# gigalens integrates 1/E(z) with an n_grid-point trapezoid (default n_grid=1000).
# Trapezoid error is O((Δz)^2) = O(1/n_grid^2) on a smooth integrand; for n_grid=1000
# over z~O(1) that is ~1e-6..1e-5 relative. astropy uses (effectively exact) adaptive
# quadrature. So a >1e-4 relative disagreement is a model/units bug, not integrator
# error. If n_grid is raised, tighten this and re-derive.
COSMO_DISTANCE_RTOL = 1e-4

# Pure integrator check (gigalens distance vs astropy on the SAME cosmology): this is
# the link that isolates "is the trapezoid fine enough," separate from "is the model
# right." Held at the trapezoid's own predicted error, not the looser bug-band above.
COSMO_INTEGRATOR_RTOL = 5e-5

# --- Multiplane ray-shooting ------------------------------------------------------
# Source-plane positions from the recursion vs lenstronomy ray_shooting. Both are
# float64 compositions of analytic deflections + distance ratios; the distances come
# from different integrators (gigalens trapezoid vs astropy), so this inherits the
# cosmo band rather than the tighter profile band.
RAY_SHOOTING_RTOL = 2e-4
RAY_SHOOTING_ATOL = 1e-9   # arcsec, for positions passing through ~0

# --- lstsq amplitude recovery -----------------------------------------------------
# Planting amplitudes by a forward render and recovering them through the normal
# equations is exact in exact arithmetic, but the recovery passes through a Gram
# matrix solve (Xᵀ X) whose condition number for a couple of well-separated sersic
# bases is ~1e2-1e4 in float64. So the recovered coefficient inherits ~cond·eps ≈
# 1e4·2e-16 ≈ 1e-12 relative from the solve, plus the supersample-pool/conv float64
# accumulation; 1e-6 relative is many orders above that floor while still catching any
# design-matrix/weighting bug (which would be O(1)). Matches the Phase-3 round-trip band.
LSTSQ_AMPLITUDE_RTOL = 1e-6

# --- Reductions / metamorphic (no oracle, pure self-consistency) ------------------
# N=1 multiplane vs single-plane, deflection_ratio vs multiplane, translation/rotation
# equivariance: these are exact identities in exact arithmetic, so they are held at
# the float64 composition floor, NOT a physical band.
REDUCTION_RTOL = 1e-10
REDUCTION_ATOL = 1e-11
LINEARITY_RTOL = 1e-12     # amplitude linearity is exact up to rounding

# --- Full image (PSF subgrid + pooling) -------------------------------------------
# The dominant discrepancy vs an independent image is lenstronomy's iterative
# subgrid_kernel refinement + conv edge handling, NOT the physics. So the end-to-end
# image is held loosely and is a backstop only: prefer the pre-convolution image
# (IMAGE_NOPSF_*) and the isolated layers above. Absolute floor is a fraction of the
# image peak so faint pixels are not judged by a meaningless relative error.
IMAGE_RTOL = 1e-3
IMAGE_ATOL_PEAK_FRAC = 1e-4

# Pre-convolution model image (no PSF, supersample=1) vs a hand-built lenstronomy
# render: pure profile+deflection composition → profile-level float64 parity.
IMAGE_NOPSF_F64_RTOL = 1e-9
IMAGE_NOPSF_F64_ATOL = 1e-10

# --- §6b refactor regression anchor (NEW scene path vs OLD prob_model/simulator) --
# This is a CODE-PATH-equivalence claim, not a physics claim: the new SceneSimulator
# + scene ProbModel must reproduce the OLD LensSimulator + BackwardProbModel on the
# SAME physical model (EPL+Shear mass, Sersic lens light, Shapelets-lstsq source, real
# PSF, supersample=2, float64). Both paths call the SAME underlying ops — identical
# profile evaluators (EPL.deriv/Shear.deriv/SersicEllipse.light/Shapelets.light), the
# SAME PSF pipeline (subgrid_kernel(num_iter=100) → _convolve_components → average_pool_2d),
# and the SAME lstsq solver (_solve_normal_eq_with_fallback / _regularize_gram), reused
# verbatim by scene_simulator.py from simulator.py. The render loops also visit light in
# the SAME order (lens light first, then source), so the float64 accumulation order is
# the same too. Hence equivalence is expected at the float64 *reduction* floor, not a
# physical band:
#
#   image:  float64 eps ~2.2e-16; a ~40k-supersampled-pixel render with O(10) ops/pixel
#           accumulates to ~1e-13 relative at worst. Held at ANCHOR_IMAGE_RTOL=1e-10
#           (~5 orders above the floor) with a small absolute floor scaled to the image
#           peak so faint ring pixels are not judged by a meaningless relative error.
#   coeffs: pass through the Gram solve (cond ~1e2-1e6 for a Sersic + n_max=6 shapelet
#           basis in float64), inheriting ~cond·eps ≲ 1e-10 relative. Both paths build the
#           SAME design matrix, so any real wiring divergence (component order, depth
#           offset, weighting) would be O(1), not ~1e-10. Held at ANCHOR_COEFF_RTOL=1e-8.
#   log_like / reduced chi2: a sum over masked pixels of ((im-obs)/err)^2; the residual
#           (im-obs) is itself ~1e-10·peak, so the chi2 difference is dominated by that,
#           giving a relative log_like agreement ≲ 1e-9. Held at ANCHOR_LOGLIKE_RTOL=1e-8.
#
# Falsifier (all three): an O(1) or spatially/structurally localized residual (ring
# region, one component's coefficient, a per-pixel dipole) ⇒ the new path is NOT
# code-equivalent to the old one — a deflection-sign, render-order, basis-depth, or
# noise-weighting bug — and would equally flag a real physics divergence. This is NOT a
# hand-picked round number: it is the float64 floor of two paths sharing their ops, set
# a few orders above eps to absorb begin/JIT/XLA reassociation without masking a bug.
ANCHOR_IMAGE_RTOL = 1e-10
ANCHOR_IMAGE_ATOL_PEAK_FRAC = 1e-11
ANCHOR_COEFF_RTOL = 1e-8
ANCHOR_COEFF_ATOL = 1e-12
ANCHOR_LOGLIKE_RTOL = 1e-8

# --- §6b prob-model FREE-param companion (closes the 6a tautology gap) ------------
# The 6a anchor above fixes ALL params, so neither ProbModel's bijector nor its prior
# runs — the log_like leg is partly tautological (a numpy χ² recomputed on already-equal
# images). This companion builds the SAME scene with FREE params (tfd priors) so the
# unconstraining+pack bijector AND the prior actually evaluate, and compares the NEW
# scene ProbModel(mode="lstsq").log_like/log_prob against the OLD BackwardProbModel.
# Bijector *orderings* differ between the two APIs (NEW: flat unique-keys; OLD: nested
# lens_mass/lens_light/source_light JointDistributionNamed), so the same raw z maps to
# different params; the test instead matches at CONSTRAINED physical params (draw z_new,
# forward to physical dict, invert through the OLD bijector to z_old).
#
# Derivation of the floor (float64):
#   log_like: both paths fit the SAME lstsq image and reduce a masked Gaussian — OLD via
#     tfd.Independent(Normal).log_prob in float64 (_independent_normal_log_prob_f64), NEW
#     via -0.5·Σ((r/σ)² + log 2πσ²). These are algebraically identical; on a ~3600-pixel
#     sum of magnitude ~1e6 the float64 reduction floor is ~cancellation·eps ~ 1e-12
#     relative. Held at 1e-10 (a couple orders above) to absorb XLA reassociation.
#   log_prior: identical 1-D marginal priors evaluated on identical scalars + the
#     default-event-space Jacobian, which is per-coordinate (ordering-invariant). Floor
#     is ~machine eps; held at 1e-10.
# Falsifier: a ≳1e-8 relative gap in either ⇒ a real semantic difference (different
# normalization, a dropped Jacobian term, or one path silently degrading to float32) —
# a FINDING, not a tolerance to loosen. Observed in the 6b run: log_like rel ≲ 1e-12,
# log_prior rel ≲ 1e-15.
ANCHOR_PROB_LOGLIKE_RTOL = 1e-10
ANCHOR_PROB_LOGPRIOR_RTOL = 1e-10
