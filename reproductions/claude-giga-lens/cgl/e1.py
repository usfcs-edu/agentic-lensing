"""E1 mock-experiment fit driver (P1b): 22-dim-class fits on the 05 drizzle mocks.

This is the P1b fit library (NOT cgl.fitting — that belongs to P2). It builds
diagonal / correlated (conv-whitened) likelihoods for the mock trio products
and runs the reduced-budget MAP -> SVI -> PHMC+ChEES pipeline.

Design decisions (documented for the campaign report; each is load-bearing):

1. FIT PRIOR == TRUTH-SAMPLING DISTRIBUTION. The fit prior reproduces
   ``cgl.mocks.sample_truth`` (the gu-2022 Eq.(8) SIMULATION distribution)
   exactly, per parameter. This is REQUIRED for E1c SBC validity (rank
   uniformity holds iff truth ~ fit prior); it also makes E1b bias/coverage a
   clean self-consistency test. gu-2022's 02_fit_system used a deliberately
   broader modelling prior — using that here would fail SBC by construction,
   not by likelihood defect. Deviation flagged in the P1b report.

2. FORWARD MODEL = THE GENERATOR'S OWN OPERATORS. Every fit renders the
   214^2 @ 0.04" scene exactly like 05_gen_drizzle_mocks (same SimulatorConfig)
   and applies the mock pipeline's exact linear maps:
     fine   : kernel=psf_eff, crop [2:210)  -> 208^2   (render gate: 2.1e-12 sigma)
     binned : fine model -> 2x2 block-sum   -> 104^2   (generator identity)
     native : kernel=psf, per frame crop [oy:oy+210) -> 3x3 block-sum -> 70^2
   so there is NO simulator-mismatch floor (the gu-2022 cross-simulator lesson).

3. MOCK "NATIVE PRODUCT" = the NINE native exposures fit JOINTLY (iid within
   and across frames; the corr fit runs the delta-kernel limit, per
   CAMPAIGN.md). Rationale: the E1b width-ratio gate sigma_fine/sigma_native
   in [0.7, 1.5] presumes same-photon products (the real native/fine/binned
   products stack the same exposures); a single-exposure "native" carries 1/9
   of the photons and fails that gate by design, not by physics.

4. DTYPE POLICY ("mixed", the default): process runs with GIGALENS_X64=1 but
   the prior/bijector/render/whitening pipeline is built in float32 (gu-2022
   f32 precedent for the 22-dim mock class); the whitened-residual reductions
   and the ridge-marg Cholesky block are computed in float64 (the part the 04
   exact-reference gate certified at f64). 'f64' mode (everything f64) is kept
   for pilot cross-validation; 'f32' (x64 off) matches gu-2022 exactly.

5. KERNELS FOR CORR FITS follow the real-data two-pass recipe (02 pattern):
   quick diag MAP -> model-subtract -> detrend_sky -> cgl.noise.masked_acf_2d
   on sky pixels -> WLS fit of the drizzle-anchored family -> whitener via
   cgl.whiten.build_whitener with the P1a-tuned ADAPTIVE s_floor policy
   (03_build_whiteners), M searched upward to e_op <= 0.02 (or a caller
   target, e.g. 0.05 for the E1d relaxed arm).

Import contract: jax / gigalens / tfp are imported lazily inside the GPU
functions so the numpy-only helpers (stats, manifest) stay CPU-importable for
unit tests.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import types
from pathlib import Path

import numpy as np
from scipy import stats as sstats

from cgl import guards
from cgl.mocks import (
    CROP0, FINE_PIX, FINE_RENDER, N_FINE, NATIVE_PIX, OFFSETS_FINE, R_INT,
    DrizzleMockPipeline,
)
from cgl.noise import (
    binned_kernel_from_fine, detrend_sky, fit_kernel, fit_kernel2,
    masked_acf_2d,
)
from cgl.paths import DATA, REPRO

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
E1_FITS = DATA / "e1_fits"
E1_KERNELS = DATA / "e1_kernels"
MOCKS_DIR = DATA / "mocks"

MASS_LABELS = ["theta_E", "gamma", "e1", "e2", "gamma1", "gamma2"]
N_NATIVE = 70
SKY_R_ARC = 3.0          # mock sky annulus inner radius (arcsec). DEVIATION
# from the real-data ARC_ANNULUS=(1.2,4.5): the mock product half-width is
# 4.16", so r>4.5" is empty; mock arcs live inside ~2.7" (theta_E<=2.2 + src
# offset), so r>3.0" is arc-free by construction.
CORE_MASK_ARC = 0.20
E_OP_GATE = 0.02          # pre-registered whitener gate (strict)
WHITEN_GRID = 512
M_GRID = {"fine": (14, 16, 18, 20), "binned": (2, 3, 4, 6, 8, 10),
          "native": (0, 1, 2, 3, 4)}
REG_LAMBDA = 0.10         # delta-regularization for NEAR-SINGULAR kernels.
# The mock fine covariance is genuinely singular (pure 9-phase drizzle stack,
# no iid component: 1-D tent spectrum has exact zeros at omega=2pi/3), so the
# spectral-FLOOR construction leaves a kinked target that truncated taps
# cannot approximate (measured: e_op stalls at ~0.2 for M<=14 at s_floor=0.05).
# Instead the whitener is built for the DECLARED noise model
#   K_reg = (K + lambda*delta) / (1 + lambda),  lambda = 0.10
# (kink-free spectrum; measured e_op: M=18 -> 0.0235, M=20 -> 0.0174 <= 0.02).
# Conservative by construction: whitened-noise variance under the true K is
# S/(S+lambda)*(1+lambda) * (1/(1+lambda)) <= 1 in every direction, so the
# likelihood never overstates information; E1b/E1c coverage+SBC certify the
# net calibration empirically. Engaged only when S_min/S_mean < REG_ENGAGE.
REG_ENGAGE = 0.02
BKG_BOX = {"fine": 20, "binned": 10, "native": 10}   # detrend box (02-scaled)
ACF_WINDOW = {"fine": 4, "binned": 3, "native": 2}   # fit-window half-width L
SHAPELET_NMAX = 4
SHAPELET_SIGMA0 = 50.0    # mock shapelet-amp prior scale (amps in gigalens
# render units; Lambda_ii=(i+1)/sigma0^2 in the fit — generator and fit share
# this EXACTLY, the SBC self-consistency requirement)

_E1D_CROP = 5             # real v2d 80^2 keep_mask -> central 70^2 port


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPRO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# pure-numpy statistics helpers (unit-tested in tests/test_e1.py)
# --------------------------------------------------------------------------- #
def z_scores(post_mean, post_std, truth):
    """(mean - truth)/std elementwise; std<=0 -> nan."""
    post_mean = np.asarray(post_mean, dtype=np.float64)
    post_std = np.asarray(post_std, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return (post_mean - truth) / np.where(post_std > 0, post_std, np.nan)


def coverage_flag(draws, truth, level):
    """1 if truth inside the central `level` credible interval of draws."""
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(np.asarray(draws, dtype=np.float64), [a, 1.0 - a])
    return bool(lo <= truth <= hi)


def thin_indices(n_total, n_use):
    """Deterministic even-stride thinning indices (n_use of n_total)."""
    if n_use >= n_total:
        return np.arange(n_total)
    idx = np.floor((np.arange(n_use) + 0.5) * n_total / n_use).astype(int)
    return np.clip(idx, 0, n_total - 1)


def sbc_rank(draws, truth, n_use=127):
    """Standard SBC rank: #(thinned draws < truth), in 0..n_use."""
    d = np.asarray(draws, dtype=np.float64).reshape(-1)
    dt = d[thin_indices(d.size, n_use)]
    return int(np.sum(dt < truth))


def rank_uniformity_chi2(ranks, n_use=127, n_bins=8):
    """Chi^2 uniformity test on SBC ranks (0..n_use -> n_bins equal bins)."""
    ranks = np.asarray(ranks, dtype=int)
    n_vals = n_use + 1
    assert n_vals % n_bins == 0, "bins must divide the rank support"
    edges = np.arange(n_bins + 1) * (n_vals // n_bins)
    obs, _ = np.histogram(ranks, bins=edges)
    exp = ranks.size / n_bins
    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    p = float(sstats.chi2.sf(chi2, df=n_bins - 1))
    return chi2, p, obs.tolist()


def recalibrate_err_40b(img, model, err, sky_mask):
    """40b-style sky-chi2=1 recalibration, mock-adapted (E1a legacy err map).

    The real 40b recipe rescales the err map so per-pixel sky chi^2 = 1 on
    source-free raw pixels. Mocks have NO source-free sky (Sersic n<=6 lens
    wings cover the frame), so the faithful mock analog uses TRUE-model-
    subtracted residuals over the r>3" annulus — same endpoint (marginal
    per-pixel chi^2_pp == 1, correlations ignored) without re-importing the
    wing-contamination incident foundry-i already documented.

    Returns (err_recal, rescale_factor, chi2_sky_before).
    """
    resid = np.asarray(img, dtype=np.float64) - np.asarray(model, np.float64)
    err = np.asarray(err, dtype=np.float64)
    sky = np.asarray(sky_mask, dtype=bool)
    chi2_before = float(np.mean((resid[sky] / err[sky]) ** 2))
    s = float(np.sqrt(max(chi2_before, 1e-12)))
    return err * s, s, chi2_before


# --------------------------------------------------------------------------- #
# mock-product geometry
# --------------------------------------------------------------------------- #
def fine_center_px():
    return (FINE_RENDER - 1) / 2.0 - CROP0            # 104.5


def r_arc_map(scale, frame=None):
    """Radial map (arcsec from the lens center) for a product plane."""
    if scale == "fine":
        c = fine_center_px()
        yy, xx = np.indices((N_FINE, N_FINE))
        return np.hypot(yy - c, xx - c) * FINE_PIX
    if scale == "binned":
        c = (fine_center_px() - 0.5) / 2.0            # 52.0
        yy, xx = np.indices((N_FINE // 2, N_FINE // 2))
        return np.hypot(yy - c, xx - c) * 2 * FINE_PIX
    if scale == "native":
        oy, ox = frame
        # native pixel k center sits at global fine coordinate o + 3k + 1;
        # the lens center is at global fine coordinate (FINE_RENDER-1)/2.
        c_glob = (FINE_RENDER - 1) / 2.0
        cy = (c_glob - 1.0 - oy) / 3.0
        cx = (c_glob - 1.0 - ox) / 3.0
        yy, xx = np.indices((N_NATIVE, N_NATIVE))
        return np.hypot(yy - cy, xx - cx) * NATIVE_PIX
    raise ValueError(scale)


def load_mock(path):
    """Load a 05 mock npz into a plain dict (json meta parsed)."""
    z = np.load(path, allow_pickle=False)
    d = {k: np.asarray(z[k]) for k in z.files}
    d["meta"] = json.loads(str(z["meta"]))
    d["truth_nested"] = json.loads(str(z["truth_json"]))
    d["flat"] = dict(zip([str(k) for k in z["flat_keys"]],
                         np.asarray(z["flat_vals"], dtype=np.float64)))
    d["path"] = str(path)
    return d


def mock_planes(mock, scale):
    """Per-plane (img, err, keep, r_arc, frame_offset) arrays for a scale.

    E1d mocks (meta['e1d']=True) are single-native-frame products with the
    ported real-v2d keep mask; trio mocks follow the 05 layout.
    """
    if mock["meta"].get("e1d", False):
        return [dict(img=mock["img"], err=mock["err_map"],
                     keep=mock["keep_mask"].astype(bool),
                     r_arc=r_arc_map("native", frame=(2, 2)), frame=(2, 2))]
    if scale == "fine":
        return [dict(img=mock["img"], err=mock["err_map"],
                     keep=mock["keep_mask"].astype(bool),
                     r_arc=r_arc_map("fine"), frame=None)]
    if scale == "binned":
        return [dict(img=mock["binned_img"], err=mock["binned_err"],
                     keep=mock["binned_keep"].astype(bool),
                     r_arc=r_arc_map("binned"), frame=None)]
    if scale == "native":
        planes = []
        for f, (oy, ox) in enumerate(OFFSETS_FINE):
            r = r_arc_map("native", frame=(oy, ox))
            planes.append(dict(img=mock["native_img"][f],
                               err=mock["native_err"][f],
                               keep=r > CORE_MASK_ARC, r_arc=r,
                               frame=(oy, ox)))
        return planes
    raise ValueError(scale)


# --------------------------------------------------------------------------- #
# fit prior == truth-sampling distribution (see module docstring, decision 1)
# --------------------------------------------------------------------------- #
def build_truth_prior(dt=np.float32, shapelets=False):
    """tfp prior matching cgl.mocks.sample_truth exactly (+ shapelet beta).

    Returns (prior, bij, ndim). Lazy tfp import (GPU/CPU-agnostic)."""
    import tensorflow_probability.substrates.jax as tfp

    tfd = tfp.distributions
    tfb = tfp.bijectors

    def LN(med, sig):
        return tfd.LogNormal(dt(np.log(med)), dt(sig))

    def N(mu, sig):
        return tfd.Normal(dt(mu), dt(sig))

    def TN(mu, sig, lo, hi):
        return tfd.TruncatedNormal(dt(mu), dt(sig), dt(lo), dt(hi))

    def U(lo, hi):
        return tfd.Uniform(dt(lo), dt(hi))

    lens_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            theta_E=LN(1.25, 0.25), gamma=TN(2.0, 0.25, 1.0, 3.0),
            e1=N(0.0, 0.1), e2=N(0.0, 0.1),
            center_x=N(0.0, 0.05), center_y=N(0.0, 0.05))),
        tfd.JointDistributionNamed(dict(gamma1=N(0.0, 0.03),
                                        gamma2=N(0.0, 0.03))),
    ])
    lens_light_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            R_sersic=LN(1.6, 0.15), n_sersic=U(2.0, 6.0),
            e1=TN(0.0, 0.05, -0.15, 0.15), e2=TN(0.0, 0.05, -0.15, 0.15),
            center_x=N(0.0, 0.01), center_y=N(0.0, 0.01),
            Ie=LN(300.0, 0.3))),
    ])
    src_blocks = [tfd.JointDistributionNamed(dict(
        R_sersic=LN(0.25, 0.15), n_sersic=U(0.5, 4.0),
        e1=TN(0.0, 0.15, -0.5, 0.5), e2=TN(0.0, 0.15, -0.5, 0.5),
        center_x=N(0.0, 0.25), center_y=N(0.0, 0.25),
        Ie=LN(150.0, 0.5)))]
    if shapelets:
        src_blocks.append(tfd.JointDistributionNamed(dict(
            beta=LN(0.15, 0.15))))
    source_light_prior = tfd.JointDistributionSequential(src_blocks)
    prior = tfd.JointDistributionSequential(
        [lens_prior, lens_light_prior, source_light_prior])

    import jax

    example = prior.sample(seed=jax.random.PRNGKey(0))
    pack_bij = tfb.pack_sequence_as(example)
    bij = tfb.Chain([prior.experimental_default_event_space_bijector(),
                     pack_bij])
    ndim = int(sum(np.size(np.asarray(v))
                   for v in jax.tree_util.tree_leaves(example)))
    return prior, bij, ndim


def sample_shapelet_truth(rng):
    """Shapelet-source extras for the E1c marg arm: beta ~ LogNormal(log 0.15,
    0.15) (center TIED to the source Sersic center), amps ~ N(0,
    sigma0/sqrt(i+1)) — the exact ridge prior the fit marginalizes under."""
    depth = (SHAPELET_NMAX + 1) * (SHAPELET_NMAX + 2) // 2
    beta = float(np.exp(rng.normal(np.log(0.15), 0.15)))
    i = np.arange(depth, dtype=np.float64)
    amps = rng.normal(0.0, SHAPELET_SIGMA0 / np.sqrt(i + 1.0))
    return beta, amps


# --------------------------------------------------------------------------- #
# whitener construction (03 adaptive-s_floor policy, extracted)
# --------------------------------------------------------------------------- #
def adaptive_s_floor(rho, grid=WHITEN_GRID, hard=0.05):
    """03_build_whiteners ADAPTIVE FLOOR policy: choose the largest floor in
    {0.05, 0.02, 0.01, 0.005} strictly below S_raw_min/S_mean so a PSD kernel
    is not biased; keep the hard 0.05 floor if the spectrum touches <= 0."""
    from cgl import exact_ref

    S_raw = exact_ref.stationary_spectrum(np.asarray(rho, np.float64), grid)
    ratio = float(S_raw.min() / S_raw.mean())
    s_floor = hard
    if 0.0 < ratio <= hard:
        for cand in (0.05, 0.02, 0.01, 0.005):
            if cand < ratio:
                s_floor = cand
                break
        else:
            s_floor = 0.5 * ratio
    return s_floor, ratio


def build_product_whitener(rho, m_grid, e_target=E_OP_GATE, grid=WHITEN_GRID,
                           allow_reg=True, _force_reg=False):
    """M-search build_whitener wrapper (03 pattern + the near-singular
    delta-regularization policy, see REG_LAMBDA above). Returns the first M
    in m_grid meeting e_target, else the best (largest-M) attempt, flagged.

    QC FALLBACK (added 2026-07-06 mid-E1, documented deviation): the original
    engage rule keyed only on the spectrum ratio (S_min/S_mean < REG_ENGAGE).
    Fitted fine-scale kernels have a noisy ratio estimate that can land just
    above the threshold while the un-regularized tap search still stalls far
    above the pre-registered e_op gate (observed: seeds 5/7/9/11 fine at
    e_op 0.020-0.19 with reg not engaged). Since the pre-registered spec is
    the e_op gate itself, the fallback now retries with delta-regularization
    whenever the un-regularized search fails e_target, and keeps the better
    build. Kernels already cached in-spec are unaffected."""
    from cgl.whiten import build_whitener

    rho = np.asarray(rho, dtype=np.float64)
    rho_in = rho
    _, ratio0 = adaptive_s_floor(rho, grid=grid)
    reg_lambda = 0.0
    if _force_reg or (allow_reg and ratio0 < REG_ENGAGE):
        reg_lambda = REG_LAMBDA
        c = (rho.shape[0] - 1) // 2
        rho = rho.copy()
        rho[c, c] += reg_lambda
        rho = rho / (1.0 + reg_lambda)
    s_floor, ratio = adaptive_s_floor(rho, grid=grid)
    chosen, tried = None, []
    for M in m_grid:
        if M == 0:
            h = np.array([[1.0]])
            from cgl import exact_ref
            S = exact_ref.stationary_spectrum(np.asarray(rho, np.float64),
                                              grid)
            Sf = np.maximum(S, s_floor * S.mean())
            # optimal scalar tap under the L_inf-symmetrized criterion
            a = float(np.sqrt(2.0 / (Sf.max() + Sf.min())))
            e_op = float(np.max(np.abs(Sf * a * a - 1.0)))
            w = dict(h=a * h, e_op=e_op, M=0, s_floor=s_floor,
                     logdet_per_pix=float(np.mean(np.log(Sf))))
        else:
            w = build_whitener(np.asarray(rho, np.float64), M,
                               s_floor=s_floor, grid=grid)
            w["s_floor"] = s_floor
        tried.append((int(M), float(w["e_op"])))
        chosen = w
        if w["e_op"] <= e_target:
            break
    chosen["e_target"] = float(e_target)
    chosen["e_target_met"] = bool(chosen["e_op"] <= e_target)
    chosen["m_search"] = tried
    chosen["s_ratio"] = ratio
    chosen["s_ratio_unreg"] = ratio0
    chosen["reg_lambda"] = float(reg_lambda)
    chosen["rho_whitened"] = rho
    # QC fallback: un-regularized search failed the pre-registered e_op gate
    # -> retry with delta-regularization and keep the better build.
    if (not chosen["e_target_met"]) and allow_reg and reg_lambda == 0.0 \
            and not _force_reg:
        reg = build_product_whitener(rho_in, m_grid, e_target=e_target,
                                     grid=grid, allow_reg=False,
                                     _force_reg=True)
        reg["reg_fallback"] = True
        if reg["e_op"] < chosen["e_op"]:
            return reg
    return chosen


def rho_drz_anchor(scale, max_lag):
    """Drizzle-anchor ACF for the kernel-fit family, per product scale."""
    if scale == "native":
        L = int(max_lag) + 8
        d = np.zeros((2 * L + 1, 2 * L + 1))
        d[L, L] = 1.0
        return d
    pipe = DrizzleMockPipeline()
    fine = pipe.fine_rho(max_lag=2 * (int(max_lag) + 8) + 1)
    if scale == "fine":
        return fine
    if scale == "binned":
        cov_b = binned_kernel_from_fine(fine)
        c = (cov_b.shape[0] - 1) // 2
        return cov_b / cov_b[c, c]
    raise ValueError(scale)


# --------------------------------------------------------------------------- #
# the E1 model (batched log-posterior with the whiten_fn seam)
# --------------------------------------------------------------------------- #
def build_e1_model(mock, *, scale, likelihood, whitener=None, err_mode="exact",
                   shapelets=False, dtype_mode="mixed"):
    """Build the batched 22/23-dim log-posterior for one mock product.

    Args:
        mock: load_mock dict.
        scale: fine | binned | native (e1d mocks: pass 'native').
        likelihood: 'diag' (masked sqrt(W), the gate-B/D-anchored path) or
            'corr' (cgl.whiten.make_conv_whitener per plane).
        whitener: for 'corr': dict with h (taps) and M (support half-width);
            keep_w is derived per plane via erode_keep. Required iff corr.
        err_mode: 'exact' | 'recal40b' (E1a legacy per-pixel recalibration).
        shapelets: marginalize a shapelet source component (n_max=4) with the
            mock ridge prior (fine scale only).
        dtype_mode: 'mixed' (f32 pipeline + f64 reductions/Cholesky; needs
            x64 enabled), 'f64', or 'f32' (x64 off; reductions f32).

    Returns a SimpleNamespace (see fields at the bottom).
    """
    import jax
    import jax.numpy as jnp

    from cgl.marg import marg_loglik
    from cgl.paths import bootstrap_vendor
    from cgl.whiten import erode_keep, make_conv_whitener

    bootstrap_vendor()
    from gigalens.jax.profiles.light import sersic, shapelets as shp_profile
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.jax.simulator import LensSimulator
    from gigalens.model import PhysicalModel
    from gigalens.simulator import SimulatorConfig

    guards.require_gpu()
    guards.require_single_device()

    x64 = bool(jax.config.jax_enable_x64)
    if dtype_mode in ("mixed", "f64") and not x64:
        raise guards.GuardError(
            f"dtype_mode={dtype_mode} needs GIGALENS_X64=1 before jax import")
    if dtype_mode == "f32" and x64:
        raise guards.GuardError("dtype_mode=f32 but x64 is enabled")
    dt = np.float64 if dtype_mode == "f64" else np.float32
    acc = jnp.float64 if dtype_mode in ("mixed", "f64") else jnp.float32

    if likelihood == "corr" and whitener is None:
        raise ValueError("corr likelihood requires a whitener dict")
    if shapelets and scale != "fine":
        raise ValueError("shapelet marg supported on the fine scale only")

    is_e1d = bool(mock["meta"].get("e1d", False))
    planes = mock_planes(mock, scale)

    # ---- E1a legacy err map -------------------------------------------------
    recal_info = None
    if err_mode == "recal40b":
        assert scale == "fine" and not is_e1d
        sky = planes[0]["keep"] & (planes[0]["r_arc"] > SKY_R_ARC)
        err_r, s, chi2b = recalibrate_err_40b(
            planes[0]["img"], mock["model"], planes[0]["err"], sky)
        planes[0]["err"] = err_r
        recal_info = dict(rescale=s, chi2_sky_before=chi2b,
                          n_sky_px=int(sky.sum()),
                          note=("legacy 40b-style per-pixel recalibration: "
                                "marginal chi2_pp=1, correlations ignored; "
                                "exact err map is the mock TRUTH"))

    # ---- per-plane whitening ------------------------------------------------
    plane_data = []
    n_data = 0
    for pl in planes:
        img = np.asarray(pl["img"], dtype=dt)
        err = np.asarray(pl["err"], dtype=np.float64)
        keep = pl["keep"]
        masked_err = np.where(keep, err, 1e10)
        if likelihood == "diag":
            sqrtW = jnp.asarray((1.0 / masked_err).astype(dt))
            fn = None
            keep_w = keep
        else:
            M = int(whitener["M"])
            keep_w = erode_keep(keep, M)
            fn = make_conv_whitener(
                np.asarray(whitener["h"], dtype=dt),
                (1.0 / masked_err).astype(dt),
                keep_w.astype(dt))
            sqrtW = jnp.asarray((1.0 / masked_err).astype(dt))
        n_data += int(keep_w.sum()) if likelihood == "corr" else int(keep.sum())
        plane_data.append(dict(
            img=jnp.asarray(img), sqrtW=sqrtW, whiten_fn=fn,
            keep=jnp.asarray(keep.astype(dt)), n_keep=int(keep.sum()),
            n_keep_w=int(np.asarray(keep_w).sum()), frame=pl["frame"]))

    # ---- prior / bijector ---------------------------------------------------
    prior, bij, ndim = build_truth_prior(dt=dt, shapelets=shapelets)
    exp_ndim = 23 if shapelets else 22
    assert ndim == exp_ndim, (ndim, exp_ndim)

    # ---- simulators (generator-identical configs) ---------------------------
    kernel = mock["psf_eff"] if scale in ("fine", "binned") else mock["psf"]
    kernel = np.asarray(kernel, dtype=dt)   # must match the render dtype
    # (lax.conv requires equal dtypes; f32 in mixed mode, f64 in f64 mode)
    # transform_pix2angle passed explicitly in the PIPELINE dtype: the default
    # (jnp.eye under x64) is f64 and silently promotes the whole render;
    # coordinate grids are f32 either way (measured; identical to the 05
    # generator's grids, so no simulator mismatch).
    cfg = SimulatorConfig(delta_pix=FINE_PIX, num_pix=FINE_RENDER,
                          supersample=1, kernel=kernel,
                          transform_pix2angle=np.eye(2, dtype=dt) * dt(FINE_PIX))
    guards.assert_psf_sampling(FINE_PIX, FINE_PIX)   # self-consistent mocks
    phys_det = PhysicalModel([epl.EPL(50), shear.Shear()],
                             [sersic.SersicEllipse(use_lstsq=False)],
                             [sersic.SersicEllipse(use_lstsq=False)])
    phys_shp = None
    depth = 0
    Lambda64 = None
    if shapelets:
        phys_shp = PhysicalModel(
            [epl.EPL(50), shear.Shear()], [],
            [shp_profile.Shapelets(n_max=SHAPELET_NMAX, use_lstsq=True,
                                   interpolate=False)])
        depth = (SHAPELET_NMAX + 1) * (SHAPELET_NMAX + 2) // 2
        i_idx = np.arange(depth, dtype=np.float64)
        Lambda64 = jnp.asarray((i_idx + 1.0) / SHAPELET_SIGMA0 ** 2,
                               dtype=acc)

    frames = [pl["frame"] for pl in planes]

    def scene_to_products(scene):
        """(B, 214, 214) -> list of per-plane (B, h, w) product models
        (the generator's exact linear operators; see decision 2)."""
        B = scene.shape[0]
        if scale == "fine":
            return [scene[:, CROP0:CROP0 + N_FINE, CROP0:CROP0 + N_FINE]]
        if scale == "binned":
            f = scene[:, CROP0:CROP0 + N_FINE, CROP0:CROP0 + N_FINE]
            return [f.reshape(B, N_FINE // 2, 2, N_FINE // 2, 2).sum((2, 4))]
        outs = []
        for (oy, ox) in frames:
            sub = scene[:, oy:oy + 3 * N_NATIVE, ox:ox + 3 * N_NATIVE]
            outs.append(sub.reshape(B, N_NATIVE, 3, N_NATIVE, 3).sum((2, 4)))
        return outs

    def _whiten_plane(pl, R):
        """(B, h, w) residual -> (B, n) whitened, plane-appropriate."""
        if pl["whiten_fn"] is None:
            return (R * pl["sqrtW"][None]).reshape(R.shape[0], -1)
        return jax.vmap(pl["whiten_fn"])(R)

    _sim_cache = {}

    def _sims(bs):
        if bs not in _sim_cache:
            sims = dict(det=LensSimulator(phys_det, cfg, bs=bs))
            if shapelets:
                sims["shp"] = LensSimulator(phys_shp, cfg, bs=bs)
            _sim_cache[bs] = sims
        return _sim_cache[bs]

    def _design_ret(sim_shp, x, B):
        """(B, 208, 208, depth) fine-scale design tensor, DATA units (the
        conversion_factor IS included so marginalized coefficients live in
        the generator's amp units — the SBC-matching convention; contrast
        cgl.likelihood which divides M_det instead)."""
        lens_params = x[0]
        shp_params = dict(center_x=x[2][0]["center_x"],
                          center_y=x[2][0]["center_y"],
                          beta=x[2][1]["beta"])
        beta_x, beta_y = sim_shp._beta(lens_params)
        im = sim_shp.phys_model.source_light[0].light(
            beta_x, beta_y, **shp_params)          # (depth, H, W, B)
        # vendored shapelets.light builds jnp.ones(...) which is f64 under
        # x64 regardless of input dtype; cast back to the pipeline dtype
        im = jnp.nan_to_num(im).astype(sim_shp.kernel.dtype)
        im = jnp.transpose(im, (3, 0, 1, 2))       # (B, depth, H, W)
        ret = jax.lax.conv_general_dilated(
            im, sim_shp.kernel, (1, 1), padding="SAME",
            feature_group_count=sim_shp.depth,
            dimension_numbers=("NCHW", "HWOI", "NCHW"))
        ret = jnp.transpose(ret, (0, 2, 3, 1))     # (B, H, W, depth)
        ret = ret * sim_shp.conversion_factor
        return ret[:, CROP0:CROP0 + N_FINE, CROP0:CROP0 + N_FINE, :]

    def _marg_parts(sims, x, B, Rw):
        ret = _design_ret(sims["shp"], x, B)
        wfn = plane_data[0]["whiten_fn"]
        if wfn is None:
            # diag likelihood (e.g. the quick diag MAP of the two-pass
            # kernel fit): whiten the design exactly like the residual,
            # ret * sqrtW (masked err reciprocal is 0 off-keep)
            sW = plane_data[0]["sqrtW"]
            Xw = (ret * sW[None, :, :, None]).reshape(
                ret.shape[0], -1, ret.shape[-1])
        else:
            Xw = jax.vmap(
                lambda r: jax.vmap(wfn, in_axes=2, out_axes=1)(r))(ret)
        Xw = Xw.astype(acc)

        def _one(Xb, Rb):
            logL, a_star, logdetA = marg_loglik(Xb, Rb, Lambda64)
            return logL, a_star

        return jax.vmap(_one)(Xw, Rw.astype(acc)), ret

    _lp_cache = {}

    def get_log_prob(bs):
        """Jitted (bs, ndim) -> (logpost (bs,), chi2_pp (bs,)). Cached per bs."""
        if bs in _lp_cache:
            return _lp_cache[bs]
        sims = _sims(bs)

        @jax.jit
        def log_prob(z):
            zl = list(z.T)
            x = bij.forward(zl)
            scene = sims["det"].simulate(x)
            scene = scene.reshape(bs, FINE_RENDER, FINE_RENDER)
            prods = scene_to_products(scene)
            usq = jnp.zeros((bs,), dtype=acc)
            chi2_num = jnp.zeros((bs,), dtype=acc)
            Rw0 = None
            for pl, mp in zip(plane_data, prods):
                R = pl["img"][None] - mp
                u = _whiten_plane(pl, R)
                if Rw0 is None:
                    Rw0 = u
                usq = usq + jnp.sum(u.astype(acc) ** 2, axis=1)
                cw = (R * pl["sqrtW"][None]) * pl["keep"][None]
                chi2_num = chi2_num + jnp.sum(cw.astype(acc) ** 2,
                                              axis=(1, 2))
            n_keep_tot = sum(pl["n_keep"] for pl in plane_data)
            chi2 = chi2_num / n_keep_tot
            if shapelets:
                (marg_lls, _), _ = _marg_parts(sims, x, bs, Rw0)
                loglike = marg_lls
            else:
                loglike = -0.5 * usq
            log_prior = prior.log_prob(x) + bij.forward_log_det_jacobian(zl)
            lp = loglike.astype(z.dtype) + log_prior
            return lp, chi2.astype(z.dtype)

        _lp_cache[bs] = log_prob
        return log_prob

    def render_products(z):
        """Single-z per-plane MODEL images (numpy), incl. the marg shapelet
        component at its conditional mode (for the two-pass kernel fit)."""
        z1 = jnp.asarray(np.asarray(z, dtype=dt).reshape(1, -1))
        sims = _sims(1)
        x = bij.forward(list(z1.T))
        scene = sims["det"].simulate(x).reshape(1, FINE_RENDER, FINE_RENDER)
        prods = scene_to_products(scene)
        outs = [np.asarray(p[0]) for p in prods]
        if shapelets:
            R = plane_data[0]["img"][None] - prods[0]
            Rw = _whiten_plane(plane_data[0], R)
            (_, a_star), ret = _marg_parts(sims, x, 1, Rw)
            shp_img = np.asarray(
                jnp.sum(ret[0] * a_star[0][None, None, :], axis=-1))
            outs[0] = outs[0] + shp_img
        return outs

    def to_physical(z_flat):
        """(N, ndim) -> (labels, cols (N, n_phys)) in bijector key order,
        block-consumed (the gu-2022 ordering-trap-safe pattern)."""
        arr = jnp.asarray(np.asarray(z_flat, dtype=dt))
        phys = bij.forward(list(arr.T))
        blocks = [(0, 0, ""), (0, 1, ""), (1, 0, "ll_"), (2, 0, "src_")]
        if shapelets:
            blocks.append((2, 1, "srcshp_"))
        labels, cols = [], []
        for (i, j, pref) in blocks:
            for key, val in phys[i][j].items():
                labels.append(pref + key)
                cols.append(np.asarray(val, dtype=np.float64))
        return labels, np.vstack(cols).T

    def truth_vector(labels):
        """Truth in the same label order (from the mock's nested truth)."""
        tn = mock["truth_nested"]
        tmap = {}
        for k, v in tn[0][0].items():
            tmap[k] = float(v)
        for k, v in tn[0][1].items():
            tmap[k] = float(v)
        for k, v in tn[1][0].items():
            tmap["ll_" + k] = float(v)
        for k, v in tn[2][0].items():
            tmap["src_" + k] = float(v)
        if shapelets:
            tmap["srcshp_beta"] = float(mock["meta"]["shapelet_beta"])
        return np.array([tmap[lab] for lab in labels])

    return types.SimpleNamespace(
        get_log_prob=get_log_prob, prior=prior, bij=bij, ndim=ndim,
        dt=dt, dtype_mode=dtype_mode, scale=scale, likelihood=likelihood,
        shapelets=shapelets, n_data=n_data,
        n_keep=[pl["n_keep"] for pl in plane_data],
        n_keep_w=[pl["n_keep_w"] for pl in plane_data],
        render_products=render_products, to_physical=to_physical,
        truth_vector=truth_vector, recal_info=recal_info,
        whitener_meta=(None if whitener is None else dict(
            M=int(whitener["M"]), e_op=float(whitener["e_op"]),
            s_floor=float(whitener.get("s_floor", np.nan)))),
    )


# --------------------------------------------------------------------------- #
# MAP -> SVI -> PHMC+ChEES (reduced-budget recipe)
# --------------------------------------------------------------------------- #
def run_map(model, n_particles=100, steps=300, lr=1e-2, seed=0,
            lbfgs_rounds=6, lbfgs_iters=500):
    """Multistart adabelief MAP + L-BFGS canyon polish.

    Stage 1: n_particles prior starts, adabelief with lr annealed lr -> lr/10
    (the gu-2022 paper's 'Adam 1e-2 -> 1e-3' schedule). Stage 2: optax.lbfgs
    (zoom linesearch) from the best particle, `lbfgs_rounds` restarts of
    `lbfgs_iters` iterations (fresh curvature memory each round).

    PILOT EVIDENCE (mock_000, recorded in the P1b report): the posterior at
    arc SNR ~1e3 is a single connected canyon (a 51-point straight-line
    logpost slice from the adabelief pocket to the truth rises MONOTONICALLY
    — no barrier), but first-order multistart stalls 350-1600 nats below the
    mode (chi2_pp 1.1-5), leaving SVI/HMC in a locally-mixing pocket 20-50
    sigma from truth. L-BFGS traverses the canyon at ~1200 nats/500 iters
    (bs=1, ~15 s/500 iters on an A16) and lands within ~15 nats of the
    truth's own logpost; a truth-started control run then shows fully
    calibrated recovery (binned corr: z(gamma)=0.06).

    Returns dict(best_z, best_lp, pop_z (n,ndim), pop_lp, chi2_best,
    best_lp_stage1, lbfgs_lp (per round), wall_s)."""
    import functools

    import jax
    import jax.numpy as jnp
    import optax

    t0 = time.time()
    log_prob = model.get_log_prob(n_particles)

    xs = model.prior.sample(n_particles, seed=jax.random.PRNGKey(seed))
    z0 = jnp.stack([jnp.reshape(v, (-1,)) for v in model.bij.inverse(xs)]).T
    z0 = z0.astype(model.dt)
    sched = optax.exponential_decay(lr, transition_steps=max(steps, 1),
                                    decay_rate=0.1)
    opt = optax.adabelief(sched, b1=0.95, b2=0.99)

    def loss(z):
        lp, chi2 = log_prob(z)
        return -jnp.mean(lp) / model.n_data, (lp, chi2)

    vg = jax.jit(jax.value_and_grad(loss, has_aux=True))

    def one_step(carry, _):
        z, opt_state = carry
        (_, (lp, chi2)), g = vg(z)
        updates, opt_state = opt.update(g, opt_state)
        return (optax.apply_updates(z, updates), opt_state), None

    @jax.jit
    def run(z0):
        (zf, _), _ = jax.lax.scan(one_step, (z0, opt.init(z0)), length=steps)
        return zf

    zf = run(z0)
    lp_f, chi2_f = log_prob(zf)
    lp_f = np.asarray(lp_f, dtype=np.float64)
    best1 = int(np.nanargmax(lp_f))
    best_lp_stage1 = float(lp_f[best1])

    # ---- stage 2: L-BFGS restarts from the best particle --------------------
    lp1 = model.get_log_prob(1)

    def fun(z):
        v = -lp1(z.astype(model.dt)[None])[0][0]
        return v.astype(z.dtype)

    opt2 = optax.lbfgs()
    vgf = optax.value_and_grad_from_state(fun)

    @functools.partial(jax.jit, static_argnums=(1,))
    def lbfgs_run(zz, n):
        def step(carry, _):
            z, s = carry
            v, g = vgf(z, state=s)
            u, s = opt2.update(g, s, z, value=v, grad=g, value_fn=fun)
            return (optax.apply_updates(z, u), s), v

        (z_out, _), vals = jax.lax.scan(step, (zz, opt2.init(zz)), length=n)
        return z_out, vals

    z_cur = jnp.asarray(np.asarray(zf[best1], dtype=np.float64)
                        if model.dtype_mode != "f32"
                        else np.asarray(zf[best1], dtype=np.float32))
    round_lps = []
    best_z = np.asarray(z_cur, dtype=np.float64)
    best_lp = best_lp_stage1
    for _ in range(int(lbfgs_rounds)):
        z_cur, vals = lbfgs_run(z_cur, int(lbfgs_iters))
        lp_round = -float(np.asarray(vals)[-1])
        round_lps.append(lp_round)
        if np.isfinite(lp_round) and lp_round > best_lp:
            best_lp = lp_round
            best_z = np.asarray(z_cur, dtype=np.float64)
        if len(round_lps) > 1 and abs(round_lps[-1] - round_lps[-2]) < 0.5:
            break

    lp_b, chi2_b = lp1(jnp.asarray(best_z.astype(model.dt))[None])
    return dict(best_z=best_z, best_lp=float(np.asarray(lp_b)[0]),
                best_lp_stage1=best_lp_stage1,
                lbfgs_lp=round_lps,
                pop_z=np.asarray(zf, dtype=np.float32),
                pop_lp=lp_f.astype(np.float32),
                chi2_best=float(np.asarray(chi2_b)[0]),
                wall_s=time.time() - t0)


def run_svi(model, start_z, n_vi=300, steps=500, lr=1e-3, seed=1):
    """Full-covariance MVN SVI (gigalens SVI loop, single-device port).

    Returns dict(loc (ndim,), cov (ndim,ndim) f64, neg_elbo_hist, wall_s)."""
    import jax
    import jax.numpy as jnp
    import optax
    import tensorflow_probability.substrates.jax as tfp

    tfd = tfp.distributions
    tfb = tfp.bijectors

    log_prob = model.get_log_prob(n_vi)
    t0 = time.time()
    ndim = model.ndim
    dt = model.dt
    cov_bij = tfb.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=dt(1e-6))
    scale0 = jnp.eye(ndim, dtype=dt) * dt(1e-3)
    params0 = jnp.concatenate([jnp.asarray(start_z, dtype=dt).reshape(-1),
                               cov_bij.inverse(scale0)])
    opt = optax.adabelief(lr, b1=0.95, b2=0.99)

    def neg_elbo(qp, key):
        mean = qp[:ndim]
        tril = cov_bij.forward(qp[ndim:])
        qz = tfd.MultivariateNormalTriL(loc=mean, scale_tril=tril)
        z = qz.sample(n_vi, seed=key)
        return jnp.mean(qz.log_prob(z) - log_prob(z)[0])

    vg = jax.jit(jax.value_and_grad(neg_elbo))

    def one_step(carry, _):
        qp, opt_state, key, best_qp, best_loss = carry
        key, k = jax.random.split(key)
        loss, g = vg(qp, k)
        better = loss < best_loss
        best_qp = jax.lax.select(better, qp, best_qp)
        best_loss = jax.lax.select(better, loss, best_loss)
        updates, opt_state = opt.update(g, opt_state)
        qp = optax.apply_updates(qp, updates)
        return (qp, opt_state, key, best_qp, best_loss), loss

    @jax.jit
    def run(params0, key0):
        carry = (params0, opt.init(params0), key0, params0,
                 jnp.asarray(np.inf, dtype=dt))
        (qp, _, _, best_qp, best_loss), hist = jax.lax.scan(
            one_step, carry, length=steps)
        return best_qp, best_loss, hist

    best_qp, best_loss, hist = run(params0, jax.random.PRNGKey(seed))
    loc = np.asarray(best_qp[:ndim], dtype=np.float64)
    tril = np.asarray(cov_bij.forward(best_qp[ndim:]), dtype=np.float64)
    cov = tril @ tril.T
    return dict(loc=loc, cov=cov, neg_elbo=float(best_loss),
                neg_elbo_hist=np.asarray(hist, dtype=np.float32),
                wall_s=time.time() - t0)


def run_chees(model, loc, cov, chains=24, burn=250, keep=750, step_size=0.3,
              init_leapfrog=3, max_leapfrog=30, seed=2):
    """Batched single-device PHMC + ChEES (GBTLA) + dual-averaging step size
    (the gu-2022 02_fit_system --gbtla stack; guards.floor_svi_covariance).

    Returns dict(draws (keep, chains, ndim) f32, leapfrogs traced (or None),
    step_size_final, n_floored, wall_s)."""
    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    from tensorflow_probability.python.internal import unnest

    tfd = tfp.distributions
    tfe = tfp.experimental

    cov_reg, chol, n_floored = guards.floor_svi_covariance(cov)
    dt = model.dt
    log_prob = model.get_log_prob(chains)

    @jax.jit
    def lp_only(z):
        return log_prob(z)[0]

    momentum = tfd.MultivariateNormalFullCovariance(
        loc=jnp.zeros(model.ndim, dtype=dt),
        covariance_matrix=jnp.asarray(np.linalg.inv(cov_reg), dtype=dt))
    num_adapt = int(0.8 * burn)

    key = jax.random.PRNGKey(seed)
    k_start, k_chain = jax.random.split(key)
    eps = jax.random.normal(k_start, (chains, model.ndim), dtype=dt)
    start = jnp.asarray(loc, dtype=dt)[None] + eps @ jnp.asarray(
        chol.T, dtype=dt)

    kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=lp_only, momentum_distribution=momentum,
        step_size=dt(step_size), num_leapfrog_steps=init_leapfrog)
    kernel = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
        kernel, num_adaptation_steps=num_adapt,
        max_leapfrog_steps=max_leapfrog)
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel, num_adaptation_steps=num_adapt)

    def trace_fn(_, pkr):
        return (unnest.get_innermost(pkr, "max_trajectory_length"),
                unnest.get_innermost(pkr, "step_size"))

    t0 = time.time()

    def _run(tf):
        @jax.jit
        def run(start, seed):
            return tfp.mcmc.sample_chain(
                num_results=keep, num_burnin_steps=burn, current_state=start,
                kernel=kernel, trace_fn=tf, seed=seed)

        return run(start, k_chain)

    try:
        draws, trace = _run(trace_fn)
        max_traj = np.asarray(trace[0], dtype=np.float32)
        step_hist = np.asarray(trace[1], dtype=np.float32)
    except Exception:
        draws = _run(None)
        max_traj = step_hist = np.array([], dtype=np.float32)
    draws = np.asarray(jax.block_until_ready(draws), dtype=np.float32)
    return dict(draws=draws, max_traj=max_traj, step_size_hist=step_hist,
                n_floored=int(n_floored), wall_s=time.time() - t0,
                num_adapt=num_adapt)


def diagnostics(draws):
    """(T, C, ndim) -> (ess, rhat) per dim (tfp, split chains)."""
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp

    s = jnp.asarray(draws)
    rhat = np.asarray(tfp.mcmc.potential_scale_reduction(
        s, independent_chain_ndims=1, split_chains=True), dtype=np.float64)
    ess = np.asarray(tfp.mcmc.effective_sample_size(
        s, cross_chain_dims=1), dtype=np.float64)
    return ess, rhat


# --------------------------------------------------------------------------- #
# two-pass kernel fit (real-data 02 recipe on the mock products)
# --------------------------------------------------------------------------- #
def kernel_pass(mock, scale, model_prods, out_path=None):
    """Pass 2 of the two-pass recipe: model-subtracted masked ACF -> WLS
    kernel fit -> whitener build. CPU-only given the MAP model images.

    Args:
        mock: load_mock dict; scale: fine|binned|native.
        model_prods: per-plane model images (from the pass-1 diag MAP).

    Returns the kernel dict (rho_fit, whitener taps, gates, provenance)."""
    planes = mock_planes(mock, scale)
    L = ACF_WINDOW[scale]
    box = BKG_BOX[scale]
    num = None
    den = None
    chi2_sky = []
    n_sky = 0
    for pl, mp in zip(planes, model_prods):
        v = (pl["img"] - mp) / pl["err"]
        sky = pl["keep"] & (pl["r_arc"] > SKY_R_ARC)
        v_det, method = detrend_sky(v, sky, box)
        rho_i, counts_i = masked_acf_2d(v_det, sky, max_lag=L + 6)
        w = np.where(np.isfinite(rho_i), counts_i, 0.0)
        r = np.where(np.isfinite(rho_i), rho_i, 0.0)
        num = r * w if num is None else num + r * w
        den = w if den is None else den + w
        chi2_sky.append(float(np.mean(v[sky] ** 2)))
        n_sky += int(sky.sum())
    rho_meas = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    counts = den

    rho_drz = rho_drz_anchor(scale, L)
    fit = fit_kernel(rho_meas, counts, rho_drz, max_lag=L)
    family = "rho_model(w,sigma_e)"
    if not fit["gate_le_0p05"]:
        fit2 = fit_kernel2(rho_meas, counts, rho_drz, max_lag=L)
        if fit2["max_abs_resid"] < fit["max_abs_resid"]:
            fit, family = fit2, "rho_model2 (fallback; 1-family failed 0.05)"

    wh = build_product_whitener(fit["rho_fit"], M_GRID[scale])
    out = dict(
        scale=scale, family=family, rho_fit=fit["rho_fit"],
        rho_meas=rho_meas, counts=counts,
        max_abs_resid=float(fit["max_abs_resid"]),
        gate_le_0p05=bool(fit["gate_le_0p05"] or fit["max_abs_resid"] <= 0.05),
        fit_params={k: float(v) for k, v in fit.items()
                    if isinstance(v, float)},
        h=wh["h"], M=int(wh["M"]), e_op=float(wh["e_op"]),
        s_floor=float(wh["s_floor"]), m_search=wh["m_search"],
        e_target_met=bool(wh["e_target_met"]),
        reg_lambda=float(wh["reg_lambda"]),
        chi2_sky=chi2_sky, n_sky_px=n_sky, detrend=method,
        model_subtracted=True,
    )
    if out_path is not None:
        meta = dict(model_subtracted=True, scale=scale, family=family,
                    seed=int(mock["meta"].get("seed", -1)),
                    detrend=method, n_sky_px=n_sky,
                    sky_annulus_arc=SKY_R_ARC, acf_window=L,
                    reg_lambda=float(wh["reg_lambda"]),
                    commit=_git_head())
        np.savez(out_path, rho_kernel=out["rho_fit"], rho_meas=rho_meas,
                 counts=counts, h=out["h"], M=out["M"], e_op=out["e_op"],
                 s_floor=out["s_floor"], reg_lambda=out["reg_lambda"],
                 max_abs_resid=out["max_abs_resid"],
                 meta=json.dumps(meta))
    return out


def analytic_whitener(mock, scale, e_target=E_OP_GATE):
    """Whitener from the mock's stored EXACT analytic kernel (ablation arm)."""
    if mock["meta"].get("e1d", False):
        rho = mock["rho_kernel"]
    elif scale == "fine":
        rho = mock["rho_kernel"]
    elif scale == "binned":
        rho = mock["rho_kernel_binned"]
    else:
        rho = np.zeros((5, 5))
        rho[2, 2] = 1.0
    wh = build_product_whitener(np.asarray(rho, np.float64), M_GRID.get(
        scale, (4, 6, 8, 10, 12, 14)), e_target=e_target)
    wh["rho_fit"] = np.asarray(rho, np.float64)
    return wh


# --------------------------------------------------------------------------- #
# fit summary
# --------------------------------------------------------------------------- #
def summarize_fit(labels, phys, truth_vec):
    """Posterior summary vs truth: means/stds/z/coverage per physical param."""
    mean = phys.mean(axis=0)
    std = phys.std(axis=0)
    z = z_scores(mean, std, truth_vec)
    cov68 = np.array([coverage_flag(phys[:, j], truth_vec[j], 0.68)
                      for j in range(phys.shape[1])], dtype=bool)
    cov95 = np.array([coverage_flag(phys[:, j], truth_vec[j], 0.95)
                      for j in range(phys.shape[1])], dtype=bool)
    q16 = np.quantile(phys, 0.16, axis=0)
    q84 = np.quantile(phys, 0.84, axis=0)
    return dict(labels=labels, mean=mean, std=std, z=z, cov68=cov68,
                cov95=cov95, q16=q16, q84=q84)


# --------------------------------------------------------------------------- #
# job manifest (07 batcher; unit-tested dry logic)
# --------------------------------------------------------------------------- #
def _job(name, out, args):
    return dict(name=name, out=str(out), args=args)


def build_job_manifest(experiment, mocks_dir=None, fits_dir=None,
                       n_mocks=8, sbc_seeds=range(64),
                       e1d_seeds=range(100, 116), skip_existing=True):
    """Job list for 07_run_e1_batch.sh. Each job: name, out npz, 06 args."""
    mocks_dir = Path(mocks_dir or MOCKS_DIR)
    fits_dir = Path(fits_dir or E1_FITS)
    jobs = []

    def add(name, args):
        out = fits_dir / f"{name}.npz"
        jobs.append(_job(name, out, args + ["--out", str(out)]))

    if experiment in ("pilot",):
        s = 0
        add(f"mock{s:03d}_fine_diag_recal",
            ["--seed", str(s), "--scale", "fine", "--likelihood", "diag",
             "--err", "recal40b", "--map-starts", "4"])
        add(f"mock{s:03d}_fine_corr_fitted",
            ["--seed", str(s), "--scale", "fine", "--likelihood", "corr",
             "--kernel", "fitted"])
        add(f"mock{s:03d}_binned_corr_fitted",
            ["--seed", str(s), "--scale", "binned", "--likelihood", "corr",
             "--kernel", "fitted"])
        add(f"mock{s:03d}_native_corr_fitted",
            ["--seed", str(s), "--scale", "native", "--likelihood", "corr",
             "--kernel", "fitted"])
        add(f"mock{s:03d}_fine_corr_analytic",
            ["--seed", str(s), "--scale", "fine", "--likelihood", "corr",
             "--kernel", "analytic"])
    if experiment in ("e1a", "all"):
        for s in range(n_mocks):
            add(f"mock{s:03d}_fine_diag_recal",
                ["--seed", str(s), "--scale", "fine", "--likelihood", "diag",
                 "--err", "recal40b", "--map-starts", "4"])
    if experiment in ("e1b", "all"):
        for s in range(n_mocks):
            for scale in ("fine", "binned", "native"):
                add(f"mock{s:03d}_{scale}_corr_fitted",
                    ["--seed", str(s), "--scale", scale,
                     "--likelihood", "corr", "--kernel", "fitted"])
        for s in (0, 1):
            add(f"mock{s:03d}_fine_corr_analytic",
                ["--seed", str(s), "--scale", "fine", "--likelihood", "corr",
                 "--kernel", "analytic"])
    if experiment in ("e1c", "all"):
        for s in sbc_seeds:
            add(f"mock{s:03d}_fine_corr_fitted",
                ["--seed", str(s), "--scale", "fine", "--likelihood", "corr",
                 "--kernel", "fitted"])
    if experiment in ("e1d", "all"):
        for s in e1d_seeds:
            for arm in ("diag", "strict", "relaxed"):
                add(f"e1d{s:03d}_{arm}",
                    ["--e1d", "--seed", str(s), "--whitener-arm", arm])

    # de-dup (e1b fine ⊂ e1c) keeping first occurrence
    seen, uniq = set(), []
    for j in jobs:
        if j["out"] in seen:
            continue
        seen.add(j["out"])
        uniq.append(j)
    if skip_existing:
        uniq = [j for j in uniq if not Path(j["out"]).exists()]
    return uniq
