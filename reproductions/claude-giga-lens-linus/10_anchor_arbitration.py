"""10_anchor_arbitration.py — P3-L0: two-stack anchor arbitration (PLAN §6).

QUESTION: does the scene API, fitting the SAME v2d native product with the
DIAGONAL (masked-err down-weighted, ridge-marginalized) likelihood, reproduce
the foundry-i gamma = 1.433 [1.400, 1.468] anchor at posterior level?
Pre-registered design checkpoint: research/checkpoints_l0.md (2026-07-15) —
written BEFORE this script ran. Falsify the premise before explaining it.

Model/likelihood = the PARITY-CERTIFIED configuration (F1–F7,
data/parity_report_scene.json): cgl2.scene_build marg view (foundry-i priors
verbatim) + CorrelatedImageData delta bundle on masked_err (the old stack's
down-weighting convention bit-for-bit) + parity grid/PSF overrides from
data/parity_refs.npz. Sampler = the B0-validated MC-SMC MAMS
(cgl2.samplers.smc_micro protocol, frozen P0 constants), PRIOR-SEEDED, via
common.run_tempered_smc with a DELEGATING logging/checkpoint kernel wrapper
(validated driver + kernel code untouched; wrapper only observes U -> z and
enforces the pre-registered wall cap).

Stages (run in order):
  sanity    : wiring gates S1/S2/S4 (term logL + log_prior vs parity refs,
              gamma column identity), S3 prior-draw finiteness, 2-stage
              timing smoke, N decision  -> data/l0_sanity_report.json
  run       : the production SMC (N from sanity decision; per-stage particle
              checkpoints under data/l0_smc_checkpoints/)
                                        -> data/l0_arbitration_smc.{npz,json}
  v3bsmoke  : L0-G2 memory feasibility (v3b correlated marg term, vmapped
              value_and_grad at 8/16 particles; MB/particle)
                                        -> data/l0_v3b_memory_smoke.json
  harvest   : PLOTS FIRST (figs/l0_gamma_hist.png, figs/l0_smc_traces.png),
              then the pre-registered gate math -> data/l0_gate_eval.json

Run (GPU 9 L4 ONLY — front assignment):
  GIGALENS_X64=1 CUDA_VISIBLE_DEVICES=9 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  XLA_FLAGS=--xla_gpu_autotune_level=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /raid/benson/.venvs/cgl2/bin/python 10_anchor_arbitration.py <stage>
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("GIGALENS_X64", "1")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CGL2_GPU", "9"))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figs"
CKPT_DIR = DATA / "l0_smc_checkpoints"

from cgl2 import guards, param_map, paths  # noqa: E402

paths.bootstrap_vendor()
guards.require_vendor_ref()
guards.require_jax_pin()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from cgl2 import scene_build  # noqa: E402
from cgl2.correlated import (CorrelatedImageData, delta_bundle,  # noqa: E402
                             load_whitener_bundle)
from cgl2.samplers import common, smc_micro  # noqa: E402
from gigalens.jax.scene_prob_model import ProbModel  # noqa: E402

# ---- frozen references (checkpoint 2026-07-15, research/checkpoints_l0.md) ----
GAMMA_ANCHOR = 1.433
ANCHOR_CI68 = (1.400, 1.468)          # foundry-i hmc_v13_v2d
SIGMA_ANCHOR = 0.034                  # CI half-width
GAMMA_KEY = "planes/0/mass/0/gamma"
SEED = 2                              # campaign production seed
N_FULL = 512
N_REDUCED = 256                       # single pre-registered halving
PROJ_CAP_H = 5.0                      # projected-wall threshold for halving
WALL_CAP_H = float(os.environ.get("L0_WALL_CAP_H", "5.5"))
EXPECTED_STAGES = 40                  # extrapolation prior (checkpoint)
BASIN_SPLIT = 1.9                     # campaign v3b convention (continuity)

S1_TOL = 1e-7
S2_TOL = 1e-5
S4_TOL = 1e-9


class WallCapReached(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def load_refs():
    return np.load(DATA / "parity_refs.npz", allow_pickle=True)


def build_pm(tag, refs, *, diagonal, checkpoint=True, remat=None):
    """ProbModel for product `tag` in the parity-certified marg configuration.

    diagonal=True  -> delta bundle on masked_err (the old down-weighting
                      convention; the arbitration likelihood).
    diagonal=False -> the ported stationary whitener bundle for `tag`
                      (correlated; L0-G2 smoke only).
    """
    cfgs = {"v2d": dict(delta_pix=0.13, supersample=2),
            "v3b": dict(delta_pix=0.08, supersample=2)}
    cfg = cfgs[tag]
    prod = scene_build.load_product(paths.FOUNDRY_DATA / f"cutout_{tag}.npz")
    # input desync fence (same as the parity harness)
    for k, arr in (("img", prod["img"]), ("err_raw", prod["err_map"]),
                   ("keep_mask", prod["keep_mask"])):
        d = float(np.max(np.abs(np.asarray(arr, dtype=np.float64)
                                - refs[f"{tag}:{k}"].astype(np.float64))))
        if d != 0.0:
            raise RuntimeError(f"{tag}: product {k} desynced from refs (|d|={d})")

    prov = json.loads(str(refs["provenance"]))
    masked_err = refs[f"{tag}:masked_err"]
    Lambda_diag = refs[f"{tag}:Lambda_diag"]
    sim_cfg = scene_build.sim_config_for(prod, **cfg)
    if remat is None:
        remat = os.environ.get("L0_REMAT") == "1"
    if remat:
        # ledgered deviation 2026-07-15 22:50 (research/checkpoints_l0.md):
        # the vendor's exact memory fix for the memory-bound ss=2 regime;
        # identical ops, gated by the remat-exactness check in `rematgate`.
        import dataclasses as _dc
        sim_cfg = _dc.replace(sim_cfg, remat_basis=True)
    mv = scene_build.build_scene_model(view="marg", near_xy=prov["near_xy"])

    if diagonal:
        bundle = delta_bundle(None, None, masked_err=masked_err,
                              source="L0 diag arbitration (old masked-err "
                                     "downweighting, parity anchor)")
    else:
        bundle = load_whitener_bundle(
            paths.CGL1_DATA / f"whitener_{tag}.npz",
            sqrt_d_inv=1.0 / np.asarray(masked_err, dtype=np.float64),
            source=f"ported whitener_{tag} (02 manifest)")

    ds = CorrelatedImageData(
        prod["img"].astype(np.float64), sim_cfg, bundle,
        error_map=refs[f"{tag}:err_raw"],
        mask=refs[f"{tag}:keep_mask"].astype(bool), sees="all",
        corr_mode="marg", Lambda_diag=Lambda_diag, checkpoint=checkpoint)
    pm = ProbModel(mv.model, ds, mode="lstsq")
    # parity grid/PSF conventions BEFORE first trace (scene_build items 1-3)
    scene_build.apply_parity_conventions(
        pm.terms[0].simulator, img_X=refs[f"{tag}:img_X32"],
        img_Y=refs[f"{tag}:img_Y32"], kernel=refs[f"{tag}:kernel_unflipped"])
    return pm, mv, refs


def make_closures(pm):
    def logprior_fn(z):
        return pm.log_prior(jnp.asarray(z, dtype=jnp.float64))

    def loglik_fn(z):
        return pm.log_like(jnp.asarray(z, dtype=jnp.float64))[0]

    return logprior_fn, loglik_fn


def prior_draws(pm, n, seed):
    x = pm.model.prior.sample(int(n), seed=jax.random.PRNGKey(int(seed)))
    z = pm.model.bijector.inverse(x)
    return np.asarray(z, dtype=np.float64)


def gamma_of(pm, Z):
    """gamma (constrained) for flat-z rows — z_param_names-keyed, never order."""
    u = pm.model.bijector.forward(jnp.asarray(Z, dtype=jnp.float64))
    return np.asarray(u[GAMMA_KEY], dtype=np.float64)


# --------------------------------------------------------------------------- #
# logging/checkpoint kernel wrapper (delegation only; no numerics)
# --------------------------------------------------------------------------- #
class LoggingKernel(common.MutationKernel):
    """Wraps the validated MAMS kernel: delegates build/tune/mutate verbatim,
    records per-stage wall/tuning stats, dumps the post-mutation ensemble
    (z-space) per stage, and enforces the pre-registered wall cap by raising
    WallCapReached AFTER writing the stage checkpoint."""

    def __init__(self, inner, ckpt_dir: Path, wall_cap_h: float):
        self.inner = inner
        self.name = inner.name + "+log"
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.wall_cap_s = float(wall_cap_h) * 3600.0
        self.t0 = time.time()
        self.stage = 0
        self.trace = []

    def build(self, logprior_fn, loglik_fn, dim):
        self.t0 = time.time()
        return self.inner.build(logprior_fn, loglik_fn, dim)

    def tune(self, key, lam, mu, chol, U_pilot, eps0):
        t = time.time()
        tuning = self.inner.tune(key, lam, mu, chol, U_pilot, eps0)
        self._t_tune = time.time() - t
        return tuning

    def mutate(self, key, lam, mu, chol, U, tuning):
        t = time.time()
        U_new, stats = self.inner.mutate(key, lam, mu, chol, U, tuning)
        t_mut = time.time() - t
        mu_np = np.asarray(mu, dtype=np.float64)
        chol_np = np.asarray(chol, dtype=np.float64)
        z = mu_np[None, :] + np.asarray(U_new, dtype=np.float64) @ chol_np.T
        np.savez(self.ckpt_dir / f"stage_{self.stage:03d}.npz",
                 z=z, lam=np.float64(lam),
                 eps=np.float64(tuning.step_size), n_int=np.int64(tuning.n_int))
        row = dict(stage=self.stage, lam=float(lam),
                   eps=float(tuning.step_size), n_int=int(tuning.n_int),
                   accept=float(stats["accept_mean"]),
                   t_tune_s=round(self._t_tune, 2), t_mutate_s=round(t_mut, 2),
                   wall_s=round(time.time() - self.t0, 1))
        self.trace.append(row)
        print(f"[l0-smc] {row}", flush=True)
        self.stage += 1
        if time.time() - self.t0 > self.wall_cap_s:
            raise WallCapReached(
                f"pre-registered wall cap {self.wall_cap_s/3600:.2f} h reached "
                f"at stage {self.stage}, lambda={lam:.6f} — checkpoints kept")
        return U_new, stats


def run_smc_logged(pm, n_particles, seed, *, max_stages=400, ckpt_dir=CKPT_DIR,
                   kern_holder=None):
    """Exactly cgl2.samplers.smc_micro.run_smc's frozen protocol (same
    defaults: target_ess 0.7, num_mcmc_steps 4, pilot 64, eps0 0.5, n_boot 200,
    boot_seed 20260715+seed, precondition=True) with the logging wrapper."""
    logprior_fn, loglik_fn = make_closures(pm)
    z0 = prior_draws(pm, n_particles, seed)
    inner = smc_micro.make_kernel("mams",
                                  num_mcmc_steps=smc_micro.NUM_MCMC_STEPS)
    kern = LoggingKernel(inner, ckpt_dir, WALL_CAP_H)
    if kern_holder is not None:
        kern_holder.append(kern)   # reachable even if the driver raises
    key = jax.random.PRNGKey(int(seed))
    res = common.run_tempered_smc(
        logprior_fn, loglik_fn, z0, key, kernel=kern,
        target_ess=smc_micro.TARGET_ESS, max_stages=max_stages,
        n_boot=200, pilot_size=64, eps0=0.5, precondition=True,
        boot_seed=20260715 + int(seed))
    return res, kern, z0


def gpu_peak_mb():
    st = jax.local_devices()[0].memory_stats() or {}
    return float(st.get("peak_bytes_in_use", 0)) / 2**20


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_sanity():
    refs = load_refs()
    pm, mv, _ = build_pm("v2d", refs, diagonal=True)
    logprior_fn, loglik_fn = make_closures(pm)
    rep = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               stage="sanity", device=str(jax.local_devices()[0]),
               CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"])

    # S1 + S2 + S4: wiring vs the parity reference artifacts
    s1 = {}
    ll_jit = jax.jit(loglik_fn)
    for pname in ("z_ref", "z_pert0", "z_pert1", "z_pert2"):
        labeled = param_map.labeled_from_vec46(refs[f"v2d:{pname}:x46"])
        z = param_map.scene_z_from_old_labels(mv.model, labeled)
        ll = float(ll_jit(jnp.asarray(z)))
        ref = float(refs[f"v2d:{pname}:logL_data"])
        s1[pname] = dict(scene=ll, ref=ref, absdiff=abs(ll - ref))
        if pname == "z_ref":
            lp = float(pm.log_prior(jnp.asarray(z)))
            lp_ref = float(refs["v2d:z_ref:log_prior"])
            rep["S2_log_prior"] = dict(scene=lp, ref=lp_ref,
                                       absdiff=abs(lp - lp_ref),
                                       tol=S2_TOL, ok=abs(lp - lp_ref) <= S2_TOL)
            g = float(gamma_of(pm, z[None, :])[0])
            rep["S4_gamma_column"] = dict(
                scene=g, ref=float(labeled["mass.gamma"]),
                absdiff=abs(g - labeled["mass.gamma"]), tol=S4_TOL,
                ok=abs(g - labeled["mass.gamma"]) <= S4_TOL)
    worst1 = max(v["absdiff"] for v in s1.values())
    rep["S1_term_logL"] = dict(points=s1, worst=worst1, tol=S1_TOL,
                               ok=worst1 <= S1_TOL)

    # S3: prior-draw finiteness at production N (chunked forward evals)
    z0 = prior_draws(pm, N_FULL, SEED)
    lp0 = np.asarray(jax.jit(jax.vmap(logprior_fn))(jnp.asarray(z0)))
    llb = jax.jit(jax.vmap(loglik_fn))
    lls = np.concatenate([np.asarray(llb(jnp.asarray(z0[i:i + 128])))
                          for i in range(0, N_FULL, 128)])
    rep["S3_prior_draws"] = dict(
        n=N_FULL, seed=SEED, logprior_finite=bool(np.all(np.isfinite(lp0))),
        loglik_finite=bool(np.all(np.isfinite(lls))),
        loglik_min=float(lls.min()), loglik_max=float(lls.max()),
        gamma_prior_med=float(np.median(gamma_of(pm, z0))))
    hard_ok = (rep["S1_term_logL"]["ok"] and rep["S2_log_prior"]["ok"]
               and rep["S4_gamma_column"]["ok"]
               and rep["S3_prior_draws"]["logprior_finite"]
               and rep["S3_prior_draws"]["loglik_finite"])
    rep["sanity_ok"] = bool(hard_ok)

    # timing smoke: 2 stages at N_FULL (stage 0 carries jit compile)
    smoke_dir = DATA / "l0_smc_smoke"
    holder = []
    t0 = time.time()
    try:
        run_smc_logged(pm, N_FULL, SEED, max_stages=2, ckpt_dir=smoke_dir,
                       kern_holder=holder)
    except RuntimeError as e:  # the expected max_stages RuntimeError (or cap)
        for f in sorted(smoke_dir.glob("stage_*.npz")):
            f.unlink()
        rep["smoke_note"] = str(e)[:200]
    rep["smoke_wall_s"] = round(time.time() - t0, 1)
    rep["smoke_peak_mb"] = gpu_peak_mb()
    rep["smoke_trace"] = holder[0].trace if holder else []
    # steady-state stage estimate: stage-1 tune+mutate (stage 0 pays jit
    # compile; new n_int values later recompile — +20% margin covers it),
    # falling back to wall/2 if the trace is unavailable
    if holder and len(holder[0].trace) >= 2:
        r1 = holder[0].trace[1]
        t_stage = 1.2 * (r1["t_tune_s"] + r1["t_mutate_s"])
    else:
        t_stage = rep["smoke_wall_s"] / 2.0
    rep["t_stage_est_s"] = round(t_stage, 1)
    proj_h = t_stage * EXPECTED_STAGES / 3600.0
    rep["projected_hours_at_N512"] = round(proj_h, 2)
    oom = "RESOURCE_EXHAUSTED" in rep.get("smoke_note", "")
    rep["oom_at_N512"] = bool(oom)
    n_decision = N_REDUCED if (oom or proj_h > PROJ_CAP_H) else N_FULL
    rep["N_decision"] = int(n_decision)
    rep["decision_rule"] = (f"oom={oom}, proj {proj_h:.2f} h vs cap "
                            f"{PROJ_CAP_H} h -> N={n_decision} "
                            "(checkpoint halving rule)")

    (DATA / "l0_sanity_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    if not hard_ok:
        sys.exit(1)


def stage_run():
    sane = json.loads((DATA / "l0_sanity_report.json").read_text())
    if not sane.get("sanity_ok"):
        raise RuntimeError("sanity gates not passed — refusing to run")
    n = int(os.environ.get("L0_N_OVERRIDE", sane["N_decision"]))
    remat = os.environ.get("L0_REMAT") == "1"
    if remat:  # ledgered deviation: its exactness gate must have passed
        rg = json.loads((DATA / "l0_remat_gate.json").read_text())
        if not rg.get("ok"):
            raise RuntimeError("remat exactness gate not passed")
    refs = load_refs()
    pm, mv, _ = build_pm("v2d", refs, diagonal=True)

    t0 = time.time()
    partial = False
    err_note = ""
    holder = []
    try:
        res, kern, z0 = run_smc_logged(pm, n, SEED, max_stages=400,
                                       kern_holder=holder)
    except WallCapReached as e:
        partial, err_note = True, str(e)
        res, kern = None, (holder[0] if holder else None)
    wall_h = (time.time() - t0) / 3600.0

    if partial:
        # newest checkpoint = the partial state; no gate math here
        summary = dict(status="PARTIAL_WALL_CAP", note=err_note,
                       wall_h=round(wall_h, 3), N=n, seed=SEED,
                       peak_mb=gpu_peak_mb(),
                       trace=(kern.trace if kern else []))
        (DATA / "l0_arbitration_smc.json").write_text(
            json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        sys.exit(3)

    Z = np.asarray(res["particles"], dtype=np.float64)
    gam = gamma_of(pm, Z)
    np.savez(DATA / "l0_arbitration_smc.npz",
             particles=Z, gamma=gam,
             lambda_schedule=np.asarray(res["lambda_schedule"]),
             ess_trace=np.asarray(res["ess_trace"]),
             unique_trace=np.asarray(res["unique_particle_trace"]),
             accept_trace=np.asarray(res["accept_trace"]),
             step_size_trace=np.asarray(res["step_size_trace"]),
             n_int_trace=np.asarray(res["n_int_trace"]),
             logZ=np.float64(res["logZ"]),
             logZ_boot_sigma=np.float64(res["logZ_boot_sigma"]),
             logZ_increments=np.asarray(res["logZ_increments"]))
    summary = dict(
        status="COMPLETE", N=n, seed=SEED, dim=int(Z.shape[1]),
        remat_basis=remat,
        kernel=res["kernel"], n_stages=int(res["n_stages"]),
        reached_lambda1=True,
        logZ=float(res["logZ"]), logZ_boot_sigma=float(res["logZ_boot_sigma"]),
        grad_evals=res["grad_evals"], n_logp=int(res["n_logp"]),
        wall_h=round(wall_h, 3), peak_mb=gpu_peak_mb(),
        final_ess=float(res["ess_trace"][-1]),
        final_unique=int(res["unique_particle_trace"][-1]),
        trace=kern.trace,
        note="raw run record — gate math happens in `harvest` AFTER plots")
    (DATA / "l0_arbitration_smc.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "trace"},
                     indent=2))


def stage_resume():
    """Resume the production SMC from the newest stage checkpoint (ledgered
    deviation, research/checkpoints_l0.md 2026-07-16 03:10 PT).

    The stage loop below is copied WITH ATTRIBUTION from
    cgl2.samplers.common.run_tempered_smc (the validated driver) — identical
    per-stage semantics and frozen constants; differences (all recorded):
    (1) starts from (z, lam) of the newest checkpoint instead of prior draws;
    (2) fresh jax key chain (PRNGKey(seed*1000+start_stage)) — tempered-SMC
    invariance does not depend on key continuation; (3) logZ accumulates only
    post-resume increments -> stored as logZ_partial_from_resume and NOT
    quotable (the pre-registered gate does not use logZ); (4) no bootstrap.
    """
    import jax as _jax
    import jax.numpy as _jnp
    from scipy.linalg import solve_triangular

    sane = json.loads((DATA / "l0_sanity_report.json").read_text())
    if not sane.get("sanity_ok"):
        raise RuntimeError("sanity gates not passed — refusing to run")
    n = int(os.environ.get("L0_N_OVERRIDE", sane["N_decision"]))
    refs = load_refs()
    pm, mv, _ = build_pm("v2d", refs, diagonal=True)
    logprior_fn, loglik_fn = make_closures(pm)

    cks = sorted(CKPT_DIR.glob("stage_*.npz"))
    if not cks:
        raise RuntimeError("no stage checkpoints to resume from")
    last = np.load(cks[-1])
    z = np.asarray(last["z"], dtype=np.float64)
    lam = float(last["lam"])
    eps_warm = float(last["eps"])
    start_stage = len(cks)
    if z.shape[0] != n:
        raise RuntimeError(f"checkpoint N={z.shape[0]} != requested {n}")
    dim = z.shape[1]
    print(f"[l0-resume] from stage {start_stage - 1}: lam={lam:.6e}, "
          f"eps={eps_warm:.4f}, N={n}", flush=True)

    inner = smc_micro.make_kernel("mams",
                                  num_mcmc_steps=smc_micro.NUM_MCMC_STEPS)
    kern = LoggingKernel(inner, CKPT_DIR, WALL_CAP_H)
    kern.stage = start_stage
    kern.build(logprior_fn, loglik_fn, dim)
    loglik_batch = _jax.jit(_jax.vmap(loglik_fn))
    key = _jax.random.PRNGKey(SEED * 1000 + start_stage)
    rng = np.random.default_rng(20260716 + start_stage)
    target_ess = smc_micro.TARGET_ESS
    max_stages = 400

    logz_partial = 0.0
    lambdas, ess_trace, uniq_trace = [], [], []
    acc_trace, eps_trace, nint_trace = [], [], []
    t0 = time.time()
    n_stages = start_stage
    status = "COMPLETE"
    # ---- loop copied with attribution from common.run_tempered_smc ---------
    try:
        while lam < 1.0 and n_stages < max_stages:
            ll = np.asarray(loglik_batch(_jnp.asarray(z)), dtype=np.float64)
            if not np.all(np.isfinite(ll)):
                raise RuntimeError("non-finite loglik at resume stage")
            delta = common.solve_delta_lambda(ll, lam, target_ess)
            lam_new = min(1.0, lam + delta)
            logz_partial += common.logmeanexp(delta * ll)
            logw = delta * ll
            w = np.exp(logw - logw.max())
            w = w / w.sum()
            ess_trace.append(common.weight_ess(w))
            mu, cov = common.weighted_mean_cov(z, w)
            cov_reg = common.regularize_cov(cov, common.weight_ess(w))
            chol = np.linalg.cholesky(cov_reg)
            key, k_u = _jax.random.split(key)
            idx = common.systematic_resample(w, float(_jax.random.uniform(k_u)))
            uniq_trace.append(int(len(np.unique(idx))))
            z = z[idx]
            mu_j, chol_j = _jnp.asarray(mu), _jnp.asarray(chol)
            U = _jnp.asarray(solve_triangular(chol, (z - mu[None, :]).T,
                                              lower=True).T)
            key, k_pilot, k_mut = _jax.random.split(key, 3)
            pilot_idx = rng.choice(n, size=min(64, n), replace=False)
            tuning = kern.tune(k_pilot, lam_new, mu_j, chol_j, U[pilot_idx],
                               eps_warm)
            eps_warm = tuning.step_size
            U_new, stats = kern.mutate(k_mut, lam_new, mu_j, chol_j, U, tuning)
            z = np.asarray(mu[None, :]
                           + np.asarray(U_new, dtype=np.float64) @ chol.T)
            acc_trace.append(float(stats["accept_mean"]))
            eps_trace.append(float(tuning.step_size))
            nint_trace.append(int(tuning.n_int))
            lam = lam_new
            lambdas.append(lam)
            n_stages += 1
    except WallCapReached as e:
        status = "PARTIAL_WALL_CAP"
        print(f"[l0-resume] {e}", flush=True)
    if lam < 1.0 and status == "COMPLETE":
        status = "PARTIAL_MAX_STAGES"

    if status == "COMPLETE":
        gam = gamma_of(pm, z)
        # prepend the pre-resume trace segments from the checkpoints
        pre = [np.load(f) for f in cks]
        lam_pre = [float(p["lam"]) for p in pre]
        np.savez(DATA / "l0_arbitration_smc.npz",
                 particles=z, gamma=gam,
                 lambda_schedule=np.asarray(lam_pre + lambdas),
                 ess_trace=np.asarray([np.nan] * start_stage + ess_trace),
                 unique_trace=np.asarray([-1] * start_stage + uniq_trace),
                 accept_trace=np.asarray([np.nan] * start_stage + acc_trace),
                 step_size_trace=np.asarray([float(p["eps"]) for p in pre]
                                            + eps_trace),
                 n_int_trace=np.asarray([int(p["n_int"]) for p in pre]
                                        + nint_trace),
                 logZ=np.float64(np.nan),
                 logZ_boot_sigma=np.float64(np.nan),
                 logZ_increments=np.asarray([]))
        summary = dict(
            status="COMPLETE", N=n, seed=SEED, dim=dim,
            remat_basis=os.environ.get("L0_REMAT") == "1",
            kernel="mams+log(resumed)",
            n_stages=int(n_stages), reached_lambda1=True,
            resumed_from_stage=start_stage - 1,
            logZ=None, logZ_boot_sigma=None,
            logZ_partial_from_resume=float(logz_partial),
            logZ_note="NOT QUOTABLE — pre-resume increments lost (ledgered "
                      "resume deviation); gate does not use logZ",
            wall_h_resume_leg=round((time.time() - t0) / 3600.0, 3),
            peak_mb=gpu_peak_mb(),
            final_ess=float(ess_trace[-1]),
            final_unique=int(uniq_trace[-1]),
            trace=kern.trace,
            note="raw run record — gate math happens in `harvest` AFTER plots")
        (DATA / "l0_arbitration_smc.json").write_text(
            json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k != "trace"},
                         indent=2))
    else:
        summary = dict(status=status, N=n, seed=SEED,
                       resumed_from_stage=start_stage - 1,
                       lambda_reached=lam,
                       wall_h_resume_leg=round((time.time() - t0) / 3600.0, 3),
                       trace=kern.trace)
        (DATA / "l0_arbitration_smc.json").write_text(
            json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        sys.exit(3)


def stage_rematgate():
    """Pre-registered exactness gate for the remat deviation (checkpoint
    2026-07-15 22:50): term logL at z_ref + 3 perts, remat ON vs OFF,
    abs diff <= 1e-10. Writes data/l0_remat_gate.json."""
    refs = load_refs()
    pm_off, mv, _ = build_pm("v2d", refs, diagonal=True, remat=False)
    pm_on, mv_on, _ = build_pm("v2d", refs, diagonal=True, remat=True)
    ll_off = jax.jit(make_closures(pm_off)[1])
    ll_on = jax.jit(make_closures(pm_on)[1])
    out, worst = {}, 0.0
    for pname in ("z_ref", "z_pert0", "z_pert1", "z_pert2"):
        labeled = param_map.labeled_from_vec46(refs[f"v2d:{pname}:x46"])
        z = jnp.asarray(param_map.scene_z_from_old_labels(mv.model, labeled))
        a = float(ll_off(z))
        b = float(ll_on(jnp.asarray(
            param_map.scene_z_from_old_labels(mv_on.model, labeled))))
        out[pname] = dict(remat_off=a, remat_on=b, absdiff=abs(a - b))
        worst = max(worst, abs(a - b))
    rep = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               stage="rematgate", tol=1e-10, worst=worst,
               ok=bool(worst <= 1e-10), points=out)
    (DATA / "l0_remat_gate.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    if not rep["ok"]:
        sys.exit(1)


def stage_v3bsmoke():
    refs = load_refs()
    pm, mv, _ = build_pm("v3b", refs, diagonal=False, checkpoint=True)
    logprior_fn, loglik_fn = make_closures(pm)
    base_mb = gpu_peak_mb()

    def logdens(z):        # what the SMC mutation differentiates (lam=1)
        return logprior_fn(z) + loglik_fn(z)

    vg = jax.jit(jax.vmap(jax.value_and_grad(logdens)))
    z8 = prior_draws(pm, 8, SEED)
    v, g = vg(jnp.asarray(z8))
    v.block_until_ready()
    peak8 = gpu_peak_mb()
    ok8 = bool(np.all(np.isfinite(np.asarray(v)))
               and np.all(np.isfinite(np.asarray(g))))
    z16 = prior_draws(pm, 16, SEED + 1)
    v16, g16 = vg(jnp.asarray(z16))
    v16.block_until_ready()
    peak16 = gpu_peak_mb()
    mb_pp_slope = max(peak16 - peak8, 0.0) / 8.0
    mb_pp_upper = (peak16 - base_mb) / 16.0
    mb_pp = mb_pp_slope if mb_pp_slope > 1.0 else mb_pp_upper
    total_mb = 23034.0
    n_feasible = int(max(0, np.floor((total_mb * 0.90 - peak8 + 8 * mb_pp)
                                     / max(mb_pp, 1e-9))))
    rep = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        stage="v3bsmoke", device=str(jax.local_devices()[0]),
        whitener="whitener_v3b.npz (ported, manifest sha 8816745c...)",
        checkpointed_whitening=True,
        baseline_peak_mb=round(base_mb, 1),
        peak_mb_at_8=round(peak8, 1), peak_mb_at_16=round(peak16, 1),
        finite_at_8=ok8,
        finite_at_16=bool(np.all(np.isfinite(np.asarray(v16)))
                          and np.all(np.isfinite(np.asarray(g16)))),
        mb_per_particle_slope=round(mb_pp_slope, 1),
        mb_per_particle_upper=round(mb_pp_upper, 1),
        mb_per_particle_used=round(mb_pp, 1),
        l4_total_mb=total_mb,
        n_particles_feasible_est=n_feasible,
        decision_rule="checkpoint: L0-G2 attempt iff n_feasible >= 96 AND "
                      "arbitration finished with >= 2.5 h margin; else defer "
                      "to Perlmutter with this measurement",
    )
    (DATA / "l0_v3b_memory_smoke.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


def stage_harvest():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(DATA / "l0_arbitration_smc.npz")
    gam = np.asarray(z["gamma"], dtype=np.float64)
    summ = json.loads((DATA / "l0_arbitration_smc.json").read_text())

    # ---------------- PLOTS FIRST (house rule) ----------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(gam, bins=48, color="#4477aa", alpha=0.85,
            label=f"scene-API diag v2d SMC (N={gam.size}, eq-weight)")
    ax.axvspan(*ANCHOR_CI68, color="#cc6677", alpha=0.25,
               label="foundry-i anchor 68% [1.400, 1.468]")
    ax.axvline(GAMMA_ANCHOR, color="#cc6677", lw=1.5)
    ax.axvline(float(np.median(gam)), color="#4477aa", ls="--", lw=1.5,
               label=f"scene median {np.median(gam):.4f}")
    ax.set_xlabel(r"$\gamma$ (EPL slope, constrained)")
    ax.set_ylabel("particles")
    ax.set_title("P3-L0 anchor arbitration: v2d native product, DIAGONAL "
                 "likelihood, scene API")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "l0_gamma_hist.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    lam = np.asarray(z["lambda_schedule"])
    axes[0, 0].plot(lam, marker="."); axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("lambda schedule"); axes[0, 0].set_xlabel("stage")
    axes[0, 1].plot(np.asarray(z["ess_trace"]), marker=".")
    axes[0, 1].set_title("incremental-weight ESS")
    axes[1, 0].plot(np.asarray(z["unique_trace"]), marker=".")
    axes[1, 0].set_title("unique particles after resample")
    axes[1, 1].plot(np.asarray(z["accept_trace"]), marker=".")
    axes[1, 1].set_title("MAMS acceptance"); axes[1, 1].axhline(0.9, ls=":")
    for a in axes.flat:
        a.set_xlabel("stage")
    fig.suptitle("P3-L0 SMC diagnostics (v2d diag, MAMS, prior-seeded)")
    fig.tight_layout()
    fig.savefig(FIGS / "l0_smc_traces.png", dpi=150)
    plt.close(fig)
    print(f"PLOTS WRITTEN FIRST: {FIGS/'l0_gamma_hist.png'}, "
          f"{FIGS/'l0_smc_traces.png'}", flush=True)

    # ---------------- gate math (pre-registered) ----------------
    med = float(np.median(gam))
    q16, q84 = (float(np.quantile(gam, q)) for q in (0.16, 0.84))
    sigma_scene = 0.5 * (q84 - q16)
    sigma_comb = float(np.hypot(SIGMA_ANCHOR, sigma_scene))
    dmed = med - GAMMA_ANCHOR
    within = abs(dmed) <= 2.0 * sigma_comb
    overlap = (q16 <= ANCHOR_CI68[1]) and (ANCHOR_CI68[0] <= q84)
    frac_hi = float(np.mean(gam > BASIN_SPLIT))
    # data-driven basin split (the fixed 1.9 v3b convention would miss a mode
    # near z_ref's 1.87): largest gap in sorted gamma, requiring a real gap
    # (>0.15) and >=2% mass on both sides; 1.9-frac stays as continuity stat
    gs = np.sort(gam)
    k0 = max(1, int(0.02 * gs.size))
    diffs = np.diff(gs)[k0 - 1: gs.size - k0]
    split, gapw = None, 0.0
    if diffs.size:
        j = int(np.argmax(diffs)) + k0 - 1
        gapw = float(gs[j + 1] - gs[j])
        if gapw > 0.15:
            split = float(0.5 * (gs[j] + gs[j + 1]))
    bimodal = split is not None
    basins = {}
    if bimodal:
        lo, hi = gam[gam <= split], gam[gam > split]
        for name, arr in (("low", lo), ("steep", hi)):
            basins[name] = dict(
                occupancy=float(arr.size / gam.size),
                med=float(np.median(arr)),
                q16=float(np.quantile(arr, 0.16)),
                q84=float(np.quantile(arr, 0.84)))
        dom = max(basins, key=lambda k: basins[k]["occupancy"])
        dmed_dom = basins[dom]["med"] - GAMMA_ANCHOR
        sig_dom = 0.5 * (basins[dom]["q84"] - basins[dom]["q16"])
        within_dom = abs(dmed_dom) <= 2.0 * np.hypot(SIGMA_ANCHOR, sig_dom)
        overlap_dom = (basins[dom]["q16"] <= ANCHOR_CI68[1]
                       and ANCHOR_CI68[0] <= basins[dom]["q84"])
    else:
        dom, within_dom, overlap_dom = "unimodal", within, overlap

    # worst-param shift between the final two stage ensembles
    cks = sorted(CKPT_DIR.glob("stage_*.npz"))
    worst_param = None
    if len(cks) >= 2:
        z_last = np.load(cks[-1])["z"]
        z_prev = np.load(cks[-2])["z"]
        sd = np.std(z_last, axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        shifts = np.abs(np.median(z_last, axis=0)
                        - np.median(z_prev, axis=0)) / sd
        worst_param = dict(max_shift_sigma=float(shifts.max()),
                           argmax=int(np.argmax(shifts)),
                           n_ckpt_stages=len(cks))

    gate = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        checkpoint="research/checkpoints_l0.md 2026-07-15 (pre-registered)",
        anchor=dict(gamma=GAMMA_ANCHOR, ci68=list(ANCHOR_CI68),
                    sigma=SIGMA_ANCHOR, source="foundry-i hmc_v13_v2d"),
        scene=dict(med=med, q16=q16, q84=q84, sigma=sigma_scene,
                   N=int(gam.size), seed=SEED,
                   n_stages=summ["n_stages"], logZ=summ.get("logZ"),
                   logZ_boot_sigma=summ.get("logZ_boot_sigma"),
                   logZ_note=summ.get("logZ_note", "")),
        gate_within_2sigma_comb=dict(
            dmed=dmed, sigma_comb=sigma_comb, band=2.0 * sigma_comb,
            ok=bool(within)),
        gate_ci68_overlap=dict(ok=bool(overlap)),
        frac_gamma_gt_1p9=frac_hi, bimodal=bool(bimodal),
        basin_split=split, basin_gap_width=gapw, basins=basins,
        dominant_basin=dom,
        dominant_within=bool(within_dom), dominant_overlap=bool(overlap_dom),
        worst_param_final_two_stages=worst_param,
        verdict=("ANCHOR STACK-ROBUST (prediction holds)"
                 if (within_dom and overlap_dom) else
                 "FALSIFIER FIRED — anchor NOT reproduced at posterior level"),
    )
    (DATA / "l0_gate_eval.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "sanity"
    {"sanity": stage_sanity, "run": stage_run, "resume": stage_resume,
     "rematgate": stage_rematgate,
     "v3bsmoke": stage_v3bsmoke, "harvest": stage_harvest}[stage]()
