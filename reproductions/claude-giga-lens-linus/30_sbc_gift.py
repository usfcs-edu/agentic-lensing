"""X2 SBC gift — first formal simulation-based calibration of the GIGALens
pipeline class (PLAN cross-pollination X2; Front D).

Design checkpoint: research/checkpoints_x2.md (written BEFORE any run; the
validity-precondition verdict, arm decision, settings ladder + budget rule,
and all gate thresholds are frozen there).

What this runs
--------------
N=64 prior-matched mock systems of the hundred_systems_GL2 config class
(EPL(50)+Shear mass + Sersic lens light + Sersic source, forward mode,
bg_rms=0.2 exp_time=100, 0.065"/80px/ss2, vendored assets/psf.npy), truth
drawn from the FITTING prior (make_default_prior re-expressed as scene
Components — cgl2.zoo._hs2_prior_components, P0 parity-audited) — the pivot
arm: the team's frozen 100SystemsStandard80px.npz was generated from the
NARROWER simulation prior (their t13_resim.py provenance + gl2_sersic.py
gl2_simulation_prior vs gl2_inference_prior), so SBC on it is invalid by
construction, and the npz is absent from the local mirror anyway.

System construction is REPLICATED (not imported) from cgl2/zoo.py::build_hs2
with the index range extended 0..63 (file-ownership rule: zoo.py untouched);
i=0..7 reproduce the certified hs2_sys{0..7} targets bit-for-bit (same
PRNGKey(1000+i) truth, same np.random.seed(3000+i) noise recipe).

Pipeline per system = the GIGALens pipeline class on OUR certified port
(vendored scene ModellingSequence; NO gigalens_research import):
multi-start MAP (adabelief 1e-2, b1=.95, b2=.99[, nesterov]) ->
full-rank Gaussian SVI (adabelief 1e-4) ->
SVI-cov preconditioned HMC (init_eps .3, init_l 3, max_leapfrog 30). f64.
Optimizer/stage defaults mirror GIGALens-Code@eb2a09b6
src/gigalens_research/inference_utils/pipeline.py (MAPStage/SVIStage/
HMCStage + _default_*_optimizer) — values copied with attribution, code not
imported. Stage seeds VARY per system (MAP i, SVI 10000+i, HMC 20000+i);
their campaign's fixed seed=0 is a documented deviation we do not copy (SBC
needs independent pipeline randomness across systems).

Rank machinery (thin_indices, sbc_rank, rank_uniformity_chi2) is copied
VERBATIM with attribution from the OLD campaign's
../claude-giga-lens/cgl/e1.py (the E1c SBC harness conventions).

Physicality arms: vacuous on this substrate (measured from vendored code —
gigalens/physicality.py validates at LensModel construction and diagnoses
posteriors, but never enters log_prob). ONE arm is run; construction-time
physicality warnings and validate_posterior_samples fractions are captured
as data per system.

Usage (cgl2 venv, GIGALENS_X64=1, CUDA_DEVICE_ORDER=PCI_BUS_ID):
  python 30_sbc_gift.py smoke                 # sys0 timing at REDUCED rung (+ guard)
  python 30_sbc_gift.py decide                # apply the pre-stated budget rule
  python 30_sbc_gift.py run --start 0 --count 16   # worker (one GPU)
  python 30_sbc_gift.py harvest               # figs FIRST, then data/x2_sbc.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = DATA / "x2_sbc_runs"
FIGS = ROOT / "figs"

OLD_E1_REPORT = (ROOT.parent / "claude-giga-lens" / "data" / "e1_report.json")

N_SYSTEMS = 64
N_USE = 127          # E1c convention: ranks in 0..127, 8 bins of 16
N_BINS = 8

# Settings ladder (frozen in the checkpoint; "reduced_a16" = Deviation 2 —
# the A16-memory-forced revision of the reduced rung, ledgered in
# research/checkpoints_x2.md before the batch: map_samples 500->128,
# n_vi 250->128, everything else unchanged. "reference" (their campaign.yaml)
# is INFEASIBLE-ON-A16: map n_samples=2000 f64 OOMs by ~6x).
SETTINGS = {
    "reference": dict(map_steps=1000, map_samples=2000, svi_steps=5000,
                      n_vi=1000, hmc_chains=64, hmc_burnin=500,
                      hmc_results=1500),
    "reduced": dict(map_steps=350, map_samples=500, svi_steps=1500,
                    n_vi=250, hmc_chains=32, hmc_burnin=300,
                    hmc_results=750),
    "reduced_a16": dict(map_steps=350, map_samples=128, svi_steps=1500,
                        n_vi=128, hmc_chains=32, hmc_burnin=300,
                        hmc_results=750),
}
# Per-stage work ratios reference/reduced for the projection rule
# (steps x batch; HMC: (burnin+results) x chains).
WORK_RATIO = dict(
    map=(1000 * 2000) / (350 * 500),
    svi=(5000 * 1000) / (1500 * 250),
    hmc=((500 + 1500) * 64) / ((300 + 750) * 32),
)
BUDGET_A16H = 36.0        # commit (cap ~40)
N_GPUS = 4
MARGIN = 1.15             # compile-amortization margin (checkpoint rule)


# --------------------------------------------------------------------------- #
# rank machinery — copied VERBATIM with attribution from
# ../claude-giga-lens/cgl/e1.py (claude-giga-lens campaign, E1c SBC harness)
# --------------------------------------------------------------------------- #
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
    from scipy import stats as sstats
    ranks = np.asarray(ranks, dtype=int)
    n_vals = n_use + 1
    assert n_vals % n_bins == 0, "bins must divide the rank support"
    edges = np.arange(n_bins + 1) * (n_vals // n_bins)
    obs, _ = np.histogram(ranks, bins=edges)
    exp = ranks.size / n_bins
    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    p = float(sstats.chi2.sf(chi2, df=n_bins - 1))
    return chi2, p, obs.tolist()


# --------------------------------------------------------------------------- #
# extra pure-numpy stats (this campaign)
# --------------------------------------------------------------------------- #
def rank_location_z(ranks, n_use=127):
    """Location z of the mean rank vs discrete-uniform{0..n_use}:
    z = (mean - n_use/2) / sqrt(Var_unif / N), Var_unif = ((n_use+1)^2-1)/12."""
    r = np.asarray(ranks, dtype=np.float64)
    var_u = ((n_use + 1) ** 2 - 1) / 12.0
    return float((r.mean() - n_use / 2.0) / np.sqrt(var_u / r.size))


# --------------------------------------------------------------------------- #
# system builder — REPLICATED with attribution from cgl2/zoo.py::build_hs2
# (P0 task #13), index range extended to 0..63. zoo.py itself is untouched
# (X2 file-ownership rule).
# --------------------------------------------------------------------------- #
def build_system(sys_idx: int):
    from lenstronomy.Util import image_util

    import jax

    from gigalens.jax.scene import LensModel, Plane
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    from gigalens.jax.scene_simulator import SceneSimulator
    from gigalens.simulator import SimulatorConfig

    from cgl2 import paths
    from cgl2.zoo import (HS2_BACKGROUND_RMS, HS2_DELTA_PIX, HS2_EXP_TIME,
                          HS2_NUM_PIX, HS2_SUPERSAMPLE, _hs2_prior_components)

    sys_idx = int(sys_idx)
    if not (0 <= sys_idx < N_SYSTEMS):
        raise ValueError(f"sys_idx must be 0..{N_SYSTEMS - 1}")

    epl, shear, lens_light, source = _hs2_prior_components()
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        model = LensModel([
            Plane(mass=[epl, shear], light=[lens_light]),
            Plane(light=[source]),
        ])
    phys_warnings = [str(w.message) for w in wrec]

    kernel = np.load(paths.VENDOR_DIR / "src" / "gigalens" / "assets"
                     / "psf.npy").astype(np.float64)
    cfg = SimulatorConfig(delta_pix=HS2_DELTA_PIX, num_pix=HS2_NUM_PIX,
                          supersample=HS2_SUPERSAMPLE, kernel=kernel)

    key = jax.random.PRNGKey(1000 + sys_idx)
    z_true = model.bijector.inverse(model.prior.sample(seed=key))
    params_true = model.constrained(z_true)
    sim = SceneSimulator(model, cfg, sees=[lens_light, source])
    img = np.asarray(sim.simulate(params_true), dtype=np.float64)
    np.random.seed(3000 + sys_idx)
    img = (img + image_util.add_poisson(img, exp_time=HS2_EXP_TIME)
           + image_util.add_background(img, sigma_bkd=HS2_BACKGROUND_RMS))

    ds = ImageData(img, cfg, background_rms=HS2_BACKGROUND_RMS,
                   exp_time=HS2_EXP_TIME, sees="all")
    pm = ProbModel(model, ds, mode="forward")

    z_true = np.asarray(z_true, dtype=np.float64).ravel()
    assert z_true.size == int(model.num_free_params), \
        (z_true.size, int(model.num_free_params))
    return pm, model, z_true, phys_warnings


def _patch_shard_map_check():
    """DOCUMENTED DEVIATION (the only vendored-behavior change, applied at
    runtime — the vendor tree stays UNPATCHED): under jax 0.6.2 the vendored
    MAP/SVI shard_map wrappers fail in the backward pass with 'cotangent type
    does not match function output ... {V:device}' — the varying-mode
    replication CHECK (check_vma) rejects the FFT-convolution cotangent
    (complex128 rfft buffers). The vendored code targets the jax 0.10 pcast
    semantics (its own _pvary shim). Fix: re-bind
    gigalens.jax.inference._shard_map with check_vma=False — this disables
    ONLY the static replication checker; the computation, sharding, RNG
    streams, and math are unchanged (single-device mesh here anyway).
    Verified: MAP/SVI/HMC run end-to-end and MAP loss decreases (smoke)."""
    import gigalens.jax.inference as gji

    orig = gji._shard_map
    if getattr(gji, "_x2_shard_map_patched", False):
        return

    def _sm_nocheck(f, **kw):
        kw.setdefault("check_vma", False)
        return orig(f, **kw)

    gji._shard_map = _sm_nocheck
    gji._x2_shard_map_patched = True


def _optimizers():
    """MAP/SVI optimizers, values copied with attribution from
    GIGALens-Code@eb2a09b6 inference_utils/pipeline.py::_default_*_optimizer."""
    import optax
    try:
        map_opt = optax.adabelief(1e-2, b1=0.95, b2=0.99, nesterov=True)
        map_id = "adabelief_1e-2_b1_0.95_b2_0.99_nesterov"
    except TypeError:
        map_opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
        map_id = "adabelief_1e-2_b1_0.95_b2_0.99"
    svi_opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
    return map_opt, map_id, svi_opt, "adabelief_1e-4_b1_0.95_b2_0.99"


# --------------------------------------------------------------------------- #
# one system end-to-end
# --------------------------------------------------------------------------- #
def run_one(i: int, s: dict, out_path: Path, do_guard: bool = False) -> dict:
    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp

    _patch_shard_map_check()
    from gigalens.jax.inference import ModellingSequence

    t_build0 = time.perf_counter()
    pm, model, z_true, phys_warnings = build_system(i)
    names = list(pm.z_param_names or [])
    dim = z_true.size
    seq = ModellingSequence.from_scene(pm)
    # SVI runs on the remat twin (exact recomputation — the same mechanism the
    # vendored MAP self-applies at ss<3); Deviation 2, A16 memory budget.
    pm_svi = pm.with_map_remat() if hasattr(pm, "with_map_remat") else pm
    seq_svi = ModellingSequence.from_scene(pm_svi)
    map_opt, map_id, svi_opt, svi_id = _optimizers()
    wall_build = time.perf_counter() - t_build0

    # chi^2/pixel at truth (render+noise consistency sanity; not a gate)
    lp_t, chisq_t = pm.log_prob(jnp.asarray(z_true)[None])
    chisq_truth = float(np.asarray(chisq_t).ravel()[0])
    lp_truth = float(np.asarray(lp_t).ravel()[0])

    t0 = time.perf_counter()
    map_samples, map_lps, map_chisqs = seq.MAP(
        map_opt, start=None, n_samples=s["map_samples"],
        num_steps=s["map_steps"], seed=i, output_type="best_step",
        pbar_interval=0)
    map_lps = np.asarray(map_lps)
    best = int(np.nanargmax(map_lps))
    z_best = np.asarray(map_samples)[best]
    wall_map = time.perf_counter() - t0

    t0 = time.perf_counter()
    qz, svi_loss = seq_svi.SVI(jnp.asarray(z_best), svi_opt, n_vi=s["n_vi"],
                               init_scales=1e-3, num_steps=s["svi_steps"],
                               seed=10000 + i, pbar_interval=0)
    svi_loss = np.asarray(svi_loss).ravel()
    wall_svi = time.perf_counter() - t0

    t0 = time.perf_counter()
    hmc = seq.HMC(qz, init_eps=0.3, init_l=3, n_hmc=s["hmc_chains"],
                  num_burnin_steps=s["hmc_burnin"],
                  num_results=s["hmc_results"], max_leapfrog_steps=30,
                  seed=20000 + i, pbar_interval=0)
    samples = np.asarray(hmc, dtype=np.float64)
    wall_hmc = time.perf_counter() - t0
    # canonicalize to (chains, draws, dim): the vendored HMC returns
    # (num_results, num_devices, n_hmc_per_device, dim) on this stack
    # (measured; their comment agrees); tolerate a 3-D (chains, draws, dim).
    if samples.ndim == 4:
        d_, dev_, ch_, dim_ = samples.shape
        samples = samples.reshape(d_, dev_ * ch_, dim_).transpose(1, 0, 2)
    assert samples.ndim == 3 and samples.shape[2] == dim, samples.shape

    # diagnostics (draws, chains, dim) — the hundred-systems metric convention
    x = jnp.asarray(np.transpose(samples, (1, 0, 2)))
    rhat = np.asarray(tfp.mcmc.potential_scale_reduction(
        x, independent_chain_ndims=1), dtype=np.float64)
    ess_pc = np.asarray(tfp.mcmc.effective_sample_size(
        x, filter_beyond_positive_pairs=True), dtype=np.float64)
    ess = ess_pc.sum(axis=0) if ess_pc.ndim == 2 else ess_pc

    pooled = samples.reshape(-1, dim)             # chains-major flatten
    ranks = np.array([sbc_rank(pooled[:, k], z_true[k], N_USE)
                      for k in range(dim)], dtype=int)
    post_mean = pooled.mean(0)
    post_std = pooled.std(0, ddof=1)

    # physicality posterior diagnosis (their layer, as data; never a gate here)
    try:
        from gigalens import physicality
        thin = pooled[thin_indices(pooled.shape[0], 512)]
        rep = physicality.validate_posterior_samples(
            model, model.constrained(jnp.asarray(thin)))
        try:
            phys_post = str(rep.summary())
        except Exception:
            phys_post = str(rep)
    except Exception as e:  # diagnosis failure is reported, not fatal
        phys_post = f"DIAGNOSIS-ERROR: {e!r}"

    guard = None
    if do_guard:
        guard = _monotone_guard(model, pooled, z_true, names)

    meta = dict(
        sys_idx=i, dim=dim, names=names, settings=s,
        map_optimizer=map_id, svi_optimizer=svi_id,
        seeds=dict(truth=1000 + i, noise=3000 + i, map=i, svi=10000 + i,
                   hmc=20000 + i),
        chisq_truth=chisq_truth, lp_truth=lp_truth,
        map_best_lp=float(map_lps[best]),
        map_best_chisq=float(np.asarray(map_chisqs)[best]),
        svi_final_loss=float(svi_loss[-1]),
        wall=dict(build=wall_build, map=wall_map, svi=wall_svi, hmc=wall_hmc,
                  total=wall_build + wall_map + wall_svi + wall_hmc),
        max_rhat=float(np.max(rhat)), min_ess=float(np.min(ess)),
        phys_construction_warnings=phys_warnings,
        phys_posterior=phys_post,
        guard=guard,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, z_true=z_true, ranks=ranks, rhat=rhat, ess=ess,
             post_mean=post_mean, post_std=post_std,
             meta=json.dumps(meta))
    print(f"[x2] sys{i:02d} done: wall {meta['wall']['total']:.1f}s "
          f"(map {wall_map:.1f} svi {wall_svi:.1f} hmc {wall_hmc:.1f}), "
          f"max_rhat {meta['max_rhat']:.3f}, min_ess {meta['min_ess']:.0f}, "
          f"chisq_truth {chisq_truth:.3f}", flush=True)
    return meta


def _monotone_guard(model, pooled, z_true, names):
    """Checkpoint guard: per-coordinate ranks are identical in constrained
    space (every hs2 param is a scalar prior with a strictly monotone
    event-space bijector). Probe the z->leaf mapping by perturbation, check
    the direction at two base points, and re-compute the ranks in constrained
    space for comparison."""
    import jax
    import jax.numpy as jnp

    dim = z_true.size

    def leaves_of(z):
        flat, _ = jax.tree_util.tree_flatten(
            model.constrained(jnp.asarray(z, dtype=jnp.float64)))
        return np.array([float(np.asarray(l).ravel()[0]) for l in flat])

    base_a = np.zeros(dim)
    base_b = 0.3 * np.ones(dim)
    la, lb = leaves_of(base_a), leaves_of(base_b)
    k2leaf, k2sign = {}, {}
    for k in range(dim):
        for base, lbase in ((base_a, la), (base_b, lb)):
            zp = base.copy()
            zp[k] += 0.25
            dl = leaves_of(zp) - lbase
            idx = np.nonzero(np.abs(dl) > 1e-12)[0]
            assert idx.size == 1, (k, idx)
            sign = float(np.sign(dl[idx[0]]))
            if k in k2leaf:
                assert k2leaf[k] == int(idx[0]) and k2sign[k] == sign, k
            k2leaf[k], k2sign[k] = int(idx[0]), sign
    # constrained-space ranks on the SAME thinned subset
    n_chk = min(pooled.shape[0], 2048)
    sub = pooled[thin_indices(pooled.shape[0], n_chk)]
    leaves_sub = np.stack([leaves_of(z) for z in sub])   # (n_chk, n_leaves)
    leaves_true = leaves_of(z_true)
    mismatches = []
    for k in range(dim):
        m = k2leaf[k]
        rz = sbc_rank(sub[:, k], z_true[k], N_USE)
        if k2sign[k] > 0:
            rc = sbc_rank(leaves_sub[:, m], leaves_true[m], N_USE)
        else:
            rc = N_USE - sbc_rank(-leaves_sub[:, m], -leaves_true[m], N_USE)
        if rz != rc:
            mismatches.append(dict(k=k, name=names[k] if names else None,
                                   rank_z=rz, rank_constrained=rc))
    return dict(all_increasing=all(v > 0 for v in k2sign.values()),
                n_checked=n_chk, mismatches=mismatches,
                passed=len(mismatches) == 0)


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def mode_smoke():
    RUNS.mkdir(parents=True, exist_ok=True)
    meta = run_one(0, SETTINGS["reduced_a16"], RUNS / "smoke_sys00.npz",
                   do_guard=True)
    w = meta["wall"]
    per_sys = w["total"]
    proj = {"reduced_a16": dict(per_system_s=per_sys,
                                projected_a16h=per_sys * N_SYSTEMS * MARGIN
                                / 3600.0),
            "reference": dict(note="INFEASIBLE-ON-A16 (Deviation 2: "
                                   "map n_samples=2000 f64 OOMs; 500 already "
                                   "asked 23.8 GB of 15.3)"),
            "reduced": dict(note="OOM at map_samples=500 (measured); "
                                 "superseded by reduced_a16")}
    out = dict(smoke_meta=meta, projection=proj, budget_a16h=BUDGET_A16H,
               rule=("reduced_a16 at N=64 if proj<=budget else shrink N to "
                     "largest multiple of 8 that fits"))
    path = RUNS / "smoke_projection.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(projection=proj, guard=meta["guard"],
                          budget_a16h=BUDGET_A16H), indent=1))
    print(f"[x2] smoke projection written: {path}")


def mode_decide():
    proj = json.loads((RUNS / "smoke_projection.json").read_text())
    p_a16 = proj["projection"]["reduced_a16"]["projected_a16h"]
    per = proj["projection"]["reduced_a16"]["per_system_s"]
    if p_a16 <= BUDGET_A16H:
        n = N_SYSTEMS
    else:
        n = int((BUDGET_A16H * 3600.0 / (per * MARGIN)) // 8 * 8)
    decision = dict(rung="reduced_a16", n_systems=n,
                    settings=SETTINGS["reduced_a16"],
                    projections=proj["projection"],
                    budget_a16h=BUDGET_A16H)
    (RUNS / "settings_decision.json").write_text(json.dumps(decision, indent=1))
    print(json.dumps(decision, indent=1))


def mode_run(start: int, count: int):
    decision = json.loads((RUNS / "settings_decision.json").read_text())
    s = decision["settings"]
    n = decision["n_systems"]
    for i in range(start, min(start + count, n)):
        out = RUNS / f"sys{i:02d}.npz"
        if out.exists():
            print(f"[x2] sys{i:02d} exists, skipping", flush=True)
            continue
        try:
            run_one(i, s, out, do_guard=(i == 0))
        except Exception as e:
            # fail-loud per system; the harvest reports missing systems
            err = RUNS / f"sys{i:02d}.ERROR"
            err.write_text(repr(e))
            print(f"[x2] sys{i:02d} FAILED: {e!r}", flush=True)


# --------------------------------------------------------------------------- #
# harvest: figs FIRST (plots-before-metrics), then data/x2_sbc.json
# --------------------------------------------------------------------------- #
HEALTH_RHAT, HEALTH_ESS = 1.05, 200.0

# param-class map (z_param_names look like planes/<i>/<mass|light>/<j>/<param>)
def _param_class(name: str) -> str:
    if "/mass/" in name:
        return "mass"
    if name.startswith("planes/0") and "/light/" in name:
        return "lens_light"
    return "source"


def _short(name: str) -> str:
    parts = name.split("/")
    if parts[0] == "planes" and len(parts) >= 5:
        plane, kind, comp, par = parts[1], parts[2], parts[3], parts[4]
        if kind == "mass":
            block = "EPL" if comp == "0" else "SHR"
        elif plane == "0":
            block = "LL"
        else:
            block = "SRC"
        return f"{block}.{par}"
    return name


def mode_harvest():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(RUNS.glob("sys[0-9][0-9].npz"))
    if not files:
        raise SystemExit("no completed systems in " + str(RUNS))
    metas, ranks, rhats, esss = [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=False)
        metas.append(json.loads(str(d["meta"])))
        ranks.append(d["ranks"])
        rhats.append(d["rhat"])
        esss.append(d["ess"])
    ranks = np.stack(ranks)          # (N, dim)
    rhats = np.stack(rhats)
    esss = np.stack(esss)
    names = metas[0]["names"]
    dim = ranks.shape[1]
    n_fits = ranks.shape[0]
    healthy = np.array([(m["max_rhat"] <= HEALTH_RHAT)
                        and (m["min_ess"] >= HEALTH_ESS) for m in metas])
    errors = sorted(p.name for p in RUNS.glob("sys*.ERROR"))

    # ---------------- figures FIRST ----------------
    blue = "#4269D0"        # single-series hue (dataviz palette, cat step 1)
    gray = "#9AA1A9"
    red = "#C4443C"         # status: FAIL flag only
    exp = n_fits / N_BINS
    # 99% band per bin count under uniformity (binomial)
    from scipy import stats as sstats
    lo99, hi99 = sstats.binom.ppf([0.005, 0.995], n_fits, 1.0 / N_BINS)

    order = np.argsort([0 if _param_class(nm) == "mass"
                        else (1 if _param_class(nm) == "lens_light" else 2)
                        for nm in names], kind="stable")
    ncol, nrow = 6, int(np.ceil(dim / 6))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.3 * nrow),
                             sharex=True, sharey=True)
    edges = np.arange(N_BINS + 1) * ((N_USE + 1) // N_BINS)
    stats_per_param = {}
    for ax_i, k in enumerate(order):
        ax = axes.ravel()[ax_i]
        obs, _ = np.histogram(ranks[:, k], bins=edges)
        chi2, p, bins = rank_uniformity_chi2(ranks[:, k], N_USE, N_BINS)
        zloc = rank_location_z(ranks[:, k], N_USE)
        stats_per_param[names[k]] = dict(
            chi2=chi2, p=p, bins=bins, z_location=zloc,
            param_class=_param_class(names[k]),
            mean_rank=float(ranks[:, k].mean()))
        ax.axhspan(lo99, hi99, color=gray, alpha=0.18, lw=0)
        ax.axhline(exp, color=gray, lw=1.0, ls="--")
        ax.bar(0.5 * (edges[:-1] + edges[1:]), obs,
               width=(edges[1] - edges[0]) * 0.92, color=blue, lw=0)
        fail = p <= 0.01
        ax.set_title(f"{_short(names[k])}  z={zloc:+.1f}", fontsize=9)
        ax.text(0.97, 0.92, f"p={p:.3g}", transform=ax.transAxes, fontsize=8,
                ha="right", va="top", color=(red if fail else "#555555"),
                fontweight=("bold" if fail else "normal"))
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for j in range(dim, nrow * ncol):
        axes.ravel()[j].axis("off")
    fig.suptitle(
        f"X2 SBC — GIGALens pipeline class (MAP->SVI->HMC, forward/diagonal), "
        f"N={n_fits} prior-matched mocks, ranks 0..{N_USE} in {N_BINS} bins "
        f"(gray: uniform mean + 99% band)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIGS.mkdir(exist_ok=True)
    f1 = FIGS / "x2_rank_hist_grid.png"
    fig.savefig(f1, dpi=140)
    plt.close(fig)

    # glass-house: our gamma vs OLD E1c gamma (n=44 FAIL) side by side
    old = json.loads(OLD_E1_REPORT.read_text())["e1c"]
    gname = [nm for nm in names
             if _param_class(nm) == "mass" and nm.endswith("/gamma")]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=False)
    kg = names.index(gname[0])
    obs, _ = np.histogram(ranks[:, kg], bins=edges)
    axs[0].axhspan(lo99, hi99, color=gray, alpha=0.18, lw=0)
    axs[0].axhline(exp, color=gray, lw=1.0, ls="--")
    axs[0].bar(range(N_BINS), obs, color=blue, width=0.92)
    p_g = stats_per_param[gname[0]]["p"]
    axs[0].set_title(f"THIS WORK: scene-API pipeline, gamma ranks "
                     f"(N={n_fits}, p={p_g:.3g})", fontsize=9)
    old_bins = old["per_param"]["gamma"]["bins"]
    n_old = old["n_fits"]
    lo99o, hi99o = sstats.binom.ppf([0.005, 0.995], n_old, 1.0 / N_BINS)
    axs[1].axhspan(lo99o, hi99o, color=gray, alpha=0.18, lw=0)
    axs[1].axhline(n_old / N_BINS, color=gray, lw=1.0, ls="--")
    axs[1].bar(range(N_BINS), old_bins, color="#B5651D", width=0.92)
    axs[1].set_title(f"GLASS-HOUSE: OUR OLD E1c (drizzle mocks, ChEES-HMC), "
                     f"gamma ranks (N={n_old}, p={old['per_param']['gamma']['p']:.2g} FAIL)",
                     fontsize=9)
    for ax in axs:
        ax.set_xlabel("rank bin")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("count")
    fig.tight_layout()
    f2 = FIGS / "x2_rank_hist_glasshouse_gamma.png"
    fig.savefig(f2, dpi=140)
    plt.close(fig)

    # ---------------- metrics AFTER figures ----------------
    def readout(mask, label):
        out = {}
        for k, nm in enumerate(names):
            r = ranks[mask][:, k]
            if r.size < N_BINS:
                out[nm] = dict(n=int(r.size), note="too few for chi2")
                continue
            chi2, p, bins = rank_uniformity_chi2(r, N_USE, N_BINS)
            out[nm] = dict(n=int(r.size), chi2=chi2, p=p, bins=bins,
                           z_location=rank_location_z(r, N_USE),
                           param_class=_param_class(nm))
        return out

    primary = readout(np.ones(n_fits, bool), "all")
    secondary = readout(healthy, "healthy") if healthy.any() else {}

    def worst_by_class(stats):
        wc = {}
        for nm, st in stats.items():
            if "z_location" not in st:
                continue
            c = st["param_class"]
            if c not in wc or abs(st["z_location"]) > abs(wc[c]["z"]):
                wc[c] = dict(param=nm, z=st["z_location"], p=st["p"])
        return wc

    report = dict(
        generated_by="30_sbc_gift.py harvest",
        n_fits=n_fits, n_requested=N_SYSTEMS, errors=errors,
        settings=metas[0]["settings"],
        rung=json.loads((RUNS / "settings_decision.json").read_text())["rung"],
        health=dict(rule=f"max_rhat<={HEALTH_RHAT} and min_ess>={HEALTH_ESS}",
                    n_healthy=int(healthy.sum()),
                    per_system=[dict(sys=m["sys_idx"],
                                     max_rhat=m["max_rhat"],
                                     min_ess=m["min_ess"],
                                     healthy=bool(h),
                                     chisq_truth=m["chisq_truth"],
                                     wall_s=m["wall"]["total"])
                                for m, h in zip(metas, healthy)]),
        primary_all_fits=primary,
        secondary_healthy_only=secondary,
        worst_abs_z_by_class=dict(all=worst_by_class(primary),
                                  healthy=worst_by_class(secondary)),
        gates=dict(
            severe_miscalibration_any=bool(any(
                abs(st.get("z_location", 0)) > 5 for st in primary.values())),
            rank_p_gt_0p01_all_mass=bool(all(
                st["p"] > 0.01 for st in primary.values()
                if st.get("param_class") == "mass" and "p" in st)),
        ),
        physicality=dict(
            construction_warnings=metas[0]["phys_construction_warnings"],
            posterior_diagnosis_sys00=metas[0]["phys_posterior"],
            arm_note=("single arm — layer is construction-time validation + "
                      "posterior diagnosis only; never enters log_prob "
                      "(vacuous ON/OFF, see checkpoint)")),
        guard_sys00=metas[0].get("guard"),
        glass_house=dict(
            old_e1c=dict(
                n_fits=old["n_fits"], per_param=old["per_param"],
                pooled_coverage68=old["pooled_coverage68"],
                source="../claude-giga-lens/data/e1_report.json e1c "
                       "(FAIL row, 2026-07-07)"),
            old_e1c_healthy_only=json.loads(
                OLD_E1_REPORT.read_text())["diagnosis"]["e1c_healthy_only"]),
        figures=[str(f1), str(f2)],
        wall_total_gpu_h=float(sum(m["wall"]["total"] for m in metas) / 3600.0),
    )
    out = DATA / "x2_sbc.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"[x2] harvest written: {out}\n figs: {f1}\n       {f2}")
    print(json.dumps(dict(worst=report["worst_abs_z_by_class"],
                          n_healthy=int(healthy.sum()), n_fits=n_fits),
                     indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "decide", "run", "harvest"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=16)
    a = ap.parse_args()
    if a.mode == "smoke":
        mode_smoke()
    elif a.mode == "decide":
        mode_decide()
    elif a.mode == "run":
        mode_run(a.start, a.count)
    else:
        mode_harvest()
