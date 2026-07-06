#!/usr/bin/env python
"""21_validate_zoo.py -- pre-registered zoo validation gates (P2a).

Gates:
  (i)   T1: zoo log_prob equals a DIRECT 02_fit_system-style construction
        (independent inline copy below) on system_000 AND system_003 at the
        3 freeze points + 5 seeded random points. Expected bit-level (same
        vendored code path, same device, same batch shape); gate <= 1e-6 rel.
  (ii)  T2: zoo logp at qz_refined equals the parity-report stored value
        (data/parity_report.json points.z_ref.logp_cgl), |d| <= 1e-8.
  (iii) T3: consistency vs stored hmc_v13_v3b.npz chains. The npz stores NO
        logp values (samples + mass_* + meta only), so the checkable set is:
        (a) to_physical(stored z) reproduces stored mass_* arrays;
        (b) masked reduced chi^2 at stored draws is finite and in the
            documented range; (c) logp finite at stored draws; (d) basin
        occupancy 45 low / 3 steep with ZERO gamma-1.8 crossings, measured
        through cgl.metrics mode machinery.
  (iv)  T0: arviz rank-R-hat/ESS on iid exact samples ~ nominal; mixture and
        ill-conditioned logZ vs independent numeric recompute (quadrature /
        scipy); funnel + mixture + illcond moments sane.
  (v)   log_prior + log_like == log_prob identity on 8 random points, ALL
        targets (<=1e-10 rel f64 / <=1e-5 rel f32). T2 has no independent
        jitted log_like (the marg data term is P0-frozen inside
        build_marg_model), so its genuine decomposition check uses
        marg_internals (eager logL_data + log_prior vs jitted logpost).

Orchestrator spawns one worker process per target (dtype isolation), writes
data/zoo_validation.json. Run on GPU 9:
  /raid/benson/.venvs/cgl/bin/python 21_validate_zoo.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPRO = Path(__file__).resolve().parent
PARTS = REPRO / "data" / "zoo_validation.parts"
WORKER_TIMEOUT_S = 3600

IDENTITY_TOL = {"float64": 1e-10, "float32": 1e-5}
GATE_I_TOL = 1e-6          # rel
GATE_II_TOL = 1e-8         # abs
GATE_III_MASS_TOL = 1e-3   # abs, f32 cross-stack forward reproduction
T2_INTERNALS_TOL = 1e-6    # abs, eager-vs-jit decomposition

# T1 systems that get the full gate-(i) independent-construction check.
GATE_I_SYSTEMS = ("gu2022_sys000", "gu2022_sys003")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--worker", default=None)
    ap.add_argument("--gpu", default=None)
    return ap.parse_args()


def _rand_points(target, n=8, seed0=555):
    if target.init.map_z is not None:
        anchor, scale = np.asarray(target.init.map_z, dtype=np.float64), 0.1
    else:
        anchor, scale = np.zeros(target.dim), 0.5
    return np.asarray([anchor + scale *
                       np.random.default_rng(seed0 + k).standard_normal(
                           target.dim) for k in range(n)])


# --------------------------------------------------------------------------- #
# gate (v): prior+like == prob identity
# --------------------------------------------------------------------------- #
def check_identity(target) -> dict:
    fdtype = np.float64 if target.dtype == "float64" else np.float32
    zs = _rand_points(target).astype(fdtype)
    lp = np.asarray(target.log_prob_batch(zs), dtype=np.float64)
    pri = np.asarray(target.log_prior_batch(zs), dtype=np.float64)
    lik = np.asarray(target.log_like_batch(zs), dtype=np.float64)
    rel = np.max(np.abs(pri + lik - lp) / np.maximum(np.abs(lp), 1.0))
    tol = IDENTITY_TOL[target.dtype]
    return dict(max_rel=float(rel), tol=tol,
                independent_loglike=bool(target.has_independent_loglike),
                n_points=int(zs.shape[0]),
                logp_range=[float(lp.min()), float(lp.max())],
                ok=bool(rel <= tol and target.has_independent_loglike))


# --------------------------------------------------------------------------- #
# gate (iv): T0 analytics
# --------------------------------------------------------------------------- #
def check_t0(target) -> dict:
    from cgl import metrics

    out = {}
    ref = target.reference
    dim = target.dim

    # arviz diagnostics on exact iid samples
    n_draw, n_chain = 1000, 8
    X = ref.exact_sample_fn(1234, n_draw * n_chain).reshape(
        n_draw, n_chain, dim)
    diag = metrics.rank_diagnostics(X, target.labels, target.labels)
    n_tot = n_draw * n_chain
    ess = np.asarray(diag["ess_bulk"])
    # arviz caps ESS at N*log10(N) (anti-thetic chains); iid draws sit AT the
    # cap, so the sane window is [0.8*N, N*log10(N)].
    ess_cap = n_tot * np.log10(n_tot)
    out["arviz"] = dict(
        max_rhat=diag["summary"]["rhat_all"]["max"],
        min_ess_frac=float(ess.min() / n_tot),
        max_ess_frac=float(ess.max() / n_tot),
        ess_cap_frac=float(ess_cap / n_tot),
        ok=bool(diag["summary"]["rhat_all"]["max"] < 1.015
                and 0.8 <= ess.min() / n_tot
                and ess.max() <= ess_cap * 1.001))

    # logZ vs an INDEPENDENT numeric recompute
    fam = target.meta.get("family")
    logz_num = None
    if fam == "gaussian_mixture":
        sp, s, x0 = (target.meta["prior_sigma"], target.meta["comp_sigma"],
                     target.meta["mode_x0"])
        if dim == 2:      # brute-force grid quadrature
            g = np.linspace(-8.0, 8.0, 3201)
            dA = (g[1] - g[0]) ** 2
            XX, YY = np.meshgrid(g, g, indexing="ij")
            pts = np.stack([XX.ravel(), YY.ravel()], 1)
            fd = np.float64 if target.dtype == "float64" else np.float32
            lp = np.asarray(target.log_prob_batch(pts.astype(fd)),
                            dtype=np.float64)
            logz_num = float(np.log(np.exp(lp).sum() * dA))
        else:             # independent closed form via scipy
            from scipy.stats import norm
            ev = norm.logpdf(0.0, loc=x0, scale=np.sqrt(s**2 + sp**2)) \
                + (dim - 1) * norm.logpdf(0.0, loc=0.0,
                                          scale=np.sqrt(s**2 + sp**2))
            logz_num = float(np.logaddexp(np.log(0.8) + ev,
                                          np.log(0.2) + ev))
    elif fam == "illconditioned_gaussian":
        from scipy.stats import norm
        lam = np.logspace(0.0, -np.log10(target.meta["cond"]), dim)
        sp2 = target.meta["prior_sigma"] ** 2
        logz_num = float(np.sum(norm.logpdf(0.0, loc=0.0,
                                            scale=np.sqrt(lam + sp2))))
    if logz_num is not None:
        cmp = metrics.compare_logZ(logz_num, ref.logZ)
        tol = 1e-5 if dim == 2 and fam == "gaussian_mixture" else 1e-9
        out["logZ"] = {**cmp, "tol": tol, "ok": bool(cmp["abs_diff"] <= tol)}
    else:   # funnel: logZ = 0 exact by construction (like := prob - prior
            # verified through the independent expression in gate v)
        out["logZ"] = dict(logz_ref=ref.logZ, note="0 exact by construction",
                           ok=bool(ref.logZ == 0.0))

    # moments + mode metrics
    flat = X.reshape(-1, dim)
    if fam == "gaussian_mixture":
        tr = ref.truth
        mean0, var0 = tr["mean"][0], tr["var"][0]
        se = np.sqrt(var0 / flat.shape[0])
        mean_ok = abs(flat[:, 0].mean() - mean0) < 5 * se
        var_ok = abs(flat[:, 0].var() / var0 - 1.0) < 0.1
        assign = metrics.assign_modes(ref, Z=X)
        occ = metrics.mode_occupancy(assign, 2, ref.mode_weights)
        rt = metrics.count_mode_round_trips(assign.reshape(n_draw, n_chain))
        out["moments"] = dict(mean0=float(flat[:, 0].mean()),
                              mean0_true=mean0, var0=float(flat[:, 0].var()),
                              var0_true=var0, ok=bool(mean_ok and var_ok))
        out["modes"] = dict(**occ, round_trips=rt["total_round_trips"],
                            ok=bool(occ["max_abs_weight_error"] < 0.02
                                    and occ["recovery_rate"] == 1.0
                                    and rt["total_round_trips"] > 0))
    elif fam == "neal_funnel":
        th = flat[:, 0]
        logx2 = np.log(flat[:, 1:] ** 2).ravel()
        out["moments"] = dict(
            var_theta=float(th.var()), mean_theta=float(th.mean()),
            mean_log_x2=float(logx2.mean()),
            ok=bool(abs(th.var() - 9.0) < 0.5 and abs(th.mean()) < 0.15
                    and abs(logx2.mean() - (-1.2703628)) < 0.05))
    elif fam == "illconditioned_gaussian":
        var_true = np.asarray(ref.truth["var"])
        ratio = flat.var(axis=0) / var_true
        out["moments"] = dict(
            max_var_ratio_err=float(np.max(np.abs(ratio - 1.0))),
            cond_posterior=ref.truth["cond_posterior"],
            ok=bool(np.max(np.abs(ratio - 1.0)) < 0.1))
    out["ok"] = all(v.get("ok", True) for v in out.values()
                    if isinstance(v, dict))
    return out


# --------------------------------------------------------------------------- #
# gate (i): T1 vs direct 02_fit_system-style construction (independent copy)
# --------------------------------------------------------------------------- #
def check_t1_direct(target, freeze_entry) -> dict:
    """Independent inline construction following gu-2022/02_fit_system.py
    lines 32-72 + 163-174 + 209-211 (deliberate code duplication: the gate
    compares the ZOO closure against a from-scratch build)."""
    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    from gigalens.jax.model import ForwardProbModel
    from gigalens.jax.profiles.light import sersic
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.jax.simulator import LensSimulator
    from gigalens.model import PhysicalModel
    from gigalens.simulator import SimulatorConfig

    tfd = tfp.distributions
    d = np.load(target.meta["mock"], allow_pickle=True)

    lens_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
            gamma=tfd.TruncatedNormal(2.0, 0.5, 1.0, 3.0),
            e1=tfd.Normal(0.0, 0.2), e2=tfd.Normal(0.0, 0.2),
            center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1))),
        tfd.JointDistributionNamed(dict(
            gamma1=tfd.Normal(0.0, 0.06), gamma2=tfd.Normal(0.0, 0.06)))])
    lens_light_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
            n_sersic=tfd.Uniform(0.5, 8.0),
            e1=tfd.TruncatedNormal(0.0, 0.1, -0.15, 0.15),
            e2=tfd.TruncatedNormal(0.0, 0.1, -0.15, 0.15),
            center_x=tfd.Normal(0.0, 0.02), center_y=tfd.Normal(0.0, 0.02),
            Ie=tfd.LogNormal(jnp.log(300.0), 0.5)))])
    source_light_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(0.25), 0.25),
            n_sersic=tfd.Uniform(0.5, 8.0),
            e1=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
            e2=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
            center_x=tfd.Normal(0.0, 0.5), center_y=tfd.Normal(0.0, 0.5),
            Ie=tfd.LogNormal(jnp.log(150.0), 0.9)))])
    prior = tfd.JointDistributionSequential(
        [lens_prior, lens_light_prior, source_light_prior])

    image = np.array(d["image"], dtype=np.float32)
    sim_config = SimulatorConfig(delta_pix=float(d["delta_pix"]),
                                 num_pix=int(d["num_pix"]),
                                 supersample=int(d["supersample"]),
                                 kernel=np.array(d["psf"], dtype=np.float32))
    phys_model = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=False)],
        [sersic.SersicEllipse(use_lstsq=False)],
    )
    prob_model = ForwardProbModel(prior, image,
                                  background_rms=float(d["sigma_bkg"]),
                                  exp_time=float(d["exp_time"]))

    zs = np.asarray([p["z"] for p in freeze_entry["points"]])
    zs = np.concatenate([zs, _rand_points(target, n=5, seed0=777)])
    zs32 = jnp.asarray(zs, dtype=jnp.float32)
    lens_sim = LensSimulator(phys_model, sim_config, bs=int(zs.shape[0]))
    lp_direct = np.asarray(prob_model.log_prob(lens_sim, zs32)[0],
                           dtype=np.float64)
    lp_zoo = np.asarray(target.log_prob_batch(zs32), dtype=np.float64)
    d_abs = np.abs(lp_direct - lp_zoo)
    rel = float(np.max(d_abs / np.maximum(np.abs(lp_direct), 1.0)))

    # labels vs the stored fit's phys_labels. FINDING (P2a): the stored
    # phys_labels is the forward-dict .items() order = per-block REVERSED
    # alphabetical; the TRUE z-index -> leaf mapping (measured by coordinate
    # probing, which the zoo uses) is per-block alphabetical. Both are
    # block-contiguous, so the stored fit's mass-set ESS aggregates are
    # unaffected; per-param attribution within a block is reversed there.
    f = np.load(target.meta["fit"], allow_pickle=True)
    stored_labels = [str(x) for x in f["phys_labels"]]
    blocks = [slice(0, 6), slice(6, 8), slice(8, 15), slice(15, 22)]
    labels_match = all(
        stored_labels[b] == list(reversed(list(target.labels)[b]))
        for b in blocks)

    # truth z-vector roundtrip through the zoo bijector
    tr = target.reference.truth
    phys = target.to_physical(np.asarray(tr["z_truth"])[None, :])
    truth_err = max(abs(float(phys[k][0]) - tr["by_label"][k])
                    / max(abs(tr["by_label"][k]), 1e-3)
                    for k in target.mass_labels)

    return dict(
        n_points=int(zs.shape[0]),
        max_abs_dlogp=float(d_abs.max()),
        max_rel_dlogp=rel,
        bit_identical=bool(d_abs.max() == 0.0),
        logp_zoo=[float(v) for v in lp_zoo],
        tol=GATE_I_TOL,
        labels_match_stored_fit_mod_block_reversal=bool(labels_match),
        label_order_note="zoo labels = probed z-aligned (per-block "
                         "alphabetical); stored fit phys_labels = .items() "
                         "order (per-block reversed); block-contiguous, so "
                         "stored mass-set aggregates are unaffected",
        stored_labels=stored_labels,
        truth_roundtrip_max_rel=float(truth_err),
        ok=bool(rel <= GATE_I_TOL and labels_match and truth_err < 1e-3),
    )


# --------------------------------------------------------------------------- #
# gate (ii) + T2 internals decomposition
# --------------------------------------------------------------------------- #
def check_t2(target, model, freeze_entry) -> dict:
    import jax.numpy as jnp

    from cgl.paths import MAP_MARG_PD, PARITY_REPORT

    parity = json.loads(PARITY_REPORT.read_text())
    stored_parity = float(parity["points"]["z_ref"]["logp_cgl"])
    stored_map = float(np.load(MAP_MARG_PD)["logp"])
    z_ref = np.asarray(np.load(MAP_MARG_PD)["qz_refined"], dtype=np.float64)

    lp = float(target.log_prob(jnp.asarray(z_ref)))
    d_parity = abs(lp - stored_parity)
    d_map = abs(lp - stored_map)

    # batched-vs-single consistency (vmap fusion sanity)
    lp_b = np.asarray(target.log_prob_batch(
        jnp.asarray(np.stack([z_ref, z_ref + 0.05]))), dtype=np.float64)
    d_batch = abs(float(lp_b[0]) - lp)

    # genuine decomposition check: eager marg_internals vs jitted logpost
    zs = np.asarray([p["z"] for p in freeze_entry["points"]])
    d_int = []
    for z in zs:
        internals = model.marg_internals(z)
        lp_j = float(target.log_prob(jnp.asarray(z)))
        d_int.append(abs(internals["logL_data"] + internals["log_prior"]
                         - lp_j))
    d_int = float(np.max(d_int))

    return dict(
        logp_at_qz_refined=lp,
        stored_parity_logp=stored_parity,
        abs_diff_vs_parity=d_parity,
        stored_map_logp=stored_map,
        abs_diff_vs_map_npz=d_map,
        tol=GATE_II_TOL,
        batch_vs_single_abs=d_batch,
        internals_decomposition_max_abs=d_int,
        internals_tol=T2_INTERNALS_TOL,
        ok=bool(d_parity <= GATE_II_TOL and d_int <= T2_INTERNALS_TOL
                and d_batch <= 1e-8),
    )


# --------------------------------------------------------------------------- #
# gate (iii): T3 vs stored chains
# --------------------------------------------------------------------------- #
def check_t3(target) -> dict:
    from cgl import metrics
    from cgl.paths import HMC_V13_V3B

    g = np.load(HMC_V13_V3B, allow_pickle=True)
    samples = g["samples"]                      # (3500, 48, 74) f32
    stored_keys = list(g.files)
    T, C, dim = samples.shape

    # (a) bijector/label parity: reproduce stored mass_* at sampled draws
    t_idx = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    mass_keys = ["theta_E", "gamma", "e1", "e2", "center_x", "center_y",
                 "gamma1", "gamma2"]
    max_d = 0.0
    for t in t_idx:
        phys = target.to_physical(samples[t])       # dict of (48,)
        for k in mass_keys:
            stored = np.asarray(g[f"mass_{k}"][t], dtype=np.float64)
            ours = np.asarray(phys[k], dtype=np.float64)
            max_d = max(max_d, float(np.max(np.abs(ours - stored))))

    # (b)+(c) chi2 / logp at stored draws
    Z = samples[T // 2]                              # (48, 74)
    chi2 = np.asarray(target.chi2_fn(Z), dtype=np.float64)
    lp = np.asarray(target.log_prob_batch(Z), dtype=np.float64)
    chi2_ok = bool(np.all(np.isfinite(chi2)) and 1.0 < np.median(chi2) < 2.5)
    lp_ok = bool(np.all(np.isfinite(lp)))

    # (d) basin occupancy through the metrics machinery
    assign = metrics.assign_modes(target.reference,
                                  phys={"gamma": np.asarray(g["mass_gamma"])})
    occ = metrics.mode_occupancy(assign, 2, target.reference.mode_weights)
    rt = metrics.count_mode_round_trips(assign)
    chain_mode = (np.asarray(g["mass_gamma"]).mean(axis=0)
                  >= target.meta["gamma_threshold"])
    basin_ok = bool(int((~chain_mode).sum()) == 45 and int(chain_mode.sum()) == 3
                    and rt["n_migrating_chains"] == 0)

    return dict(
        stored_npz_keys=stored_keys,
        stored_shape=list(samples.shape),
        logp_values_stored=False,
        checked="mass_* reproduction (a), chi2 (b), logp finite (c), "
                "basin occupancy + zero migrations (d); bit-check of the "
                "density impossible (npz stores no logp)",
        mass_reproduction_max_abs=max_d,
        mass_tol=GATE_III_MASS_TOL,
        chi2=dict(min=float(chi2.min()), median=float(np.median(chi2)),
                  max=float(chi2.max()), ok=chi2_ok),
        logp=dict(min=float(lp.min()), max=float(lp.max()), ok=lp_ok),
        basins=dict(n_low=int((~chain_mode).sum()),
                    n_steep=int(chain_mode.sum()),
                    occupancy=occ["occupancy"],
                    n_migrating_chains=rt["n_migrating_chains"],
                    total_round_trips=rt["total_round_trips"],
                    gamma_low_mean=target.reference.truth[
                        "chain_mean_gamma_low"],
                    gamma_steep_mean=target.reference.truth[
                        "chain_mean_gamma_steep"],
                    ok=basin_ok),
        ok=bool(max_d <= GATE_III_MASS_TOL and chi2_ok and lp_ok and basin_ok),
    )


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
def run_worker(name: str, gpu):
    from cgl.zoo import get_target_info
    from cgl.zoo.runtime import setup_process_env

    info = get_target_info(name)
    setup_process_env(info.dtype, gpu)

    import jax  # noqa: E402

    from cgl.paths import ZOO_FREEZE
    from cgl.zoo import get_target
    from cgl.zoo.runtime import freeze_entry_for, load_freeze, setup_jax_cache

    setup_jax_cache(REPRO)
    print(f"[worker {name}] devices={jax.devices()} "
          f"x64={jax.config.jax_enable_x64}", flush=True)

    freeze = load_freeze(ZOO_FREEZE)
    entry = freeze_entry_for(freeze, name)
    report = dict(name=name, tier=info.tier, dtype=info.dtype)
    t0 = time.time()

    if info.tier == "T2":
        from cgl.zoo.foundry_marg import build, build_raw_model
        model = build_raw_model()
        target = build(model=model)
    else:
        target, model = get_target(name), None

    # freeze re-check in this fresh process
    zs = np.asarray([p["z"] for p in entry["points"]])
    want = np.asarray([p["logp"] for p in entry["points"]])
    fd = np.float64 if info.dtype == "float64" else np.float32
    got = np.asarray(target.log_prob_batch(zs.astype(fd)), dtype=np.float64)
    frz_rel = float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1.0)))
    report["freeze_recheck_max_rel"] = frz_rel

    if info.tier == "T2":
        report["gate_ii"] = check_t2(target, model, entry)
        # identity gate (v) evidence for T2 = the internals decomposition
        report["identity"] = dict(
            max_rel=report["gate_ii"]["internals_decomposition_max_abs"]
            / abs(report["gate_ii"]["logp_at_qz_refined"]),
            tol=IDENTITY_TOL["float64"],
            independent_loglike=False,
            note="via marg_internals (eager) -- see gate_ii; the dataclass "
                 "log_like is derived (documented)",
            ok=report["gate_ii"]["internals_decomposition_max_abs"]
            <= T2_INTERNALS_TOL)
    else:
        report["identity"] = check_identity(target)
        if info.tier == "T0":
            report["gate_iv"] = check_t0(target)
        elif info.tier == "T1" and name in GATE_I_SYSTEMS:
            report["gate_i"] = check_t1_direct(target, entry)
        elif info.tier == "T3":
            report["gate_iii"] = check_t3(target)

    report["wall_s"] = time.time() - t0
    PARTS.mkdir(parents=True, exist_ok=True)
    out = PARTS / f"{name}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[worker {name}] done ({report['wall_s']:.0f}s) -> {out}",
          flush=True)
    return 0


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def run_orchestrator(names, gpu):
    from cgl.paths import ZOO_VALIDATION
    from cgl.zoo import get_target_info, list_targets
    from cgl.zoo.runtime import child_env

    if not names:
        names = [t["name"] for t in list_targets(available_only=True)]

    failures = {}
    for name in names:
        info = get_target_info(name)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", name]
        if gpu is not None:
            cmd += ["--gpu", str(gpu)]
        print(f"[spawn] {name} ...", flush=True)
        r = subprocess.run(cmd, env=child_env(info.dtype, gpu),
                           timeout=WORKER_TIMEOUT_S)
        if r.returncode != 0:
            failures[name] = r.returncode

    # merge ALL existing parts (a --targets rerun refreshes only its workers;
    # earlier part files remain part of the report)
    parts = {}
    for p in sorted(PARTS.glob("*.json")):
        parts[p.stem] = json.loads(p.read_text())

    # ---- gate verdicts --------------------------------------------------------
    gates = {}
    gi = {n: parts[n].get("gate_i") for n in GATE_I_SYSTEMS if n in parts}
    if gi and all(v for v in gi.values()):
        gates["i"] = dict(
            per_system={n: dict(max_rel=v["max_rel_dlogp"],
                                bit_identical=v["bit_identical"],
                                labels_match=v[
                                    "labels_match_stored_fit_mod_block_reversal"])
                        for n, v in gi.items()},
            tol=GATE_I_TOL,
            achieved=max(v["max_rel_dlogp"] for v in gi.values()),
            passed=bool(all(v["ok"] for v in gi.values())
                        and set(gi) == set(GATE_I_SYSTEMS)))
    t2 = next((parts[n]["gate_ii"] for n in parts
               if parts[n].get("gate_ii")), None)
    if t2:
        gates["ii"] = dict(achieved=t2["abs_diff_vs_parity"],
                           tol=GATE_II_TOL, passed=t2["ok"],
                           logp=t2["logp_at_qz_refined"],
                           internals_max_abs=t2[
                               "internals_decomposition_max_abs"])
    t3 = next((parts[n]["gate_iii"] for n in parts
               if parts[n].get("gate_iii")), None)
    if t3:
        gates["iii"] = dict(
            mass_reproduction_max_abs=t3["mass_reproduction_max_abs"],
            chi2_median=t3["chi2"]["median"],
            basins=t3["basins"], passed=t3["ok"])
    t0s = {n: parts[n]["gate_iv"] for n in parts if parts[n].get("gate_iv")}
    if t0s:
        gates["iv"] = dict(
            per_target={n: dict(ok=v["ok"], max_rhat=v["arviz"]["max_rhat"],
                                logZ_absdiff=v["logZ"].get("abs_diff"))
                        for n, v in t0s.items()},
            passed=all(v["ok"] for v in t0s.values()))
    idents = {n: parts[n]["identity"] for n in parts
              if parts[n].get("identity")}
    gates["v"] = dict(
        worst_rel=max(v["max_rel"] for v in idents.values()),
        per_target={n: dict(max_rel=v["max_rel"], ok=v["ok"])
                    for n, v in idents.items()},
        passed=all(v["ok"] for v in idents.values()))

    from cgl.zoo import list_targets as _lt
    n_available = len(_lt(available_only=True))
    all_pass = (all(g.get("passed") for g in gates.values())
                and not failures and len(parts) >= n_available)
    report = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        n_targets=len(parts), gates=gates, targets=parts,
        failures=failures, all_gates_pass=bool(all_pass))
    ZOO_VALIDATION.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {ZOO_VALIDATION}", flush=True)

    print("\n=== ZOO VALIDATION SUMMARY ===", flush=True)
    for k in ("i", "ii", "iii", "iv", "v"):
        if k in gates:
            g = gates[k]
            print(f"gate {k:>3s}: {'PASS' if g.get('passed') else 'FAIL'}  "
                  f"{ {kk: vv for kk, vv in g.items() if kk not in ('per_system', 'per_target', 'targets', 'basins')} }",
                  flush=True)
    print(f"\nALL GATES {'PASS' if all_pass else 'FAIL'}", flush=True)
    return 0 if all_pass else 1


def main():
    args = parse_args()
    if args.worker:
        return run_worker(args.worker, args.gpu)
    return run_orchestrator(args.targets, args.gpu)


if __name__ == "__main__":
    sys.exit(main())
