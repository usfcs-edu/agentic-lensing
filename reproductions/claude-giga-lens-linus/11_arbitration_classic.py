"""11_arbitration_classic.py — P3-L0 anchor arbitration, CLASSIC-RECIPE ARM.

PRIMARY arbitration vehicle per the 2026-07-20 protocol amendment in
research/checkpoints_l0.md (written BEFORE this script ran): three MC-SMC
attempts wall-capped below lambda=1 (L4 ~0.45 @ ~18 h; A100 hbm80g 0.587 @
3.5 h), so the pre-registered optional arm (b) — the classic scene-API
MAP -> SVI -> HMC recipe, the SAME algorithm class that produced the 1.433
anchor on the old stack — is promoted to primary. Gates and bands UNCHANGED;
the MC-SMC partials stay as secondary prior-seeded evidence.

Target    : v2d native product, DIAGONAL masked-err ridge-marg likelihood —
            10_anchor_arbitration.build_pm("v2d", diagonal=True), the
            F-gate-certified configuration (imported by path, verbatim).
Recipe    : B3 reference-arm replication (22_run_b3.py, imported by path):
            _map (warm single-start refine, their adabelief(1e-2), 350 steps)
            -> _svi_replicated (250 draws x 500 steps, adabelief 1e-4) with
            the B3 OOM halving ladder 250->128->64
            -> _hmc_replicated (50 chains, 250/750, init_l 3, max_l 30) with
            the single pre-registered 500/1500 escalation on Rhat/ESS failure
            (RHAT_MAX 1.05, ESS_MIN 200); stage seeds 0/1/2 (B3 convention).
Warm start: mapped foundry-i MAP — param_map.scene_z_from_old_labels of
            refs["v2d:z_ref:x46"] (the S1 wiring-certified object). The
            multi-start MAP is replaced by a single-start warm refine
            (multi-start from one fixed point is degenerate — declared).
Outputs   : data/l0_arbitration_classic.npz  (HMC chains, gamma samples,
            map/qz arrays, rhat/ess vectors)
            data/l0_arbitration_classic.json (config echo, stage walls,
            worst-param Rhat / min ESS, gamma quantiles)
            written STAGE-WISE (post-SVI / post-HMC / post-escalation) so a
            walltime kill loses at most the running stage.
NO GATE MATH here — harvest stays plots-first (house rule).

Run (Perlmutter, plain gpu — see slurm/p3_l0arb_classic.slurm):
    python 11_arbitration_classic.py
Local CPU toy test of the driver (synthetic quadratic prob-model, tiny
constants; the lensing likelihood is NEVER evaluated on CPU):
    python 11_arbitration_classic.py --toy
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ---- import the certified builders by path (env-before-jax order preserved:
# module 10 sets GIGALENS_X64/JAX_ENABLE_X64 defaults + runs the vendor/jax
# guards at import, exactly as the SMC arm did) -------------------------------


def _import_by_path(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


anchor_arb = _import_by_path("anchor_arb", "10_anchor_arbitration.py")
run_b3 = _import_by_path("run_b3", "22_run_b3.py")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import tensorflow_probability.substrates.jax as tfp  # noqa: E402

from cgl2 import param_map  # noqa: E402

DATA = anchor_arb.DATA
OUT_STEM = "l0_arbitration_classic"

# frozen constants — B3 reference arm (22_run_b3.py header), never redefined
REF = dict(map_steps=run_b3.REF_MAP_STEPS,
           svi_nvi=run_b3.REF_SVI_NVI, svi_steps=run_b3.REF_SVI_STEPS,
           hmc_n=run_b3.REF_HMC_N, hmc_burn=run_b3.REF_HMC_BURN,
           hmc_res=run_b3.REF_HMC_RES, hmc_init_l=run_b3.REF_HMC_INIT_L,
           hmc_max_l=run_b3.REF_HMC_MAX_L,
           esc_burn=run_b3.REF_ESCALATED_BURN, esc_res=run_b3.REF_ESCALATED_RES,
           rhat_max=run_b3.RHAT_MAX, ess_min=run_b3.ESS_MIN,
           svi_ladder=(run_b3.REF_SVI_NVI, 128, 64),
           seed_map=0, seed_svi=1, seed_hmc=2)


def _jsonable(x):
    return run_b3._jsonable(x)


# --------------------------------------------------------------------------- #
# warm single-start MAP refine (structure replicated with attribution from
# 22_run_b3.py::_map_replicated — same loss, optimizer application and
# post-update (params, lp) pairing; the ONLY change is start = the mapped
# foundry-i MAP instead of n prior samples, per the 2026-07-20 amendment)
# --------------------------------------------------------------------------- #
def _map_warm_refine(prob_model, z_start, optimizer, num_steps):
    import optax

    params = jnp.asarray(np.asarray(z_start, dtype=np.float64)[None, :])
    map_prob = (prob_model.with_map_remat()
                if hasattr(prob_model, "with_map_remat") else prob_model)

    def loss(z):
        lp, chisq = map_prob.log_prob(z)
        return -jnp.mean(lp) / map_prob.loss_normalization, (lp, chisq)

    loss_and_grad = jax.value_and_grad(loss, has_aux=True)

    def one_step(carry, _):
        p, opt_state = carry
        (_, (lp, chisq)), grads = loss_and_grad(p)
        updates, opt_state = optimizer.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        i = jnp.nanargmax(lp)              # their post-update pairing, kept
        return (p, opt_state), (p[i], lp[i], chisq[i])

    @jax.jit
    def run(p0):
        opt_state = optimizer.init(p0)
        _, best = jax.lax.scan(one_step, (p0, opt_state), length=num_steps)
        return best

    bp, blp, bchi = run(params)
    j = int(jnp.nanargmax(blp))
    return np.asarray(bp[j]), float(blp[j]), np.asarray(bchi)


def _hmc_diagnosed(pm, qz, burn, res, cfg):
    """One HMC stage + Rhat/ESS diagnostics (block replicated with attribution
    from 22_run_b3.py::run_ref_arm's _hmc closure — identical kernel stack and
    statistics; only the constants dict differs in origin)."""
    t0 = time.time()
    raw = run_b3._hmc_replicated(pm, qz, cfg["hmc_n"], burn, res,
                                 init_eps=0.3, init_l=cfg["hmc_init_l"],
                                 max_l=cfg["hmc_max_l"], seed=cfg["seed_hmc"])
    wall = time.time() - t0
    chains = run_b3._hmc_chains_first_to_samples(raw, cfg["hmc_n"], res)
    rhat = np.asarray(tfp.mcmc.potential_scale_reduction(
        jnp.asarray(chains), independent_chain_ndims=1, split_chains=True))
    ess = np.asarray(tfp.mcmc.effective_sample_size(
        jnp.asarray(chains), cross_chain_dims=1))
    log = dict(n_hmc=cfg["hmc_n"], num_burnin_steps=burn, num_results=res,
               seed=cfg["seed_hmc"], init_l=cfg["hmc_init_l"],
               max_leapfrog_steps=cfg["hmc_max_l"], wall_s=round(wall, 1),
               rhat_worst=float(np.max(rhat)),
               rhat_argmax=int(np.argmax(rhat)),
               ess_min=float(np.min(ess)), ess_argmin=int(np.argmin(ess)),
               rhat=rhat.tolist(), ess=ess.tolist())
    return chains, rhat, ess, log


def _gamma_from_chains(gamma_fn, chains, batch=8192):
    flat = np.asarray(chains, dtype=np.float64).reshape(-1, chains.shape[-1])
    out = [gamma_fn(flat[i:i + batch]) for i in range(0, flat.shape[0], batch)]
    return np.concatenate(out)


def _write_outputs(log, arrays, out_stem):
    (DATA / f"{out_stem}.json").write_text(json.dumps(_jsonable(log), indent=2))
    np.savez(DATA / f"{out_stem}.npz", **arrays)
    print(f"[l0-classic] wrote data/{out_stem}.json + .npz "
          f"(stage={log.get('last_stage_written')})", flush=True)


# --------------------------------------------------------------------------- #
def drive(pm, z_warm, gamma_fn, cfg, out_stem, extra_prov=None):
    """MAP -> SVI -> HMC(+escalation) with stage-wise artifact writes."""
    log = dict(script="11_arbitration_classic.py", arm="classic_warm",
               amendment="research/checkpoints_l0.md 2026-07-20 (vehicle "
                         "amendment; gates/bands unchanged)",
               recipe="warm MAP refine -> SVI -> HMC, B3 reference-arm "
                      "frozen settings (replicated with attribution; vendored "
                      "MAP raises jax-0.6.2 shard_map cotangent TypeError)",
               config=dict(cfg),
               date=time.strftime("%Y-%m-%d %H:%M:%S"),
               jax_version=jax.__version__, backend=jax.default_backend(),
               device=str(jax.devices()[0]),
               x64=bool(jax.config.jax_enable_x64),
               status="RUNNING")
    if extra_prov:
        log.update(extra_prov)
    arrays = dict(z_warm=np.asarray(z_warm, dtype=np.float64))
    oom_fallbacks = []

    def _oom_retry(fn, label, *sizes):
        # replicated with attribution from 22_run_b3.py::run_ref_arm
        for s in sizes:
            try:
                return s, fn(s)
            except Exception as e:  # noqa: BLE001
                if "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e):
                    oom_fallbacks.append(dict(stage=label, failed_size=s,
                                              error=str(e)[:200]))
                    continue
                raise
        raise RuntimeError(f"{label}: all sizes OOM ({sizes})")

    # ---- MAP (warm single-start refine) -----------------------------------
    t0 = time.time()
    map_best, map_lp, map_chisq = _map_warm_refine(
        pm, z_warm, run_b3._their_map_optimizer(), cfg["map_steps"])
    log["map"] = dict(warm_start="mapped foundry-i MAP (see provenance)",
                      n_samples=1, num_steps=cfg["map_steps"],
                      seed=cfg["seed_map"], wall_s=round(time.time() - t0, 1),
                      best_lp=float(map_lp),
                      final_chisq_min=float(np.nanmin(map_chisq)),
                      grad_evals=cfg["map_steps"])
    arrays["map_best"] = np.asarray(map_best)
    print(f"[l0-classic] MAP done in {log['map']['wall_s']} s "
          f"(lp={map_lp:.3f})", flush=True)

    # ---- SVI (B3 OOM halving ladder) --------------------------------------
    t0 = time.time()
    n_vi, (qz, svi_loss) = _oom_retry(
        lambda n: run_b3._svi_replicated(pm, map_best,
                                         run_b3._their_svi_optimizer(), n,
                                         cfg["svi_steps"],
                                         seed=cfg["seed_svi"]),
        "svi", *cfg["svi_ladder"])
    log["svi"] = dict(n_vi=n_vi, num_steps=cfg["svi_steps"],
                      seed=cfg["seed_svi"],
                      wall_s=round(time.time() - t0, 1),
                      final_neg_elbo=float(np.asarray(svi_loss)[-1]),
                      best_neg_elbo=float(np.nanmin(np.asarray(svi_loss))),
                      grad_evals=n_vi * cfg["svi_steps"])
    arrays["qz_mean"] = np.asarray(qz.mean())
    arrays["qz_cov"] = np.asarray(qz.covariance())
    log["oom_fallbacks"] = oom_fallbacks
    log["last_stage_written"] = "svi"
    _write_outputs(log, arrays, out_stem)          # timeout insurance
    print(f"[l0-classic] SVI done in {log['svi']['wall_s']} s "
          f"(n_vi={n_vi})", flush=True)

    # ---- HMC + single pre-registered escalation ---------------------------
    chains, rhat, ess, hlog = _hmc_diagnosed(pm, qz, cfg["hmc_burn"],
                                             cfg["hmc_res"], cfg)
    log["hmc"] = hlog
    accepted = (hlog["rhat_worst"] < cfg["rhat_max"]
                and hlog["ess_min"] >= cfg["ess_min"])
    arrays.update(chains=chains, rhat=rhat, ess=ess,
                  gamma=_gamma_from_chains(gamma_fn, chains))
    log["accepted_stage"] = "hmc" if accepted else "PENDING-ESCALATION"
    log["last_stage_written"] = "hmc"
    _write_outputs(log, arrays, out_stem)          # escalation insurance
    print(f"[l0-classic] HMC done: rhat_worst={hlog['rhat_worst']:.4f} "
          f"ess_min={hlog['ess_min']:.1f} accepted={accepted}", flush=True)

    if not accepted:
        chains2, rhat2, ess2, hlog2 = _hmc_diagnosed(
            pm, qz, cfg["esc_burn"], cfg["esc_res"], cfg)
        log["hmc_escalated"] = hlog2
        if (hlog2["rhat_worst"] < cfg["rhat_max"]
                and hlog2["ess_min"] >= cfg["ess_min"]):
            accepted = True
            log["accepted_stage"] = "hmc_escalated"
            arrays.update(chains=chains2, rhat=rhat2, ess=ess2,
                          gamma=_gamma_from_chains(gamma_fn, chains2))
        else:
            # keep BOTH stages' chains; accepted stays False (honest record)
            log["accepted_stage"] = "NONE (blocked-by-reference)"
            arrays.update(chains_escalated=chains2, rhat_escalated=rhat2,
                          ess_escalated=ess2,
                          gamma_escalated=_gamma_from_chains(gamma_fn,
                                                             chains2))
        print(f"[l0-classic] HMC escalated: "
              f"rhat_worst={hlog2['rhat_worst']:.4f} "
              f"ess_min={hlog2['ess_min']:.1f} accepted={accepted}",
              flush=True)

    log["reference_accepted"] = bool(accepted)
    gam = np.asarray(arrays["gamma"], dtype=np.float64)
    log["gamma_summary"] = dict(       # record only — gate math at harvest
        n=int(gam.size), med=float(np.median(gam)),
        q16=float(np.quantile(gam, 0.16)), q84=float(np.quantile(gam, 0.84)),
        frac_gt_1p9=float(np.mean(gam > anchor_arb.BASIN_SPLIT)))
    log["status"] = "COMPLETE"
    log["last_stage_written"] = "final"
    log["wall_s_total"] = round(
        log["map"]["wall_s"] + log["svi"]["wall_s"] + log["hmc"]["wall_s"]
        + log.get("hmc_escalated", {}).get("wall_s", 0.0), 1)
    _write_outputs(log, arrays, out_stem)
    # ops summary only on stdout (gamma numbers live in the artifacts;
    # harvest is plots-first per house rule)
    print(json.dumps({k: log[k] for k in
                      ("status", "accepted_stage", "reference_accepted",
                       "wall_s_total", "oom_fallbacks")}, indent=2),
          flush=True)
    return log


# --------------------------------------------------------------------------- #
def main_production():
    sane = json.loads((DATA / "l0_sanity_report.json").read_text())
    if not sane.get("sanity_ok"):
        raise RuntimeError("sanity gates not passed — refusing to run")

    refs = anchor_arb.load_refs()
    pm, mv, _ = anchor_arb.build_pm("v2d", refs, diagonal=True)

    labeled = param_map.labeled_from_vec46(refs["v2d:z_ref:x46"])
    z_warm = param_map.scene_z_from_old_labels(mv.model, labeled)

    # warmup probe (fail-loud wiring fence, 24_run_p2 pattern) + provenance
    lp = float(pm.log_prior(jnp.asarray(z_warm)))
    ll = float(pm.log_like(jnp.asarray(z_warm))[0])
    if not (np.isfinite(lp) and np.isfinite(ll)):
        raise RuntimeError(f"non-finite warm start: log_prior={lp} "
                           f"log_like={ll}")
    g_warm = float(anchor_arb.gamma_of(pm, z_warm[None, :])[0])
    print(f"[l0-classic] warm start OK: log_prior={lp:.3f} "
          f"log_like={ll:.3f} gamma_warm={g_warm:.4f} "
          f"(z_ref, known ~1.8655)", flush=True)

    prov = dict(
        target="v2d native, DIAGONAL masked-err ridge-marg (parity-certified "
               "build_pm('v2d', diagonal=True), refs data/parity_refs.npz)",
        warm_start_provenance="refs['v2d:z_ref:x46'] (foundry-i map_marg_pd "
                              "qz_refined) -> param_map.scene_z_from_old_"
                              "labels (S1 wiring-certified path)",
        warm_start_gamma=g_warm,
        warm_start_log_prior=lp, warm_start_log_like=ll,
        gamma_key=anchor_arb.GAMMA_KEY,
        anchor=dict(gamma=anchor_arb.GAMMA_ANCHOR,
                    ci68=list(anchor_arb.ANCHOR_CI68),
                    sigma=anchor_arb.SIGMA_ANCHOR))

    def gamma_fn(Z):
        return anchor_arb.gamma_of(pm, Z)

    drive(pm, z_warm, gamma_fn, REF, OUT_STEM, extra_prov=prov)


# --------------------------------------------------------------------------- #
def main_toy():
    """CPU driver test on a synthetic quadratic prob-model (dim 3). The
    lensing likelihood is never touched; verifies MAP/SVI/HMC wiring, the
    OOM-ladder pass-through, stage-wise writes and the output schema."""
    import tensorflow_probability.substrates.jax as tfp_  # local alias
    tfd = tfp_.distributions
    tfb = tfp_.bijectors

    dim = 3
    mu_true = jnp.asarray(np.array([0.7, -0.4, 1.2]))
    w = jnp.asarray(np.array([4.0, 1.0, 9.0]))

    class ToyPM:
        high_precision = True
        loss_normalization = 1.0
        prior = tfd.MultivariateNormalDiag(loc=jnp.zeros(dim),
                                           scale_diag=3.0 * jnp.ones(dim))
        bij = tfb.Identity()

        def log_prob(self, z):
            lp = -0.5 * jnp.sum(w * (z - mu_true) ** 2, axis=-1)
            return lp, -2.0 * lp

        def log_prior(self, z):
            return self.prior.log_prob(z)

        def log_like(self, z):
            return self.log_prob(z)

    cfg = dict(REF)
    cfg.update(map_steps=40, svi_nvi=16, svi_steps=60, svi_ladder=(16, 8),
               hmc_n=4, hmc_burn=40, hmc_res=60, hmc_init_l=3, hmc_max_l=5,
               esc_burn=80, esc_res=120, ess_min=8.0, rhat_max=1.2)
    pm = ToyPM()
    z_warm = np.array([0.0, 0.0, 0.0])

    def gamma_fn(Z):        # toy stand-in: column 0
        return np.asarray(Z)[:, 0]

    log = drive(pm, z_warm, gamma_fn, cfg, "l0_toy_classic",
                extra_prov=dict(toy=True))
    got = np.load(DATA / "l0_toy_classic.npz")
    assert "gamma" in got and "chains" in got and "rhat" in got, \
        "toy: output schema incomplete"
    assert np.all(np.isfinite(got["chains"])), "toy: non-finite chains"
    med = np.median(np.asarray(got["chains"]).reshape(-1, dim), axis=0)
    err = np.max(np.abs(med - np.asarray(mu_true)))
    assert err < 0.2, f"toy: posterior median off truth by {err}"
    print(f"[l0-classic] TOY PASS: median-vs-truth max err {err:.4f}, "
          f"accepted_stage={log['accepted_stage']}", flush=True)
    for suff in (".npz", ".json"):
        (DATA / f"l0_toy_classic{suff}").unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toy", action="store_true",
                    help="CPU driver test (synthetic target, tiny constants)")
    args = ap.parse_args()
    if args.toy:
        main_toy()
    else:
        main_production()
