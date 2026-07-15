"""Stage 2 -- MCMC: the certifiable recovery test for the 3-plane demo.

SVI (build a Gaussian surrogate from the MAP point) -> HMC (sample, using the surrogate
covariance as the momentum preconditioner). This is the step that turns the MAP's
degeneracy finding into a certifiable -- or explicitly withdrawn -- recovery claim, by
producing posterior widths + convergence diagnostics + coverage of the truth.

Pre-registered criteria (UNCERTIFIED until the human grades; AGENTS.md rule 1/3):
  * convergence : Max R-hat < 1.01 (accept < 1.1), Min ESS >> n_chains.
  * coverage    : each IDENTIFIABLE truth param inside its 95% credible interval,
                  |pull| = |truth - mean|/std < 2. The dominant z1 mass should pass cleanly.
  * degeneracy  : the weak z2 mass and the compact source (R_sersic, n_sersic) are
                  expected WIDE / prior-influenced (method-discipline §1: report the
                  identifiable combination, not R and n individually). A wide posterior
                  that still CONTAINS truth is success here, not failure.

Memory: SVI/HMC batch chains through the supersampled multiplane sim x GEPL(50), float64
(~0.19 GiB/chain). n_vi=100, n_hmc=50 fit one 40GB A100; do not bump without watching OOM.

Run (inside the canonical Shifter container; from ~/gigalens):
    /usr/bin/python3 -m tests.multiplane_demo.mcmc
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp

from gigalens.jax.scene_prob_model import ImageData, ProbModel
from gigalens.jax.inference import ModellingSequence

from .recover import build_recovery_model, constrained

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

SVI_SEED, HMC_SEED = 0, 0
N_VI, SVI_STEPS = 100, 500
N_HMC, BURNIN, RESULTS = 50, 200, 400


def main():
    nnpz = np.load(os.path.join(ART, "demo_3plane_noisy.npz"))
    mnpz = np.load(os.path.join(ART, "demo_3plane_map.npz"))
    noisy, sigma = nnpz["noisy"], nnpz["sigma"]
    z_map = jnp.asarray(mnpz["z_map"]); z_true = jnp.asarray(mnpz["z_true"])

    model, cfg = build_recovery_model()
    keys = list(model.prior.model.keys())
    n_params = len(keys)
    ds = ImageData(noisy, cfg, error_map=sigma, sees="all")
    prob = ProbModel(model, ds, mode="lstsq")
    seq = ModellingSequence.from_scene(prob)

    print(f"Stage 2 MCMC: {n_params} params, devices={len(jax.devices())}")
    print(f"  SVI: n_vi={N_VI} steps={SVI_STEPS} seed={SVI_SEED}")
    q_z, loss_hist = seq.SVI(start=z_map, optimizer=optax.adam(1e-2),
                             n_vi=N_VI, num_steps=SVI_STEPS, seed=SVI_SEED,
                             pbar_interval=0)
    print(f"  SVI final -ELBO = {float(np.asarray(loss_hist)[-1]):.3f}")

    # Sanitize the SVI surrogate: rebuild from HOST arrays so it carries no Explicit-axis
    # sharding tag (a JAX 0.10 shard_map artifact from SVI) -- otherwise q_z.sample inside
    # HMC's pmap (Manual axis) raises a mesh/axis-type mismatch in the scale_tril matvec.
    loc = np.asarray(jax.device_get(q_z.mean()))
    cov = np.asarray(jax.device_get(q_z.covariance()))
    q_z = tfp.distributions.MultivariateNormalTriL(
        loc=jnp.asarray(loc), scale_tril=jnp.asarray(np.linalg.cholesky(cov)))

    print(f"  HMC: n_hmc={N_HMC} burnin={BURNIN} results={RESULTS} seed={HMC_SEED}")
    samples = np.asarray(seq.HMC(q_z, n_hmc=N_HMC, num_burnin_steps=BURNIN,
                                 num_results=RESULTS, seed=HMC_SEED, pbar_interval=0))
    # collapse any device axis -> (n_chains, n_draws, n_params); then (draws, chains, p)
    samples = samples.reshape(-1, samples.shape[-2], samples.shape[-1]) \
        if samples.ndim == 4 else samples
    if samples.shape[-1] != n_params:                      # safety: last dim = params
        samples = np.moveaxis(samples, np.argmax([s == n_params for s in samples.shape]), -1)
    # to (draws, chains, params)
    draws = np.swapaxes(samples, 0, 1) if samples.shape[0] != RESULTS else samples
    if draws.shape[0] != RESULTS:                          # ensure draws-first
        draws = np.moveaxis(draws, np.argmin([abs(s - RESULTS) for s in draws.shape[:2]]), 0)
    print(f"  HMC samples shape (draws, chains, params) = {draws.shape}")

    # --- convergence (unconstrained space, what HMC sampled) ---
    rhat = np.asarray(tfp.mcmc.potential_scale_reduction(draws, independent_chain_ndims=1))
    ess = np.asarray(tfp.mcmc.effective_sample_size(draws, cross_chain_dims=1))
    print("\n----- convergence (UNCERTIFIED) -----")
    print(f"  Max R-hat = {np.nanmax(rhat):.4f}  (target <1.01, accept <1.1)")
    print(f"  Min ESS   = {np.nanmin(ess):.1f}   (n_chains={draws.shape[1]})")
    worst_r = keys[int(np.nanargmax(rhat))]; worst_e = keys[int(np.nanargmin(ess))]
    print(f"  worst R-hat: {worst_r} = {np.nanmax(rhat):.4f}")
    print(f"  worst ESS  : {worst_e} = {np.nanmin(ess):.1f}")

    # --- coverage (constrained space) ---
    Zflat = jnp.asarray(draws.reshape(-1, n_params))
    cdict = model.bijector.forward(Zflat)          # key -> (M,) constrained
    ctru = constrained(model, z_true)
    print("\n----- coverage: truth vs marginal posterior (UNCERTIFIED) -----")
    print(f"  {'param':40s} {'truth':>8s} {'mean':>8s} {'std':>8s} {'pull':>6s} "
          f"{'in68':>5s} {'in95':>5s}")
    cov = {}
    for k in keys:
        s = np.asarray(cdict[k]); mu = s.mean(); sd = s.std()
        lo68, hi68 = np.percentile(s, [16, 84]); lo95, hi95 = np.percentile(s, [2.5, 97.5])
        t = ctru[k]; pull = (t - mu) / sd if sd > 0 else np.nan
        in68 = lo68 <= t <= hi68; in95 = lo95 <= t <= hi95
        cov[k] = (mu, sd, pull, in68, in95)
        short = k.replace("planes/", "p").replace("/mass/0/", ".M.").replace("/light/0/", ".L.")
        print(f"  {short:40s} {t:8.4f} {mu:8.4f} {sd:8.4f} {pull:6.2f} "
              f"{'Y' if in68 else '.':>5s} {'Y' if in95 else 'NO':>5s}")
    n95 = sum(v[4] for v in cov.values())
    print(f"  coverage: {n95}/{n_params} params contain truth in 95% CI "
          f"(expect most; degenerate ones should still be inside)")

    _figure(draws, keys, cdict, ctru, n_params)
    np.savez(os.path.join(ART, "demo_3plane_mcmc.npz"),
             samples=draws, keys=np.array(keys), rhat=rhat, ess=ess,
             z_true=np.asarray(z_true), svi_seed=SVI_SEED, hmc_seed=HMC_SEED)
    print(f"\nsaved: {os.path.join(ART, 'demo_3plane_mcmc.npz')}")
    print("\nUNCERTIFIED (Mode B). Grade the R-hat/ESS, the coverage table, and the "
          "(R_sersic, n_sersic) joint plot before any recovery claim.")


def _figure(draws, keys, cdict, ctru, n_params):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # trace plots for a representative few + the source (R,n) joint posterior
    show = ["planes/0/mass/0/theta_E", "planes/1/mass/0/theta_E",
            "planes/2/light/0/R_sersic", "planes/2/light/0/n_sersic"]
    show = [k for k in show if k in keys]
    fig, ax = plt.subplots(1, len(show) + 1, figsize=(4.2 * (len(show) + 1), 4.0))
    ki = {k: keys.index(k) for k in show}
    for j, k in enumerate(show):
        tr = draws[:, :, ki[k]]                            # (draws, chains)
        ax[j].plot(tr, lw=0.4, alpha=0.5)
        short = k.replace("planes/", "p").replace("/mass/0/", ".M.").replace("/light/0/", ".L.")
        ax[j].set_title(f"trace: {short}"); ax[j].set_xlabel("draw")
        ax[j].set_ylabel("unconstrained")
    # source (R_sersic, n_sersic) joint, constrained, with truth
    kR, kN = "planes/2/light/0/R_sersic", "planes/2/light/0/n_sersic"
    if kR in keys and kN in keys:
        R = np.asarray(cdict[kR]); n = np.asarray(cdict[kN])
        a = ax[-1]
        a.scatter(R, n, s=2, alpha=0.15, color="steelblue")
        a.scatter([ctru[kR]], [ctru[kN]], marker="*", s=260, color="red",
                  edgecolor="k", zorder=5, label="truth")
        a.set_xlabel("source R_sersic [\"]"); a.set_ylabel("source n_sersic")
        a.set_title("source shape posterior\n(degenerate; truth marked)"); a.legend()
    plt.tight_layout()
    path = os.path.join(ART, "demo_3plane_mcmc.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
