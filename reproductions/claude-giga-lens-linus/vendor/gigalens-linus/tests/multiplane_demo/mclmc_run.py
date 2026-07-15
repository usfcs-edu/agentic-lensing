"""Stage 2 (take 2) -- MAP -> MCLMC (no SVI): the certifiable recovery test.

The SVI->HMC attempt (mcmc.py) did not converge (Max R-hat 1.92): the degenerate /
correlated directions (z2 lens-light position, z1 theta_E, z2 slope) mixed too slowly.
MCLMC (microcanonical Langevin Monte Carlo) is a more efficient sampler that adapts its
own step size, trajectory length L, and mass matrix during burn-in, so it copes better
with a crude start/preconditioner.

Surrogate: built directly from the MAP with a small DEFAULT covariance (1e-6 * I), the
canonical pattern (repro_mclmc_238.py) -- MCLMC's windowed-Welford adaptation replaces it
during burn-in. No SVI.

Same pre-registered grading criteria as before (UNCERTIFIED until the human grades):
Max R-hat < 1.01 (accept <1.1), Min ESS >> n_chains; coverage of identifiable truth in
95% CI (|pull|<2); the source (R_sersic, n_sersic) reported as the identifiable
combination, z2 mass expected wide.

Run (gigalens_env conda python, native JAX 0.9.1 -- no container needed):
    python -m tests.multiplane_demo.mclmc_run        # from ~/gigalens
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from gigalens.jax.scene_prob_model import ImageData, ProbModel
from gigalens.jax.inference import ModellingSequence
from gigalens.jax.experimental.mclmc import MCLMC_JIT

from .recover import build_recovery_model, constrained

tfd = tfp.distributions
HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

SEED = 0
# MCLMC is efficient per chain: 8 chains with generous burn-in (so L / step-size / mass
# matrix fully adapt) and equal results is a better config than many short chains.
# (The first run used 32/1000/2000; per guidance 8/2000/2000 is preferred going forward.)
N_CHAINS = 8
N_BURNIN = 2000
N_RESULTS = 2000


def main():
    nnpz = np.load(os.path.join(ART, "demo_3plane_noisy.npz"))
    mnpz = np.load(os.path.join(ART, "demo_3plane_map.npz"))
    noisy, sigma = nnpz["noisy"], nnpz["sigma"]
    z_map = jnp.asarray(mnpz["z_map"]); z_true = jnp.asarray(mnpz["z_true"])

    model, cfg = build_recovery_model()
    keys = list(model.prior.model.keys()); n_params = len(keys)
    ds = ImageData(noisy, cfg, error_map=sigma, sees="all")
    prob = ProbModel(model, ds, mode="lstsq")
    seq = ModellingSequence.from_scene(prob)

    # surrogate from MAP with small DEFAULT covariance (MCLMC adapts the rest)
    default_start = jnp.diag(jnp.ones((n_params,), jnp.float64)) * 1e-3
    qz = tfd.MultivariateNormalTriL(loc=jnp.squeeze(z_map), scale_tril=default_start)

    print(f"Stage 2 (MCLMC): {n_params} params, {N_CHAINS} chains, "
          f"burnin={N_BURNIN} results={N_RESULTS} seed={SEED}, devices={len(jax.devices())}")
    samples = MCLMC_JIT(seq, qz, n_hmc=N_CHAINS, num_burnin_steps=N_BURNIN,
                        num_results=N_RESULTS, seed=SEED, progress_bar=False)
    samples = np.asarray(samples)                          # (n_chains, n_results, n_params)
    print(f"  raw MCLMC samples shape = {samples.shape}")
    draws = np.swapaxes(samples, 0, 1)                     # (draws, chains, params)
    finite = np.isfinite(draws).all(axis=(0, 2))           # drop any all-NaN chain
    if not finite.all():
        print(f"  WARNING: {int((~finite).sum())}/{draws.shape[1]} chains non-finite -> dropped")
        draws = draws[:, finite, :]
    print(f"  usable (draws, chains, params) = {draws.shape}")

    # --- convergence (unconstrained) ---
    rhat = np.asarray(tfp.mcmc.potential_scale_reduction(draws, independent_chain_ndims=1))
    ess = np.asarray(tfp.mcmc.effective_sample_size(draws, cross_chain_dims=1))
    print("\n----- convergence (UNCERTIFIED) -----")
    print(f"  Max R-hat = {np.nanmax(rhat):.4f}  (target <1.01, accept <1.1)")
    print(f"  Min ESS   = {np.nanmin(ess):.1f}   (n_chains={draws.shape[1]})")
    o = np.argsort(rhat)[::-1]
    for i in o[:6]:
        short = keys[i].replace("planes/", "p").replace("/mass/0/", ".M.").replace("/light/0/", ".L.")
        print(f"    {short:30s} R-hat={rhat[i]:.3f}  ESS={ess[i]:.0f}")

    # --- coverage (constrained) ---
    Zflat = jnp.asarray(draws.reshape(-1, n_params))
    cdict = model.bijector.forward(Zflat)
    ctru = constrained(model, z_true)
    print("\n----- coverage: truth vs marginal posterior (UNCERTIFIED) -----")
    print(f"  {'param':40s} {'truth':>8s} {'mean':>8s} {'std':>8s} {'pull':>6s} {'in95':>5s}")
    n95 = 0
    for k in keys:
        s = np.asarray(cdict[k]); mu = s.mean(); sd = s.std()
        lo95, hi95 = np.percentile(s, [2.5, 97.5]); t = ctru[k]
        pull = (t - mu) / sd if sd > 0 else np.nan; in95 = lo95 <= t <= hi95
        n95 += int(in95)
        short = k.replace("planes/", "p").replace("/mass/0/", ".M.").replace("/light/0/", ".L.")
        print(f"  {short:40s} {t:8.4f} {mu:8.4f} {sd:8.4f} {pull:6.2f} {'Y' if in95 else 'NO':>5s}")
    print(f"  coverage: {n95}/{n_params} params contain truth in 95% CI")

    _figure(draws, keys, cdict, ctru)
    np.savez(os.path.join(ART, "demo_3plane_mclmc.npz"),
             samples=draws, keys=np.array(keys), rhat=rhat, ess=ess,
             z_true=np.asarray(z_true), seed=SEED, n_chains=N_CHAINS,
             n_burnin=N_BURNIN, n_results=N_RESULTS)
    print(f"\nsaved: {os.path.join(ART, 'demo_3plane_mclmc.npz')}")
    print("\nUNCERTIFIED (Mode B). Grade R-hat/ESS, the coverage table, and the "
          "(R_sersic, n_sersic) joint before any recovery claim.")


def _figure(draws, keys, cdict, ctru):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["planes/0/mass/0/theta_E", "planes/1/light/0/center_x",
            "planes/1/mass/0/theta_E", "planes/2/light/0/R_sersic"]
    show = [k for k in show if k in keys]
    fig, ax = plt.subplots(1, len(show) + 1, figsize=(4.2 * (len(show) + 1), 4.0))
    nshow = min(12, draws.shape[1])
    for j, k in enumerate(show):
        tr = draws[:, :nshow, keys.index(k)]
        ax[j].plot(tr, lw=0.5, alpha=0.6)
        short = k.replace("planes/", "p").replace("/mass/0/", ".M.").replace("/light/0/", ".L.")
        ax[j].set_title(f"trace: {short}"); ax[j].set_xlabel("draw"); ax[j].set_ylabel("uncon")
    kR, kN = "planes/2/light/0/R_sersic", "planes/2/light/0/n_sersic"
    if kR in keys and kN in keys:
        a = ax[-1]
        a.scatter(np.asarray(cdict[kR]), np.asarray(cdict[kN]), s=2, alpha=0.12, color="steelblue")
        a.scatter([ctru[kR]], [ctru[kN]], marker="*", s=260, color="red", edgecolor="k", zorder=5, label="truth")
        a.set_xlabel("source R_sersic [\"]"); a.set_ylabel("source n_sersic")
        a.set_title("source shape posterior\n(MCLMC; truth marked)"); a.legend()
    plt.tight_layout()
    path = os.path.join(ART, "demo_3plane_mclmc.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
