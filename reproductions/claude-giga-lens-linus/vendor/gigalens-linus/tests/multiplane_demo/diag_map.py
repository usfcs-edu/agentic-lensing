"""MAP diagnostics (no re-fit): load the saved MAP, then LOOK before trusting numbers.

Two questions the MAP param table alone cannot answer (method-discipline §5: plots before
metrics; project-standards §2/§7: a good chi^2 is not evidence of correct recovery):

  1. Is the MAP an *equivalent* fit to truth, or did it beat/lose to truth? Compare
     reduced chi^2(truth) vs reduced chi^2(MAP) on the SAME noisy data. If they are equal
     to ~the chi^2 spread, the data cannot distinguish them -> the off params (z2 mass,
     source R/n) are a DEGENERACY (wide posterior), not a point-estimate bug. If MAP is
     much lower, the prior is pulling / truth isn't optimal -> investigate.
  2. Does the MAP residual look like N(0,1)? Structure (arcs, dipoles, point-in-ring)
     near the ring or the source would betray real mismodelling that chi^2~1 averaged away.

Run (inside the canonical Shifter container):
    /usr/bin/python3 -m tests.multiplane_demo.diag_map        # from ~/gigalens
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.jax.scene_prob_model import ImageData, ProbModel

from .recover import build_recovery_model

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")


def _model_image(sim, model, ds, z):
    params = model.to_params(model.bijector.forward(list(jnp.atleast_2d(z).T)))
    return np.asarray(sim.lstsq_simulate(params, ds.image, ds.error_map, ds.mask))


def main():
    nnpz = np.load(os.path.join(ART, "demo_3plane_noisy.npz"))
    mnpz = np.load(os.path.join(ART, "demo_3plane_map.npz"))
    noisy, sigma = nnpz["noisy"], nnpz["sigma"]
    z_map = jnp.asarray(mnpz["z_map"]); z_true = jnp.asarray(mnpz["z_true"])

    model, cfg = build_recovery_model()
    sim = SceneSimulator(model, cfg)
    ds = ImageData(noisy, cfg, error_map=sigma, sees="all")
    prob = ProbModel(model, ds, mode="lstsq")
    N = ds.event_size
    spread = np.sqrt(2.0 / N)

    _, rc_true = prob.log_like(z_true, simulator=sim)
    _, rc_map = prob.log_like(z_map, simulator=sim)
    rc_true = float(np.asarray(rc_true).reshape(-1)[0])
    rc_map = float(np.asarray(rc_map).reshape(-1)[0])

    print("\n----- MAP vs truth fit on the SAME noisy data -----")
    print(f"  reduced chi^2(truth) = {rc_true:.4f}")
    print(f"  reduced chi^2(MAP)   = {rc_map:.4f}")
    print(f"  difference           = {rc_map - rc_true:+.4f}   (chi^2 spread sqrt(2/N) = {spread:.4f})")
    if abs(rc_map - rc_true) < spread:
        print("  => MAP and truth are statistically EQUIVALENT fits: the data cannot")
        print("     distinguish them -> off params are a DEGENERACY (wide posterior),")
        print("     NOT a point-estimate failure. The MCMC posterior width is the test.")
    elif rc_map < rc_true - spread:
        print("  => MAP fits NOTABLY better than truth: prior pull or a too-flexible")
        print("     direction is absorbing noise -> scrutinize before trusting widths.")
    else:
        print("  => MAP fits WORSE than truth: MAP did not converge -> more steps/starts.")

    # images at MAP and truth
    im_map = _model_image(sim, model, ds, z_map)
    im_true = _model_image(sim, model, ds, z_true)
    nres_map = (noisy - im_map) / sigma
    nres_true = (noisy - im_true) / sigma
    print(f"\n  normalized-residual stats (should be ~N(0,1)):")
    print(f"    MAP   : mean={nres_map.mean():+.3f}  std={nres_map.std():.3f}  "
          f"max|.|={np.abs(nres_map).max():.2f}  frac|.|>3={np.mean(np.abs(nres_map)>3):.4f}")
    print(f"    truth : mean={nres_true.mean():+.3f}  std={nres_true.std():.3f}  "
          f"max|.|={np.abs(nres_true).max():.2f}  frac|.|>3={np.mean(np.abs(nres_true)>3):.4f}")

    # figure: data / MAP model / normalized residual (MAP) / normalized residual (truth)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    import tests.multiplane_demo.config as Cc
    half = Cc.NUM_PIX * Cc.DELTA_PIX / 2
    ext = [-half, half, -half, half]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    vmax = noisy.max()
    a0 = ax[0].imshow(noisy, origin="lower", extent=ext, cmap="magma",
                      norm=PowerNorm(0.5, vmin=0, vmax=vmax)); ax[0].set_title("noisy data (sqrt)")
    a1 = ax[1].imshow(im_map, origin="lower", extent=ext, cmap="magma",
                      norm=PowerNorm(0.5, vmin=0, vmax=vmax)); ax[1].set_title("MAP model (sqrt)")
    a2 = ax[2].imshow(nres_map, origin="lower", extent=ext, cmap="RdBu_r", vmin=-4, vmax=4)
    ax[2].set_title("MAP normalized residual\n(data-model)/sigma")
    a3 = ax[3].imshow(nres_true, origin="lower", extent=ext, cmap="RdBu_r", vmin=-4, vmax=4)
    ax[3].set_title("truth normalized residual\n(data-model)/sigma")
    for a, im in zip(ax, (a0, a1, a2, a3)):
        a.set_xlabel("arcsec"); a.set_ylabel("arcsec"); plt.colorbar(im, ax=a, fraction=0.046)
    plt.tight_layout()
    path = os.path.join(ART, "demo_3plane_map_resid.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
