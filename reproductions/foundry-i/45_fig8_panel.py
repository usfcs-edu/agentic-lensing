"""Stage F: render the Fig.-8 analogue — observed | model + critical curves |
reduced residual | unlensed source + caustics (cf. Huang 2025a Fig. 8).

Critical curves: zero contour of det(A) where A is the Jacobian of the lens
mapping, computed by central differences of the gigalens deflection field on a
fine grid. Caustics: the critical-curve vertices mapped through the lens
equation beta = theta - alpha(theta). Source panel: the fitted Sersic +
shapelet source evaluated in the source plane (no deflection).

Input: an HMC posterior (data/hmc_v13_<tag>.npz, uses the posterior median) or
a MAP fit (data/map_v11_<tag>.npz, uses best_z).

Run:  python 45_fig8_panel.py --params data/hmc_v13_prod.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

import _data_lib as D

REPRO = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default="data/hmc_v13_prod.npz")
    ap.add_argument("--out", type=str, default="figs/ours_foundry-i_fig8.png")
    ap.add_argument("--n-max", type=int, default=D.N_MAX)
    ap.add_argument("--data", type=str, default="cutout_v2.npz")
    ap.add_argument("--grid", type=int, default=480, help="critical-curve grid")
    ap.add_argument("--src-fov", type=float, default=2.4, help="source panel FOV (arcsec)")
    args = ap.parse_args()

    D.bootstrap_vendor()
    import jax
    import jax.experimental.shard_map  # noqa: F401
    import jax.numpy as jnp
    from gigalens.jax.simulator import LensSimulator

    d, prior, phys, prob, sim_config = D.build_all(n_max=args.n_max, data_file=args.data)
    img, err, mask = d["img"], d["err_map"], d["keep_mask"]
    npix = img.shape[0]
    fov = npix * float(d["meta"].get("delta_pix", D.DELTA_PIX))

    z = np.load(REPRO / args.params)
    if "samples" in z.files:          # HMC posterior -> median in z-space
        s = z["samples"]              # (draws, chains, dim)
        z_best = np.median(s.reshape(-1, s.shape[-1]), axis=0).astype(np.float32)
        src_label = f"posterior median ({Path(args.params).name})"
    else:                              # MAP
        z_best = z["best_z"].astype(np.float32)
        src_label = f"MAP best ({Path(args.params).name})"

    # ---- forward model at the best parameters (bs=2 to dodge the squeeze) --
    sim = LensSimulator(phys, sim_config, bs=2)
    zb = jnp.asarray(np.stack([z_best, z_best]))
    x = prob.bij.forward(list(zb.T))
    model = np.asarray(sim.simulate(x))[0]

    chi_map = (model - img) / err
    red_chi2 = float(np.mean(chi_map[mask] ** 2))

    # ---- physical params (batch element 0) ---------------------------------
    def first(v):
        return float(np.asarray(v).reshape(-1)[0])
    lens_params = [{k: first(v) for k, v in p.items()} for p in x[0]]
    # source params keep a length-1 batch axis: the shapelets einsums expect
    # amp arrays of shape (n_amps, batch) and grids with a trailing batch dim
    src_params = [{k: np.asarray(v, dtype=np.float32).reshape(-1)[:1]
                   for k, v in p.items()} for p in x[2]]

    # ---- critical curves: det(Jacobian) on a fine grid ---------------------
    g = args.grid
    half = fov / 2.0
    lin = np.linspace(-half, half, g, dtype=np.float64)
    gx, gy = np.meshgrid(lin, lin)

    def deflection(gx, gy):
        ax = np.zeros_like(gx)
        ay = np.zeros_like(gy)
        for lens, p in zip(phys.lenses, lens_params):
            fxi, fyi = lens.deriv(jnp.asarray(gx, dtype=jnp.float32),
                                  jnp.asarray(gy, dtype=jnp.float32), **p)
            ax = ax + np.asarray(fxi, dtype=np.float64)
            ay = ay + np.asarray(fyi, dtype=np.float64)
        return ax, ay

    ax, ay = deflection(gx, gy)
    h = lin[1] - lin[0]
    daxdx = np.gradient(ax, h, axis=1)
    daxdy = np.gradient(ax, h, axis=0)
    daydx = np.gradient(ay, h, axis=1)
    daydy = np.gradient(ay, h, axis=0)
    detA = (1 - daxdx) * (1 - daydy) - daxdy * daydx

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # extract critical curves as contour vertices, then map to caustics
    cs = plt.contour(gx, gy, detA, levels=[0.0])
    raw_paths = [p.vertices for p in cs.collections[0].get_paths()] \
        if hasattr(cs, "collections") else [p.vertices for p in cs.get_paths()]
    plt.close()
    # split contour paths on coordinate jumps (matplotlib joins disjoint loops),
    # and drop vertices too close to the profile center (numerically singular)
    crit_paths = []
    for v in raw_paths:
        r = np.hypot(v[:, 0], v[:, 1])
        v = v[r > 0.15]
        if len(v) < 4:
            continue
        jumps = np.where(np.hypot(*np.diff(v, axis=0).T) > 0.4)[0]
        for seg in np.split(v, jumps + 1):
            if len(seg) >= 4:
                crit_paths.append(seg)
    caustic_paths = []
    for v in crit_paths:
        cax, cay = deflection(v[:, 0][None, :], v[:, 1][None, :])
        b = np.column_stack([v[:, 0] - cax.ravel(), v[:, 1] - cay.ravel()])
        b = b[np.hypot(b[:, 0], b[:, 1]) < 3.0]
        if len(b) >= 4:
            caustic_paths.append(b)

    # ---- source-plane reconstruction ---------------------------------------
    sh = args.src_fov / 2.0
    slin = np.linspace(-sh, sh, 300, dtype=np.float32)
    sgx, sgy = np.meshgrid(slin, slin)
    src_img = np.zeros_like(sgx, dtype=np.float64)
    for light, p in zip(phys.source_light, src_params):
        contrib = light.light(jnp.asarray(sgx)[..., None],
                              jnp.asarray(sgy)[..., None], **p)
        src_img += np.asarray(jnp.sum(contrib, axis=-1) if contrib.ndim == 3
                              else contrib, dtype=np.float64)

    # ---- figure -------------------------------------------------------------
    ext = [-half, half, -half, half]
    sext = [-sh, sh, -sh, sh]
    fig, axes = plt.subplots(1, 4, figsize=(19, 5))
    scale = 3 * np.median(err)
    v0 = np.arcsinh(img / scale)
    v1 = np.arcsinh(model / scale)
    vmax = np.percentile(v0, 99.7)

    axes[0].imshow(v0, origin="lower", cmap="magma", extent=ext, vmax=vmax)
    axes[0].set_title("HST F140W (sky-subtracted)")
    axes[1].imshow(v1, origin="lower", cmap="magma", extent=ext, vmax=vmax)
    for v in crit_paths:
        axes[1].plot(v[:, 0], v[:, 1], "r-", lw=1.2)
    axes[1].set_title("best-fit model + critical curves")
    resid_show = np.where(mask, chi_map, np.nan)
    im2 = axes[2].imshow(resid_show, origin="lower", cmap="RdBu_r",
                         vmin=-4, vmax=4, extent=ext)
    axes[2].set_title(f"reduced residual (masked red. $\\chi^2$={red_chi2:.2f})")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    axes[3].imshow(np.arcsinh(src_img / max(src_img.max() / 50, 1e-6)),
                   origin="lower", cmap="magma", extent=sext)
    for v in caustic_paths:
        axes[3].plot(v[:, 0], v[:, 1], "-", color="lime", lw=1.2)
    axes[3].set_xlim(sext[0], sext[1])
    axes[3].set_ylim(sext[2], sext[3])
    axes[3].set_title("source reconstruction + caustics")
    for a in axes:
        a.set_xlabel(r"$\Delta\alpha$ [arcsec]")
    axes[0].set_ylabel(r"$\Delta\delta$ [arcsec]")
    fig.suptitle(f"DESI-165.4754$-$06.0423 — {src_label}", y=1.02)
    fig.tight_layout()
    out = REPRO / args.out
    fig.savefig(out, dpi=140, bbox_inches="tight")

    mass = {k: float(v[0]) for k, v in
            D.mass_params_from_z(prob, z_best[None, :]).items()}
    n_crit = len(crit_paths)
    summary = dict(params=args.params, red_chi2_masked=red_chi2,
                   n_critical_curves=n_crit,
                   inner_critical_curve=bool(n_crit >= 2),
                   gamma=mass["gamma"], theta_E=mass["theta_E"],
                   out=str(out))
    (REPRO / "data" / "fig8_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
