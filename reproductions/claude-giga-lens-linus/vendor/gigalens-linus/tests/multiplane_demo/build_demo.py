"""Build the 3-lens-plane demo image with lenstronomy (the validation oracle), render
the same scene with gigalens, and check they agree. Saves artifacts + a figure.

Run (inside the canonical Shifter container; see GIGALens-Code/docs/env_setup.md):

    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:\
/global/u1/l/linusu/gigalens/src:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    /usr/bin/python3 -m tests.multiplane_demo.build_demo          # from ~/gigalens

Self-checks (printed; must pass before the image is trusted):
  A. lenstronomy ray_shooting_partial(0->z3) == full ray_shooting   (beta convention)
  B. lenstronomy beta == gigalens scene trace beta @ z2 and z3       (independent trace)
Forward-model identity (gigalens render vs lenstronomy render) is reported; it is
expected to plateau at the Sersic b_n convention floor (~2e-3 relative), NOT machine
precision (decision 2a: accept the shipped gigalens Sersic, quantify the offset).
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import astropy.units as u
from astropy.cosmology import FlatwCDM

import jax
jax.config.update("jax_enable_x64", True)

from gigalens.simulator import SimulatorConfig, LensWCS
from gigalens.jax.cosmo import wCDM_Cosmo
from gigalens.jax.profiles.mass.epl import EPL as GEPL
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.scene_simulator import SceneSimulator

from lenstronomy.LensModel.lens_model import LensModel as LLensModel
from lenstronomy.LightModel.light_model import LightModel as LLightModel
from lenstronomy.Util.kernel_util import subgrid_kernel
from scipy.signal import fftconvolve

from . import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")


# ------------------------------------------------------------------ PSF
def gaussian_kernel(fwhm_arcsec=C.PSF_FWHM, pix=C.DELTA_PIX, half_pix=C.PSF_HALF_PIX):
    sigma = fwhm_arcsec / (2.0 * np.sqrt(2.0 * np.log(2.0))) / pix
    ax = np.arange(-half_pix, half_pix + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return (k / k.sum()).astype(np.float64)


# ------------------------------------------------------------------ gigalens scene
def build_gigalens():
    """Return (sim, params, components) for the gigalens forward model of the demo."""
    cosmo = Component(wCDM_Cosmo(z_lens=C.Z1, z_source_ref=10.0),
                      dict(H0=C.COSMO["H0"], Om0=C.COSMO["Om0"], k=0.0, w0=C.COSMO["w0"]))
    c_lens = Component(SersicEllipse(use_lstsq=False), dict(C.LENS_LIGHT_Z2))
    c_sz3 = Component(SersicEllipse(use_lstsq=False), dict(C.SRC_LIGHT_Z3))
    scene = LensModel([
        Plane(redshift=C.Z1, mass=[Component(GEPL(50), dict(C.EPL_Z1))]),
        Plane(redshift=C.Z2, mass=[Component(GEPL(50), dict(C.EPL_Z2))], light=[c_lens]),
        Plane(redshift=C.Z3, light=[c_sz3]),
    ], cosmo=cosmo)
    cfg = SimulatorConfig(delta_pix=C.DELTA_PIX, num_pix=C.NUM_PIX,
                          supersample=C.SUPERSAMPLE, kernel=gaussian_kernel(),
                          likelihood_precision="float64")
    sim = SceneSimulator(scene, cfg)
    assert sim.trace_mode == "multiplane", sim.trace_mode
    return sim, scene.to_params({}), (c_lens, c_sz3)


# ------------------------------------------------------------------ lenstronomy oracle
def lenstronomy_oracle(grid_x, grid_y):
    """Per-plane lensed surface brightness (supersampled) from lenstronomy physics.

    ONE multi-plane model, z_source=z3 (theta_E referenced to z3). Intermediate-plane
    (z2) light is placed via ray_shooting_partial(0->z2) on the SAME model, so the
    deflectors keep one consistent physical mass. Returns a dict of supersampled SB
    arrays and the per-plane beta fields (for the trace self-check)."""
    ap = FlatwCDM(H0=C.COSMO["H0"] * u.km / u.s / u.Mpc, Om0=C.COSMO["Om0"],
                  w0=C.COSMO["w0"], Tcmb0=C.TCMB0, Neff=C.NEFF)
    llens = LLensModel(lens_model_list=["EPL", "EPL"], multi_plane=True,
                       lens_redshift_list=[C.Z1, C.Z2], z_source=C.Z3, cosmo=ap)
    mp = llens.lens_model
    kw = [C.EPL_Z1, C.EPL_Z2]

    def beta_at(z_stop):
        zero = np.zeros_like(grid_x)
        # lenstronomy 1.13.1: ray_shooting_partial returns the ANGULAR position at
        # z_stop directly (self-check A guards this). Deflectors with z < z_stop only.
        x, y, _, _ = mp.ray_shooting_partial(zero, zero, grid_x, grid_y, 0, z_stop, kw,
                                             include_z_start=False)
        return x, y

    bz2 = beta_at(C.Z2)
    bz3 = beta_at(C.Z3)
    fb = mp.ray_shooting(grid_x, grid_y, kw)            # full -> z3, for check A

    light = LLightModel(["SERSIC_ELLIPSE"])

    def sb(beta, p):
        kw_l = dict(amp=p["Ie"], R_sersic=p["R_sersic"], n_sersic=p["n_sersic"],
                    e1=p["e1"], e2=p["e2"], center_x=p["center_x"], center_y=p["center_y"])
        return light.surface_brightness(beta[0], beta[1], [kw_l])

    return dict(
        sb_lens=sb(bz2, C.LENS_LIGHT_Z2),   # z2 lens light (foreground-lensed by z1)
        sb_sz3=sb(bz3, C.SRC_LIGHT_Z3),     # z3 source (lensed by z1 + z2)
        beta_z2=bz2, beta_z3=bz3, beta_z3_full=fb,
    )


def lenstronomy_noz1(grid_x, grid_y):
    """Reference 'z2-plane only' scene: z1 deleted. The z2 lens light is then UNLENSED
    (nothing in front of it) and the z3 source is lensed by the z2 mass alone (single
    plane). Shows what z1 contributes, by difference."""
    ap = FlatwCDM(H0=C.COSMO["H0"] * u.km / u.s / u.Mpc, Om0=C.COSMO["Om0"],
                  w0=C.COSMO["w0"], Tcmb0=C.TCMB0, Neff=C.NEFF)
    llens = LLensModel(lens_model_list=["EPL"], multi_plane=True,
                       lens_redshift_list=[C.Z2], z_source=C.Z3, cosmo=ap)
    bz3x, bz3y = llens.lens_model.ray_shooting(grid_x, grid_y, [C.EPL_Z2])
    light = LLightModel(["SERSIC_ELLIPSE"])

    def sb(bx, by, p):
        kw = dict(amp=p["Ie"], R_sersic=p["R_sersic"], n_sersic=p["n_sersic"],
                  e1=p["e1"], e2=p["e2"], center_x=p["center_x"], center_y=p["center_y"])
        return light.surface_brightness(bx, by, [kw])

    p = C.LENS_LIGHT_Z2
    return dict(
        sb_lens=sb(grid_x, grid_y, p),               # unlensed lens galaxy
        sb_sz3=sb(bz3x, bz3y, C.SRC_LIGHT_Z3),       # z3 source, lensed by z2 only
    )


def lenstronomy_noz2(grid_x, grid_y):
    """Reference scene with the z2 MASS deleted (z1 mass + both lights kept). The z2
    lens light is unchanged (it is only ever lensed by the foreground z1, never by z2's
    own mass), so only the z3 source changes: it is now lensed by z1 alone. The
    (full) - (no z2) difference therefore isolates z2's lensing contribution to the
    z3 ring."""
    ap = FlatwCDM(H0=C.COSMO["H0"] * u.km / u.s / u.Mpc, Om0=C.COSMO["Om0"],
                  w0=C.COSMO["w0"], Tcmb0=C.TCMB0, Neff=C.NEFF)
    llens = LLensModel(lens_model_list=["EPL"], multi_plane=True,
                       lens_redshift_list=[C.Z1], z_source=C.Z3, cosmo=ap)
    bz3x, bz3y = llens.lens_model.ray_shooting(grid_x, grid_y, [C.EPL_Z1])
    light = LLightModel(["SERSIC_ELLIPSE"])
    p = C.SRC_LIGHT_Z3
    kw = dict(amp=p["Ie"], R_sersic=p["R_sersic"], n_sersic=p["n_sersic"],
              e1=p["e1"], e2=p["e2"], center_x=p["center_x"], center_y=p["center_y"])
    return dict(sb_sz3=light.surface_brightness(bz3x, bz3y, [kw]))  # z3 lensed by z1 only


def finalize(sb, kernel):
    """supersampled SB -> native image: subgrid PSF conv + mean-pool + pixel area."""
    ksub = subgrid_kernel(kernel, C.SUPERSAMPLE, odd=True, num_iter=100)
    ksub = ksub / ksub.sum()
    conv = fftconvolve(sb, ksub, mode="same")
    pooled = conv.reshape(C.NUM_PIX, C.SUPERSAMPLE, C.NUM_PIX, C.SUPERSAMPLE).mean(axis=(1, 3))
    return pooled * (C.DELTA_PIX ** 2)                 # conversion_factor = det = dpix^2


# ------------------------------------------------------------------ figure
def make_figure(imgs, resid, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm, SymLogNorm
    img = imgs["total"]
    half = C.NUM_PIX * C.DELTA_PIX / 2
    ext = [-half, half, -half, half]
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))

    def show(ax, a, title, stretch="sqrt", cmap="magma", vmax=None):
        vmax = img.max() if vmax is None else vmax
        if stretch == "sqrt":
            norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
        else:  # linear
            norm = None
        im = ax.imshow(a, origin="lower", extent=ext, norm=norm,
                       vmin=(0.0 if norm is None else None),
                       vmax=(vmax if norm is None else None), cmap=cmap)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)

    def show_diverging(ax, a, title):
        rr = float(np.max(np.abs(a))) + 1e-30
        im = ax.imshow(a, origin="lower", extent=ext, cmap="RdBu_r",
                       norm=SymLogNorm(linthresh=rr * 1e-3, vmin=-rr, vmax=rr))
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)

    # row 0: the full 3-plane system
    show(axes[0, 0], img, "TOTAL with z1 (sqrt)")
    show(axes[0, 1], img, "TOTAL with z1 (linear)", stretch="linear")
    show(axes[0, 2], imgs["lens"], "z2 lens light (z1-lensed, sqrt)")
    show(axes[0, 3], imgs["sz3"], "z3 source: z1+z2 ring (sqrt)", vmax=imgs["sz3"].max())
    # row 1: the z2-only reference + what z1 adds + validation
    show(axes[1, 0], imgs["noz1"], "z2-plane only: NO z1 (sqrt)")
    show(axes[1, 1], imgs["noz1"], "z2-plane only: NO z1 (linear)", stretch="linear")
    show_diverging(axes[1, 2], imgs["z2_effect"], "z2 contribution\n(with z2) - (no z2)")
    show_diverging(axes[1, 3], resid,
                   f"gigalens - lenstronomy\nmax|r|={np.max(np.abs(resid)):.2e} (b_n floor)")
    for ax in axes.flat:
        if ax.has_data():
            ax.set_xlabel("arcsec"); ax.set_ylabel("arcsec")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    os.makedirs(ART, exist_ok=True)
    kernel = gaussian_kernel()

    sim, params, _ = build_gigalens()
    wcs = LensWCS(n=C.NUM_PIX, supersample=C.SUPERSAMPLE, pix_scale=C.DELTA_PIX)
    gx, gy = (np.asarray(a, np.float64) for a in wcs.pixel_grid())

    orc = lenstronomy_oracle(gx, gy)

    # --- self-check A: partial(0->z3) == full ray_shooting ---
    errA = max(np.max(np.abs(orc["beta_z3"][0] - orc["beta_z3_full"][0])),
               np.max(np.abs(orc["beta_z3"][1] - orc["beta_z3_full"][1])))
    # --- self-check B: lenstronomy beta == gigalens trace beta ---
    tr = sim.trace(params)
    gz2 = (np.asarray(tr[1][0])[..., 0], np.asarray(tr[1][1])[..., 0])
    gz3 = (np.asarray(tr[2][0])[..., 0], np.asarray(tr[2][1])[..., 0])
    errB2 = max(np.max(np.abs(gz2[0] - orc["beta_z2"][0])),
                np.max(np.abs(gz2[1] - orc["beta_z2"][1])))
    errB3 = max(np.max(np.abs(gz3[0] - orc["beta_z3"][0])),
                np.max(np.abs(gz3[1] - orc["beta_z3"][1])))
    print(f"[check A] partial(0->z3) vs full ray_shooting : max|dbeta| = {errA:.2e}")
    print(f"[check B] gigalens vs lenstronomy beta @z2     : max|dbeta| = {errB2:.2e}")
    print(f"[check B] gigalens vs lenstronomy beta @z3     : max|dbeta| = {errB3:.2e}")

    # --- render ---
    img_lens = finalize(orc["sb_lens"], kernel)
    img_sz3 = finalize(orc["sb_sz3"], kernel)
    img = img_lens + img_sz3
    gimg = np.asarray(sim.simulate(params))
    resid = gimg - img

    # --- 'z2-plane only' reference (z1 deleted) ---
    noz1 = lenstronomy_noz1(gx, gy)
    img_noz1 = finalize(noz1["sb_lens"], kernel) + finalize(noz1["sb_sz3"], kernel)
    # --- z2 contribution: (full) - (z2 mass deleted). Lens light is unchanged by
    #     removing z2's mass, so this isolates z2's lensing of the z3 ring. ---
    noz2 = lenstronomy_noz2(gx, gy)
    img_noz2 = img_lens + finalize(noz2["sb_sz3"], kernel)
    z2_effect = img - img_noz2

    print(f"\n[forward-model identity] gigalens render vs lenstronomy render:")
    print(f"   max|resid|/max(img) = {np.max(np.abs(resid)) / img.max():.3e}")
    print(f"   rms(resid)/rms(img) = "
          f"{np.sqrt((resid**2).mean()) / np.sqrt((img**2).mean()):.3e}  (Sersic b_n floor)")

    print("\n[physicality]")
    print(f"   z(1,2,3) = {C.Z1}, {C.Z2}, {C.Z3}   theta_E = "
          f"{C.EPL_Z1['theta_E']}\", {C.EPL_Z2['theta_E']}\"")
    print(f"   FOV {C.NUM_PIX*C.DELTA_PIX:.1f}\"  @ {C.DELTA_PIX}\"/pix  "
          f"supersample {C.SUPERSAMPLE}  PSF FWHM {C.PSF_FWHM}\"")
    for name, a in [("z2 lens light", img_lens),
                    ("z3 source", img_sz3), ("TOTAL", img)]:
        print(f"   {name:14s} total_flux={a.sum():10.3f}  peak={a.max():9.4f}")
    print(f"   dynamic range (max/median+) = {img.max()/np.median(img[img>0]):.1f}")

    fig_path = os.path.join(ART, "demo_3plane.png")
    np.savez(os.path.join(ART, "demo_3plane.npz"),
             img=img, img_lens=img_lens, img_sz3=img_sz3,
             img_noz1=img_noz1, img_noz2=img_noz2, z2_effect=z2_effect,
             gimg=gimg, resid=resid, kernel=kernel)
    make_figure(dict(total=img, lens=img_lens, sz3=img_sz3,
                     noz1=img_noz1, z2_effect=z2_effect), resid, fig_path)
    print(f"\nsaved: {fig_path}")
    print(f"saved: {os.path.join(ART, 'demo_3plane.npz')}")


if __name__ == "__main__":
    main()
