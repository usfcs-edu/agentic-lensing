"""Track-A Stage-A binned product: 2x2 block-sum of the 0.04" fine skycell cutout.

Builds cutout_v3b.npz, a 130x130 @ 0.08"/px rebin of cutout_v3.npz:
  * IMAGE: 2x2 block-SUM (the simulator outputs integrated flux per output
    pixel, so binned data = sum of fine fluxes).
  * MASK: a binned pixel is kept only if all 4 fine pixels are kept.
  * NOISE: the stored fine err_map is decomposed into Poisson + sky terms;
    the sky variance is block-summed in quadrature (knowingly wrong under
    drizzle-induced correlation) and then recalibrated against binned sky
    pixels in the spirit of 40b_make_cutout_fine.py l.127-146. The gap between
    the naive-quadrature sky chi^2 and 1.0 quantifies intra-block covariance.
  * PSF: empirical_psf_04.npy (51x51 @ 0.04) degraded x2 with the
    flux-conserving lenstronomy degrade_kernel -> 27x27 @ 0.08, the
    simulator-convention kernel for the supersample=2 config (sampled at the
    OUTPUT delta_pix, upsampled internally by subgrid_kernel).
    NOTE: the subgrid_kernel(psf_08, 2, odd=True) round-trip cosine vs
    empirical_psf_04 has a HARD CEILING ~0.872 for ANY 27x27 kernel: the
    F140W PSF core (FWHM ~0.13" ~ 1.6 px @ 0.08") is undersampled, and
    subgrid_kernel output is constrained to bilinear upsamples of the input
    (re_size_array uses kx=ky=1). The script computes that ceiling by exact
    projection and reports it; the decisive PSF-fidelity test is the
    binned-domain self-consistency render check (binning low-passes exactly
    the frequencies the 0.08 kernel cannot represent).

Outputs: data/cutout_v3b.npz, data/empirical_psf_08.npy,
         data/cutout_v3b_stats.json, figs/cutout_v3b_masks.png

Run:  python 40c_make_cutout_binned.py                   # build + GPU render check
      JAX_PLATFORMS=cpu python 40c_make_cutout_binned.py --skip-render-check
      python 40c_make_cutout_binned.py --fallback-supersample1  # re-emit at supersample=1
"""
import argparse
import json
from pathlib import Path

import numpy as np
from astropy.stats import sigma_clipped_stats

import _data_lib as D

REPRO = Path(__file__).resolve().parent
DATA = REPRO / "data"
FIGS = REPRO / "figs"

PARENT = "cutout_v3.npz"
DELTA_PIX = 0.08
SKY_R_ARC = 4.5                # arcsec, outside the arc annulus (40b ARC_ANNULUS[1])
N_MAX_BY_DIM = {74: 6, 91: 8}  # unconstrained-z dimension -> shapelet order


def blocksum(a):
    """2x2 block-sum of a square array with even side (float64 path)."""
    n = a.shape[0] // 2
    return a.reshape(n, 2, a.shape[1] // 2, 2).sum(axis=(1, 3))


def blockall(m):
    """2x2 block-AND of a boolean mask."""
    n = m.shape[0] // 2
    return m.reshape(n, 2, m.shape[1] // 2, 2).all(axis=(1, 3))


def kernel_com_arcsec(k, delta_pix):
    """Center-of-mass offset of a kernel from its center pixel, in arcsec."""
    yy, xx = np.indices(k.shape)
    cen = (k.shape[0] - 1) / 2.0
    s = k.sum()
    return (((k * xx).sum() / s - cen) * delta_pix,
            ((k * yy).sum() / s - cen) * delta_pix)


def roundtrip_cos_ceiling(psf_fine):
    """Max achievable flatten-cosine of subgrid_kernel(k08, 2, odd=True) vs psf_fine.

    subgrid_kernel's output is always re_size_array(...) of a 27x27 kernel,
    i.e. a BILINEAR upsample (RectBivariateSpline kx=ky=1) onto the 53x53
    subgrid. Project psf_fine onto that linear range (central 51x51 crop) by
    least squares -- no choice of psf_08 can beat this cosine.
    """
    from scipy.interpolate import RectBivariateSpline
    n_in, n_out = 27, 53
    dx = 1.0 / n_in
    x_in = np.linspace(dx / 2, 1 - dx / 2, n_in)
    d2 = 1.0 / n_out
    x_out = np.linspace(d2 / 2, 1 - d2 / 2, n_out)
    L = np.zeros((n_out * n_out, n_in * n_in))
    e = np.zeros((n_in, n_in))
    for i in range(n_in):
        for j in range(n_in):
            e[i, j] = 1.0
            f = RectBivariateSpline(x_in, x_in, e, kx=1, ky=1, s=0)
            L[:, i * n_in + j] = f(x_out, x_out).ravel()
            e[i, j] = 0.0
    idx = np.arange(n_out * n_out).reshape(n_out, n_out)[1:-1, 1:-1].ravel()
    Lc = L[idx]
    t = psf_fine.ravel()
    k_ls, *_ = np.linalg.lstsq(Lc, t, rcond=None)
    u = Lc @ k_ls
    return float(u @ t / np.sqrt((u @ u) * (t @ t)))


def build_psf_08(psf_fine):
    """Flux-conserving x2 degrade of the 0.04" empirical PSF -> 27x27 @ 0.08.

    lenstronomy degrade_kernel pads 51->53 and block-averages with the
    half/quarter edge weights that are the exact adjoint of the centered
    even-factor convention used by subgrid_kernel -- it preserves total flux
    and the center-of-mass exactly.
    """
    from lenstronomy.Util.kernel_util import degrade_kernel, subgrid_kernel

    psf_08 = degrade_kernel(psf_fine.astype(np.float64), 2)
    sum_before = float(psf_08.sum())
    psf_08 = psf_08 / psf_08.sum()                       # check (iii): renorm to 1

    # check (i): round-trip cosine vs the fine PSF on a common 51x51 center
    up = subgrid_kernel(psf_08, 2, odd=True)             # 53x53 @ 0.04
    c, h = up.shape[0] // 2, psf_fine.shape[0] // 2
    crop = up[c - h:c + h + 1, c - h:c + h + 1].ravel()
    t = psf_fine.ravel()
    cos = float(crop @ t / np.sqrt((crop @ crop) * (t @ t)))

    # check (ii): centroid shift in arcsec
    cx4, cy4 = kernel_com_arcsec(psf_fine, 0.04)
    cx8, cy8 = kernel_com_arcsec(psf_08, DELTA_PIX)
    shift = float(np.hypot(cx8 - cx4, cy8 - cy4))

    checks = dict(
        psf_method=("lenstronomy degrade_kernel(empirical_psf_04, 2) -> 27x27 "
                    "@ 0.08, renormalized to sum=1"),
        psf_roundtrip_cos=cos,
        psf_roundtrip_cos_ceiling=roundtrip_cos_ceiling(psf_fine),
        psf_centroid_shift_arcsec=shift,
        psf_sum_before_renorm=sum_before,
        psf_sum=float(psf_08.sum()),
        psf_gate_cos_ok=bool(cos >= 0.99),
        psf_gate_centroid_ok=bool(shift <= 0.005),
    )
    return psf_08, checks


def render_check(out_path, err_b, keep_b, params_file, tol):
    """Self-consistency: blocksum2x2(render at v3 config) vs render at v3b config.

    GPU only -- never run the gigalens simulator on CPU. Lazy imports keep the
    --skip-render-check path jax-free.
    """
    D.bootstrap_vendor()
    import jax  # noqa: F401
    import jax.experimental.shard_map  # noqa: F401
    import jax.numpy as jnp
    from gigalens.jax.simulator import LensSimulator

    z = np.load(REPRO / params_file)
    z_best = z["best_z"].astype(np.float32)
    dim = int(z_best.shape[-1])
    if dim not in N_MAX_BY_DIM:
        raise SystemExit(f"render check: unknown z dim {dim} "
                         f"(expected one of {sorted(N_MAX_BY_DIM)})")
    n_max = N_MAX_BY_DIM[dim]

    def render(data_file):
        d, prior, phys, prob, cfg = D.build_all(n_max=n_max, data_file=data_file)
        sim = LensSimulator(phys, cfg, bs=2)             # bs=2 dodges the squeeze
        zb = jnp.asarray(np.stack([z_best, z_best]))
        x = prob.bij.forward(list(zb.T))
        return np.asarray(sim.simulate(x))[0].astype(np.float64), cfg

    r3, cfg3 = render(PARENT)
    r3b, cfg3b = render(str(Path(out_path).resolve()))
    dev = (blocksum(r3) - r3b) / err_b
    dev_kept = dev[keep_b]
    max_dev = float(np.max(np.abs(dev_kept)))
    ok = max_dev < tol

    result = dict(params=str(params_file), n_max=n_max,
                  supersample_v3b=int(cfg3b.supersample),
                  max_abs_dev_over_err=max_dev, tol=tol, ok=bool(ok))
    if not ok:
        # is the deviation a coherent dipole (grid half-pixel offset signature)?
        gy, gx = np.gradient(r3b / err_b)
        sel = keep_b
        def corr(a, b):
            a = a[sel] - a[sel].mean()
            b = b[sel] - b[sel].mean()
            return float((a @ b) / np.sqrt((a @ a) * (b @ b) + 1e-30))
        iy, ix = np.unravel_index(np.argmax(np.abs(np.where(keep_b, dev, 0.0))),
                                  dev.shape)
        result.update(
            dev_mean=float(dev_kept.mean()),
            dev_p01=float(np.percentile(dev_kept, 1)),
            dev_p99=float(np.percentile(dev_kept, 99)),
            frac_positive=float((dev_kept > 0).mean()),
            argmax_yx=[int(iy), int(ix)],
            corr_with_dmodel_dx=corr(dev, gx),
            corr_with_dmodel_dy=corr(dev, gy),
        )
    return result


def main():
    ap = argparse.ArgumentParser(
        description="2x2-binned 0.08\" Stage-A product from cutout_v3.npz")
    ap.add_argument("--poisson", choices=("auto", "keep", "drop"), default="auto",
                    help="auto = read D2_drop_poisson from data/noise_audit.json "
                         "if it exists, else keep")
    ap.add_argument("--out", type=str, default="data/cutout_v3b.npz")
    ap.add_argument("--params", type=str, default="data/map_v11_v3cold.npz",
                    help="MAP npz (best_z) for the self-consistency render check")
    ap.add_argument("--render-tol", type=float, default=0.1)
    ap.add_argument("--skip-render-check", action="store_true",
                    help="CPU-only dry run: never imports jax / the simulator")
    ap.add_argument("--fallback-supersample1", action="store_true",
                    help="re-emit the npz with meta supersample=1 (kernel "
                         "psf_08 convolved at the output scale, no internal "
                         "upsampling) and re-run the render check there")
    ap.add_argument("--calib", choices=("resid", "img"), default="resid",
                    help="calibrate the binned sky term on model-subtracted "
                         "residuals (resid; default — the 46_noise_audit "
                         "finding: diffuse lens-wing flux contaminates an "
                         "img-based calibration, inflating sigma) or on raw "
                         "img fluctuations (img; the legacy 40b convention)")
    ap.add_argument("--calib-model", type=str,
                    default="data/model_map_v3cold.npy",
                    help="fine-scale model render (from 46_noise_audit.py) "
                         "used for the resid calibration")
    args = ap.parse_args()
    out_path = (REPRO / args.out) if not Path(args.out).is_absolute() \
        else Path(args.out)
    supersample = 1 if args.fallback_supersample1 else 2

    # ------------------------------------------------------------ parent load
    par = D.load_v2(PARENT)
    pmeta = par["meta"]
    exp_time = float(pmeta.get("exp_time", 1197.7))
    img = par["img"].astype(np.float64)
    err = par["err_map"].astype(np.float64)
    keep = par["keep_mask"].astype(bool)
    assert img.shape[0] % 2 == 0 and img.shape[0] == img.shape[1]
    crop_b = img.shape[0] // 2

    # ----------------------------------------------------- poisson term mode
    poisson_mode = args.poisson
    if poisson_mode == "auto":
        audit = DATA / "noise_audit.json"
        if audit.exists():
            drop = bool(json.loads(audit.read_text()).get("D2_drop_poisson", False))
            poisson_mode = "drop" if drop else "keep"
        else:
            poisson_mode = "keep"
    keep_poisson = poisson_mode == "keep"

    # ------------------------------------------------- bin image, mask, noise
    img_b = blocksum(img)                                  # integrated flux/px
    keep_b = blockall(keep)

    poisson_fine = np.clip(img, 0, np.inf) / exp_time
    sky_var_fine = err ** 2 - poisson_fine                 # invert 40b recipe
    n_negvar = int((sky_var_fine < 0).sum())
    sky_var_fine = np.clip(sky_var_fine, 0, np.inf)
    sky_var_b = blocksum(sky_var_fine)                     # naive quadrature
    poisson_b = (np.clip(img_b, 0, np.inf) / exp_time) if keep_poisson \
        else np.zeros_like(img_b)

    # --------------------------- sky-chi2 recalibration on the binned product
    yy, xx = np.indices(img_b.shape)
    cen = (crop_b - 1) / 2.0
    r_arc = np.hypot(xx - cen, yy - cen) * DELTA_PIX
    base_err_b = np.sqrt(sky_var_b + poisson_b)
    sky_px = keep_b & (r_arc > SKY_R_ARC) & \
        (np.abs(img_b) < 5.0 * np.median(base_err_b))
    chi2_sky_naive = float(np.mean((img_b[sky_px] / base_err_b[sky_px]) ** 2))

    # calibration signal: model-subtracted residuals (default) so diffuse
    # lens-wing flux in the sky annulus does not inflate sigma (46_noise_audit
    # measured a 0.30 variance factor at fine scale from this contamination)
    calib_mode = args.calib
    calib_model_file = REPRO / args.calib_model
    if calib_mode == "resid" and not calib_model_file.exists():
        print(f"WARNING: {calib_model_file} missing (run 46_noise_audit.py "
              f"first); falling back to --calib img")
        calib_mode = "img"
    if calib_mode == "resid":
        model_b = blocksum(np.load(calib_model_file).astype(np.float64))
        cal_b = img_b - model_b
    else:
        cal_b = img_b
    chi2_sky_naive_calib = float(
        np.mean((cal_b[sky_px] / base_err_b[sky_px]) ** 2))

    norm = cal_b[sky_px] / np.sqrt(sky_var_b[sky_px])
    _, _, rstd = sigma_clipped_stats(norm, sigma=5.0, maxiters=10)
    clipped = np.clip(norm, -5.0 * rstd, 5.0 * rstd)
    rescale_b = float(np.sqrt(np.mean(clipped ** 2)))
    n_iter = 0
    for _ in range(4):
        err_b = np.sqrt(rescale_b ** 2 * sky_var_b + poisson_b)
        c = float(np.mean((cal_b[sky_px] / err_b[sky_px]) ** 2))
        if abs(c - 1.0) <= 0.01:
            break
        rescale_b *= np.sqrt(max(c, 1e-6))
        n_iter += 1
    err_b = np.sqrt(rescale_b ** 2 * sky_var_b + poisson_b)
    chi2_sky = float(np.mean((cal_b[sky_px] / err_b[sky_px]) ** 2))
    chi2_sky_img = float(np.mean((img_b[sky_px] / err_b[sky_px]) ** 2))
    gate_ok = abs(chi2_sky - 1.0) <= 0.05

    # --------------------------------------------------------------- PSF @0.08
    psf_fine = np.load(DATA / "empirical_psf_04.npy").astype(np.float64)
    psf_fine = psf_fine / psf_fine.sum()
    psf_08, psf_checks = build_psf_08(psf_fine)
    np.save(DATA / "empirical_psf_08.npy", psf_08.astype(np.float32))
    if not psf_checks["psf_gate_cos_ok"]:
        print(f"WARNING: PSF round-trip cos = {psf_checks['psf_roundtrip_cos']:.4f}"
              f" < 0.99, but the achievable ceiling for ANY 27x27 @0.08 kernel is"
              f" {psf_checks['psf_roundtrip_cos_ceiling']:.4f} (undersampled PSF"
              f" core; subgrid_kernel = bilinear upsample). The binned-domain"
              f" render check is the decisive fidelity gate.")

    # ------------------------------------------------------------- meta + npz
    meta = dict(
        crop=crop_b, delta_pix=DELTA_PIX, supersample=supersample,
        exp_time=exp_time, parent=PARENT, source=pmeta.get("source"),
        bin_factor=2, sky=pmeta.get("sky"),
        rescale=rescale_b, rescale_parent=pmeta.get("rescale"),
        chi2_sky=chi2_sky, chi2_sky_naive_quadrature=chi2_sky_naive,
        chi2_sky_naive_calib=chi2_sky_naive_calib, chi2_sky_img=chi2_sky_img,
        calib_mode=calib_mode,
        calib_model=(str(args.calib_model) if calib_mode == "resid" else None),
        n_recal_iter=n_iter, gate_sky_chi2_ok=bool(gate_ok),
        poisson_term=bool(keep_poisson), poisson_mode=poisson_mode,
        n_px=int(keep_b.size), n_masked_px=int((~keep_b).sum()),
        n_sky_px=int(sky_px.sum()),
        n_negative_skyvar_fine_px_clipped=n_negvar,
        nearby_arcsec=pmeta["nearby_arcsec"],
        **psf_checks,
    )
    np.savez(out_path, img=img_b.astype(np.float32),
             err_map=err_b.astype(np.float32), keep_mask=keep_b,
             psf=psf_08.astype(np.float32), meta=json.dumps(meta))
    (DATA / "cutout_v3b_stats.json").write_text(json.dumps(meta, indent=2))

    # ------------------------------------------------------------- QA figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    stretch = np.arcsinh(img_b / (3 * np.median(err_b)))
    axes[0].imshow(stretch, origin="lower", cmap="magma")
    axes[0].set_title(f"cutout v3b ({crop_b}x{crop_b} @ {DELTA_PIX}\", asinh)")
    over = np.ma.masked_where(keep_b, np.ones_like(img_b))
    axes[1].imshow(stretch, origin="lower", cmap="gray")
    axes[1].imshow(over, origin="lower", cmap="autumn", alpha=0.7, vmin=0, vmax=1)
    axes[1].set_title(f"keep_mask ({int((~keep_b).sum())} px masked)")
    sky_over = np.ma.masked_where(~sky_px, np.ones_like(img_b))
    axes[2].imshow(stretch, origin="lower", cmap="gray")
    axes[2].imshow(sky_over, origin="lower", cmap="cool", alpha=0.6, vmin=0, vmax=1)
    axes[2].set_title(f"sky px ({int(sky_px.sum())}, rescale={rescale_b:.3f}, "
                      f"chi2={chi2_sky:.3f})")
    fig.tight_layout()
    fig.savefig(FIGS / "cutout_v3b_masks.png", dpi=120)

    # ----------------------------------------------------------- render check
    render = dict(skipped=True)
    if not args.skip_render_check:
        render = render_check(out_path, err_b, keep_b, args.params,
                              args.render_tol)

    summary = dict(
        out=str(out_path), shape=list(img_b.shape),
        n_kept_px=int(keep_b.sum()),
        n_kept_expected_quarter_v3=int(keep.sum()) // 4,
        chi2_sky_naive_quadrature=chi2_sky_naive,
        rescale_b=rescale_b, chi2_sky=chi2_sky,
        gate_sky_chi2_ok=bool(gate_ok),
        poisson_mode=poisson_mode,
        psf_roundtrip_cos=psf_checks["psf_roundtrip_cos"],
        psf_roundtrip_cos_ceiling=psf_checks["psf_roundtrip_cos_ceiling"],
        psf_centroid_shift_arcsec=psf_checks["psf_centroid_shift_arcsec"],
        psf_sum=psf_checks["psf_sum"],
        render_check=render,
    )
    print(json.dumps(summary, indent=2))
    print(f"\nGATE sky chi2 = {chi2_sky:.4f} -> {'PASS' if gate_ok else 'FAIL'}")

    if not gate_ok:
        raise SystemExit("FAIL: binned sky chi2 outside 1.00 +/- 0.05")
    if not psf_checks["psf_gate_centroid_ok"]:
        raise SystemExit("FAIL: psf_08 centroid shift > 0.005 arcsec")
    if not render.get("skipped") and not render["ok"]:
        print(json.dumps(render, indent=2))
        msg = (f"FAIL: render check max|dev|/err = "
               f"{render['max_abs_dev_over_err']:.3f} >= {args.render_tol}. ")
        if max(abs(render.get("corr_with_dmodel_dx", 0.0)),
               abs(render.get("corr_with_dmodel_dy", 0.0))) > 0.5:
            msg += ("Deviation correlates with the model gradient (coherent "
                    "dipole = grid half-pixel offset signature). ")
        msg += ("Fallback: rerun with --fallback-supersample1 to re-emit the "
                "npz with meta supersample=1 and psf=psf_08, re-running the "
                "render check at supersample=1.")
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
