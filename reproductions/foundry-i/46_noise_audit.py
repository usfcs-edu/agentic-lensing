"""Stage G: noise-correlation audit for the Foundry-I error model.

(A) Mask-aware 2-D autocorrelation of the source-free sky for cutout_v2
    (0.13"/px) and cutout_v3 (0.04"/px fine skycell), zero-padded to (2N,2N)
    so there is no circular wraparound. Computed on the RAW sky and on a
    DETRENDED sky (local background from photutils Background2D, ~1" boxes,
    removed) to separate large-scale background structure from the
    drizzle-kernel correlation. f_corr = sum(rho) over a centered window
    (|rho| >= 0.02 only) at TWO windows: short (drizzle-scale: the
    upsampling footprint) and full (spec). The drizzle-relevant headline is
    f_corr(detrended, short); the raw full-window value quantifies the
    large-scale background structure as a separate finding.
(B) Effective reduced chi^2 for the gated MAPs (v3cold dim-74, v3nm8_p dim-91,
    nm8_d dim-91) with nu_eff = N_kept / f_corr - n_free, using the
    detrended-short f_corr. CONVENTION CAVEAT (stored in the JSON): because
    E[per-pixel chi^2] = 1 for pure correlated noise, a perfect model gives
    chi2_eff ~ f_corr under this convention, NOT 1; the diagnostic ratio is
    chi2_eff / f_corr (~ per-pixel chi2). Also reports chi2_pp_noP, the
    per-pixel chi^2 with the Poisson term removed from sigma. Models are
    rendered through the 45_fig8_panel.py forward path (GPU required).
(C) Arc vs sky chi^2 with/without the data-based Poisson term, from the MAP
    residual (v3cold for v3, nm8_d for v2). The no-Poisson sigma is recovered
    by inverting the cutout recipe: sigma_noP = sqrt(err^2 - clip(img,0)/t_exp).
    Drizzling spreads source photons over the correlation footprint, so the
    per-pixel Poisson variance of a 3.25x-upsampled product is strongly
    suppressed relative to clip(img,0)/t_exp — if the data-based term is
    double-counted/unsuppressed, arc sigma is overestimated exactly where the
    signal is and chi2_arc < 1 while chi2_sky = 1 (the observed pattern).
(D) Decisions: D1 (does correlation + Poisson over-counting explain the v3
    chi^2 ~ 0.45? — smoking gun: chi2_arc_noP returns to ~1) and
    D2 (should the Poisson term be dropped in new products?), stored in the
    JSON. The spec'd chi2_eff-band rule is kept as D1_spec for the record.

Outputs: data/noise_audit.json, figs/noise_audit.png

Run:  python 46_noise_audit.py                                 (full; needs GPU)
      JAX_PLATFORMS=cpu python 46_noise_audit.py --no-render   (A + the
          no-model parts of C, pure numpy; writes a partial JSON + figure)
"""
import argparse
import json
from pathlib import Path

import numpy as np

import _data_lib as D

REPRO = Path(__file__).resolve().parent

ARC_ANNULUS = (1.2, 4.5)            # arcsec, as in 40/40b
RHO_FLOOR = 0.02                    # |rho| below this is treated as noise
NMAX_FROM_DIM = {74: 6, 91: 8}      # unconstrained-z dim -> shapelet order
WINDOW_L = {"cutout_v2.npz": 4, "cutout_v3.npz": 12}        # full (spec)
WINDOW_L_SHORT = {"cutout_v2.npz": 2, "cutout_v3.npz": 6}   # drizzle-scale
BKG_BOX = {"cutout_v2.npz": 10, "cutout_v3.npz": 26}        # ~1.3"/1.0" tiles
EXPECTED_CHI2 = {"v3cold": 0.4515, "v3nm8_p": 0.4326, "nm8_d": 3.4246}
MAPS = [("v3cold", "cutout_v3.npz"),
        ("v3nm8_p", "cutout_v3.npz"),
        ("nm8_d", "cutout_v2.npz")]
RESIDUAL_MAP = {"cutout_v3.npz": "v3cold", "cutout_v2.npz": "nm8_d"}
CONVENTION = ("variance-of-the-sample-mean convention: N_eff = N / sum(rho), "
              "stationary signal-free sky assumed. CAVEAT: E[per-pixel chi2] "
              "= 1 for pure correlated noise, so a perfect model gives "
              "chi2_eff ~ f_corr under this convention (not 1); the "
              "scale-free diagnostic is chi2_eff / f_corr ~ per-pixel chi2. "
              "The drizzle-relevant f_corr is the detrended short-window "
              "value; the raw full-window value includes genuine large-scale "
              "background structure (reported separately).")
V2_CAVEAT = ("v2 absolute chi2 levels carry the known ~3.4 scale-limited "
             "misfit (PSF-sampling convention defect); only the arc-sky "
             "CONTRAST is meaningful for this dataset")


# --------------------------------------------------------------------- (A)
def masked_autocorr(img, sky):
    """Mask-aware sky autocorrelation, zero-padded to (2N,2N).

    n = img - mean(img[sky]) on sky px (0 elsewhere); w = sky indicator.
    A = IFFT2(|FFT2(n)|^2), W = IFFT2(|FFT2(w)|^2); rho = (A/W)/(A/W)[0,0],
    valid only where W > 0.05*W[0,0] (NaN elsewhere). Returned fftshifted,
    center lag (0,0) at index (N, N).
    """
    n_pix = img.shape[0]
    n = np.where(sky, img - img[sky].mean(), 0.0)
    pad_n = np.zeros((2 * n_pix, 2 * n_pix))
    pad_w = np.zeros((2 * n_pix, 2 * n_pix))
    pad_n[:n_pix, :n_pix] = n
    pad_w[:n_pix, :n_pix] = sky.astype(np.float64)
    acf = np.real(np.fft.ifft2(np.abs(np.fft.fft2(pad_n)) ** 2))
    wgt = np.real(np.fft.ifft2(np.abs(np.fft.fft2(pad_w)) ** 2))
    valid = wgt > 0.05 * wgt[0, 0]
    cov = np.where(valid, acf / np.where(valid, wgt, 1.0), np.nan)
    rho = cov / cov[0, 0]
    return np.fft.fftshift(rho), n_pix      # center index = n_pix


def radial_profile(rho, center, r_max):
    """nan-aware mean of rho in integer-rounded radius bins 0..r_max (px)."""
    win = rho[center - r_max - 1:center + r_max + 2,
              center - r_max - 1:center + r_max + 2]
    yy, xx = np.indices(win.shape)
    r = np.hypot(xx - (r_max + 1), yy - (r_max + 1))
    rbin = np.rint(r).astype(int)
    prof = np.full(r_max + 1, np.nan)
    for k in range(r_max + 1):
        vals = win[(rbin == k) & np.isfinite(win)]
        if vals.size:
            prof[k] = vals.mean()
    return np.arange(r_max + 1, dtype=float), prof


def crossing(lag, prof, thr):
    """First downward crossing of `thr` in the radial profile (linear interp)."""
    for i in range(1, len(prof)):
        if not (np.isfinite(prof[i]) and np.isfinite(prof[i - 1])):
            continue
        if prof[i - 1] >= thr > prof[i]:
            frac = (prof[i - 1] - thr) / (prof[i - 1] - prof[i])
            return float(lag[i - 1] + frac * (lag[i] - lag[i - 1]))
    return None


def f_corr_window(rho, center, half_window):
    """Sum of rho over |dx|,|dy| <= L counting only finite |rho| >= RHO_FLOOR."""
    win = rho[center - half_window:center + half_window + 1,
              center - half_window:center + half_window + 1]
    keep = np.isfinite(win) & (np.abs(win) >= RHO_FLOOR)
    return float(win[keep].sum()), int(keep.sum())


def detrend_sky(img, sky, box):
    """Remove a ~1"-scale local background estimated from the sky pixels.

    photutils Background2D with non-sky pixels masked; falls back to a global
    sky-median subtraction if the tiling fails (note recorded by the caller).
    """
    try:
        from photutils.background import Background2D, MedianBackground
        bkg = Background2D(img, box, mask=~sky, filter_size=3,
                           bkg_estimator=MedianBackground(),
                           exclude_percentile=95.0)
        return img - bkg.background, "background2d"
    except Exception as exc:                          # pragma: no cover
        return img - np.median(img[sky]), f"global-median ({exc})"


def autocorr_stats(img, sky, half_window, half_window_short):
    """rho map + radial profile + dual-window f_corr for one sky image."""
    rho, center = masked_autocorr(img, sky)
    r_max = min(max(4 * half_window, 20), img.shape[0] - 2)
    lag_px, prof = radial_profile(rho, center, r_max)
    f_full, n_full = f_corr_window(rho, center, half_window)
    f_short, n_short = f_corr_window(rho, center, half_window_short)
    return dict(rho=rho, center=center, lag_px=lag_px, prof=prof,
                f_corr_full=f_full, n_lags_full=n_full,
                f_corr_short=f_short, n_lags_short=n_short,
                len_half_px=crossing(lag_px, prof, 0.5),
                len_1e_px=crossing(lag_px, prof, 1.0 / np.e))


def audit_dataset(data_file):
    """Pure-numpy stage: sky/arc regions, autocorrelation, Poisson inversion."""
    d = D.load_v2(data_file)
    img = d["img"].astype(np.float64)
    err = d["err_map"].astype(np.float64)
    keep = d["keep_mask"]
    meta = d["meta"]
    n_pix = img.shape[0]
    delta_pix = float(meta.get("delta_pix", D.DELTA_PIX))
    exp_time = float(meta.get("exp_time", D.EXP_TIME))

    # geometry exactly as in 40b l.52-54 / 40 l.60-64
    yy, xx = np.indices(img.shape)
    cen = (n_pix - 1) / 2.0
    r_arc = np.hypot(xx - cen, yy - cen) * delta_pix

    sky = keep & (r_arc > ARC_ANNULUS[1]) & (np.abs(img) < 5.0 * np.median(err))
    arc = keep & (r_arc > ARC_ANNULUS[0]) & (r_arc < ARC_ANNULUS[1]) \
        & (img > 2.0 * err)

    half_window = WINDOW_L[data_file]
    half_short = WINDOW_L_SHORT[data_file]
    raw = autocorr_stats(img, sky, half_window, half_short)
    img_det, detrend_method = detrend_sky(img, sky, BKG_BOX[data_file])
    det = autocorr_stats(img_det, sky, half_window, half_short)

    # Poisson inversion (recipe: err^2 = (rescale*sig_wht)^2 + clip(img,0)/t)
    var_noP = np.clip(err ** 2 - np.clip(img, 0, None) / exp_time, 1e-20, None)
    sig_noP = np.sqrt(var_noP)
    poisson_frac_arc = float(np.median(
        (np.clip(img, 0, None) / exp_time / err ** 2)[arc]))

    report = dict(
        data_file=data_file, delta_pix=delta_pix, exp_time=exp_time,
        n_pix=int(img.size), n_kept_px=int(keep.sum()),
        n_sky_px=int(sky.sum()), n_arc_px=int(arc.sum()),
        autocorr=dict(
            window_L_px=half_window, window_L_short_px=half_short,
            rho_floor=RHO_FLOOR, detrend_method=detrend_method,
            detrend_box_px=BKG_BOX[data_file],
            # headline for the drizzle question:
            f_corr=det["f_corr_short"],
            neff_over_n=1.0 / det["f_corr_short"],
            # all four variants for the record:
            f_corr_detrended_short=det["f_corr_short"],
            f_corr_detrended_full=det["f_corr_full"],
            f_corr_raw_short=raw["f_corr_short"],
            f_corr_raw_full=raw["f_corr_full"],
            n_lags_counted=det["n_lags_short"],
            corr_len_half_px=det["len_half_px"],
            corr_len_half_arcsec=(None if det["len_half_px"] is None
                                  else det["len_half_px"] * delta_pix),
            corr_len_1e_px=det["len_1e_px"],
            corr_len_1e_arcsec=(None if det["len_1e_px"] is None
                                else det["len_1e_px"] * delta_pix),
            corr_len_half_px_raw=raw["len_half_px"],
            radial_lag_px=[float(v) for v in det["lag_px"]],
            radial_rho=[None if not np.isfinite(v) else round(float(v), 6)
                        for v in det["prof"]],
            radial_rho_raw=[None if not np.isfinite(v) else round(float(v), 6)
                            for v in raw["prof"]],
        ),
        poisson=dict(poisson_frac_arc=poisson_frac_arc),
        convention_note=CONVENTION,
    )
    arrays = dict(img=img, err=err, keep=keep, sky=sky, arc=arc,
                  sig_noP=sig_noP, rho=det["rho"], center=det["center"],
                  lag_px=det["lag_px"], prof=det["prof"],
                  prof_raw=raw["prof"], lag_px_raw=raw["lag_px"],
                  delta_pix=delta_pix)
    return report, arrays


# --------------------------------------------------------------------- (B)
def render_map(tag, data_file):
    """Forward-render a stored MAP via the 45_fig8_panel.py path (GPU)."""
    D.bootstrap_vendor()
    import jax  # noqa: F401
    import jax.experimental.shard_map  # noqa: F401
    import jax.numpy as jnp
    from gigalens.jax.simulator import LensSimulator

    z = np.load(REPRO / "data" / f"map_v11_{tag}.npz")
    z_best = z["best_z"].astype(np.float32).reshape(-1)
    n_max = NMAX_FROM_DIM[z_best.size]
    d, prior, phys, prob, sim_config = D.build_all(n_max=n_max,
                                                   data_file=data_file)
    sim = LensSimulator(phys, sim_config, bs=2)
    zb = jnp.asarray(np.stack([z_best, z_best]))
    x = prob.bij.forward(list(zb.T))
    model = np.asarray(sim.simulate(x))[0].astype(np.float64)
    # persist for downstream consumers (40c model-subtracted calibration)
    np.save(REPRO / "data" / f"model_map_{tag}.npy", model.astype(np.float32))
    return model, int(z_best.size), n_max


def chi2_eff_report(tag, data_file, model, n_free, n_max, arrays, f_corr):
    img, err, keep = arrays["img"], arrays["err"], arrays["keep"]
    chi = (img - model) / err
    chi_noP = (img - model) / arrays["sig_noP"]
    n_kept = int(keep.sum())
    chi2_pp = float(np.mean(chi[keep] ** 2))
    expected = EXPECTED_CHI2[tag]
    assert abs(chi2_pp - expected) <= 0.02, \
        f"{tag}: chi2_pp={chi2_pp:.4f} vs gate {expected:.4f} (>0.02 apart)"
    nu_eff = n_kept / f_corr - n_free
    chi2_eff = float(np.sum(chi[keep] ** 2) / nu_eff)
    return dict(tag=tag, data_file=data_file, n_max=n_max, n_free=n_free,
                N_kept=n_kept, f_corr=f_corr, nu_eff=float(nu_eff),
                chi2_pp=chi2_pp, chi2_pp_gate=expected,
                chi2_pp_noP=float(np.mean(chi_noP[keep] ** 2)),
                chi2_eff=chi2_eff,
                chi2_eff_over_fcorr=float(chi2_eff / f_corr))


# --------------------------------------------------------------------- (C)
def arc_sky_report(data_file, model, arrays, poisson_frac_arc):
    img, err, sig_noP = arrays["img"], arrays["err"], arrays["sig_noP"]
    sky, arc = arrays["sky"], arrays["arc"]
    resid = img - model

    def chi2(region, sigma):
        return float(np.mean((resid[region] / sigma[region]) ** 2))

    rep = dict(
        residual_map=RESIDUAL_MAP[data_file],
        chi2_arc_full=chi2(arc, err), chi2_arc_noP=chi2(arc, sig_noP),
        chi2_sky_full=chi2(sky, err), chi2_sky_noP=chi2(sky, sig_noP),
        n_arc_px=int(arc.sum()), n_sky_px=int(sky.sum()),
        poisson_frac_arc=poisson_frac_arc,
    )
    rep["contrast_full"] = abs(rep["chi2_arc_full"] - rep["chi2_sky_full"])
    rep["contrast_noP"] = abs(rep["chi2_arc_noP"] - rep["chi2_sky_noP"])
    if data_file == "cutout_v2.npz":
        rep["caveat"] = V2_CAVEAT
    return rep


def honest_report(data_file, model, arrays, exp_time):
    """Recalibrated ('honest') chi^2: the cutout's sky term was calibrated on
    img fluctuations that include diffuse lens-halo wing flux the model fits
    (40b's segmentation found no sources at r>4.5" on the fine product), so
    the stored sigma over-states the noise. Re-derive the sky-term factor
    from MODEL-SUBTRACTED sky residuals and rebuild sigma with the Poisson
    term unchanged: var_honest = f_sky*(err^2 - pois) + pois.
    """
    img, err, keep = arrays["img"], arrays["err"], arrays["keep"]
    sky, arc = arrays["sky"], arrays["arc"]
    resid = img - model
    pois = np.clip(img, 0, None) / exp_time
    sky_var = np.clip(err ** 2 - pois, 1e-20, None)
    f_sky = float(np.mean(resid[sky] ** 2 / sky_var[sky]))
    var_h = f_sky * sky_var + pois
    chi2_h = resid ** 2 / var_h
    rep = dict(
        sky_term_variance_factor=f_sky,
        sky_term_sigma_factor=float(np.sqrt(f_sky)),
        chi2_kept_honest=float(np.mean(chi2_h[keep])),
        chi2_arc_honest=float(np.mean(chi2_h[arc])),
        chi2_sky_honest=float(np.mean(chi2_h[sky])),
        note=("f_sky from model-subtracted sky residuals on the audit sky "
              "region; Poisson term kept as stored (data-based)"),
    )
    if data_file == "cutout_v2.npz":
        rep["caveat"] = V2_CAVEAT
    return rep


# ------------------------------------------------------------------ figure
def make_figure(out_fig, audits, arrays, maps_out, arc_sky, decisions,
                honest=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # rho 2-D maps, +/-8 px window, log stretch
    for ax, name in zip(axes[0, :2], ("cutout_v2.npz", "cutout_v3.npz")):
        arr = arrays[name]
        c, win = arr["center"], 8
        sub = arr["rho"][c - win:c + win + 1, c - win:c + win + 1]
        im = ax.imshow(sub, origin="lower", cmap="RdBu_r",
                       extent=[-win - .5, win + .5, -win - .5, win + .5],
                       norm=SymLogNorm(linthresh=RHO_FLOOR, vmin=-1, vmax=1))
        ax.set_title(f"sky autocorr rho — {name.split('.')[0]} "
                     f"({arr['delta_pix']:.3f}\"/px)")
        ax.set_xlabel("lag x [px]")
        ax.set_ylabel("lag y [px]")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # radial profiles overlaid (in arcsec, so the two scales are comparable);
    # solid = detrended (the drizzle-relevant one), faint dashed = raw
    ax = axes[0, 2]
    for name, color in (("cutout_v2.npz", "tab:orange"),
                        ("cutout_v3.npz", "tab:blue")):
        arr = arrays[name]
        ac = audits[name]["autocorr"]
        ax.plot(arr["lag_px"] * arr["delta_pix"], arr["prof"], "o-", ms=3,
                color=color,
                label=(f"{name.split('.')[0]} detr "
                       f"f_short={ac['f_corr_detrended_short']:.2f}"))
        ax.plot(arr["lag_px_raw"] * arr["delta_pix"], arr["prof_raw"], "--",
                lw=0.9, alpha=0.5, color=color,
                label=f"{name.split('.')[0]} raw "
                      f"f_full={ac['f_corr_raw_full']:.1f}")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.axhline(1.0 / np.e, color="gray", ls=":", lw=0.8)
    ax.axhline(RHO_FLOOR, color="gray", ls="-", lw=0.6)
    ax.set_xlabel("lag [arcsec]")
    ax.set_ylabel(r"radially-averaged $\rho$")
    ax.set_xlim(0, 1.6)
    ax.set_title("radial rho profiles (-- 0.5, : 1/e)")
    ax.legend(fontsize=9)

    # bar charts: arc/sky chi2 full vs no-Poisson
    for ax, name in zip(axes[1, :2], ("cutout_v3.npz", "cutout_v2.npz")):
        short = name.split(".")[0]
        if arc_sky is None:
            ax.text(0.5, 0.5, "render skipped (--no-render)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        rep = arc_sky[short]
        vals = [rep["chi2_arc_full"], rep["chi2_arc_noP"],
                rep["chi2_sky_full"], rep["chi2_sky_noP"]]
        ax.bar(range(4), vals,
               color=["firebrick", "salmon", "navy", "skyblue"])
        ax.set_xticks(range(4))
        ax.set_xticklabels(["arc full", "arc noP", "sky full", "sky noP"])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        title = f"arc vs sky chi2 — {short} ({rep['residual_map']})"
        if name == "cutout_v2.npz":
            title += "\n(scale-limited misfit: contrast only)"
        ax.set_title(title)

    # text panel
    ax = axes[1, 2]
    ax.set_axis_off()
    lines = []
    for name in ("cutout_v2.npz", "cutout_v3.npz"):
        ac = audits[name]["autocorr"]
        lh = ac["corr_len_half_arcsec"]
        lines.append(f"{name.split('.')[0]}: "
                     f"f_detr(short)={ac['f_corr_detrended_short']:.2f}  "
                     f"f_raw(full)={ac['f_corr_raw_full']:.1f}")
        lines.append(f"   N_eff/N={ac['neff_over_n']:.4f}  len(rho=0.5)="
                     f"{'n/a' if lh is None else f'{lh:.3f}\"'} (detr)")
    lines.append("")
    if maps_out is not None:
        for rep in maps_out.values():
            lines.append(f"{rep['tag']}: chi2_pp={rep['chi2_pp']:.3f} "
                         f"noP={rep['chi2_pp_noP']:.3f} "
                         f"eff/f={rep['chi2_eff_over_fcorr']:.3f}")
        lines.append("")
        v3rep = arc_sky["cutout_v3"]
        lines.append(f"D1 arc_noP in [0.8,1.2]: "
                     f"{decisions['D1_correlation_confirmed']} "
                     f"(arc_noP={v3rep['chi2_arc_noP']:.3f})")
        lines.append(f"D2 drop Poisson term:    "
                     f"{decisions['D2_drop_poisson']}")
        lines.append(f"   poisson_frac_arc(v3)={v3rep['poisson_frac_arc']:.3f}")
        if honest is not None:
            h3 = honest["cutout_v3"]
            lines.append("")
            lines.append(f"HONEST (sky term recalibrated on model-")
            lines.append(f"subtracted sky, f_var={h3['sky_term_variance_factor']:.3f}):")
            lines.append(f"   v3 kept={h3['chi2_kept_honest']:.3f} "
                         f"arc={h3['chi2_arc_honest']:.3f}")
            lines.append(f"D6 recalibration explains: "
                         f"{decisions['D6_sky_recalibration_explains']}")
    else:
        lines.append("chi2_eff / decisions: n/a (--no-render)")
    ax.text(0.02, 0.95, "\n".join(lines), transform=ax.transAxes,
            family="monospace", fontsize=10, va="top")
    ax.set_title("summary")

    fig.suptitle("Foundry-I noise-correlation audit (46_noise_audit.py)",
                 y=0.995)
    fig.tight_layout()
    out_fig.parent.mkdir(exist_ok=True)
    fig.savefig(out_fig, dpi=130, bbox_inches="tight")


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=str, default="data/noise_audit.json")
    ap.add_argument("--out-fig", type=str, default="figs/noise_audit.png")
    ap.add_argument("--no-render", action="store_true",
                    help="skip the GPU model renders: write a partial JSON "
                         "with (A) + the no-model parts of (C) only")
    args = ap.parse_args()

    audits, arrays = {}, {}
    for data_file in ("cutout_v2.npz", "cutout_v3.npz"):
        audits[data_file], arrays[data_file] = audit_dataset(data_file)

    maps_out, arc_sky, decisions, honest = None, None, None, None
    if not args.no_render:
        models = {}
        maps_out = {}
        for tag, data_file in MAPS:
            model, n_free, n_max = render_map(tag, data_file)
            models[tag] = model
            maps_out[tag] = chi2_eff_report(
                tag, data_file, model, n_free, n_max, arrays[data_file],
                audits[data_file]["autocorr"]["f_corr"])

        arc_sky = {}
        honest = {}
        for data_file in ("cutout_v3.npz", "cutout_v2.npz"):
            short = data_file.split(".")[0]
            arc_sky[short] = arc_sky_report(
                data_file, models[RESIDUAL_MAP[data_file]], arrays[data_file],
                audits[data_file]["poisson"]["poisson_frac_arc"])
            honest[short] = honest_report(
                data_file, models[RESIDUAL_MAP[data_file]], arrays[data_file],
                float(audits[data_file]["exp_time"]))

        v3rep = arc_sky["cutout_v3"]
        v3hon = honest["cutout_v3"]
        decisions = dict(
            # smoking gun: arc chi2 returns to ~1 once the (drizzle-
            # suppressed) Poisson term is removed from sigma
            D1_correlation_confirmed=bool(
                0.8 <= v3rep["chi2_arc_noP"] <= 1.2),
            D1_rule="0.8 <= chi2_arc_noP(v3) <= 1.2 (Poisson over-counting "
                    "+ correlation explain the fitted-region chi2 < 1)",
            # the spec'd chi2_eff band, kept for the record (see
            # convention_note: a perfect model gives chi2_eff ~ f_corr)
            D1_spec_chi2_eff_band=bool(
                0.7 <= maps_out["v3cold"]["chi2_eff"] <= 1.5),
            D1_spec_rule="0.7 <= chi2_eff(v3cold) <= 1.5",
            D2_drop_poisson=bool(
                (v3rep["contrast_noP"] < v3rep["contrast_full"])
                and (v3rep["poisson_frac_arc"] > 0.10)),
            D2_rule="|chi2_arc_noP - chi2_sky_noP| < "
                    "|chi2_arc_full - chi2_sky_full| on v3 "
                    "AND poisson_frac_arc(v3) > 0.10",
            # the explanation that actually closes: the stored sigma's sky
            # term was calibrated on wing-contaminated img fluctuations;
            # recalibrated on model-subtracted sky (Poisson kept), arc and
            # kept-region chi2 should both return to ~1
            D6_sky_recalibration_explains=bool(
                0.8 <= v3hon["chi2_arc_honest"] <= 1.2
                and 0.8 <= v3hon["chi2_kept_honest"] <= 1.3),
            D6_rule="0.8 <= chi2_arc_honest(v3) <= 1.2 AND "
                    "0.8 <= chi2_kept_honest(v3) <= 1.3",
            D6_sky_term_variance_factor_v3=v3hon["sky_term_variance_factor"],
        )

    out = dict(
        generated_by="46_noise_audit.py",
        partial=bool(args.no_render),
        convention_note=CONVENTION,
        rho_floor=RHO_FLOOR,
        datasets={k.split(".")[0]: v for k, v in audits.items()},
        maps=maps_out,
        arc_sky=arc_sky,
        honest=honest,
        decisions=decisions,
    )

    out_json = REPRO / args.out_json
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))

    make_figure(REPRO / args.out_fig, audits, arrays, maps_out, arc_sky,
                decisions, honest)

    print(json.dumps(out, indent=2))
    print(f"wrote {out_json} and {REPRO / args.out_fig}")


if __name__ == "__main__":
    main()
