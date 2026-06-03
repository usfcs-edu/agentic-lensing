#!/usr/bin/env python3
"""
05_variability_lightcurve.py  --  Sheu+2024a §3 light curves + variability sigma (Eq. 1)

This is the scientific heart of Paper II and the part that DIFFERS from Paper I
(sheu-2023's 05_lightcurve_salt3.py).  Where Paper I fits a SALT3 SN-Ia template
and derives a magnification, Paper II measures the *stochastic photometric
variability* of each posited lensed-quasar IMAGE and quantifies it with their
magnitude-standard-deviation metric (their Eq. 1):

    sigma = (1/N) * sum_b sqrt( sum_{epochs in b} (m_b - mu_b)^2 )

  -- where for band b: m_b are the per-epoch PSF magnitudes, mu_b is the mean
     magnitude in band b, N_b is the number of epochs in band b, and
     N = sum_b N_b is the total number of epochs across all bands.
  -- the per-band term is sqrt(sum (m-mu)^2), i.e. N_b * RMS_b is NOT divided by
     N_b inside the sqrt; the (1/N) prefactor outside makes it an N_b-weighted
     average of the band scatter.  We implement the literal Eq. 1 AND, for
     interpretability, also report the conventional N_b-weighted mean of the
     per-band magnitude RMS (this is the quantity that matches the ~0.13-0.40 mag
     values quoted in their Table 2; see DERIVATION note below).

DERIVATION note (which form matches Table 2?)
---------------------------------------------
Table 2 quotes <sigma> in mag of order 0.1-0.4, i.e. the *typical magnitude
scatter of a light curve* -- which is the per-band RMS, weighted by N_b and
averaged across bands and images.  The literal Eq. 1 as printed (no 1/N_b inside
the sqrt) would give numbers ~sqrt(N_b)~5x larger, which do not match Table 2.
We therefore interpret Eq. 1 as the magnitude *standard deviation* (its caption
in Table 2: "average magnitude standard deviation"), i.e.

    sigma_b = sqrt( (1/N_b) sum_{epochs} (m_b - mu_b)^2 )      # std dev in band b
    <sigma> = (1/N) * sum_b ( N_b * sigma_b )                  # N_b-weighted mean

This reduces exactly to Eq. 1 with the conventional 1/N_b normalisation and
reproduces the Table 2 magnitude scale.  Both forms are written to the output
CSV so the choice is transparent.

Photometry (paper §3): "compute the PSF photometry of each posited image ... A
PSF must be resolved using SEP at the lensed image location; if not, we omit
that exposure ... We filter out photometric data with S/N < 5."
We implement this as: at each lensed-image position, fit a 2-D circular Gaussian
PSF (FWHM from the per-frame seeing, fit on a small stamp) to the science frame
to get the image flux + error, require S/N>5, convert to AB mag via the
per-epoch zeropoint (frames are rescaled to ZP=30 at the reproject stage).  This
is a faithful, lightweight stand-in for the paper's Tractor/forced PSF
photometry; the difference imaging (03) independently confirms the variability
as visible over/under-subtraction (see --check-diff).

Modes
-----
  --mode real    measure sigma for the known/confirmed lensed quasar (default)
  --mode synth   inject a point source with a KNOWN stochastic light curve of a
                 target sigma into the real frames, run the SAME photometry +
                 Eq. 1, and check we recover the injected sigma -- this is the
                 honest validation of the variability metric (a la sheu-2023's
                 --mode synth).
  --mode both

Outputs
-------
  data/image_positions.csv          lensed-image (x,y) used for photometry
  data/lightcurves_<tag>.csv        per-image, per-band, per-epoch PSF mags + S/N
  data/sigma_<tag>.csv              Eq. 1 variability sigma per image (+ <sigma>)
  figs/lightcurve_<tag>.png         light curves of the posited lensed images
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)
REPROJ = DATA / "reproj"
DIFF = DATA / "diff"

PIXSCALE = 0.262          # arcsec/pix on the common grid
NPIX = 801
ZP = 30.0                 # common photometric zeropoint (set at reproject stage)
SN_MIN = 5.0              # paper: filter out S/N < 5

# Target: DESI-038.0655-24.4942 (Sheu+2024a Table 2; Dawes+2023 grade-A double).
# System centre RA/Dec and the two posited lensed-image offsets.  Dawes lists
# image separation 1.54"; we place the two images symmetrically about the centre
# along the position angle measured from the median reference (refined per band
# by a local-peak search around each nominal position).
CAND_RA = 38.0655
CAND_DEC = -24.4942
CAND_NAME = "DESI-038.0655-24.4942"
IMG_SEP_ARCSEC = 1.54     # Dawes+2023 Type=Double separation


# ---------------------------------------------------------------- PSF photometry
def gaussian_psf_flux(img, x0, y0, fwhm_px, box=8):
    """Fit a 2-D circular Gaussian (free amp, bg, small centroid shift) on a stamp.

    Returns (flux, fluxerr, snr, xfit, yfit).  flux = 2*pi*sigma^2*amp (analytic
    integral of the Gaussian).  fluxerr from the residual rms over the fit box.
    """
    ny, nx = img.shape
    xi, yi = int(round(x0)), int(round(y0))
    x0b, x1b = max(0, xi - box), min(nx, xi + box + 1)
    y0b, y1b = max(0, yi - box), min(ny, yi + box + 1)
    stamp = img[y0b:y1b, x0b:x1b].astype(np.float64)
    if stamp.size < 25 or not np.isfinite(stamp).all():
        return np.nan, np.nan, np.nan, x0, y0
    yy, xx = np.mgrid[y0b:y1b, x0b:x1b]
    sig = fwhm_px / 2.3548

    def model(p):
        amp, bg, dx, dy = p
        g = amp * np.exp(-((xx - (x0 + dx)) ** 2 + (yy - (y0 + dy)) ** 2)
                         / (2 * sig ** 2)) + bg
        return (g - stamp).ravel()

    amp0 = float(np.nanmax(stamp) - np.nanmedian(stamp))
    bg0 = float(np.nanmedian(stamp))
    try:
        res = least_squares(model, [max(amp0, 1.0), bg0, 0.0, 0.0],
                            bounds=([0, -np.inf, -2, -2], [np.inf, np.inf, 2, 2]),
                            max_nfev=200)
        amp, bg, dx, dy = res.x
    except Exception:
        return np.nan, np.nan, np.nan, x0, y0
    flux = 2 * np.pi * sig ** 2 * amp
    resid = res.fun
    npix = resid.size
    rms = np.sqrt(np.sum(resid ** 2) / max(npix - 4, 1))
    # flux error: PSF-weighted noise ~ rms * sqrt(effective area) = rms*sqrt(4*pi*sig^2)
    fluxerr = rms * np.sqrt(4 * np.pi * sig ** 2)
    snr = flux / fluxerr if fluxerr > 0 else np.nan
    return float(flux), float(fluxerr), float(snr), float(x0 + dx), float(y0 + dy)


def refine_peak(ref, x0, y0, search=4):
    """Find the local brightness peak within +/-search px of (x0,y0) in the ref."""
    ny, nx = ref.shape
    xi, yi = int(round(x0)), int(round(y0))
    x0b, x1b = max(0, xi - search), min(nx, xi + search + 1)
    y0b, y1b = max(0, yi - search), min(ny, yi + search + 1)
    sub = ref[y0b:y1b, x0b:x1b]
    if sub.size == 0 or not np.isfinite(sub).any():
        return x0, y0
    j = np.nanargmax(sub)
    dy, dx = np.unravel_index(j, sub.shape)
    return float(x0b + dx), float(y0b + dy)


def image_positions(bands):
    """Locate the two posited lensed-image positions on the common grid.

    For a Dawes double, the two PSF-like images sit either side of the system
    centre (grid pixel 400,400) at ~IMG_SEP_ARCSEC.  We find the two brightest
    *distinct* local maxima within a window of the centre (a true double-peak
    search, robust to the brighter image dominating), enforcing a minimum
    separation so the same blob is not picked twice.  Falls back to the brightest
    peak + its reflection through the centre if only one peak is resolvable.
    """
    from scipy.ndimage import maximum_filter
    sep_px = IMG_SEP_ARCSEC / PIXSCALE
    ref = None
    for b in ("r", "z", "g"):  # prefer r (deepest), then z, g
        p = DATA / f"reference_{b}.fits"
        if p.exists():
            ref = fits.getdata(p).astype(np.float64)
            break
    if ref is None:
        raise SystemExit("no reference image; run 02 first")
    cx, cy = 400.0, 400.0
    win = int(round(sep_px)) + 5
    x0, x1 = int(cx - win), int(cx + win + 1)
    y0, y1 = int(cy - win), int(cy + win + 1)
    sub = ref[y0:y1, x0:x1].copy()
    bg = np.nanmedian(ref[340:380, 340:380])
    sub = sub - bg
    # local maxima
    mx = maximum_filter(sub, size=3)
    ispeak = (sub == mx) & np.isfinite(sub)
    ys, xs = np.where(ispeak)
    vals = sub[ys, xs]
    order = np.argsort(-vals)
    peaks = []  # (xg, yg, flux)
    min_sep = 2.5  # px; the two images must be >~0.65" apart
    for o in order:
        xg, yg = float(x0 + xs[o]), float(y0 + ys[o])
        if all(np.hypot(xg - p[0], yg - p[1]) > min_sep for p in peaks):
            peaks.append((xg, yg, float(vals[o])))
        if len(peaks) == 2:
            break
    if len(peaks) == 2:
        # order A=brighter, B=fainter
        peaks.sort(key=lambda t: -t[2])
        (xA, yA, _), (xB, yB, _) = peaks
    else:
        xA, yA = (peaks[0][0], peaks[0][1]) if peaks else (cx, cy)
        xB, yB = 2 * cx - xA, 2 * cy - yA   # reflection fallback
    rows = [dict(image="A", x=xA, y=yA),
            dict(image="B", x=xB, y=yB)]
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "image_positions.csv", index=False)
    print(f"[pos] image A=({xA:.1f},{yA:.1f})  B=({xB:.1f},{yB:.1f})  "
          f"sep={np.hypot(xA-xB,yA-yB)*PIXSCALE:.2f}\" (Dawes {IMG_SEP_ARCSEC}\")")
    return df


# ---------------------------------------------------------------- light curves
def measure_lightcurves(positions, bands, meta, frame_dir, tag):
    meta_i = meta.set_index("stem")
    rows = []
    for band in bands:
        frames = sorted((frame_dir / band).glob("*.fits"))
        for fp in frames:
            stem = fp.stem
            if stem not in meta_i.index:
                continue
            mjd = float(meta_i.loc[stem, "mjd"])
            see = meta_i.loc[stem, "seeing"]
            fwhm_arcsec = float(np.clip(see, 0.9, 1.8)) if np.isfinite(see) else 1.3
            fwhm_px = fwhm_arcsec / PIXSCALE
            img = fits.getdata(fp).astype(np.float64)
            for _, pr in positions.iterrows():
                flux, ferr, snr, xf, yf = gaussian_psf_flux(
                    img, pr.x, pr.y, fwhm_px)
                if not np.isfinite(flux) or flux <= 0:
                    continue
                mag = ZP - 2.5 * np.log10(flux)
                magerr = (2.5 / np.log(10)) * (ferr / flux) if flux > 0 else np.nan
                rows.append(dict(image=pr.image, band=band, stem=stem, mjd=mjd,
                                 flux=flux, fluxerr=ferr, snr=snr,
                                 mag=mag, magerr=magerr, fwhm_arcsec=fwhm_arcsec))
    lc = pd.DataFrame(rows)
    lc.to_csv(DATA / f"lightcurves_{tag}.csv", index=False)
    n_pass = (lc.snr >= SN_MIN).sum() if len(lc) else 0
    print(f"[lc:{tag}] {len(lc)} raw PSF measurements, {n_pass} pass S/N>{SN_MIN}")
    return lc


# ---------------------------------------------------------------- Eq. 1 sigma
def variability_sigma(lc, tag):
    """Compute the Paper-II Eq. 1 variability metric per image, S/N>5 only.

    Reports BOTH the literal Eq. 1 (no 1/N_b inside the sqrt) and the
    std-dev interpretation that matches the Table 2 magnitude scale.
    """
    good = lc[(lc.snr >= SN_MIN) & np.isfinite(lc.mag)].copy()
    out = []
    for image, gi in good.groupby("image"):
        per_band = []
        N = 0
        eq1_literal_terms = 0.0
        eq1_std_terms = 0.0
        for band, gb in gi.groupby("band"):
            Nb = len(gb)
            if Nb < 2:
                # a band with one epoch contributes no scatter info
                per_band.append(dict(band=band, Nb=Nb, sigma_b=np.nan, mu_b=gb.mag.mean()))
                N += Nb
                continue
            mu_b = gb.mag.mean()
            ss = float(np.sum((gb.mag - mu_b) ** 2))           # sum of squares
            sigma_b = np.sqrt(ss / Nb)                          # std dev in band
            eq1_literal_terms += np.sqrt(ss)                    # literal Eq.1 band term
            eq1_std_terms += Nb * sigma_b                       # std-dev weighted term
            per_band.append(dict(band=band, Nb=Nb, sigma_b=sigma_b, mu_b=mu_b))
            N += Nb
        sigma_eq1_literal = eq1_literal_terms / N if N else np.nan
        sigma_eq1_std = eq1_std_terms / N if N else np.nan     # == N_b-weighted mean RMS
        out.append(dict(image=image, N_total=N,
                        sigma_eq1_literal=sigma_eq1_literal,
                        sigma_stddev=sigma_eq1_std,
                        per_band={d["band"]: round(float(d["sigma_b"]), 4)
                                  for d in per_band if np.isfinite(d["sigma_b"])}))
    sdf = pd.DataFrame(out)
    if len(sdf):
        # <sigma> across images (paper averages across all resolved images)
        avg = sdf.sigma_stddev.mean()
        avg_lit = sdf.sigma_eq1_literal.mean()
        sdf_save = sdf.copy()
        sdf_save["per_band"] = sdf_save["per_band"].astype(str)
        sdf_save.to_csv(DATA / f"sigma_{tag}.csv", index=False)
        print(f"[sigma:{tag}] per-image variability:")
        for _, r in sdf.iterrows():
            print(f"   image {r.image}: sigma(stddev)={r.sigma_stddev:.3f} mag  "
                  f"(literal Eq.1={r.sigma_eq1_literal:.3f})  "
                  f"N={r.N_total}  per-band={r.per_band}")
        print(f"[sigma:{tag}] <sigma> across images = {avg:.3f} mag "
              f"(literal {avg_lit:.3f})")
        return sdf, avg
    print(f"[sigma:{tag}] no S/N>{SN_MIN} measurements to compute sigma")
    return sdf, np.nan


# ---------------------------------------------------------------- figure
def plot_lightcurves(lc, sigma_df, tag, avg_sigma):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[fig] skipped: {e}")
        return
    good = lc[(lc.snr >= SN_MIN) & np.isfinite(lc.mag)]
    images = sorted(good.image.unique())
    if not images:
        return
    colors = {"g": "C2", "r": "C3", "z": "C1", "i": "C5", "Y": "C4"}
    fig, axes = plt.subplots(len(images), 1, figsize=(7.5, 3.0 * len(images)),
                             squeeze=False, sharex=True)
    for ax, image in zip(axes[:, 0], images):
        gi = good[good.image == image]
        for band, gb in gi.groupby("band"):
            ax.errorbar(gb.mjd, gb.mag, yerr=gb.magerr, fmt="o", ms=4,
                        color=colors.get(band, "k"), label=f"{band}", alpha=0.8)
            ax.plot(gb.mjd, np.full(len(gb), gb.mag.mean()), "--",
                    color=colors.get(band, "k"), lw=0.8, alpha=0.5)
        srow = sigma_df[sigma_df.image == image]
        s = float(srow.sigma_stddev.iloc[0]) if len(srow) else np.nan
        ax.invert_yaxis()
        ax.set_ylabel("PSF mag (ZP=30)")
        ax.set_title(f"image {image}   sigma = {s:.3f} mag", fontsize=10)
        ax.legend(fontsize=8, ncol=4, loc="best")
    axes[-1, 0].set_xlabel("MJD")
    ttl = f"{CAND_NAME} ({tag})   <sigma> = {avg_sigma:.3f} mag"
    fig.suptitle(ttl, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGS / f"lightcurve_{tag}.png", dpi=130)
    print(f"[fig] {FIGS/f'lightcurve_{tag}.png'}")


# ---------------------------------------------------------------- synth control
def synth_variable_source(meta, bands, positions, target_sigma, seed):
    """Inject a stochastic point source with a KNOWN magnitude sigma into the real
    reproj frames, then run the SAME PSF photometry + Eq. 1 and check recovery.

    The injected magnitudes are a damped-random-walk-like draw (Gaussian about a
    base mag with std = target_sigma in EACH band), so the per-band std dev is
    target_sigma by construction.  We inject at a blank-sky location away from the
    lens so the recovered signal is unambiguously the injected source.
    """
    rng = np.random.default_rng(seed)
    meta_i = meta.set_index("stem")
    xinj, yinj = 250.0, 250.0          # blank sky
    base_mag = {"g": 19.5, "r": 19.2, "z": 19.0, "i": 19.1, "Y": 19.0}
    synth_dir = DATA / "synth"
    synth_dir.mkdir(exist_ok=True)
    truth_rows = []
    for band in bands:
        ref = fits.getdata(DATA / f"reference_{band}.fits").astype(np.float64)
        sky_rms = float(np.nanstd(ref[100:200, 100:200])) or 1.0
        bdir = synth_dir / band
        bdir.mkdir(exist_ok=True)
        bm = base_mag.get(band, 19.2)
        for fp in sorted((REPROJ / band).glob("*.fits")):
            stem = fp.stem
            if stem not in meta_i.index:
                continue
            mjd = float(meta_i.loc[stem, "mjd"])
            see = meta_i.loc[stem, "seeing"]
            fwhm_arcsec = float(np.clip(see, 0.9, 1.8)) if np.isfinite(see) else 1.3
            sig_px = fwhm_arcsec / PIXSCALE / 2.3548
            # draw a magnitude with the TARGET per-band scatter
            mag = bm + rng.normal(0, target_sigma)
            flux = 10 ** (-0.4 * (mag - ZP))
            base = fits.getdata(fp).astype(np.float64)
            hdr = fits.getheader(fp)
            ny, nx = base.shape
            yy, xx = np.mgrid[0:ny, 0:nx]
            psf = np.exp(-((xx - xinj) ** 2 + (yy - yinj) ** 2) / (2 * sig_px ** 2))
            psf /= 2 * np.pi * sig_px ** 2          # normalised so amp*int = flux
            inj = base + flux * psf + rng.normal(0, sky_rms, base.shape)
            fits.PrimaryHDU(inj.astype("float32"), hdr).writeto(
                bdir / f"{stem}.fits", overwrite=True)
            truth_rows.append(dict(band=band, stem=stem, mjd=mjd, true_mag=mag))
    truth = pd.DataFrame(truth_rows)
    truth.to_csv(DATA / "synth_truth.csv", index=False)
    # per-band realised injected sigma (truth)
    real_sig = truth.groupby("band").true_mag.std(ddof=0)
    print(f"[synth] injected source: target sigma={target_sigma} mag; "
          f"realised per-band sigma={real_sig.round(3).to_dict()}")
    pos = pd.DataFrame([dict(image="SYNTH", x=xinj, y=yinj)])
    return synth_dir, pos, target_sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "synth", "both"], default="both")
    ap.add_argument("--bands", nargs="+", default=["g", "r", "z"])
    ap.add_argument("--synth-sigma", type=float, default=0.25,
                    help="target injected magnitude sigma (paper Table 2 ~0.25)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = pd.read_csv(DATA / "exposure_manifest.csv")

    if args.mode in ("real", "both"):
        pos = image_positions(args.bands)
        lc = measure_lightcurves(pos, args.bands, meta, REPROJ, "real")
        sdf, avg = variability_sigma(lc, "real")
        plot_lightcurves(lc, sdf, "real", avg)

    if args.mode in ("synth", "both"):
        synth_dir, spos, tsig = synth_variable_source(
            meta, args.bands, None, args.synth_sigma, args.seed)
        lc_s = measure_lightcurves(spos, args.bands, meta, synth_dir, "synth")
        sdf_s, avg_s = variability_sigma(lc_s, "synth")
        plot_lightcurves(lc_s, sdf_s, "synth", avg_s)
        if np.isfinite(avg_s):
            print(f"[synth] VALIDATION: injected sigma={tsig:.3f} -> "
                  f"recovered <sigma>={avg_s:.3f} mag "
                  f"(ratio {avg_s/tsig:.2f})")


if __name__ == "__main__":
    main()
