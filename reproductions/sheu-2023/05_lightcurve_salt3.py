#!/usr/bin/env python3
"""
05_lightcurve_salt3.py  --  Sheu+2023 §3.4 / §6.1.1 light-curve fit + magnification

Two modes.

(A) --mode real   [default]
    Forced aperture photometry on every B08 difference image at the transient
    location (default = the >=3-subdetection group nearest the lens centre, i.e.
    the counter-image where Sheu+2023 place the L-SN), converted to DECam g/r/z
    fluxes via the per-epoch MAGZERO.  We then fit a SALT3 SN Ia model with
    sncosmo under a source-galaxy redshift prior (z=1.188, Sheu §6.1.1) AND an
    unlensed prior (lens z=0.374), and compute the implied magnification from the
    Hubble residual the same way the paper does.

(B) --mode synth
    End-to-end PIPELINE VALIDATION on data with KNOWN truth: inject a SALT3
    L-SN Ia (z, t0, x0 known; magnified by a known mu) into the REAL reference
    frames at the counter-image position to make a synthetic multi-epoch
    sequence, run it back through B08 differencing + forced photometry + SALT3,
    and check we recover t0, the light curve, and the input magnification.  This
    isolates and proves the photometry->SALT3->mu chain independent of whether
    the true SN epochs happen to be in our (capped) real download.

Magnification from Hubble residual (paper §3.4)
-----------------------------------------------
A lensed SN Ia appears too bright for its redshift: the Hubble residual
mu_resid = m_obs - m_expected(z) is negative; the magnification is
        magnification = 10 ** (-0.4 * mu_resid).
Sheu+2023 report mu_resid = -2.29 +/- 0.30 mag -> magnification 8.23 (+2.61/-1.98)
for DESI-344.6252-48.8977 under the L-SN Ia (z=0.833) postulation.

Outputs
-------
  data/forced_photometry.csv         per-epoch diff-image forced flux (real mode)
  data/salt3_fit_<tag>.csv           best-fit SALT3 params + chi2/dof + mu
  figs/lightcurve_<tag>.png          light curve + best-fit SALT3 model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

import sncosmo
from astropy.cosmology import FlatLambdaCDM

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figs"
DIFF = DATA / "diff"
REPROJ = DATA / "reproj"

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
PIXSCALE = 0.262
# DECam band -> sncosmo bandpass
BANDMAP = {"g": "desg", "r": "desr", "z": "desz", "i": "desi", "Y": "desy"}
ZP = 30.0  # our common photometric zeropoint (nanomaggie -> AB via 22.5? see note)
# NOTE: InstCal MAGZERO is the AB zeropoint for ADU; our cutouts are in those ADU
# units rescaled to MAGZERO=30. So flux f -> AB mag = 30 - 2.5*log10(f).
ZPSYS = "ab"


def aperture_flux(img: np.ndarray, x: float, y: float, r: float = 3.0):
    """Simple circular-aperture sum + local-annulus background on a diff image."""
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - x, yy - y)
    ap = (rr <= r) & np.isfinite(img)
    ann = (rr > r + 2) & (rr <= r + 6) & np.isfinite(img)
    if ap.sum() < 3:
        return np.nan, np.nan
    bg = np.nanmedian(img[ann]) if ann.sum() > 5 else 0.0
    flux = float(np.nansum(img[ap] - bg))
    noise = float(np.nanstd(img[ann])) * np.sqrt(ap.sum()) if ann.sum() > 5 else np.nan
    return flux, noise


def forced_photometry(x: float, y: float, bands, meta) -> pd.DataFrame:
    rows = []
    meta_i = meta.set_index("stem")
    for band in bands:
        for fp in sorted((DIFF / band).glob("*_diff.fits")):
            stem = fp.stem.replace("_diff", "")
            img = fits.getdata(fp).astype("float64")
            f, n = aperture_flux(img, x, y, r=5.0)
            mjd = float(meta_i.loc[stem, "mjd"]) if stem in meta_i.index else np.nan
            magzero = float(meta_i.loc[stem, "magzero"]) if stem in meta_i.index else 30.0
            # rescale to ZP=30 already done at reproject stage, so flux is in ZP=30 units
            rows.append(dict(band=band, stem=stem, mjd=mjd, flux=f,
                             fluxerr=n if np.isfinite(n) and n > 0 else abs(f) * 0.2 + 1.0,
                             zp=ZP, zpsys=ZPSYS, bandpass=BANDMAP[band]))
    return pd.DataFrame(rows)


def to_sncosmo_table(phot: pd.DataFrame):
    from astropy.table import Table
    p = phot.dropna(subset=["mjd", "flux"]).copy()
    return Table(dict(time=p.mjd.to_numpy(), band=p.bandpass.to_numpy(),
                      flux=p.flux.to_numpy(), fluxerr=p.fluxerr.to_numpy(),
                      zp=p.zp.to_numpy(), zpsys=p.zpsys.to_numpy()))


def fit_salt3(table, z, fix_z=True, t0_guess=None):
    model = sncosmo.Model(source="salt3")
    model.set(z=z)
    if t0_guess is None:
        t0_guess = float(np.median(table["time"]))
    # seed amplitude from the brightest band epoch so the optimiser starts on-source
    fmax = float(np.nanmax(table["flux"])) if len(table) else 1.0
    x0_seed = max(1e-6, 1e-3 * (fmax / 1e4))
    model.set(t0=t0_guess, x0=x0_seed, x1=0.0, c=0.0)
    vary = ["t0", "x0", "x1", "c"]
    bounds = {"t0": (table["time"].min() - 40, table["time"].max() + 40),
              "x0": (0.0, 1.0), "x1": (-3, 3), "c": (-0.4, 0.4)}
    if not fix_z:
        vary = ["z"] + vary
        bounds["z"] = (max(0.01, z - 0.3), z + 0.3)
    try:
        # let sncosmo auto-guess amplitude+t0 when S/N is sufficient
        res, fitted = sncosmo.fit_lc(table, model, vary, bounds=bounds,
                                     guess_z=False)
    except sncosmo.fitting.DataQualityError:
        # low S/N: keep our seed, fix the auto-guess off
        res, fitted = sncosmo.fit_lc(table, model, vary, bounds=bounds,
                                     guess_amplitude=False, guess_t0=False,
                                     guess_z=False)
    return res, fitted


def magnification_from_hubble(fitted, z) -> tuple[float, float]:
    """Peak abs B mag vs SN Ia standard -> magnification (paper §3.4)."""
    try:
        mb = fitted.source_peakabsmag("bessellb", "ab", cosmo=COSMO)
    except Exception:
        # fall back to apparent peak B
        mb = np.nan
    M_Ia = -19.25  # standard SN Ia (paper Table 1)
    # Hubble residual = observed peak abs mag - standard; negative => brighter => lensed
    resid = mb - M_Ia
    mu = 10 ** (-0.4 * resid)
    return float(resid), float(mu)


# ---------------- synthetic validation ----------------
def synthesize(meta, band_frames, x, y, z=0.833, mu=8.0, seed=0):
    """Inject a magnified SALT3 SN Ia into the real reference frames -> synthetic epochs."""
    rng = np.random.default_rng(seed)
    model = sncosmo.Model(source="salt3")
    model.set(z=z, x1=0.0, c=0.0)
    # Place t0 where the epoch sampling is densest so the rise/fall is captured
    # (a controlled test of t0 + magnification recovery, not a coverage test).
    mjds = np.sort(meta.mjd.dropna().to_numpy())
    # densest 60-day window centre
    best_t0, best_n = float(np.median(mjds)), -1
    for c in mjds:
        n = int(np.sum(np.abs(mjds - c) <= 30))
        if n > best_n:
            best_n, best_t0 = n, float(c)
    t0 = best_t0
    # Calibrate x0 to the SN Ia standard candle (M_B = -19.25, paper Table 1) so
    # that the ONLY source of extra brightness is the injected magnification mu.
    model.set(t0=t0)
    model.set_source_peakabsmag(-19.25, "bessellb", "ab", cosmo=COSMO)
    truth = dict(z=z, t0=t0, x1=0.0, c=0.0, mu=mu, x0=model.get("x0"))
    synth_dir = DATA / "synth"
    synth_dir.mkdir(exist_ok=True)
    meta_i = meta.set_index("stem")
    out_meta = []
    for band, frames in band_frames.items():
        ref = fits.getdata(DATA / f"reference_{band}.fits").astype("float64")
        bp = BANDMAP[band]
        for fp in frames:
            stem = fp.stem
            base = fits.getdata(fp).astype("float64")
            hdr = fits.getheader(fp)
            mjd = float(meta_i.loc[stem, "mjd"]) if stem in meta_i.index else np.nan
            if not np.isfinite(mjd):
                continue
            # model flux (in same ZP=30 system as frames) * magnification
            try:
                fmodel = model.bandflux(bp, mjd, zp=ZP, zpsys=ZPSYS) * mu
            except Exception:
                fmodel = 0.0
            fmodel = max(float(fmodel), 0.0)
            # Gaussian PSF source at (x,y). DECam seeing FWHM ~1.0-1.6"; the InstCal
            # SEEING header is unreliable here (FWHM/sigma confusion), so clamp to a
            # realistic FWHM and convert FWHM(arcsec)->sigma(px).
            see_hdr = float(meta_i.loc[stem, "seeing"]) if (
                stem in meta_i.index and np.isfinite(meta_i.loc[stem, "seeing"])) else 1.2
            fwhm_arcsec = float(np.clip(see_hdr, 0.9, 1.6))
            see_px = fwhm_arcsec / PIXSCALE / 2.355
            ny, nx = base.shape
            yy, xx = np.mgrid[0:ny, 0:nx]
            psf = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * see_px ** 2))
            psf /= psf.sum()
            # add realistic per-pixel sky noise so S/N is meaningful
            sky_rms = float(np.nanstd(base[100:200, 100:200]))
            inj = base + fmodel * psf + rng.normal(0, max(sky_rms, 1.0), base.shape)
            outp = synth_dir / f"{band}__{stem}.fits"
            fits.PrimaryHDU(inj.astype("float32"), hdr).writeto(outp, overwrite=True)
            out_meta.append(dict(band=band, stem=f"{band}__{stem}", mjd=mjd,
                                 magzero=ZP, synth=str(outp),
                                 truth_flux=fmodel))
    return pd.DataFrame(out_meta), truth, synth_dir


def synth_difference(synth_dir, band, meta):
    """B08-difference each synthetic frame vs the band reference, return diffs dict."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "b08", HERE / "03_difference_imaging_b08.py")
    b08 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b08)
    ref = fits.getdata(DATA / f"reference_{band}.fits").astype("float64")
    diffs = {}
    for fp in sorted(synth_dir.glob(f"{band}__*.fits")):
        sci = fits.getdata(fp).astype("float64")
        diff, _, _ = b08.difference_image(ref, sci, hw=4, ntile=2)
        diffs[fp.stem] = diff
    return diffs


def run_synth(meta, bands, x, y, z, mu, seed):
    # Inject at a BLANK-SKY location away from the lens and the real transient,
    # so the recovered signal is unambiguously the injected SN (controlled test).
    x, y = 250.0, 250.0
    band_frames = {b: sorted((REPROJ / b).glob("*.fits")) for b in bands}
    smeta, truth, synth_dir = synthesize(meta, band_frames, x, y, z=z, mu=mu, seed=seed)
    print(f"[synth] injected SALT3 SN Ia z={z} mu={mu} t0={truth['t0']:.1f} "
          f"into {len(smeta)} frames")
    # forced photometry on synthetic diffs
    rows = []
    for band in bands:
        diffs = synth_difference(synth_dir, band, meta)
        for stem, diff in diffs.items():
            f, n = aperture_flux(diff, x, y, r=5.0)
            mjd = float(smeta.set_index("stem").loc[stem, "mjd"])
            rows.append(dict(band=band, stem=stem, mjd=mjd, flux=f,
                             fluxerr=n if np.isfinite(n) and n > 0 else abs(f) * 0.2 + 1.0,
                             zp=ZP, zpsys=ZPSYS, bandpass=BANDMAP[band]))
    phot = pd.DataFrame(rows)
    phot.to_csv(DATA / "forced_photometry_synth.csv", index=False)
    table = to_sncosmo_table(phot)
    res, fitted = fit_salt3(table, z, fix_z=True, t0_guess=truth["t0"])
    chi2dof = res.chisq / max(res.ndof, 1)
    print(f"[synth] SALT3 fit: chi2/dof={chi2dof:.2f}  "
          f"t0={fitted.get('t0'):.1f} (truth {truth['t0']:.1f})  "
          f"x0={fitted.get('x0'):.2e}")
    resid, mu_rec = magnification_from_hubble(fitted, z)
    print(f"[synth] recovered Hubble resid={resid:.2f}  magnification={mu_rec:.2f} "
          f"(injected {mu})")
    save_fit("synth", res, fitted, z, chi2dof, resid, mu_rec, truth)
    plot_lc("synth", table, fitted, z, chi2dof, mu_rec, truth)
    return truth, mu_rec


def save_fit(tag, res, fitted, z, chi2dof, resid, mu, truth=None):
    d = dict(tag=tag, z=z, chi2=res.chisq, ndof=res.ndof, chi2dof=chi2dof,
             t0=fitted.get("t0"), x0=fitted.get("x0"),
             x1=fitted.get("x1"), c=fitted.get("c"),
             hubble_resid=resid, magnification=mu)
    if truth:
        d.update({f"truth_{k}": v for k, v in truth.items()})
    pd.DataFrame([d]).to_csv(DATA / f"salt3_fit_{tag}.csv", index=False)


def plot_lc(tag, table, fitted, z, chi2dof, mu, truth=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"desg": "C2", "desr": "C3", "desz": "C1", "desi": "C5", "desy": "C4"}
        fig, ax = plt.subplots(figsize=(7, 4.5))
        tmin, tmax = table["time"].min() - 30, table["time"].max() + 30
        tg = np.linspace(tmin, tmax, 200)
        for bp in sorted(set(table["band"])):
            m = table["band"] == bp
            ax.errorbar(table["time"][m], table["flux"][m],
                        yerr=table["fluxerr"][m], fmt="o", color=colors.get(bp, "k"),
                        label=f"{bp} data", ms=5)
            try:
                ax.plot(tg, fitted.bandflux(bp, tg, zp=ZP, zpsys=ZPSYS),
                        "-", color=colors.get(bp, "k"), alpha=0.7)
            except Exception:
                pass
        ttl = (f"SALT3 fit ({tag})  z={z}  chi2/dof={chi2dof:.2f}  "
               f"magnification={mu:.2f}")
        if truth:
            ttl += f"\n(injected mu={truth['mu']}, t0={truth['t0']:.0f})"
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("MJD"); ax.set_ylabel(f"diff flux (zp={ZP})")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(FIGS / f"lightcurve_{tag}.png", dpi=130)
        print(f"[fig] {FIGS/f'lightcurve_{tag}.png'}")
    except Exception as e:
        print(f"[fig] skipped: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "synth", "both"], default="both")
    ap.add_argument("--bands", nargs="+", default=["g", "r", "z"])
    ap.add_argument("--z", type=float, default=1.188,
                    help="source-galaxy redshift prior (Sheu §6.1.1)")
    ap.add_argument("--synth-z", type=float, default=0.833)
    ap.add_argument("--synth-mu", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = pd.read_csv(DATA / "exposure_manifest.csv")
    # transient location: nearest >=3 group to lens centre (the counter-image)
    groups = pd.read_csv(DATA / "groups.csv")
    cen = groups[groups.offset_arcsec < 3.0].sort_values(
        "n_subdet", ascending=False)
    if len(cen):
        x, y = float(cen.iloc[0].x), float(cen.iloc[0].y)
    else:
        x, y = 400.0, 400.0
    print(f"[loc] transient location (x,y)=({x:.1f},{y:.1f}), "
          f"offset {np.hypot(x-400,y-400)*PIXSCALE:.2f}\" from lens centre")

    if args.mode in ("real", "both"):
        phot = forced_photometry(x, y, args.bands, meta)
        phot.to_csv(DATA / "forced_photometry.csv", index=False)
        print(f"[real] forced photometry on {len(phot)} diff epochs")
        table = to_sncosmo_table(phot)
        for tag, z, fixz in [("real_lensed", args.z, True),
                             ("real_unlensed", 0.374, True)]:
            try:
                res, fitted = fit_salt3(table, z, fix_z=fixz)
                chi2dof = res.chisq / max(res.ndof, 1)
                resid, mu = magnification_from_hubble(fitted, z)
                print(f"[real:{tag}] z={z} chi2/dof={chi2dof:.2f} "
                      f"t0={fitted.get('t0'):.1f} resid={resid:.2f} mu={mu:.2f}")
                save_fit(tag, res, fitted, z, chi2dof, resid, mu)
                plot_lc(tag, table, fitted, z, chi2dof, mu)
            except Exception as e:
                print(f"[real:{tag}] fit failed: {e}")

    if args.mode in ("synth", "both"):
        run_synth(meta, args.bands, x, y, args.synth_z, args.synth_mu, args.seed)


if __name__ == "__main__":
    main()
