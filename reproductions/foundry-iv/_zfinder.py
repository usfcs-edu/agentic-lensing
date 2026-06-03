"""_zfinder.py - automated redshift finder for MUSE 1D spectra.

The methodological value-add of this reproduction: Lin et al. 2025 determined every
redshift by *manual* identification of spectral features in extracted 1D spectra. Here
we build an *automated* line-ID / redshift estimator and run it on the same public MUSE
cubes.

Two engines, matching the two physical regimes in the paper:

  find_z_absorption(...)  -- for passive lens galaxies (and metal-absorption sources).
      Continuum-normalize the spectrum, then for a grid of trial z, sum the
      ABSORPTION strength (1 - normalized flux, clipped >0) at the redshifted
      positions of a rest-frame line list. The best z maximizes total absorption
      depth landing on real lines. Robust to a missing line or two.

  find_z_emission(...)    -- for star-forming / Lyman-break sources with emission.
      Continuum-subtract, detect emission peaks, then for a grid of trial z score the
      summed emission flux landing on a redshifted emission line list ([OII], Hb,
      [OIII], Ha, ...). Also reports the strongest single detected line so a [OII]
      doublet (the paper's workhorse) drives the solution.

Both return (z_best, score_curve(z_grid), zgrid, detail). The line lists below are
exactly the rest wavelengths the paper quotes in Section 4.1 (air wavelengths, Angstrom).

This module has NO astronomy-archive dependencies, so it is unit-testable on synthetic
spectra (see 03_synthetic_unittest.py).
"""
import numpy as np

# ---- rest-frame line lists (air, Angstrom) -------------------------------------------
# Early-type galaxy absorption features (lens galaxies) + the 4000A break edge.
ABS_LINES_GAL = {
    "CaII_K":  3933.7,
    "CaII_H":  3968.5,
    "G_band":  4304.4,   # paper quotes ~4308
    "Hbeta":   4861.3,
    "MgI_b":   5175.4,   # Mg b triplet center
    "NaI_D":   5892.9,
}
# Rest-UV ISM / stellar absorption lines seen in high-z lensed sources (paper Sect 4.1).
ABS_LINES_UV = {
    "SiIV_1393":  1393.8,
    "SiIV_1402":  1402.8,
    "SiII_1526":  1526.7,
    "CIV_1549":   1549.5,
    "FeII_1608":  1608.5,
    "AlII_1671":  1670.8,
}
# Emission lines for star-forming sources.
EMIS_LINES = {
    "[OII]_3727":  3727.1,
    "[OII]_3729":  3729.9,
    "[NeIII]":     3869.8,
    "Hgamma":      4340.5,
    "Hbeta":       4861.3,
    "[OIII]_4959": 4958.9,
    "[OIII]_5007": 5006.8,
    "Halpha":      6562.8,
    "[SII]_6716":  6716.4,
    "[SII]_6731":  6730.8,
}
# A 'semi-forbidden' emission line that appears in UV-absorption sources.
EMIS_LINES_UV = {"[CIII]_1909": 1908.7}

MUSE_MIN, MUSE_MAX = 4750.0, 9350.0  # MUSE WFM spectral window (Angstrom)

# Bright telluric/airglow emission lines whose imperfectly-subtracted residuals
# dominate MUSE spectra redward of ~5500 A (OH bands, Na, O2). We mask +/- a few A
# around each so they cannot masquerade as galaxy features. List is the strong-line
# subset of the UVES/MUSE sky atlas (air, Angstrom).
SKY_LINES = np.array([
    5577.3, 5889.95, 5895.92, 6300.3, 6363.8, 6498.7, 6533.0, 6553.6, 6863.9,
    6912.6, 6923.2, 6939.5, 7244.0, 7276.4, 7316.3, 7340.9, 7358.7, 7392.2,
    7402.4, 7524.1, 7571.7, 7750.6, 7794.1, 7821.5, 7841.2, 7913.7, 7949.2,
    7964.6, 7993.3, 8014.6, 8025.7, 8344.6, 8399.2, 8430.2, 8452.3, 8465.4,
    8493.4, 8504.6, 8778.3, 8827.1, 8885.9, 8919.6, 8943.4, 8988.4, 9001.6,
    9038.1, 9337.9, 5460.7, 6235.0, 6257.9, 6287.4, 6307.0,
])
SKY_HALF_A = 4.0   # mask half-width around each sky line


def build_mask(wave, var=None, sky_lines=True, var_clip=8.0):
    """Boolean 'good pixel' mask. Drops NaNs, sky-line windows, and (if a variance
    spectrum is given) pixels with anomalously high variance (sky residuals)."""
    good = np.isfinite(wave)
    if sky_lines:
        for s in SKY_LINES:
            good &= np.abs(wave - s) > SKY_HALF_A
    if var is not None:
        v = np.asarray(var, float)
        good &= np.isfinite(v) & (v > 0)
        med = np.median(v[good]) if good.any() else np.nan
        if np.isfinite(med):
            good &= v < var_clip * med
    return good


# ---- continuum estimation ------------------------------------------------------------
def _running_median_continuum(wave, flux, win_A=120.0, good=None):
    """Median-filter continuum on a wavelength window (handles uneven grids).
    `good` (bool) excludes masked pixels from the running median."""
    cont = np.empty_like(flux)
    half = win_A / 2.0
    f = flux.copy()
    if good is not None:
        f = np.where(good, flux, np.nan)
    for i in range(len(wave)):
        lo = np.searchsorted(wave, wave[i] - half)
        hi = np.searchsorted(wave, wave[i] + half)
        seg = f[lo:hi]
        seg = seg[np.isfinite(seg)]
        cont[i] = np.median(seg) if seg.size else np.nan
    # fill any all-masked gaps by interpolation
    bad = ~np.isfinite(cont)
    if bad.any() and (~bad).any():
        cont[bad] = np.interp(wave[bad], wave[~bad], cont[~bad])
    return cont


def normalize(wave, flux, win_A=120.0, good=None):
    """Return continuum-normalized flux (flux/continuum) and the continuum."""
    cont = _running_median_continuum(wave, flux, win_A, good)
    norm = flux / cont
    norm[~np.isfinite(norm)] = 1.0
    return norm, cont


def continuum_subtract(wave, flux, win_A=120.0, good=None):
    cont = _running_median_continuum(wave, flux, win_A, good)
    sub = flux - cont
    sub[~np.isfinite(sub)] = 0.0
    return sub, cont


# ---- core: noise-weighted cross-correlation against a line template ------------------
def _line_template(wave, rest_lines, z, sigma_A):
    """A zero-mean multi-Gaussian template of the line list at redshift z, evaluated
    on the data wavelength grid. Sign convention: +1 at each line center (we flip the
    data so absorption -> positive; emission stays positive)."""
    t = np.zeros_like(wave)
    obs = np.asarray(rest_lines) * (1.0 + z)
    inband = (obs > wave[0] + 2 * sigma_A) & (obs < wave[-1] - 2 * sigma_A)
    for lam in obs[inband]:
        t += np.exp(-0.5 * ((wave - lam) / sigma_A) ** 2)
    return t, int(inband.sum())


def _xcorr_scan(wave, signal, ivar, good, rest_lines, zgrid, sigma_A, min_lines):
    """Noise-weighted, template-normalized cross-correlation over a z grid.

    score(z) = sum( w * T * S ) / sqrt( sum( w * T^2 ) )   with w = ivar*good
    where S is the (zero-mean) signal (absorption depth or emission residual) and
    T is the unit-amplitude line template at z. This is the standard matched-filter
    significance: noisy/masked pixels are down-weighted by ivar, and the
    normalization makes scores comparable across z (different numbers of in-band
    lines) and equal to ~SNR of the combined line detection.
    """
    w = ivar * good
    score = np.zeros_like(zgrid)
    nlines = np.zeros_like(zgrid, dtype=int)
    for k, z in enumerate(zgrid):
        T, nl = _line_template(wave, rest_lines, z, sigma_A)
        if nl < min_lines:
            continue
        nlines[k] = nl
        num = np.sum(w * T * signal)
        den = np.sqrt(np.sum(w * T * T)) + 1e-12
        score[k] = num / den
    return score, nlines


# ---- absorption-line redshift finder -------------------------------------------------
def find_z_absorption(wave, flux, var=None, zmin=0.0, zmax=1.2, dz=0.0002,
                      line_list=None, win_A=120.0, sigma_A=2.5, min_lines=2,
                      sky_mask=True):
    """Redshift from a noise-weighted cross-correlation of the continuum-normalized
    absorption signal against the redshifted galaxy/UV line template.

    Robust to noise spikes (ivar weighting), sky residuals (sky+variance mask), and
    bad pixels. Returns (z_best, score(z), zgrid, per-line report).
    """
    if line_list is None:
        line_list = ABS_LINES_GAL
    good = build_mask(wave, var, sky_lines=sky_mask).astype(float)
    norm, cont = normalize(wave, flux, win_A, good > 0)
    # absorption signal: positive where flux dips below continuum, clipped to physical
    depth = np.clip(1.0 - norm, -0.5, 1.0)
    depth = depth - np.median(depth[good > 0])         # zero-mean
    if var is not None:
        ivar = good / (np.asarray(var, float) / np.maximum(cont, 1.0) ** 2 + 1e-12)
        ivar[~np.isfinite(ivar)] = 0.0
    else:
        ivar = good.copy()
    zgrid = np.arange(zmin, zmax + dz, dz)
    rest = np.array(list(line_list.values()))
    score, nlines = _xcorr_scan(wave, depth, ivar, good, rest, zgrid, sigma_A, min_lines)
    best = int(np.argmax(score))
    detail = _line_report(wave, norm, line_list, zgrid[best], sigma_A,
                          kind="absorption", good=good > 0)
    return zgrid[best], score, zgrid, detail


# ---- emission-line redshift finder ---------------------------------------------------
def find_z_emission(wave, flux, var=None, zmin=0.0, zmax=3.5, dz=0.0002,
                    line_list=None, win_A=150.0, sigma_A=2.0, min_lines=1,
                    sky_mask=True):
    """Redshift from a noise-weighted cross-correlation of the continuum-subtracted
    emission signal against the redshifted emission line template."""
    if line_list is None:
        line_list = EMIS_LINES
    good = build_mask(wave, var, sky_lines=sky_mask).astype(float)
    sub, cont = continuum_subtract(wave, flux, win_A, good > 0)
    emis = np.clip(sub, 0.0, None)            # emission -> positive only
    emis = emis - np.median(emis[good > 0])
    if var is not None:
        ivar = good / (np.asarray(var, float) + 1e-12)
        ivar[~np.isfinite(ivar)] = 0.0
    else:
        ivar = good.copy()
    zgrid = np.arange(zmin, zmax + dz, dz)
    rest = np.array(list(line_list.values()))
    score, nlines = _xcorr_scan(wave, emis, ivar, good, rest, zgrid, sigma_A, min_lines)
    best = int(np.argmax(score))
    # noise for the per-line report
    noise = 1.4826 * np.median(np.abs(sub[good > 0] - np.median(sub[good > 0]))) + 1e-6
    detail = _line_report(wave, sub / noise, line_list, zgrid[best], sigma_A,
                          kind="emission", noise=1.0, good=good > 0)
    return zgrid[best], score, zgrid, detail


def _line_report(wave, signal, line_list, z, sigma_A, kind, noise=None, good=None):
    """Per-line SNR-ish report at the best z."""
    if good is None:
        good = np.ones_like(wave, dtype=bool)
    rep = {}
    for name, lam0 in line_list.items():
        lam = lam0 * (1 + z)
        if not (wave[0] < lam < wave[-1]):
            rep[name] = None
            continue
        sel = (np.abs(wave - lam) < 3 * sigma_A) & good
        if not sel.any():
            rep[name] = None
            continue
        if kind == "emission":
            peak = np.max(signal[sel])
            snr = peak / noise if noise else np.nan
            rep[name] = {"obs_A": round(float(lam), 1), "peak": float(peak), "snr": float(snr)}
        else:
            depth = float(np.max(np.clip(1 - signal[sel], 0, None)))
            rep[name] = {"obs_A": round(float(lam), 1), "depth": depth}
    return rep


# ---- joint spatial+spectral emission-line source finder (cube-level) -----------------
def find_emission_source_in_cube(data, wave, lens_yx, pixscale, var=None,
                                 annulus_arcsec=(0.8, 7.0), zmin=0.2, zmax=1.6,
                                 dz=0.002, nb_half_A=5.0, cont_off_A=(15.0, 40.0),
                                 anchor_lines=(3727.4, 5006.8, 4861.3)):
    """Find a lensed EMISSION source jointly in (z, sky-position) inside a MUSE cube.

    The continuum-residual 'emap' used elsewhere is dominated by bright interlopers
    across the 60" FoV. Here we instead scan trial redshifts; at each z we build a
    *narrow-band* line map for each anchor emission line (default [OII] 3727,
    [OIII] 5007, Hbeta), continuum-subtracted using off-band side windows, sum the
    maps, and record the brightest compact peak inside the annulus around the lens.
    The (z, position) that maximizes this annular line flux is the source.

    This is the automated analog of the paper's manual "pick the arc spaxels and
    look for the [OII] doublet" -- it finds BOTH where the arc is and its redshift.

    Returns dict(z, iy, ix, sep_arcsec, peak, zgrid, curve).
    """
    nz, ny, nx = data.shape
    yc, xc = lens_yx
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - yc, xx - xc) * pixscale
    ann = (r > annulus_arcsec[0]) & (r < annulus_arcsec[1])

    # Precompute a spectral cumulative sum so any narrow-band integral is an O(1)
    # difference of two cumsum planes (instead of re-slicing the 3 GB cube per z).
    clean = np.nan_to_num(data, nan=0.0)
    csum = np.empty((nz + 1, ny, nx), dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(clean, axis=0, out=csum[1:])
    dw = float(np.median(np.diff(wave)))  # ~1.25 A/pix

    def band_mean(lam_lo, lam_hi):
        i0 = int(np.searchsorted(wave, lam_lo))
        i1 = int(np.searchsorted(wave, lam_hi))
        if i1 <= i0:
            return None
        return (csum[i1] - csum[i0]) / (i1 - i0)

    zgrid = np.arange(zmin, zmax + dz, dz)
    best = dict(z=np.nan, iy=yc, ix=xc, peak=-np.inf)
    curve = np.full_like(zgrid, -np.inf)
    lo_a, hi_a = cont_off_A
    for k, z in enumerate(zgrid):
        linemap = np.zeros((ny, nx))
        n_used = 0
        for l0 in anchor_lines:
            lam = l0 * (1 + z)
            if not (wave[0] + hi_a < lam < wave[-1] - hi_a):
                continue
            on = band_mean(lam - nb_half_A, lam + nb_half_A)
            off_b = band_mean(lam - hi_a, lam - lo_a)
            off_r = band_mean(lam + lo_a, lam + hi_a)
            if on is None or off_b is None or off_r is None:
                continue
            linemap += on - 0.5 * (off_b + off_r)
            n_used += 1
        if n_used == 0:
            continue
        lm = linemap.copy()
        lm[~ann] = -np.inf
        iy, ix = np.unravel_index(np.argmax(lm), lm.shape)
        val = float(linemap[iy, ix])
        curve[k] = val
        if val > best["peak"]:
            best = dict(z=float(z), iy=int(iy), ix=int(ix), peak=val)
    best["sep_arcsec"] = float(np.hypot(best["iy"] - yc, best["ix"] - xc) * pixscale)
    best["zgrid"] = zgrid
    best["curve"] = curve
    return best
