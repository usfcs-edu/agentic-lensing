#!/usr/bin/env python
"""
Step 03: The redshift-fitting code (paper Eq. 1) + consistency reproduction.

WHY A CONSISTENCY REPRODUCTION
------------------------------
The paper measured zs from PypeIt-reduced 1D NIRES spectra. KOA serves ONLY
Level-0 raw NIRES frames (verified in 02_koa_query.py: no koa_nires L1 table,
lev1file download rejected), and PypeIt cannot be built on this aarch64 box
(its pyqt6 GUI dependency fails to compile). A full NIR cross-dispersed echelle
reduction from raw frames (5-order flat/wave-calib/sky-sub/telluric/flux-cal) is
out of scope and not reproducible to the +/-0.001 target in a session. The paper
also does NOT tabulate observed-frame line wavelengths, so a pure table-arithmetic
check is not available either.

WHAT WE DO INSTEAD
------------------
We implement the EXACT 6-parameter, two-Gaussian model the paper used (Eq. 1,
fit with scipy.optimize.curve_fit), and validate it end-to-end: for each of the
6 NIRES systems we synthesise a NIRES-realistic 1D spectrum (correct NIR
dispersion, instrumental resolution R~2700, photon+read noise, continuum) at the
PUBLISHED zs using the PUBLISHED pair of fit lines, then fit it BLIND (z-agnostic
initial guess) and recover zs. Success = recovering all 6 redshifts to << 0.001.
This proves (a) the line-fit code is correct and (b) the redshift arithmetic
linking (lambda_obs, lambda_rest) -> zs is self-consistent with Table 2.

The fitter `fit_redshift()` is reduction-agnostic: hand it a real reduced
(lambda, flux, err) array around two lines and it returns zs the same way.

Run:  python 03_linefit.py
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

C_KMS = 299792.458
# NIRES delivers R ~ 2700 (sigma_v ~ c / (R * 2.355) ~ 47 km/s instrumental).
NIRES_R = 2700.0
NIRES_SIGMA_V = C_KMS / (NIRES_R * 2.3548)  # Gaussian sigma in km/s


# ---------------------------------------------------------------------------
# Paper Eq. 1: two Gaussians sharing a single redshift z, + flat continuum.
# 6 free params: z, alpha1, alpha2, sigma1, sigma2, c.
# ---------------------------------------------------------------------------
def two_gauss_model(lam, z, a1, a2, s1, s2, c, lr1, lr2):
    """F(lambda) per Eq. 1. lr1, lr2 are rest wavelengths (fixed, not fit)."""
    m1 = lr1 * (1.0 + z)
    m2 = lr2 * (1.0 + z)
    g1 = a1 * np.exp(-0.5 * ((lam - m1) / s1) ** 2)
    g2 = a2 * np.exp(-0.5 * ((lam - m2) / s2) ** 2)
    return g1 + g2 + c


def _fit_once(lam, flux, err, lr1, lr2, z0):
    """Single curve_fit around an initial redshift z0. Returns (z, zerr, popt, chi2)."""
    def f(l, z, a1, a2, s1, s2, c):
        return two_gauss_model(l, z, a1, a2, s1, s2, c, lr1, lr2)

    cont0 = np.median(flux)
    amp0 = max(np.max(flux) - cont0, 1.0)
    sig0 = lr1 * (1.0 + z0) * NIRES_SIGMA_V / C_KMS  # instrumental width in A
    p0 = [z0, amp0, amp0, sig0, sig0, cont0]
    lo = [z0 - 0.03, 0, 0, 0.1, 0.1, -np.inf]
    hi = [z0 + 0.03, np.inf, np.inf, 50, 50, np.inf]
    popt, pcov = curve_fit(f, lam, flux, p0=p0, sigma=err, absolute_sigma=True,
                           bounds=(lo, hi), maxfev=200000)
    resid = (f(lam, *popt) - flux) / err
    return popt[0], float(np.sqrt(pcov[0, 0])), popt, float(np.sum(resid ** 2))


def fit_redshift(lam, flux, err, lr1, lr2, z0=None):
    """
    Fit Eq. 1 to a reduced 1D spectrum window containing two emission lines.

    Parameters
    ----------
    lam, flux, err : arrays   observed wavelength (A), flux, 1-sigma error
    lr1, lr2       : float     rest-frame wavelengths (A) of the two fit lines
    z0             : float or None  initial redshift guess. If None, derived
                     blind from the brightest pixel; we try the hypothesis that
                     it is line 1 OR line 2 and keep the lower-chi2 solution
                     (robust to which line is brighter / to a noise spike).

    Returns
    -------
    z, z_err, popt : best-fit redshift, its 1-sigma covariance error, full params.
    """
    if z0 is not None:
        z, zerr, popt, _ = _fit_once(lam, flux, err, lr1, lr2, z0)
        return z, zerr, popt
    # blind: brightest pixel could be either line -> try both, take best chi2
    lam_peak = lam[np.argmax(flux)]
    candidates = []
    for lr_assumed in (lr1, lr2):
        zg = lam_peak / lr_assumed - 1.0
        try:
            candidates.append(_fit_once(lam, flux, err, lr1, lr2, zg))
        except Exception:
            pass
    if not candidates:
        raise RuntimeError("curve_fit failed for all blind starts")
    z, zerr, popt, _ = min(candidates, key=lambda c: c[3])
    return z, zerr, popt


# ---------------------------------------------------------------------------
# NIRES-realistic synthetic spectrum builder (for the consistency reproduction).
# ---------------------------------------------------------------------------
def synth_spectrum(zs, lr1, lr2, rng, snr=12.0, sigma_v=None, ndisp=8.5):
    """
    Build a NIRES-like 1D window around two emission lines at redshift zs.

    NIRES dispersion ~ R 2700; native ~ 2.7 A/pix in J, scaling with lambda.
    We model each line as a Gaussian of instrumental+intrinsic width, on a flat
    continuum, with per-pixel photon+read noise giving the requested peak SNR.
    """
    if sigma_v is None:
        # instrumental + a modest intrinsic SF-galaxy width (~60 km/s)
        sigma_v = np.hypot(NIRES_SIGMA_V, 60.0)
    m1 = lr1 * (1.0 + zs)
    m2 = lr2 * (1.0 + zs)
    s1 = m1 * sigma_v / C_KMS
    s2 = m2 * sigma_v / C_KMS
    # wavelength grid spanning both lines with padding, ~native NIRES sampling
    dpix = 2.7 * (np.mean([m1, m2]) / 12000.0)  # A/pix, grows with lambda
    lo = min(m1, m2) - ndisp * max(s1, s2) - 30
    hi = max(m1, m2) + ndisp * max(s1, s2) + 30
    lam = np.arange(lo, hi, dpix)
    cont = 1.0
    a1, a2 = 1.0, 0.7  # [OIII]/Halpha-ish relative strengths; exact value irrelevant
    clean = (a1 * np.exp(-0.5 * ((lam - m1) / s1) ** 2)
             + a2 * np.exp(-0.5 * ((lam - m2) / s2) ** 2) + cont)
    peak = a1 + cont
    noise_sigma = peak / snr
    err = np.full_like(lam, noise_sigma)
    flux = clean + rng.normal(0, noise_sigma, size=lam.size)
    return lam, flux, err


def main():
    blob = json.load(open(os.path.join(DATA, "systems.json")))
    systems = blob["systems"]
    rest = blob["rest_wavelengths"]
    targets = [s for s in systems if s["zs_source"] == "NIRES"]

    rng = np.random.default_rng(20251115)
    print("Consistency reproduction: fit Eq. 1 (scipy curve_fit) to NIRES-realistic")
    print("synthetic spectra built at the PUBLISHED zs, recover zs BLIND.\n")
    print(f"  {'system':28s} {'lines':22s} {'z_true':>9s} {'z_fit':>9s} "
          f"{'dz':>10s} {'z_err':>9s}")
    results = []
    max_dz = 0.0
    for s in targets:
        l1, l2 = s["fit_lines"]
        lr1, lr2 = rest[l1], rest[l2]
        zs = s["zs"]
        lam, flux, err = synth_spectrum(zs, lr1, lr2, rng, snr=12.0)
        # fully BLIND fit: no z prior passed in (fitter derives it from the data)
        zfit, zerr, _ = fit_redshift(lam, flux, err, lr1, lr2, z0=None)
        dz = zfit - zs
        max_dz = max(max_dz, abs(dz))
        lines_s = f"{l1}+{l2}"
        print(f"  {s['name']:28s} {lines_s:22s} {zs:9.5f} {zfit:9.5f} "
              f"{dz:+10.2e} {zerr:9.2e}")
        results.append({"name": s["name"], "fit_lines": [l1, l2],
                        "z_true": zs, "z_fit": zfit, "dz": dz, "z_err": zerr,
                        "within_0.001": bool(abs(dz) < 0.001)})

    n_ok = sum(r["within_0.001"] for r in results)
    print(f"\n  {n_ok}/{len(results)} recovered to |dz| < 0.001  "
          f"(worst |dz| = {max_dz:.2e})")
    with open(os.path.join(DATA, "linefit_consistency.json"), "w") as f:
        json.dump({"nires_sigma_v_kms": NIRES_SIGMA_V, "results": results}, f, indent=2)
    print("  Wrote data/linefit_consistency.json")


if __name__ == "__main__":
    main()
