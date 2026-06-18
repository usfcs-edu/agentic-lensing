#!/usr/bin/env python3
"""make_residual_figures.py — figures for papers/residual.tex (old vs new residual).

Every figure is an explicit OLD (residual_legacy, the Gaussian high-pass) vs NEW
(residual, the signed chi=(data-model)/sigma) comparison on the SAME pixels, using the
real renderers in common/render.py so the figures show exactly what the pipeline produces.
Real cubes come from the staged DR11 cutouts (config.CUTOUT_DIRS["dr11"]); the butterfly,
over-subtraction and noise panels use synthetic PSF-convolved Sersic galaxies (mirrors
tests/test_residual_honesty.py). Outputs single composite PNGs to papers/figures/resid_*.png.

Run (lensjudge venv with matplotlib):
    ~/.venvs/lensjudge/bin/python reproductions/lensjudge/eval/make_residual_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.modeling.models import Sersic2D
from scipy.ndimage import gaussian_filter

REPRO = Path(__file__).resolve().parents[2]            # reproductions/
sys.path.insert(0, str(REPRO))
from lensjudge import config                            # noqa: E402
from lensjudge.common import render                     # noqa: E402

FIG = REPRO / "lensjudge" / "papers" / "figures"
CDIR = config.CUTOUT_DIRS["dr11"]
N = 101
PSF_SIGMA = 1.95                                         # ~1.2" FWHM at 0.262"/px


def cube(name):
    return render.load_cube(CDIR / f"{name}.fits")


def arr(img):
    return np.asarray(img.convert("RGB"))


def _model_rgb(c):
    """Lupton RGB of the per-band elliptical smooth-light model (what gets subtracted)."""
    shape = render._estimate_shape(c)
    m = np.stack([render._model_band(c[b], shape, "median") for b in range(3)])
    return render.lupton(m)


# --------------------------------------------------------------------- examples
EXAMPLES = [
    ("s_441355_1188", "KNOWN: CSWA 1 (Cosmic Horseshoe) — near-complete Einstein ring"),
    ("s_351303_461",  "KNOWN: Huang+2020 DESI-194.5344+03.5358 — tangential arc"),
    ("s_230539_4159", "NEW (unconfirmed): lrg+companion mimic — no real arc"),
    ("s_234689_542",  "NEW (unconfirmed): lrg+companion mimic — no real arc"),
]


def fig_examples():
    rows = [(n, l) for n, l in EXAMPLES if (CDIR / f"{n}.fits").exists()]
    fig, ax = plt.subplots(len(rows), 3, figsize=(7.2, 2.45 * len(rows)))
    cols = ["DESI grz", "OLD residual\n(Gaussian high-pass)", "NEW residual\n(signed $\\chi$)"]
    for i, (nm, lab) in enumerate(rows):
        c = cube(nm)
        for j, im in enumerate([render.lupton(c), render.residual_legacy(c),
                                render.residual_chi_luminance(c)]):
            ax[i, j].imshow(arr(im)); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(cols[j], fontsize=8.5)
        ax[i, 0].set_ylabel(lab.replace(" — ", "\n"), fontsize=7.0, rotation=0,
                            ha="right", va="center", labelpad=4)
    fig.tight_layout()
    fig.savefig(FIG / "resid_examples.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("  resid_examples.png", [n for n, _ in rows])


def fig_anatomy():
    nm = "s_441355_1188"
    c = cube(nm)
    fig = plt.figure(figsize=(7.2, 4.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.06)
    top = [("DESI grz", render.lupton(c)), ("elliptical smooth model", _model_rgb(c)),
           ("OLD residual (high-pass)", render.residual_legacy(c)),
           ("NEW residual $\\chi$ ($r{+}z$)", render.residual_chi_luminance(c))]
    for j, (t, im) in enumerate(top):
        a = fig.add_subplot(gs[0, j]); a.imshow(arr(im)); a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=8)
    big = fig.add_subplot(gs[1, :])
    big.imshow(arr(render.residual(c))); big.set_xticks([]); big.set_yticks([])
    big.set_title("NEW primary view — per-band signed $\\chi$ montage:  g  |  r  |  z   "
                  "(a real arc is RED and coherent across g and r; red=excess, blue=over-subtraction, $\\pm5\\sigma$)",
                  fontsize=8)
    fig.savefig(FIG / "resid_anatomy.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("  resid_anatomy.png")


# --------------------------------------------------------------------- synthetic
def _sersic(eps=0.4, theta=0.5, amp=120.0):
    yy, xx = np.mgrid[0:N, 0:N]
    m = Sersic2D(amplitude=amp, r_eff=8.0, n=3.0, x_0=N / 2, y_0=N / 2, ellip=eps, theta=theta)
    return gaussian_filter(np.asarray(m(xx, yy), float), PSF_SIGMA)


def fig_butterfly():
    # elongated galaxy + realistic noise; compare the two MODELS' residuals on a common chi scale
    # (the same noise sigma), so the old isotropic model's major-axis quadrupole is visible and the
    # new elliptical model's residual is flat noise. (A noiseless galaxy would divide by sigma->0.)
    rng = np.random.default_rng(0)
    gal = _sersic()
    data = gal + rng.normal(0, 2.0, (N, N))
    c = np.stack([data, data, data]).astype(np.float32)
    shape = render._estimate_shape(c)
    sig = render._robust_sigma(data - render._model_band(data, shape, "median"))
    old_chi = (data - gaussian_filter(data, 3.0)) / sig          # isotropic (legacy) model
    new_chi = (data - render._model_band(data, shape, "median")) / sig
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))
    ax[0].imshow(arr(render.lupton(c))); ax[0].set_title("elongated galaxy (+noise)", fontsize=8.5)
    ax[1].imshow(render._diverging_rgb(old_chi)[::-1])
    ax[1].set_title("OLD isotropic model residual:\n4-lobe 'butterfly' (mimics arcs)", fontsize=8.5)
    ax[2].imshow(render._diverging_rgb(new_chi)[::-1])
    ax[2].set_title("NEW elliptical model residual:\nflat $\\chi$ (no major-axis quadrupole)", fontsize=8.5)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(FIG / "resid_butterfly.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("  resid_butterfly.png")


def fig_sign_noise():
    rng = np.random.default_rng(0)
    gal = _sersic(eps=0.3)
    c = np.stack([gal + rng.normal(0, 2.0, (N, N)) for _ in range(3)]).astype(np.float32)
    shape = render._estimate_shape(c)
    # (a) over-subtraction: scale model up 5%; OLD re-floor hides it, NEW shows a blue bowl
    model = render._model_band(c[1], shape, "median")
    chi_over = (c[1] - 1.05 * model) / render._robust_sigma(c[1] - 1.05 * model)
    # (b) chi histogram outside the inner core vs N(0,1)
    chi = render._chi_band(c[1], shape)
    yy, xx = np.mgrid[0:N, 0:N]
    out = np.hypot(xx - N / 2, yy - N / 2) >= 8

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    ax[0].imshow(arr(render.residual_legacy(c))); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title("OLD residual (re-floored):\nover-subtraction is invisible", fontsize=8.2)
    ax[1].imshow(render._diverging_rgb(chi_over)[::-1]); ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title("NEW residual, model $\\times1.05$:\nover-subtraction = visible blue bowl", fontsize=8.2)
    ax[2].hist(chi[out], bins=60, range=(-5, 5), density=True, color="#4060c0", alpha=0.8, label="new $\\chi$ (r$\\geq$8px)")
    xs = np.linspace(-5, 5, 200); ax[2].plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1.2, label="$\\mathcal{N}(0,1)$")
    ax[2].set_title("noise calibration:\n$\\chi$ tracks $\\mathcal{N}(0,1)$", fontsize=8.2)
    ax[2].set_xlabel("$\\chi=(\\mathrm{data}-\\mathrm{model})/\\sigma$", fontsize=8); ax[2].legend(fontsize=7); ax[2].set_yticks([])
    fig.tight_layout(); fig.savefig(FIG / "resid_sign_noise.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("  resid_sign_noise.png")


def _harmonic_model(band, x0, y0, eps, theta, mmax=2):
    """Aggressive per-annulus low-order harmonic fit (illustration only) -- removes the core
    quadrupole but ABSORBS broad arcs/rings; shown to justify the median default."""
    n = band.shape[0]; yy, xx = np.mgrid[0:n, 0:n].astype(float)
    dx, dy = xx - x0, yy - y0; ct, st = np.cos(theta), np.sin(theta)
    xr = dx * ct + dy * st; yr = -dx * st + dy * ct; q = max(1 - eps, 0.1)
    r_ell = np.sqrt(xr * xr + (yr / q) ** 2); phi = np.arctan2(yr, xr)
    rmax = r_ell.max(); nb = max(int(rmax) + 1, 4); edges = np.linspace(0, rmax + 1e-6, nb + 1)
    idx = np.clip(np.digitize(r_ell.ravel(), edges) - 1, 0, nb - 1)
    flat = band.astype(float).ravel(); fphi = phi.ravel(); model = np.zeros_like(flat)
    for b in range(nb):
        m = idx == b
        if m.sum() < 6:
            if m.any(): model[m] = np.median(flat[m])
            continue
        I = flat[m]; ph = fphi[m]; cols = [np.ones_like(ph)]
        for k in range(1, mmax + 1): cols += [np.cos(k * ph), np.sin(k * ph)]
        A = np.vstack(cols).T; keep = np.ones(len(I), bool)
        for _ in range(3):
            coef = np.linalg.lstsq(A[keep], I[keep], rcond=None)[0]; r = I - A @ coef
            s = (np.median(np.abs(r - np.median(r))) * 1.4826) or 1; nk = np.abs(r - np.median(r)) < 3 * s
            if nk.sum() < 6 or (nk == keep).all(): break
            keep = nk
        model[m] = A @ coef
    return model.reshape(band.shape)


def fig_tradeoff():
    nm = "s_441355_1188"; c = cube(nm); lum = (c[1] + c[2]).astype(float)
    shape = render._estimate_shape(c)
    chi_med = (lum - render._model_band(lum, shape, "median")) / render._robust_sigma(lum - render._model_band(lum, shape, "median"))
    hm = _harmonic_model(lum, *shape)
    chi_har = (lum - hm) / render._robust_sigma(lum - hm)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))
    ax[0].imshow(arr(render.lupton(c))); ax[0].set_title("DESI grz (the horseshoe ring)", fontsize=8.5)
    ax[1].imshow(render._diverging_rgb(chi_med)[::-1]); ax[1].set_title("median model (chosen):\nring PRESERVED", fontsize=8.5)
    ax[2].imshow(render._diverging_rgb(chi_har)[::-1]); ax[2].set_title("aggressive harmonic fit:\nring ABSORBED (dishonest)", fontsize=8.5)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(FIG / "resid_tradeoff.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("  resid_tradeoff.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {FIG}")
    fig_examples(); fig_anatomy(); fig_butterfly(); fig_sign_noise(); fig_tradeoff()
    print("done")


if __name__ == "__main__":
    main()
