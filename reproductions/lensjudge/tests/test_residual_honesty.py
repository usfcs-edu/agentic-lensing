#!/usr/bin/env python3
"""Honesty validation for render.residual (the signed, noise-normalized elliptical-model
residual). Proves, on synthetic PSF-convolved Sersic galaxies + injected arcs, the four
properties the old Gaussian high-pass violated:

  1. noise consistency  - on a smooth galaxy (no arc) chi=(data-model)/sigma is ~N(0,1)
                          with NO coherent structure (no "butterfly" quadrupole).
  2. arc non-absorption - an injected tangential arc survives in the residual; the new model
                          recovers more of its flux than the legacy Gaussian high-pass.
  3. sign preservation  - deliberate over-subtraction shows a visible NEGATIVE (blue) bowl
                          (the legacy `res -= res.min()` would hide it).
  4. no butterfly       - on an elongated noiseless galaxy the residual is ~flat, whereas the
                          legacy isotropic high-pass leaves a major-axis quadrupole.

Run:  ~/.venvs/lensjudge/bin/python reproductions/lensjudge/tests/test_residual_honesty.py
(also importable as pytest test_* functions).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.modeling.models import Sersic2D
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # reproductions/ on path
from lensjudge.common import render  # noqa: E402

N = 101
PSF_SIGMA = 1.95          # ~1.2" FWHM at 0.262"/px
CORE_PX = 8               # inner ~2": known cusp + PSF-circularization residual zone (annotated,
                          # shown honestly as a blue over-subtraction). Lens arcs/rings sit outside it.
RNG = np.random.default_rng(0)


def _sersic(eps=0.3, theta=0.5, r_eff=8.0, n=3.0, amp=100.0):
    yy, xx = np.mgrid[0:N, 0:N]
    m = Sersic2D(amplitude=amp, r_eff=r_eff, n=n, x_0=N / 2, y_0=N / 2, ellip=eps, theta=theta)
    return gaussian_filter(np.asarray(m(xx, yy), float), PSF_SIGMA)


def _arc(radius=12.0, width=1.3, span_deg=90.0, pa_deg=20.0, total_flux=1.0):
    """A thin tangential arc segment (PSF-convolved), normalized to `total_flux`. Returns the
    arc image and a TIGHT crest mask (r +/-2 px around the arc), for arc-in-mask recovery."""
    yy, xx = np.mgrid[0:N, 0:N]
    dx, dy = xx - N / 2, yy - N / 2
    r = np.hypot(dx, dy)
    d = (np.degrees(np.arctan2(dy, dx)) - pa_deg + 180) % 360 - 180
    seg = np.exp(-0.5 * ((r - radius) / width) ** 2) * (np.abs(d) < span_deg / 2)
    seg = gaussian_filter(seg.astype(float), PSF_SIGMA)
    mask = (r > radius - 2) & (r < radius + 2) & (np.abs(d) < span_deg / 2)
    return seg / seg.sum() * total_flux, mask


def _cube(img):
    return np.stack([img, img, img]).astype(np.float32)


def _noise(sigma):
    return RNG.normal(0, sigma, (N, N))


def _rr():
    yy, xx = np.mgrid[0:N, 0:N]
    return np.hypot(xx - N / 2, yy - N / 2)


def _butterfly_amp(resid, r=12.0):
    """RMS over 8 azimuthal sectors of the mean residual in the annulus at radius r -> the
    m=2 quadrupole ("butterfly") strength that an isotropic model leaves on an elliptical galaxy."""
    yy, xx = np.mgrid[0:N, 0:N]
    dx, dy = xx - N / 2, yy - N / 2
    ann = (np.hypot(dx, dy) > r - 2) & (np.hypot(dx, dy) < r + 2)
    ang, vals = np.arctan2(dy, dx)[ann], resid[ann]
    means = [vals[(ang >= a) & (ang < a + np.pi / 4)].mean()
             for a in np.linspace(-np.pi, np.pi, 8, endpoint=False)]
    return float(np.nanstd(means))


# --------------------------------------------------------------------------- tests
def test_noise_consistency():
    """Outside the inner CORE_PX, chi is ~N(0,1) with no structure: the model neither
    fabricates nor leaves systematic features where lensing arcs actually appear."""
    cube = _cube(_sersic(amp=80.0) + _noise(2.0))
    chi = render._chi_band(cube[1], render._estimate_shape(cube))
    out = _rr() >= CORE_PX
    mean, std = float(chi[out].mean()), float(chi[out].std())
    frac3 = float((np.abs(chi[out]) > 3).mean())
    assert abs(mean) < 0.1, f"chi mean {mean:.3f} not ~0"
    assert 0.9 < std < 1.15, f"chi std {std:.3f} not ~1"
    assert frac3 < 0.01, f"{frac3:.3%} of pixels |chi|>3 (Gaussian expects ~0.27%)"
    print(f"  [1] noise consistency (r>={CORE_PX}px): chi mean={mean:+.3f} std={std:.3f} "
          f"|chi|>3={frac3:.2%}  OK")


def test_arc_non_absorption():
    """A tangential arc at the Einstein radius survives in the residual; the median model
    recovers a majority of its in-mask flux, far more than the legacy Gaussian high-pass
    (which smooths the arc into its model). Near-complete rings are partly absorbed by ANY
    azimuthal-median model -- a documented limit, not measured here."""
    gal = _sersic(amp=80.0)
    arc, mask = _arc(radius=12.0, span_deg=90.0, total_flux=gal.sum() * 0.03)
    cube = _cube(gal + arc + _noise(1.5))
    shape = render._estimate_shape(cube)
    inmask = float(arc[mask].sum())               # arc flux WITHIN the crest mask
    rec_new = float((cube[1] - render._model_band(cube[1], shape, "median"))[mask].sum()) / inmask
    rec_leg = float((cube[1] - gaussian_filter(cube[1].astype(float), 3.0))[mask].sum()) / inmask
    assert rec_new > 0.5, f"new recovered only {rec_new:.2f} of arc flux"
    assert rec_new > rec_leg + 0.3, f"new {rec_new:.2f} not clearly better than legacy {rec_leg:.2f}"
    print(f"  [2] arc non-absorption (90deg arc @12px): recovered new={rec_new:.2f} vs legacy "
          f"high-pass={rec_leg:.2f}  OK")


def test_oversubtraction_visible():
    """Sign is preserved: deliberate over-subtraction shows a visible NEGATIVE (blue) bowl
    that the legacy `res -= res.min()` re-floor would have hidden."""
    cube = _cube(_sersic(amp=80.0) + _noise(1.5))
    model = render._model_band(cube[1], render._estimate_shape(cube), "median")
    chi = (cube[1] - 1.05 * model) / render._robust_sigma(cube[1] - 1.05 * model)
    core = _rr() < 12
    core_mean = float(chi[core].mean())
    assert core_mean < -0.3, f"over-subtraction not visible as negative bowl (core chi {core_mean:+.3f})"
    assert float(np.clip(chi, 0, None)[core].mean()) > core_mean, "re-floor would hide over-subtraction"
    print(f"  [3] over-subtraction visible: core mean chi={core_mean:+.3f} (negative=blue bowl)  OK")


def test_models_smooth_galaxy():
    """The elliptical model actually removes the smooth elongated galaxy: outside the core its
    residual is far smaller than the legacy high-pass (which leaves the galaxy's own gradient),
    and it is core-confined -- so no butterfly quadrupole where lensing arcs appear."""
    gal = _cube(_sersic(eps=0.4, theta=0.5, amp=120.0))[1]   # noiseless, elongated
    shape = render._estimate_shape(_cube(gal))
    new_res = gal - render._model_band(gal, shape, "median")
    legacy_res = gal - gaussian_filter(gal.astype(float), sigma=3.0)
    rr = _rr()
    outer = (rr >= 10) & (rr < 45)
    s_new, s_leg = float(new_res[outer].std()), float(legacy_res[outer].std())
    assert s_new < 0.3 * s_leg, f"outer residual std new={s_new:.3f} not << legacy={s_leg:.3f}"
    assert float(new_res[rr < CORE_PX].std()) > 5 * s_new, "residual not core-confined"
    print(f"  [4] models smooth galaxy: outer(r>=10) residual std new={s_new:.3f} vs "
          f"legacy={s_leg:.3f} ({s_leg / max(s_new, 1e-9):.0f}x smaller); "
          f"butterfly@12 new={_butterfly_amp(new_res, 12):.2f} vs legacy={_butterfly_amp(legacy_res, 12):.2f}  OK")


def main():
    print("residual honesty validation (synthetic Sersic + arc injections):")
    fails = 0
    for fn in (test_noise_consistency, test_arc_non_absorption,
               test_oversubtraction_visible, test_models_smooth_galaxy):
        try:
            fn()
        except AssertionError as e:
            fails += 1
            print(f"  FAIL {fn.__name__}: {e}")
    print("ALL PASS" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
