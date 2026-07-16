"""Minimal, from-scratch strong-lensing numerics for the guide.

Deliberately NOT a wrapper over lenstronomy or gigalens. The guide's whole
pedagogical claim is that these objects are small enough to build yourself, so
they are built here in ~200 lines of numpy, with the conventions matched to the
repo's production stack (gigalens' vendored EPL/SIE/Sersic profiles) so the
numbers the guide computes are the numbers the campaigns compute.

Conventions (matched to reproductions/foundry-i/vendor/gigalens-sean/):
  * Angles in arcsec throughout. The lens-modeling likelihood in this repo is
    entirely angular — no cosmology enters it. See cosmo.py for the three
    places cosmology is real.
  * Deflection is the gradient of the potential: alpha = grad(psi).
  * Lens equation: beta = theta - alpha(theta).
  * Ellipticity is carried as (e1, e2), converted to (q, phi) by
    ``ellip_to_q_phi`` exactly as gigalens' epl.py does.
  * gamma is the 3-D density slope, rho ~ r^-gamma. gamma = 2 is isothermal.
    It is NOT the shear. The shear is gamma_ext / (gamma1, gamma2). This module
    never uses the bare name ``gamma`` for shear.
"""
from __future__ import annotations

import numpy as np

ARCSEC_PER_RAD = 206264.80624709636
C_KM_S = 299792.458


# --------------------------------------------------------------------------- #
# Ellipticity
# --------------------------------------------------------------------------- #
def ellip_to_q_phi(e1: float, e2: float) -> tuple[float, float]:
    """(e1, e2) -> (axis ratio q, position angle phi in radians).

    The gigalens convention (epl.py): phi = atan2(e2, e1) / 2, and with
    c = |e|, q = (1 - c) / (1 + c). The half-angle is the signature of a
    spin-2 quantity: an ellipse is unchanged by a 180-degree rotation, so its
    orientation lives on a circle of circumference pi, not 2*pi.
    """
    phi = np.arctan2(e2, e1) / 2.0
    c = np.clip(np.hypot(e1, e2), 0.0, 1.0)
    q = (1.0 - c) / (1.0 + c)
    return float(q), float(phi)


def _rotate(x, y, phi):
    c, s = np.cos(phi), np.sin(phi)
    return c * x + s * y, -s * x + c * y


# --------------------------------------------------------------------------- #
# Deflection fields  (alpha = grad psi)
# --------------------------------------------------------------------------- #
def sis_deflection(x, y, theta_E):
    """Singular isothermal sphere. alpha = theta_E * r_hat — constant modulus.

    The flat rotation curve of a galaxy in one line: the deflection does not
    care how far out you are, only which way.
    """
    r = np.hypot(x, y)
    r = np.where(r == 0, 1e-12, r)
    return theta_E * x / r, theta_E * y / r


def sie_deflection(x, y, theta_E, q, phi=0.0, s=1e-4):
    """Singular isothermal ellipsoid (Kormann+ 1994 closed form).

    ``s`` is a core radius that regularises the central singularity; it is the
    same trick gigalens' sie.py uses (``s_scale``). At q -> 1 this reduces to
    ``sis_deflection`` (checked in tests).
    """
    xr, yr = _rotate(x, y, phi)
    q = float(np.clip(q, 1e-4, 0.99999))
    f = np.sqrt(1.0 - q * q)
    r = np.sqrt(q * q * (xr * xr + s * s) + yr * yr)
    ax = theta_E * q / f * np.arctan(f * xr / (r + s))
    ay = theta_E * q / f * np.arctanh(f * yr / (r + q * q * s))
    return _rotate(ax, ay, -phi)


def epl_kappa(x, y, theta_E, slope, q, phi=0.0):
    """EPL convergence. kappa ~ R^(1-gamma); gamma=2 recovers the SIE.

    The elliptical radius convention matches the repo's EPL:
        kappa = (3-gamma)/2 * (theta_E / sqrt(q x^2 + y^2/q))^(gamma-1)
    """
    xr, yr = _rotate(x, y, phi)
    R = np.sqrt(q * xr * xr + yr * yr / q)
    R = np.where(R == 0, 1e-12, R)
    return 0.5 * (3.0 - slope) * (theta_E / R) ** (slope - 1.0)


def shear_deflection(x, y, gamma1, gamma2):
    """External shear. Linear in position, hence uniform in its derivatives."""
    return gamma1 * x + gamma2 * y, gamma2 * x - gamma1 * y


# --------------------------------------------------------------------------- #
# The Jacobian, and everything that falls out of it
# --------------------------------------------------------------------------- #
def lens_jacobian(defl_fn, x, y, h=None):
    """A = d(beta)/d(theta), by central differences on the lens equation.

    Returns (a11, a12, a21, a22). Deliberately numerical: the guide's point is
    that magnification is a Jacobian determinant, and differentiating the map
    numerically makes that concrete rather than quoting an analytic special case.
    """
    if h is None:
        h = 1e-5 * max(np.ptp(np.asarray(x)), 1.0)

    def beta(px, py):
        ax, ay = defl_fn(px, py)
        return px - ax, py - ay

    bx_px, by_px = beta(x + h, y)
    bx_mx, by_mx = beta(x - h, y)
    bx_py, by_py = beta(x, y + h)
    bx_my, by_my = beta(x, y - h)
    a11 = (bx_px - bx_mx) / (2 * h)
    a21 = (by_px - by_mx) / (2 * h)
    a12 = (bx_py - bx_my) / (2 * h)
    a22 = (by_py - by_my) / (2 * h)
    return a11, a12, a21, a22


def kappa_gamma_from_jacobian(a11, a12, a21, a22):
    """A = (1-kappa) I - Gamma  ->  recover kappa and the shear components.

    kappa   = 1 - (a11 + a22)/2                (the trace: an isotropic squeeze)
    gamma_1 = -(a11 - a22)/2                   (the traceless part: a stretch)
    gamma_2 = -(a12 + a21)/2
    """
    kappa = 1.0 - 0.5 * (a11 + a22)
    g1 = -0.5 * (a11 - a22)
    g2 = -0.5 * (a12 + a21)
    return kappa, g1, g2


def magnification(a11, a12, a21, a22):
    """mu = 1 / det A. Signed: negative mu means a parity-flipped image.

    This is the SAME change-of-variables factor that a normalizing flow applies
    as -log|det J|, and a cousin of the -1/2 logdet A Occam term in
    cgl/marg.py. Three log-dets, one idea.
    """
    return 1.0 / (a11 * a22 - a12 * a21)


def det_a(defl_fn, X, Y):
    a11, a12, a21, a22 = lens_jacobian(defl_fn, X, Y)
    return a11 * a22 - a12 * a21


# --------------------------------------------------------------------------- #
# Light
# --------------------------------------------------------------------------- #
def sersic_bn(n: float) -> float:
    """b_n via the Capaccioli approximation used by gigalens' sersic.py."""
    return 1.9992 * n - 0.3271


def sersic(R, I_e, R_e, n):
    """I(R) = I_e exp(-b_n[(R/R_e)^(1/n) - 1]).  n=1 exponential, n=4 de Vauc."""
    return I_e * np.exp(-sersic_bn(n) * ((R / R_e) ** (1.0 / n) - 1.0))


# --------------------------------------------------------------------------- #
# Einstein radius
# --------------------------------------------------------------------------- #
def theta_e_sis(sigma_v_kms, d_ds_over_d_s):
    """theta_E = 4 pi (sigma_v/c)^2 D_ds/D_s, in arcsec.

    Hsu+2025 Eq. 1; implemented in reproductions/hsu-2025/07_classify_einstein_dimple.py
    and reproductions/lensjudge/tools/spectrum.py.
    """
    return 4.0 * np.pi * (sigma_v_kms / C_KM_S) ** 2 * d_ds_over_d_s * ARCSEC_PER_RAD


def mean_kappa_within(kappa_fn, theta, n=200_000):
    """Mean convergence inside radius ``theta``, by direct quadrature.

    kappa_bar(theta) = (2/theta^2) * int_0^theta kappa(t) t dt

    Setting kappa_bar = 1 IS the definition of the Einstein radius, so this
    returning 1.0 at theta_E is a consistency check, not a coincidence.
    """
    t = np.linspace(theta / n, theta, n)
    return float(np.trapezoid(kappa_fn(t) * t, t) * 2.0 / theta**2)


# --------------------------------------------------------------------------- #
# Drizzle
# --------------------------------------------------------------------------- #
def drizzle_lag1(r: float) -> float:
    """Phase-averaged along-axis lag-1 noise correlation of a square drizzle
    kernel at pixfrac=1: t(1) = (r-1)/(r-1/3), r = native/output pixel ratio.

    The closed form the campaign anchors its kernel on: at the fine skycell's
    r = 3.2075 this gives 0.76805, which the numerically enumerated drizzle
    operator reproduces as 0.76799 (data/noise_kernel_report.json).

    This one line is why drizzled data has correlated noise, and therefore why
    a diagonal likelihood is mis-specified on it.
    """
    return (r - 1.0) / (r - 1.0 / 3.0)
