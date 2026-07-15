"""NFW_ELLIPSE_SLOPE -- NFW ellipse natively parameterized by OBSERVABLES:
(theta_E, s_E) instead of (Rs, alpha_Rs) or (theta_E, Rs).

Moved here from GIGALens-Code/experiments/why_hard_to_sample/ (branch
why-hard-t0t1, 2026-07-04) at user request; gate battery (all passing:
render identity vs NFW_ELLIPSE_EINSTEIN 1.4e-9, AD-vs-AD tangent 4.0e-11)
lives there as t29_slope_class_gpu.py; full derivation + noise model in
GIGALens-Code/docs/logs/why-hard-to-sample.md (T29).

    theta_E : Einstein radius (same convention as NFW_ELLIPSE_EINSTEIN:
              spherical-equivalent |alpha|(theta_E) = theta_E).
    s_E     : deflection log-slope AT the (live) Einstein radius,
              s_E = d ln|alpha| / d ln r |_{r=theta_E}  = 2 - gamma_local.
              s_E = 0 is the isothermal (SIS) slope; s_E -> 1 the mass-sheet
              limit; s_E -> -1 the point-mass limit. Same NFW family --
              (theta_E, s_E) <-> (Rs, alpha_Rs) is a smooth bijection.

Motivation (docs/logs/why-hard-to-sample.md, T25-T28): sampling Rs directly
produces a funnel + prior-edge pathology; the slope coordinate s cured it
(T28: minESS 1187 vs baseline 143) and priors on (theta_E, s_E) are the ones
physics/observation actually inform. This class moves that reparameterization
INTO the profile, so priors stay independent per-parameter slots and no
custom bijector/spline machinery is needed at all.

KEY REDUCTION: the NFW deflection shape is |alpha|(r) = 4 rho0 Rs^2 g(x)/x
with x = r/Rs, so the log-slope at any radius is a UNIVERSAL 1-D function of
x alone:

    sigma(x) = d ln( g(x)/x ) / d ln x        (strictly decreasing in x)

and "s_E at the live theta_E" is solved by ONE scalar inversion, with the
theta_E-dependence a division:

    x* = sigma^{-1}(s_E),   Rs = theta_E / x*,
    rho0 = theta_E^2 / (4 Rs^3 g(x*))          (EINSTEIN-class convention).

DEFINITION DETAIL (registered): sigma uses the same central FD in ln r as the
T28 observable (relative step H_LNR = 1e-4, matching t25_transforms.FD_REL),
so s_E here is numerically the SAME observable T28 put its prior on (exact
log-derivative differs by O(H^2) ~ 1e-9 -- irrelevant, but stated). sigma is
a smooth closed-form jnp expression, so all its gradients are exact
gradients OF this definition.

GRADIENTS THROUGH THE SOLVE: sigma^{-1} is computed by fixed-count bisection
in ln x (80 iterations; bracket x in [1e-4, 1e4] covers s_E in ~(-0.999,
0.9995)) wrapped in jax.custom_jvp with the implicit-function tangent
dx*/ds = 1 / sigma'(x*), sigma' by exact autodiff of sigma. custom_jvp
supports BOTH forward and reverse mode (reverse = linearize + transpose), so
the inherited autodiff hessian/convergence of MassProfile works. NEVER
differentiate through bisection iterates -- they are piecewise constant.

SUPPORTED RANGE + NUMERICAL FLOOR (measured, T29 diagnostics 2026-07-04):
sigma -> +-1 only LOGARITHMICALLY (sigma ~ 1 - 1/ln(2/x) as x->0), and the
gigalens g_(x) form suffers catastrophic cancellation at small x (relative
error ~2 eps/x^2), which the FD window amplifies by 1/(2*H_LNR) = 5e3 and the
inversion by ln^2(2/x). Supported s_E range: [-0.8, +0.8] (x in [7.4e-3, 148],
i.e. Rs from theta_E/148 to 135*theta_E -- every physically sane lens; the
carousel prior [0, 0.75] sits well inside). Within it the x* floor is
~1e-9..1e-7 in ln Rs at the extremes (micro-arcseconds on Rs; ~1e-5 nats of
log-density texture -- 1000x below the scale that ever mattered, cf. T20).
Out-of-range s_E clamps to the bracket edge: keep prior support inside.
AD gradients are exact derivatives of the implemented expressions and are NOT
limited by this floor (verified by the noise-free AD-vs-AD gate GC3 in
t29_slope_class_gpu.py); only FD references feel it. Run float64.
"""
import functools

import jax
import jax.numpy as jnp
from jax import jit

from gigalens.jax.profile import MassProfile

# universal-slope solver constants
H_LNR = 1e-4          # central FD relative step in ln r (== t25_transforms.FD_REL)
_XLO, _XHI = 1e-4, 1e4   # bisection bracket in x = r/Rs (sigma range ~(-0.999, 0.9995))
_NBISECT = 80            # ln-bracket width 18.4 / 2^80 -> far below f64 eps
_C = 0.000001            # g_ clamp, identical to NFW._c


def _g(x):
    """Exact mirror of gigalens NFW.g_ (nfw.py:37-51): branchwise domain clamp."""
    x = jnp.maximum(_C, x)
    x_lt1 = jnp.clip(x, _C, 1.0 - 1e-6)
    x_gt1 = jnp.clip(x, 1.0 + 1e-6, jnp.inf)
    g_lt1 = jnp.log(x_lt1 / 2.0) + 1.0 / jnp.sqrt(1.0 - x_lt1 ** 2) * jnp.arccosh(1.0 / x_lt1)
    g_gt1 = jnp.log(x_gt1 / 2.0) + 1.0 / jnp.sqrt(x_gt1 ** 2 - 1.0) * jnp.arccos(1.0 / x_gt1)
    return jnp.where(x < 1.0, g_lt1, g_gt1)


def sigma_of_x(x):
    """Universal NFW deflection log-slope sigma(x) = d ln(g(x)/x)/d ln x at
    x = r/Rs. Central FD in ln x with H_LNR (T28-identical definition);
    smooth closed form, strictly decreasing, exact-gradient-able."""
    x = jnp.asarray(x)
    xp = x * (1.0 + H_LNR)
    xm = x * (1.0 - H_LNR)
    ap = _g(xp) / xp
    am = _g(xm) / xm
    return (jnp.log(ap) - jnp.log(am)) / (jnp.log(xp) - jnp.log(xm))


_dsigma_dx = jnp.vectorize(jax.grad(lambda x: sigma_of_x(x)))


@jax.custom_jvp
def x_of_s(s):
    """x* = sigma^{-1}(s), elementwise. Fixed-count bisection in ln x
    (primal is NOT differentiated; see the custom JVP below)."""
    s = jnp.asarray(s)
    lo = jnp.full_like(s, jnp.log(_XLO))
    hi = jnp.full_like(s, jnp.log(_XHI))

    def body(_, carry):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        smid = sigma_of_x(jnp.exp(mid))
        # sigma decreasing: sigma(mid) > s means x too small -> move lo up
        go_right = smid > s
        return jnp.where(go_right, mid, lo), jnp.where(go_right, hi, mid)

    lo, hi = jax.lax.fori_loop(0, _NBISECT, body, (lo, hi))
    return jnp.exp(0.5 * (lo + hi))


@x_of_s.defjvp
def _x_of_s_jvp(primals, tangents):
    (s,), (s_dot,) = primals, tangents
    x = x_of_s(s)
    # implicit function theorem on sigma(x*) = s:  dx*/ds = 1 / sigma'(x*)
    return x, s_dot / _dsigma_dx(x)


def s_of_Rs_thetaE(Rs, theta_E):
    """Convenience inverse: the slope observable s_E for given (Rs, theta_E).
    Use to convert existing (Rs, ...) fits/priors into this class's params."""
    return sigma_of_x(jnp.asarray(theta_E) / jnp.asarray(Rs))


def Rs_of_s_thetaE(s_E, theta_E):
    """Convenience forward: physical Rs for given (s_E, theta_E)."""
    return jnp.asarray(theta_E) / x_of_s(s_E)


class NFW_ELLIPSE_SLOPE(MassProfile):
    """NFW ellipse with native observable parameters (theta_E, s_E).
    Renders identically to NFW_ELLIPSE_EINSTEIN at matched parameters
    (gate GA in t29_slope_class_gpu.py)."""

    _name = 'NFW_ELLIPSE_SLOPE'
    _params = ['theta_E', 's_E', 'e1', 'e2', 'center_x', 'center_y']
    _r_min = 0.0000001

    def __init__(self):
        super(NFW_ELLIPSE_SLOPE, self).__init__()

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, theta_E, s_E, e1, e2, center_x, center_y):
        x_star = x_of_s(s_E)
        Rs = theta_E / x_star
        # EINSTEIN-class convention (nfw.py:146), with g(theta_E/Rs) = g(x*)
        rho0 = theta_E ** 2 / (4.0 * Rs ** 3 * _g(x_star))
        e, phi = self._param_conv(e1, e2)

        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        x, y = x * jnp.sqrt(1 - e), y * jnp.sqrt(1 + e)
        R = jnp.sqrt(x ** 2 + y ** 2)

        fx, fy = self._nfwAlpha(R, Rs, rho0, x, y)
        fx *= jnp.sqrt(1 - e)
        fy *= jnp.sqrt(1 + e)
        fx, fy = self._rotate(fx, fy, -phi)
        return fx, fy

    @functools.partial(jit, static_argnums=(0,))
    def _nfwAlpha(self, R, Rs, rho0, ax_x, ax_y):
        # mirror of NFW.nfwAlpha (nfw.py:28-34)
        R = jnp.maximum(self._r_min, R)
        Rs = jnp.maximum(self._r_min, Rs)
        x = R / Rs
        gx = _g(x)
        a = 4 * rho0 * Rs * gx / x ** 2
        return a * ax_x, a * ax_y

    @functools.partial(jit, static_argnums=(0,))
    def _rotate(self, x, y, phi):
        cos_phi = jnp.cos(phi)
        sin_phi = jnp.sin(phi)
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi

    @functools.partial(jit, static_argnums=(0,))
    def _param_conv(self, e1, e2):
        phi = jnp.arctan2(e2, e1) / 2
        c = jnp.minimum(jnp.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
        q = (1 - c) / (1 + c)
        e = jnp.abs(1 - q ** 2) / (1 + q ** 2)
        return e, phi
