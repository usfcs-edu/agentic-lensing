import functools

import jax.numpy as jnp
from jax import jit

import gigalens.profile
from gigalens.jax.profiles.light import combined_profile
from gigalens.physicality import Domain, ellipticity_constraint

_SERSIC_BASE_DOMAINS = {
    "R_sersic": Domain(lo=0.0, lo_open=True, rationale=(
        "ratio = R/R_sersic feeds the NaN-safety mask (sersic.py light): "
        "R_sersic=0 renders identically zero, and R_sersic<0 silently zeroes "
        "every off-center pixel, collapsing the profile to its central spike "
        "with an unchanged peak value (verified 2026-07-10).")),
    "n_sersic": Domain(lo=0.0, lo_open=True, rationale=(
        "bn = exp(0.6950 + log(n_sersic) - 0.1789/n_sersic) (sersic.py): "
        "n<0 -> NaN via log; n=0 -> bn=0 silently renders a constant-1 sheet "
        "(both verified 2026-07-10).")),
    "center_x": Domain(rationale="any finite position is valid."),
    "center_y": Domain(rationale="any finite position is valid."),
    "Ie": Domain(lo=0.0, rationale=(
        "negative peak surface brightness is unphysical (definition-based, "
        "flagged for human review; lstsq-solved amplitudes never enter "
        "profile.params and are exempt).")),
}


class Sersic(gigalens.profile.LightProfile):
    _name = "SERSIC"
    _params = ["R_sersic", "n_sersic", "center_x", "center_y"]
    _amp = "Ie"
    _domains = _SERSIC_BASE_DOMAINS

    def __init__(self, use_lstsq=False, is_source=False, **kwargs):
        super(Sersic, self).__init__(use_lstsq=use_lstsq, is_source=is_source, **kwargs)

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, R_sersic, n_sersic, center_x, center_y, Ie=None):
        R = self.distance(x, y, center_x, center_y)
        bn = jnp.exp(0.6950 + jnp.log(n_sersic) - 0.1789/n_sersic)

        ratio = R/R_sersic

        #* Zero out brighnesses anywhere exponent_factor gets too large
        #* If it is too large, brightness will just be 0, but
        #* it can cause 0*NaN problems in the gradient calculation
        is_safe = (1.0 / n_sersic) * jnp.log10(ratio + 1e-10) < 30
        safe_ratio = jnp.where(is_safe, ratio, 0.0)

        exponent_factor = bn * (safe_ratio ** (1 / n_sersic) - 1.0)
        ret = jnp.exp(-exponent_factor)

        ret = jnp.where(is_safe, ret, 0.0)

        return ret[jnp.newaxis, ...] if self.use_lstsq else (Ie * ret)

    @functools.partial(jit, static_argnums=(0,))
    def distance(self, x, y, cx, cy, e1=None, e2=None):
        if e1 is None:
            e1 = jnp.zeros_like(cx)
        if e2 is None:
            e2 = jnp.zeros_like(cx)
        phi = jnp.arctan2(e2, e1) / 2
        c = jnp.sqrt(e1 ** 2 + e2 ** 2)
        q = (1 - c) / (1 + c)
        dx, dy = x - cx, y - cy
        cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
        xt1 = (cos_phi * dx + sin_phi * dy) * jnp.sqrt(q)
        xt2 = (-sin_phi * dx + cos_phi * dy) / jnp.sqrt(q)
        return jnp.sqrt(xt1 ** 2 + xt2 ** 2)


class SersicEllipse(Sersic):
    _name = "SERSIC_ELLIPSE"
    _params = ["R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"]
    _domains = {
        **_SERSIC_BASE_DOMAINS,
        "e1": Domain(rationale="bounded jointly with e2 (see _joint_constraints)."),
        "e2": Domain(rationale="bounded jointly with e1 (see _joint_constraints)."),
    }
    _joint_constraints = (
        ellipticity_constraint(1.0, rationale=(
            "distance() computes q=(1-c)/(1+c) with NO clip on c=sqrt(e1^2+e2^2) "
            "(sersic.py): |e| >= 1 makes q <= 0, sqrt(q) NaN/0, and the light "
            "silently renders identically zero (verified 2026-07-10).")),
    )

    def __init__(self, use_lstsq=False, is_source=False, **kwargs):
        super(SersicEllipse, self).__init__(use_lstsq=use_lstsq, is_source=is_source, **kwargs)

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, R_sersic, n_sersic, e1, e2, center_x, center_y, Ie=None):
        R = self.distance(x, y, center_x, center_y, e1, e2)
        bn = jnp.exp(0.6950 + jnp.log(n_sersic) - 0.1789/n_sersic)

        ratio = R/R_sersic

        #* Zero out brighnesses anywhere exponent_factor gets too large
        #* If it is too large, brightness will just be 0, but
        #* it can cause 0*NaN problems in the gradient calculation
        is_safe = (1.0 / n_sersic) * jnp.log10(ratio + 1e-10) < 30
        safe_ratio = jnp.where(is_safe, ratio, 0.0)

        exponent_factor = bn * (safe_ratio ** (1 / n_sersic) - 1.0)
        ret = jnp.exp(-exponent_factor)

        ret = jnp.where(is_safe, ret, 0.0)

        return ret[jnp.newaxis, ...] if self.use_lstsq else (Ie * ret)


class CoreSersic(Sersic):
    _name = "CORE_SERSIC"
    _params = [
        "R_sersic",
        "n_sersic",
        "Rb",
        "alpha",
        "gamma",
        "e1",
        "e2",
        "center_x",
        "center_y",
    ]

    def __init__(self, use_lstsq=False, is_source=False, **kwargs):
        super(CoreSersic, self).__init__(use_lstsq=use_lstsq, is_source=is_source, **kwargs)

    @functools.partial(jit, static_argnums=(0,))
    def light(
            self,
            x,
            y,
            R_sersic,
            n_sersic,
            Rb,
            alpha,
            gamma,
            e1,
            e2,
            center_x,
            center_y,
            Ie=None,
    ):
        R = self.distance(x, y, center_x, center_y, e1, e2)
        bn = jnp.exp(0.6950 + jnp.log(n_sersic) - 0.1789/n_sersic)
        ret = ((1 + (Rb / R) ** alpha) ** (gamma / alpha) * jnp.exp(-bn * (
                (R ** alpha + Rb ** alpha)
                / R_sersic ** alpha ** 1.0
                / (alpha * n_sersic)
        ) - 1.0))
        return ret[jnp.newaxis, ...] if self.use_lstsq else (Ie * ret)


class DoubleSersic(combined_profile.CombinedProfile):

    _name = "DOUBLE_SERSIC"
    _params = []
    _amp = "Ie"

    def __init__(self, use_lstsq=False, is_source=False, **kwargs):
        super(DoubleSersic, self).__init__(
            profiles=[SersicEllipse(use_lstsq=use_lstsq, is_source=is_source),
                      SersicEllipse(use_lstsq=use_lstsq, is_source=is_source)],
            use_lstsq=use_lstsq,
            is_source=is_source,
            **kwargs
        )
