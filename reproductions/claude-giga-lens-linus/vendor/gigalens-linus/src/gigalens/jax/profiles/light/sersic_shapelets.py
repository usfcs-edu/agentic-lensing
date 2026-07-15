import functools

import jax.numpy as jnp
from jax import jit

import gigalens.profile
from gigalens.jax.profiles.light import sersic, shapelets
# from gigalens.jax.profiles.light import combined_profile


class SersicShapelets(gigalens.profile.LightProfile):
    _name = "SERSIC_SHAPELETS"
    _params = []
    _amp = "Ie"

    def __init__(self, n_max, use_lstsq=False, is_source=False, interpolate=False, **kwargs):
        super(SersicShapelets, self).__init__(use_lstsq=use_lstsq, is_source=is_source, **kwargs)
        self.sersic = sersic.SersicEllipse(use_lstsq=use_lstsq, is_source=is_source)
        self.shapelets = shapelets.Shapelets(n_max, use_lstsq=use_lstsq, is_source=is_source, interpolate=interpolate)
        self.shared_params = ["center_x", "center_y", "e1", "e2"]
        self.params = []
        self.depth = self.shapelets.depth + self.sersic.depth
        for param in self.sersic.params + self.shapelets.params:
            if param not in self.params:
                self.params.append(param)

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, **params):
        # Forward to each sub-profile only the params it was actually given.
        # ``sersic.params`` / ``shapelets.params`` also list the lstsq amplitude
        # tokens ("amp", "amp00", ...), which the simulator solves separately and
        # never passes in ``params``; selecting ``if p in params`` drops them
        # (and is robust to the shared-class ``_params`` list picking up extra
        # "amp" entries across profile instantiations).
        sersic_kwargs = {p: params[p] for p in self.sersic.params if p in params}
        shapelets_kwargs = {p: params[p] for p in self.shapelets.params if p in params}
        if self.use_lstsq:
            ret = jnp.zeros((0, *x.shape))
            ret = jnp.concatenate((ret, self.sersic.light(x, y, **sersic_kwargs)), axis=0)
            ret = jnp.concatenate((ret, self.shapelets.light(x, y, **shapelets_kwargs)), axis=0)
            return ret
        else:
            ret = jnp.zeros_like(x)
            ret += self.sersic.light(x, y, **sersic_kwargs)
            ret += self.shapelets.light(x, y, **shapelets_kwargs)
            return ret


# TODO: use combined profiles for SersicShapelets
# class SersicShapelets(combined_profile.CombinedProfile):
#     _name = "SERSIC_SHAPELETS"
#     _params = []
#     _amp = "Ie"
#
#     def __init__(self, n_max, use_lstsq=False, is_source=False, interpolate=True):
#         super(SersicShapelets, self).__init__(
#             profiles=[sersic.SersicEllipse(use_lstsq=use_lstsq, is_source=is_source),
#                       shapelets.Shapelets(n_max, use_lstsq=use_lstsq, is_source=is_source, interpolate=interpolate)],
#             use_lstsq=use_lstsq,
#             is_source=is_source
#         )
#         self.n_max = n_max
