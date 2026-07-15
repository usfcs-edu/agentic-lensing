import functools

import jax.numpy as jnp
from jax import jit

import gigalens.profile


class CombinedProfile(gigalens.profile.LightProfile):
    _name = "COMBINED_PROFILE"
    _params = []
    _amp = "Ie"

    def __init__(self,
                 profiles,
                 shared_params=None,
                 use_lstsq=False,
                 is_source=False,
                 **kwargs
                 ):
        super(CombinedProfile, self).__init__(use_lstsq=use_lstsq, is_source=is_source, **kwargs)
        if shared_params is None:
            shared_params = ["center_x", "center_y", "e1", "e2"]
        self.shared_params = shared_params
        self.profiles = profiles
        self.params = []
        self.depth = 0
        for i, profile in enumerate(self.profiles):
            self.depth += profile.depth
            for param in profile.params:
                if param not in self.shared_params:
                    self.params.append(param + f"_{i}")
                elif param not in self.params:
                    self.params.append(param)

    def _parse_params(self, params):
        parsed_params = []
        # each element has the params of each profile
        # include shared parameters + profile specific parameters without the suffix
        for i, profile in enumerate(self.profiles):
            profile_params = {}
            for param in profile.params:
                if param in self.shared_params:
                    profile_params[param] = params[param]
                else:
                    profile_params[param] = params[param + f"_{i}"]
            parsed_params.append(profile_params)
        return parsed_params

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, **params):
        params = self._parse_params(params)
        if self.use_lstsq:
            rets = jnp.zeros((0, *x.shape))
            for i, profile in enumerate(self.profiles):
                rets = jnp.concatenate((rets, profile.light(x, y, **params[i])), axis=0)
            return rets

        else:
            ret = jnp.zeros_like(x)
            for i, profile in enumerate(self.profiles):
                ret += profile.light(x, y, **params[i])
            return ret
