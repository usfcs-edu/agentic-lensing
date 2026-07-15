import functools

import jax.numpy as jnp
from jax import jit

import gigalens.profile


class CombinedProfile(gigalens.profile.MassProfile):
    _name = "COMBINED_PROFILE"
    _params = []

    def __init__(self,
                 profiles,
                 shared_params=None,
                 ):
        super(CombinedProfile, self).__init__()
        if shared_params is None:
            shared_params = ["center_x", "center_y", "e1", "e2"]
        self.shared_params = shared_params
        self.profiles = profiles
        self.params = []
        for i, profile in enumerate(self.profiles):
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
    def deriv(self, x, y, **params):
        params = self._parse_params(params)
        f_x, f_y = jnp.zeros_like(x), jnp.zeros_like(y)
        for i, lens in enumerate(self.profiles):
            f_xi, f_yi = lens.deriv(x, y, **params[i])
            f_x += f_xi
            f_y += f_yi
        return f_x, f_y