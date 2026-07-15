from collections import namedtuple


# ``Prior`` is the lightweight (profile, params) record still consumed by
# ``gigalens.multiband``. The heavier prior-construction machinery
# (``ProfilePrior`` / ``CompoundPrior`` / ``LensPrior`` / ``make_prior_and_model``)
# was retired in Phase 6b — superseded by the scene API's derived prior
# (``gigalens.jax.scene.LensModel``); it had zero in-scope importers.
Prior = namedtuple("Prior", ["profile", "params"])
