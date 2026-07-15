"""Experimental / in-development JAX code for GIGA-Lens.

This namespace collects sampler and inference implementations that are not yet
part of the stable public API, including the validated MCLMC and MAMS samplers
(:mod:`gigalens.jax.experimental.mclmc`, :mod:`gigalens.jax.experimental.mams`)
and their shared BlackJAX helpers (:mod:`.blackjax_updated_utils`).

Submodules are intentionally *not* imported here: several of them pull in heavy
optional dependencies (BlackJAX), so importing this package stays cheap. Import
the specific submodule you need, e.g.::

    from gigalens.jax.experimental.mclmc import MCLMC_JIT
    from gigalens.jax.experimental.mams import MAMS_JIT
"""
