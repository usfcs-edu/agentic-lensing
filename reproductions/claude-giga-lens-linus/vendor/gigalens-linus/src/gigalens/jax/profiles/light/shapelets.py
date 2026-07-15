import functools
import warnings

import jax.numpy as jnp
from jax import jit

import gigalens.physicality
import gigalens.profile

_PI_NEG_QUARTER = float(jnp.pi) ** -0.25


def _phi_basis_1d(x, n_max):
    """Compute the orthonormal Hermite (shapelet) basis ``phi_n(x)`` for
    ``n = 0, ..., n_max`` using the numerically stable three-term recurrence
    on the *normalized* functions:

        phi_0(x) = pi^(-1/4) * exp(-x^2/2)
        phi_1(x) = sqrt(2) * x * phi_0(x)
        phi_n(x) = sqrt(2/n) * x * phi_{n-1}(x) - sqrt((n-1)/n) * phi_{n-2}(x)

    Unlike the textbook approach (compute H_n(x) via its recurrence, then
    multiply by the tiny prefactor 1/sqrt(2^n * n! * sqrt(pi)) and the
    Gaussian envelope exp(-x^2/2)), this recurrence keeps every intermediate
    value bounded uniformly in n and x (|phi_n(x)| <= O(1)), so it does not
    overflow in float32 for any n_max or x, and its VJP is finite everywhere.

    The orders are accumulated in a Python list and ``jnp.stack``-ed (lowering
    to a concatenate) rather than written into a preallocated buffer with
    ``.at[n].set``: the buffer form lowered to scatter fusions that the
    2026-07-09 kernel traces measured at ~13-18% of the whole likelihood
    gradient (GIGALens-Code docs/logs/compute-profiling.md, C-14). Same
    recurrence, same op order per element — values are unchanged.
    """
    phi = [_PI_NEG_QUARTER * jnp.exp(-x ** 2 / 2)]
    if n_max >= 1:
        phi.append(jnp.sqrt(2.0) * x * phi[0])
    for n in range(2, n_max + 1):
        phi.append(
            jnp.sqrt(2.0 / n) * x * phi[n - 1]
            - jnp.sqrt((n - 1) / n) * phi[n - 2]
        )
    return jnp.stack(phi)


def _phi_basis_1d_buffer(x, n_max):
    """Pre-2026-07-09 buffer/scatter implementation of ``_phi_basis_1d``.

    Retained as the equivalence-gate reference (wip/validate_fold_stack.py)."""
    phi = jnp.empty((n_max + 1, *x.shape), dtype=x.dtype)
    phi = phi.at[0].set(_PI_NEG_QUARTER * jnp.exp(-x ** 2 / 2))
    if n_max >= 1:
        phi = phi.at[1].set(jnp.sqrt(2.0) * x * phi[0])
    for n in range(2, n_max + 1):
        phi = phi.at[n].set(
            jnp.sqrt(2.0 / n) * x * phi[n - 1]
            - jnp.sqrt((n - 1) / n) * phi[n - 2]
        )
    return phi


class Shapelets(gigalens.profile.LightProfile):
    _name = "SHAPELETS"
    _params = ["beta", "center_x", "center_y"]
    _domains = {
        "beta": gigalens.physicality.Domain(lo=0.0, lo_open=True, rationale=(
            "coordinates divide by beta (shapelets.py light): beta=0 -> NaN "
            "(verified 2026-07-10); beta<0 silently renders the parity-mirrored "
            "source (odd modes flip sign, verified).")),
        "center_x": gigalens.physicality.Domain(
            rationale="any finite position is valid."),
        "center_y": gigalens.physicality.Domain(
            rationale="any finite position is valid."),
    }
    # ampNN parameters are generated dynamically (n_max-dependent), so they
    # cannot be enumerated in the class dict: the fallback covers them.
    _domain_fallback = gigalens.physicality.Domain(rationale=(
        "shapelet amplitudes are sign-indefinite basis coefficients by design "
        "— unbounded."))

    def __init__(self, n_max, use_lstsq=False, is_source=False, interpolate=False, **kwargs):
        super(Shapelets, self).__init__(use_lstsq=use_lstsq, is_source=is_source, **kwargs)
        # NOTE: `interpolate` is deprecated and ignored. Shapelet interpolation was
        # removed because the interpolated basis produced inaccurate light and
        # sampling instabilities in some regimes; the analytic recurrence
        # (`_phi_basis_1d`) is now always used. The argument is retained only so
        # existing call sites do not break.
        if interpolate:
            warnings.warn(
                "From Linus: interpolate=True has been removed from shapelets, since it "
                "causes issues in sampling and can generate unphysical sources. My bad if "
                "there's a use case for it. Falling back on interpolate=False behavior."
            )
        if not self.use_lstsq:
            del self.params[self.params.index(self._amp)]
        self.n_layers = int((n_max + 1) * (n_max + 2) / 2)
        self.n_max = n_max
        n1 = 0
        n2 = 0
        N1 = []
        N2 = []
        decimal_places = len(str(self.n_layers))
        self._amp_names = []
        for i in range(self.n_layers):
            if not self.use_lstsq:
                self.params.append(f"amp{str(i).zfill(decimal_places)}")
            self._amp_names.append(f"amp{str(i).zfill(decimal_places)}")
            N1.append(n1)
            N2.append(n2)
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        self.N1 = jnp.array(N1)
        self.N2 = jnp.array(N2)
        self.depth = len(self._amp_names)

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, center_x, center_y, beta, **amp):
        x = (x - center_x) / beta
        y = (y - center_y) / beta
        XX = _phi_basis_1d(x, self.n_max)
        YY = _phi_basis_1d(y, self.n_max)
        if self.use_lstsq:
            return XX[self.N1, ...] * YY[self.N2, ...]
        else:
            amps = jnp.stack([amp[k] for k in self._amp_names], axis=0)
            basis = XX[self.N1, ...] * YY[self.N2, ...]
            if jnp.ndim(amps) == 1:
                # Fixed (unbatched) amplitudes -- e.g. rendering a known/truth source. The
                # sampled path below contracts the shapelet order i against a shared trailing
                # sample axis j; a scalar amplitude has no j, so sum over i and let it
                # broadcast over the basis' trailing (sample) axis. The batched path is
                # unchanged (byte-identical).
                return jnp.einsum('i,i...->...', amps, basis)
            return jnp.einsum('ij,i...j->...j', amps, basis)
