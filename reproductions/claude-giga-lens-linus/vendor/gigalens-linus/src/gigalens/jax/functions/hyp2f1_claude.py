import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, gamma, digamma   # digamma = psi
from functools import partial

@partial(jax.jit, static_argnames=("num_terms",))
def hyp2f1_series(a, b, c, z, num_terms=64):
    """Evaluate 2F1(a, b; c; z) by summing a fixed number of series terms.

    Reverse-mode differentiable (uses lax.scan with a static trip count).

    Recurrence:  t_{n+1} = t_n * (a+n)(b+n) / ((c+n)(n+1)) * z
    Converges for |z| < 1.
    """
    def step(carry, n):
        term, total = carry
        ratio = (a + n) * (b + n) / ((c + n) * (n + 1.0)) * z
        next_term = term * ratio
        return (next_term, total + next_term), None

    # match the carry shape/dtype to the broadcasted result of the recurrence
    one = jnp.ones_like(z)
    init = (one, one)
    ns = jnp.arange(num_terms)
    (_, total), _ = jax.lax.scan(step, init, ns)
    return total

@partial(jax.jit, static_argnames=("num_terms", "inner"))
def hyp2f1_goursat(a, b, z, num_terms=16, inner=hyp2f1_series):
    """2F1(a, b; a + b + 1/2; z) via the Goursat quadratic transformation (A.13)

        2F1(a, b; a + b + 1/2; z)
            = (2 / (sqrt(1 - z) + 1))^(2a)
              * inner(2a, a - b + 1/2; a + b + 1/2;
                      (sqrt(1 - z) - 1) / (sqrt(1 - z) + 1))

    Valid for the parameter relation c = a + b + 1/2.
    Accepts complex inputs.

    `inner` is the evaluator used on the mapped argument; it must have the
    signature inner(a, b, c, z, num_terms=...). Pass hyp2f1_series for the
    direct series, hyp2f1_one_minus_z for the 1-z connection formula, or any
    composition of them.
    """
    s = jnp.sqrt(1.0 - z)
    w = (s - 1.0) / (s + 1.0)
    prefactor = (2.0 / (s + 1.0)) ** (2.0 * a)
    return prefactor * inner(2.0 * a, a - b + 0.5,
                             a + b + 0.5, w, num_terms=num_terms)

@partial(jax.jit, static_argnames=("num_terms",))
def hyp2f1_one_minus_z(a, b, c, z, num_terms=16):
    """2F1(a, b; c; z) via the z -> 1 - z connection formula.

        2F1(a, b; c; z)
          = G(c)G(c-a-b)/(G(c-a)G(c-b)) * 2F1(a, b; a+b-c+1; 1-z)
          + (1-z)^(c-a-b) * G(c)G(a+b-c)/(G(a)G(b))
                          * 2F1(c-a, c-b; c-a-b+1; 1-z)

    Valid when c - a - b is not an integer.
    Assumes all gamma arguments are positive reals.
    """
    w = 1.0 - z

    log_coeff1 = (gammaln(c) + gammaln(c - a - b)
                  - gammaln(c - a) - gammaln(c - b))
    log_coeff2 = (gammaln(c) + gammaln(a + b - c)
                  - gammaln(a) - gammaln(b))

    coeff1 = jnp.exp(log_coeff1)
    coeff2 = jnp.exp(log_coeff2)

    term1 = coeff1 * hyp2f1_series(a, b, a + b - c + 1.0, w,
                                   num_terms=num_terms)
    term2 = (w ** (c - a - b)) * coeff2 * hyp2f1_series(
        c - a, c - b, c - a - b + 1.0, w, num_terms=num_terms)

    return term1 + term2

@jax.jit
def hyp2f1_half_one_threehalf(z):
    """F(1/2, 1; 3/2; z) = arctanh(sqrt(z)) / sqrt(z)
                         = (1 / (2 sqrt(z))) * ln((1 + sqrt(z)) / (1 - sqrt(z))).

    Boundary handling:
        z = 0  -> 1   (removable singularity)
        z = 1  -> inf (the log diverges)
    """
    # Shield both z=0 and z=1 from the main evaluation
    safe_z = jnp.where(z == 0.0, 1.0, jnp.where(z == 1.0, 0.5, z))
    safe_z = jnp.sqrt(safe_z)
    
    main = jnp.log((1.0 + safe_z) / (1.0 - safe_z)) / (2.0 * safe_z)

    result = jnp.where(z == 0.0, jnp.ones_like(z), main)
    result = jnp.where(z == 1.0, jnp.inf, result)
    return result

@partial(jax.jit, static_argnames=("num_terms",))
def _phi_series(a, z, num_terms=64):
    """sum_{n>=0} z^n / (n + a),  valid for |z| < 1.

    Carries a running power zn = z**n (one multiply/step) instead of
    recomputing z**n (a transcendental) every iteration.
    """
    def step(carry, n):
        total, zn = carry           # zn = z**n
        total = total + zn / (n + a)
        return (total, zn * z), None

    ns = jnp.arange(num_terms)
    init = (jnp.zeros_like(z * a), jnp.ones_like(z * a))
    (total, _), _ = jax.lax.scan(step, init, ns)
    return total


@partial(jax.jit, static_argnames=("num_terms",))
def _near_one_series(a, z, num_terms=64):
    """
    Degenerate (c-a-b=0) connection expansion of 2F1(a,1;a+1;z) in (1-z):

        a * sum_n (a)_n/n! * [psi(n+1) - psi(a+n) - log(1-z)] (1-z)^n

    Converges fast for |1-z| < 1; use near z=1.

    Everything that depended only on the loop index or on a is carried by
    recurrence, removing all per-step transcendentals:
        coeff:   (a)_n/n!          c_{n+1} = c_n * (a+n)/(n+1)
        psi_n1:  psi(n+1)          psi(n+2)   = psi(n+1)   + 1/(n+1)
        psi_an:  psi(a+n)          psi(a+n+1) = psi(a+n)   + 1/(a+n)
        wn:      (1-z)**n          wn_{n+1}   = wn * (1-z)
    """
    w = 1.0 - z
    log1mz = jnp.log(w)              # principal branch; w = 1-z

    def step(carry, n):
        total, coeff, psi_n1, psi_an, wn = carry
        bracket = psi_n1 - psi_an - log1mz
        total = total + coeff * bracket * wn
        coeff_next = coeff * (a + n) / (n + 1.0)
        psi_n1_next = psi_n1 + 1.0 / (n + 1.0)
        psi_an_next = psi_an + 1.0 / (a + n)
        wn_next = wn * w
        return (total, coeff_next, psi_n1_next, psi_an_next, wn_next), None

    ns = jnp.arange(num_terms)
    init = (
        jnp.zeros_like(z * a),                       # total
        jnp.ones_like(z * a),                        # coeff = (a)_0/0! = 1
        jnp.full_like(z * a, digamma(1.0)),          # psi(1)
        jnp.broadcast_to(digamma(a), (z * a).shape) + 0j,  # psi(a)
        jnp.ones_like(z * a),                        # wn = w**0 = 1
    )
    (total, _, _, _, _), _ = jax.lax.scan(step, init, ns)
    return a * total


@partial(jax.jit, static_argnames=("num_terms", "delta", "a_eps"))
def hyp2f1_a_1_ap1(a, z, num_terms=16, delta=0.4, a_eps=1e-9):
    """
    2F1(a, 1; a+1; z) for a in (0, 3/2), complex z.

    Three branches, each fed sanitized inputs so the unused arms stay in-domain:
      - |1 - z| < delta        : (1-z)-expansion (handles the z->1 log point)
      - |z| < 1 (else)         : Maclaurin series  a * Phi(z,1,a)
      - |z| >= 1 (else)        : 1/z connection formula
    a == 1 (within a_eps) is the integer-difference singular case; routed to the
    closed form -log(1-z)/z which is exact everywhere. a_eps is tiny so only
    true a == 1 is captured; nearby a (e.g. 0.999) falls through to the generic
    branches, which now handle a ~ 1 without the gratuitous Gamma(a-1) pole.
    """
    z = jnp.asarray(z) + 0j

    near_one = jnp.abs(1.0 - z) < delta
    use_small = (~near_one) & (jnp.abs(z) < 1.0)
    use_large = (~near_one) & (jnp.abs(z) >= 1.0)
    
    # --- sanitize inputs per branch, choosing SAFE values for each evaluator ---
    # near branch: avoid z=1 exactly (log(1-z) = -inf); use interior of the disk
    safe_near = jnp.asarray(1.0 - 0.5 * delta, dtype=z.dtype)
    z_near = jnp.where(near_one, z, safe_near)
    # small branch: 0.5 is well inside |z| < 1
    z_small = jnp.where(use_small, z, jnp.asarray(0.5 + 0j, dtype=z.dtype))
    # large branch: off the negative real axis to avoid the (-z)**(-a) branch cut
    z_large = jnp.where(use_large, z, jnp.asarray(2.0 + 0.5j, dtype=z.dtype))

    # --- branch 1: near z = 1 (degenerate connection) ---
    near = _near_one_series(a, z_near, num_terms=num_terms)

    # --- branch 2: small |z| Maclaurin ---
    small = a * _phi_series(a, z_small, num_terms=num_terms)

    # --- branch 3: large |z| (1/z connection), pole-free near a=1 ---
    neg = -z_large
    inv = 1.0 / z_large
    piece1 = gamma(a + 1.0) * gamma(1.0 - a) * neg ** (-a)
    # coeff2*(1-a) = gamma(a+1)*gamma(a-1)*(1-a)/gamma(a)**2 * neg**(-1)
    # collapses to -a/neg via gamma(a)=(a-1)gamma(a-1) and gamma(a+1)/gamma(a)=a:
    #   gamma(a-1)*(1-a) = -(a-1)gamma(a-1) = -gamma(a)
    #   => -gamma(a+1)/gamma(a) * neg**(-1) = -a * neg**(-1)
    piece2 = -a / neg * _phi_series(1.0 - a, inv, num_terms=num_terms)
    large = piece1 + piece2

    generic = jnp.where(near_one, near, jnp.where(use_small, small, large))

    # --- a == 1 closed form: 2F1(1,1;2;z) = -log(1-z)/z, exact everywhere ---
    a_is_one = jnp.abs(a - 1.0) < a_eps
    use_closed = a_is_one          # closed form is exact everywhere, even near z=1
    z_safe = jnp.where(z == 0, jnp.ones_like(z), z)
    closed_a1 = jnp.where(z == 0, jnp.ones_like(z), -jnp.log(1.0 - z) / z_safe)

    return jnp.where(use_closed, closed_a1, generic)