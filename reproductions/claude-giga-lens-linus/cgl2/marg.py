"""Pure array-level ridge-marginalization math. No gigalens imports.

COPIED WITH ATTRIBUTION from
/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens/cgl/marg.py
(claude-giga-lens campaign; UNCHANGED — imports only), itself the
per-log-prob-eval core of foundry-i
``_hmc_lib_marg.build_model_marg._logpost``
(/raid/benson/git/agentic-lensing/reproductions/foundry-i/_hmc_lib_marg.py),
factored to an array-only function so both the diagonal and the correlated
(conv-whitened) likelihoods share one audited implementation.

Math (all smooth, never pinv):
    b       = Xw^T Rw
    A       = Xw^T Xw + diag(Lambda)          (PD for Lambda > 0)
    chol    = cholesky(A)
    a*      = cho_solve((chol, lower=True), b)
    logdetA = 2 sum(log(diag(chol)))
    logL    = -1/2 ||Rw||^2 + 1/2 b . a* - 1/2 logdetA

where Xw is the WHITENED design matrix (n_pix, k), Rw the WHITENED residual
(n_pix,), and Lambda the Tikhonov / Gaussian-prior precision of the k
marginalized amplitudes. The -1/2 logdetA is the Gaussian-evidence / Occam
term gigalens's lstsq path omits.

float64-safe and jit-friendly: pure jax.numpy, dtype follows the inputs, no
python branching on traced values.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def marg_loglik(Xw, Rw, Lambda_diag):
    """Ridge-marginalized Gaussian log-likelihood (data part, consts dropped).

    Args:
        Xw: (n_pix, k) whitened design matrix (each column already passed
            through the SAME whitening operator as the residual).
        Rw: (n_pix,) whitened residual.
        Lambda_diag: (k,) positive ridge precision Lambda_ii (for the
            foundry-i shapelet priors Normal(0, 5/sqrt(i+1)): (i+1)/25).

    Returns:
        (logL_data_part, a_star, logdetA):
            logL_data_part: scalar -1/2||Rw||^2 + 1/2 b.a* - 1/2 logdetA
            a_star: (k,) marginal-mode amplitudes (Cholesky solve, never pinv)
            logdetA: scalar log det(Xw^T Xw + diag(Lambda))
    """
    b = Xw.T @ Rw
    A = Xw.T @ Xw + jnp.diag(Lambda_diag)
    chol = jnp.linalg.cholesky(A)
    a_star = jax.scipy.linalg.cho_solve((chol, True), b)
    logdetA = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    logL_data_part = (
        -0.5 * jnp.sum(Rw ** 2) + 0.5 * jnp.dot(b, a_star) - 0.5 * logdetA
    )
    return logL_data_part, a_star, logdetA
