import functools

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

"""
AI GENERATED >:(
"""

def hyp2f1_series(a, b, c, z, niter=100):
    """Standard Maclaurin series for 2F1."""
    def body_fun(i, carry):
        n, term, total = carry
        next_term = term * ((a + n) * (b + n)) / ((c + n) * (n + 1.0)) * z
        next_total = total + next_term
        return (n + 1.0, next_term, next_total)
    
    # Initial state: n=0, term=1.0, total=1.0
    init_carry = (0.0, jnp.ones_like(z), jnp.ones_like(z))
    _, _, final_total = jax.lax.fori_loop(0, niter, body_fun, init_carry)
    return final_total

def hyp2f1_near_one(a, b, c, z, niter=100):
    """Analytic continuation for points physically near z = 1."""
    s_raw = c - a - b
    
    # --- The Scaled Perturbation Fix ---
    # Broad integer detection for float32 fuzziness
    is_integer = jnp.isclose(s_raw, jnp.round(s_raw), atol=1e-4)
    
    # Dynamically scale eps by magnitude of c to prevent float absorption
    eps = 1e-4 * jnp.maximum(1.0, jnp.abs(c))
    c_safe = jnp.where(is_integer, c + eps, c)
    
    s = c_safe - a - b
    one_minus_z = 1.0 - z
    
    # Compute the two hypergeometric components
    f1 = hyp2f1_series(a, b, a + b - c_safe + 1.0, one_minus_z, niter=niter)
    f2 = hyp2f1_series(c_safe - a, c_safe - b, c_safe - a - b + 1.0, one_minus_z, niter=niter)
    
    # Compute Coefficients with safe Gamma functions
    log_coeff1 = (jsp.gammaln(c_safe) - jsp.gammaln(c_safe - a) 
                  - jsp.gammaln(c_safe - b) - jsp.gammaln(a + b - c_safe + 1.0))
    sign1 = (jsp.gammasgn(c_safe) * jsp.gammasgn(c_safe - a) * jsp.gammasgn(c_safe - b) * jsp.gammasgn(a + b - c_safe + 1.0))
    coeff1 = sign1 * jnp.exp(log_coeff1)
    
    log_coeff2 = (jsp.gammaln(c_safe) - jsp.gammaln(a) 
                  - jsp.gammaln(b) - jsp.gammaln(c_safe - a - b + 1.0))
    sign2 = (jsp.gammasgn(c_safe) * jsp.gammasgn(a) * jsp.gammasgn(b) * jsp.gammasgn(c_safe - a - b + 1.0))
    coeff2 = sign2 * jnp.exp(log_coeff2) * (one_minus_z ** s)
    
    rhs = (coeff1 * f1) - (coeff2 * f2)
    return rhs * jnp.pi / jnp.sin(jnp.pi * s)

def hyp2f1_large_z(a, b, c, z, niter=100):
    """Analytic continuation for large |z| using the 1/z transformation."""
    # --- The Scaled Perturbation Fix for a - b ---
    ab_diff = a - b
    is_integer = jnp.isclose(ab_diff, jnp.round(ab_diff), atol=1e-4)
    
    # Dynamically scale eps to prevent float absorption
    eps = 1e-4 * jnp.maximum(1.0, jnp.abs(a))
    a_safe = jnp.where(is_integer, a + eps, a)
    
    inv_z = 1.0 / z
    minus_z = -z
    
    # Compute the two hypergeometric components for 1/z
    f1 = hyp2f1_series(a_safe, a_safe - c + 1.0, a_safe - b + 1.0, inv_z, niter=niter)
    f2 = hyp2f1_series(b, b - c + 1.0, b - a_safe + 1.0, inv_z, niter=niter)
    
    # Coeff 1
    log_coeff1 = (jsp.gammaln(c) + jsp.gammaln(b - a_safe) 
                  - jsp.gammaln(b) - jsp.gammaln(c - a_safe))
    sign1 = (jsp.gammasgn(c) * jsp.gammasgn(b - a_safe) * jsp.gammasgn(b) * jsp.gammasgn(c - a_safe))
    coeff1 = sign1 * jnp.exp(log_coeff1) * (minus_z ** (-a_safe))
    
    # Coeff 2
    log_coeff2 = (jsp.gammaln(c) + jsp.gammaln(a_safe - b) 
                  - jsp.gammaln(a_safe) - jsp.gammaln(c - b))
    sign2 = (jsp.gammasgn(c) * jsp.gammasgn(a_safe - b) * jsp.gammasgn(a_safe) * jsp.gammasgn(c - b))
    coeff2 = sign2 * jnp.exp(log_coeff2) * (minus_z ** (-b))
    
    return (coeff1 * f1) + (coeff2 * f2)

@functools.partial(jax.jit, static_argnames=['niter'])
def hyp2f1_complex(a, b, c, z, niter=100):
    """Main router for the Gauss Hypergeometric Function."""
    z = z + 0j
    
    # 1. Termination Check
    is_a_neg_int = jnp.logical_and(jnp.real(a) <= 0.0, jnp.isclose(a, jnp.round(jnp.real(a)), atol=1e-7))
    is_b_neg_int = jnp.logical_and(jnp.real(b) <= 0.0, jnp.isclose(b, jnp.round(jnp.real(b)), atol=1e-7))
    is_terminating = jnp.logical_or(is_a_neg_int, is_b_neg_int)
    
    # 2. Distance Calculations
    mag_z = jnp.abs(z)
    mag_1_minus_z = jnp.abs(1.0 - z)
    
    # 3. Compute all mathematical branches
    res_series = hyp2f1_series(a, b, c, z, niter=niter)
    res_near_1 = hyp2f1_near_one(a, b, c, z, niter=niter)
    res_large_z = hyp2f1_large_z(a, b, c, z, niter=niter)
    
    # Pfaff Transformation
    res_pfaff = (1.0 - z)**(-a) * hyp2f1_series(a, c - b, c, z / (z - 1.0), niter=niter)
    
    # 4. Route the results based on geometric boundaries
    conditions = [
        is_terminating,            # Priority 1: Exact polynomials
        mag_z <= 0.9,              # Priority 2: Safe Standard Unit Circle
        mag_1_minus_z <= 0.9,      # Priority 3: Near 1 Singularity
        mag_z > 1.1,               # Priority 4: Large z outside unit circle
    ]
    
    choices = [
        res_series, 
        res_series, 
        res_near_1, 
        res_large_z,
    ]
    
    # Fall back to Pfaff for the awkward "ring" around |z| ~ 1
    return jnp.select(conditions, choices, default=res_pfaff)

hyp2f1 = jnp.vectorize(hyp2f1_complex, excluded=(4,))

"""
tests
"""

import numpy as np
import jax
import jax.numpy as jnp
import scipy.special as sp

def run_scipy_baseline(a, b, c, z_array):
    """Evaluates the baseline using SciPy's Fortran implementation."""
    # SciPy works with standard numpy arrays
    z_np = np.asarray(z_array)
    return sp.hyp2f1(a, b, c, z_np)

def test_against_scipy(a, b, c):
    print(f"Testing parameters: a={a}, b={b}, c={c}")
    print("-" * 50)
    
    # 1. Generate a diverse set of test points across the complex plane
    test_points = [
        # Region 1: Inside unit circle (|z| <= 0.9)
        0.5 + 0.1j, 
        -0.5 - 0.5j,
        
        # Region 2: Near 1 (|1-z| <= 0.9)
        0.95 + 0.05j,
        1.2 - 0.1j,
        
        # Region 3: Negative half-plane (Re(z) < 0)
        -5.0 + 2.0j,
        -0.1 + 5.0j,
        
        # Region 4: Outside unit circle (|z| > 1, Re(z) > 0)
        2.0 + 3.0j,
        5.0 - 0.5j,
        
        # The "Bermuda Triangle" boundary points
        0.5 + 0.866j, 
        0.5 - 0.866j
    ]
    
    z_array = jnp.array(test_points)
    
    # 2. Compute results
    jax_results = hyp2f1(a, b, c, z_array)
    scipy_results = run_scipy_baseline(a, b, c, z_array)
    
    # 3. Calculate Errors
    # Absolute error: |JAX - SciPy|
    abs_errors = np.abs(jax_results - scipy_results)
    
    # Relative error: |JAX - SciPy| / |SciPy|
    # Add a tiny epsilon to prevent division by zero
    rel_errors = abs_errors / (np.abs(scipy_results) + 1e-15)
    
    # 4. Report findings
    for i, z in enumerate(test_points):
        print(f"z = {z:12}")
        print(f"  JAX:   {jax_results[i]}")
        print(f"  SciPy: {scipy_results[i]}")
        print(f"  Abs Error: {abs_errors[i]:.2e}")
        print(f"  Rel Error: {rel_errors[i]:.2e}\n")
        
    print("-" * 50)
    print(f"Max Absolute Error: {np.max(abs_errors):.2e}")
    print(f"Max Relative Error: {np.max(rel_errors):.2e}")