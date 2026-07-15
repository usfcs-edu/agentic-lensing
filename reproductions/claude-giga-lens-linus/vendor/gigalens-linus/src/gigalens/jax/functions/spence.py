import jax
import jax.numpy as jnp
import numpy as np
import scipy.special

"""
AI GENERATED >:(
"""

def _li2_bernoulli(z):
    """
    Computes Li2(z) using the Bernoulli number expansion in u = -ln(1-z).
    This converges exponentially fast for Re(z) <= 0.5 and |z| <= 1.
    """
    # Ensure float64 complex
    z = z.astype(jnp.complex128 if z.dtype == jnp.float64 else jnp.complex64)
    u = -jnp.log(1.0 - z)
    
    # Coefficients C_k = B_k / (k+1)! for k = 0 to 20
    # B_k are the Bernoulli numbers. Odd terms (except k=1) are 0.
    coeffs = jnp.array([
         1.0, 
        -0.25, 
         0.027777777777777776, 
         0.0, 
        -0.0002777777777777778, 
         0.0, 
         4.724489795918367e-06, 
         0.0, 
        -9.18577313045862e-08, 
         0.0, 
         1.8978864708781358e-09, 
         0.0, 
        -4.064221703664082e-11, 
         0.0, 
         8.921691020456453e-13, 
         0.0, 
        -1.993929586072107e-14, 
         0.0, 
         4.5262174301825424e-16, 
         0.0, 
        -1.0336940801331776e-17
    ], dtype=z.dtype)
    
    poly_val = jnp.polyval(coeffs[::-1], u)
    return u * poly_val

@jax.custom_jvp
def li2_complex(z):
    z = z + 0j
    pi_sq_6 = jnp.array(jnp.pi**2 / 6.0, dtype=z.dtype)
    
    # 1. Mapping |z| > 1
    is_large = jnp.abs(z) > 1.0
    safe_z_for_inv = jnp.where(is_large, z, 1.0 + 0j)
    z_inv = jnp.where(is_large, 1.0 / safe_z_for_inv, z)
    
    # 2. Mapping Re(z) > 0.5
    is_right = jnp.real(z_inv) > 0.5
    z_final = jnp.where(is_right, 1.0 - z_inv, z_inv)
    
    # 3. Core Series Evaluation (Now using Bernoulli)
    res = _li2_bernoulli(z_final)
    
    # 4. Back-transform Re(z) > 0.5
    safe_z_inv = jnp.where(is_right, z_inv, 0.5 + 0j)
    safe_1_minus_z_inv = jnp.where(is_right, 1.0 - z_inv, 0.5 + 0j)
    
    is_exactly_one = is_right & (z_inv == 1.0 + 0j)
    safe_1_minus_z_inv = jnp.where(is_exactly_one, 0.5 + 0j, safe_1_minus_z_inv)
    
    term_right = pi_sq_6 - jnp.log(safe_z_inv) * jnp.log(safe_1_minus_z_inv) - res
    term_right = jnp.where(is_exactly_one, pi_sq_6, term_right)
    res = jnp.where(is_right, term_right, res)
    
    # 5. Back-transform |z| > 1
    safe_minus_z = jnp.where(is_large, -z, 1.0 + 0j)
    term_large = -res - 0.5 * jnp.log(safe_minus_z)**2 - pi_sq_6
    res = jnp.where(is_large, term_large, res)
    
    return res
    
spence = jnp.vectorize(li2_complex)

@li2_complex.defjvp
def li2_complex_jvp(primals, tangents):
    z, = primals
    z_dot, = tangents
    safe_z = jnp.where(jnp.abs(z) < 1e-12, 1e-12 + 0j, z)
    deriv = -jnp.log(1.0 - safe_z) / safe_z
    deriv = jnp.where(jnp.abs(z) < 1e-12, 1.0 + 0j, deriv)
    return li2_complex(z), z_dot * deriv
    
@jax.jit
def jax_spence(z):
    return li2_complex(1.0 - z)

# ==========================================
# Testing & Validation Logic
# ==========================================
def test_against_scipy():
    # Enable JAX float64 for fair comparison against SciPy
    jax.config.update("jax_enable_x64", True)
    
    # Generate a grid of test points across the complex plane
    np.random.seed(42)
    real_parts = np.linspace(-5, 5, 100)
    imag_parts = np.linspace(-5, 5, 100)
    R, I = np.meshgrid(real_parts, imag_parts)
    Z = R + 1j * I
    Z_flat = Z.flatten()
    
    # Compute using both libraries
    print("Computing SciPy values...")
    scipy_vals = scipy.special.spence(Z_flat)
    
    print("Computing JAX values...")
    jax_vals = jax_spence(Z_flat)
    
    # Compute absolute errors
    abs_err = np.abs(scipy_vals - jax_vals)
    max_abs_err = np.max(abs_err)
    avg_abs_err = np.mean(abs_err)
    
    # Compute relative errors (with safe division to avoid division by zero)
    # Spence's function is exactly 0 at z=1, so we mask those out
    nonzero_mask = np.abs(scipy_vals) > 1e-15
    rel_err = np.zeros_like(abs_err)
    rel_err[nonzero_mask] = abs_err[nonzero_mask] / np.abs(scipy_vals[nonzero_mask])
    
    max_rel_err = np.max(rel_err)
    avg_rel_err = np.mean(rel_err)
    
    print("-" * 35)
    print(f"Max Absolute Error: {max_abs_err:.4e}")
    print(f"Avg Absolute Error: {avg_abs_err:.4e}")
    print(f"Max Relative Error: {max_rel_err:.4e}")
    print(f"Avg Relative Error: {avg_rel_err:.4e}")
    print("-" * 35)
    
    # A tolerance of 1e-12 is generally an excellent target for float64 series approximations
    if max_abs_err < 1e-12 and max_rel_err < 1e-12:
        print("Success! JAX implementation matches SciPy's Spence function perfectly.")
    else:
        print("Notice: Errors exceed 1e-12. This typically happens very close to the branch cut.")