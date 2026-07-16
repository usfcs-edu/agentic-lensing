"""P2 deployment-scout evidence E1 (research/p2_deployment_plan.md): vendor-hidden import test.
and run a tiny CPU SMC on an inline conjugate Gaussian (no cgl2.zoo -- zoo imports gigalens
inside builders only, but we avoid it entirely to prove the samplers subtree is vendor-free)."""
import sys, importlib.abc

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "gigalens" or name.startswith("gigalens."):
            raise ImportError(f"BLOCKED (vendor hidden test): {name}")
        return None

sys.meta_path.insert(0, Blocker())

import cgl2.samplers.smc_micro as sm
import cgl2.samplers.common as common
print("IMPORT OK:", sm.__file__)
assert "gigalens" not in sys.modules, [m for m in sys.modules if "gigalens" in m]
print("gigalens NOT in sys.modules: True")

# functional check: tiny MAMS SMC on a 2-D conjugate Gaussian, CPU
import numpy as np, jax, jax.numpy as jnp
mu0 = np.array([0.5, -0.3]); s2, sp2 = 1.0, 4.0
c_like = -0.5*2*(np.log(2*np.pi)+np.log(s2))
logZ_true = float(np.sum(-0.5*(np.log(2*np.pi)+np.log(s2+sp2)) - 0.5*mu0**2/(s2+sp2)))
logprior = lambda z: -0.5*2*(jnp.log(2*jnp.pi)+jnp.log(sp2)) - 0.5/sp2*jnp.sum(z**2)
loglik = lambda z: c_like - 0.5/s2*jnp.sum((z-jnp.asarray(mu0))**2)
z0 = 2.0*np.asarray(jax.random.normal(jax.random.PRNGKey(0),(256,2)))
out = sm.run_smc(logprior, loglik, z0, kernel="mams", seed=0, n_boot=50)
print(f"FUNCTIONAL OK: logZ={out['logZ']:.4f} true={logZ_true:.4f} err={abs(out['logZ']-logZ_true):.4f} sig={out['logZ_boot_sigma']:.4f} stages={out['n_stages']}")
assert "gigalens" not in sys.modules
print("VENDOR-FREE TEST PASS")
