"""Generate tests/validation/fixtures/anchor_golden.npz from the OLD API path.

NON-RUNNABLE after the old gigalens API was dropped: this script constructs the OLD
LensSimulator / BackwardProbModel / PhysicalModel, which are now import-safe stubs that
raise on construction. To RE-RUN it you must check out the pre-deletion code (git
b82397c). This is intentional: the golden .npz is FROZEN, and
``test_regression_anchor.py`` compares the NEW scene path against the frozen golden
(scene-only), so the anchor does not need this generator at test time.

Run ONCE (Phase 6b Step 2) before deleting the old API. Freezes the OLD
LensSimulator + BackwardProbModel outputs so the rewritten anchor can assert the NEW
scene path reproduces them without importing any old code.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

# Import the anchor module's data-synthesis + free-prior builders so the golden uses
# EXACTLY the same scene/draws as the test will.
sys.path.insert(0, "/global/u1/l/linusu/gigalens/tests")
from validation import test_regression_anchor as A

clean, noisy, err = A._synthesize_data()
cfg = A._cfg()

# ---- OLD fixed-param leg (image, coeffs, log_like, red_chi2) ----
old = A._build_old(noisy, err)
old_image = np.asarray(old["image"])
old_coeffs = np.asarray(old["coeffs"])
old_log_like = float(old["log_like"])
old_red_chi2 = float(old["red_chi2"])
old_event_size = int(old["event_size"])

# ---- OLD free-param leg (per-draw log_like / log_prior at matched z) ----
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.profiles.light.shapelets import Shapelets
from gigalens.jax.physical_model import PhysicalModel
from gigalens.jax.simulator import LensSimulator
from gigalens.jax.prob_model import BackwardProbModel

# Build the NEW model just to draw z and produce matched physical params (this is what
# the frozen test will also do — only the OLD evaluation is being frozen here).
model = A._new_free_model()
old_prior = A._old_free_prior()
src = Shapelets(n_max=A.N_MAX, use_lstsq=True, is_source=True, cosmo_sample=False)
phys = PhysicalModel(lenses=[EPL(50), Shear()],
                     lens_light=[SersicEllipse(use_lstsq=True)],
                     source_light=[src])
sim_old = LensSimulator(phys, cfg, bs=1)
prob_old = BackwardProbModel(old_prior, jnp.asarray(noisy), error_map=jnp.asarray(err))

N_DRAWS = 8
key = jax.random.PRNGKey(7)
ll_old_v, lp_old_v = [], []
for _ in range(N_DRAWS):
    key, sk = jax.random.split(key)
    z_new = jax.random.normal(sk, (model.num_free_params,), dtype=jnp.float64)
    x_new = model.bijector.forward(z_new[None, :])
    params_new = model.to_params(dict(x_new))
    pl = params_new['planes']
    def _scalars(d):
        return {k: jnp.float64(np.asarray(v).squeeze()) for k, v in d.items()}
    old_x = {
        'lens_mass': {'0': _scalars(pl[0]['mass'][0]), '1': _scalars(pl[0]['mass'][1])},
        'lens_light': {'0': _scalars(pl[0]['light'][0])},
        'source_light': {'0': _scalars(pl[1]['light'][0])},
    }
    z_old = jnp.asarray(prob_old.bij.inverse(old_x))
    z_old = z_old if z_old.ndim == 2 else z_old[None, :]
    ll_old, _ = prob_old.log_like(sim_old, z_old)
    lp_old = prob_old.log_prior(z_old)
    ll_old_v.append(float(np.asarray(ll_old).squeeze()))
    lp_old_v.append(float(np.asarray(lp_old).squeeze()))

out = "/global/u1/l/linusu/gigalens/tests/validation/fixtures/anchor_golden.npz"
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(
    out,
    # fixed-param leg
    image=old_image, coeffs=old_coeffs,
    log_like=np.float64(old_log_like), red_chi2=np.float64(old_red_chi2),
    event_size=np.int64(old_event_size),
    # data the NEW path also regenerates deterministically (stored for provenance/check)
    noisy=np.asarray(noisy), err=np.asarray(err),
    # free-param leg (aligned to PRNGKey(7) draws)
    prob_seed=np.int64(7), prob_n_draws=np.int64(N_DRAWS),
    prob_log_like=np.asarray(ll_old_v, dtype=np.float64),
    prob_log_prior=np.asarray(lp_old_v, dtype=np.float64),
)
print("WROTE", out)
print("fixed: log_like=%.10e red_chi2=%.6f event_size=%d coeffs.shape=%s image.shape=%s"
      % (old_log_like, old_red_chi2, old_event_size, old_coeffs.shape, old_image.shape))
print("free log_like:", np.array2string(np.asarray(ll_old_v), precision=6))
print("free log_prior:", np.array2string(np.asarray(lp_old_v), precision=6))
