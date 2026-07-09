"""CPU unit tests for the P2b sampler-adapter machinery (no GPU, no
benchmark statistics): registry completeness, frozen-policy hash assertion
logic, the REMC tempering ladder + per-replica target correctness vs a
hand-built beta-tempered logp, and flow-pullback logp correctness on a 2-D
flow against numerical Jacobians.
"""
import numpy as np
import pytest

from cgl.samplers import _ADAPTERS, get_scale_keys, list_samplers
from cgl.samplers import common

EXPECTED_SAMPLERS = {
    "s0_baseline", "bj_nuts", "bj_mclmc", "flowmc_runner", "bj_smc",
    "nautilus_runner", "remc_pt", "neutra", "glnt",
}


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_completeness():
    assert set(list_samplers()) == EXPECTED_SAMPLERS
    for name in EXPECTED_SAMPLERS:
        module, fn = _ADAPTERS[name]
        assert fn == "run_cell"
        # every adapter has Track-B scale keys defined
        assert isinstance(get_scale_keys(name), tuple)


def test_registry_adapters_importable():
    # import-light check on the two pure-python ones (full imports of the
    # jax-heavy adapters happen lazily in GPU processes)
    from cgl.samplers import get_run_cell

    for name in ("bj_nuts", "remc_pt", "glnt"):
        assert callable(get_run_cell(name))


# --------------------------------------------------------------------------- #
# frozen-policy hash assertion logic
# --------------------------------------------------------------------------- #
def _fake_policies():
    cfg = {"eps0": 0.3, "mode": "chees"}
    bud = {"n_chains": 16, "n_keep": 800}
    return {
        "methods": {
            "bj_nuts": {
                "T0": {
                    "config": cfg, "budget": bud,
                    "sha_policy": common.policy_sha(cfg, bud),
                    "sha_config": common.policy_sha(cfg, None),
                },
            },
        },
    }, cfg, bud


def test_frozen_policy_accepts_exact_match():
    policies, cfg, bud = _fake_policies()
    run_budget = {**bud, "track": "A", "n_grad_budget": 200000}
    entry = common.assert_frozen_policy(policies, "bj_nuts", "T0",
                                        dict(cfg), run_budget, "A")
    assert entry["sha_policy"] == common.policy_sha(cfg, bud)


def test_frozen_policy_rejects_config_drift():
    policies, cfg, bud = _fake_policies()
    bad = {**cfg, "eps0": 0.31}
    with pytest.raises(RuntimeError, match="FROZEN-POLICY VIOLATION"):
        common.assert_frozen_policy(policies, "bj_nuts", "T0", bad,
                                    {**bud, "track": "A"}, "A")


def test_frozen_policy_rejects_budget_drift_track_a():
    policies, cfg, bud = _fake_policies()
    bad = {**bud, "n_keep": 4 * bud["n_keep"], "track": "A"}
    with pytest.raises(RuntimeError, match="FROZEN-POLICY VIOLATION"):
        common.assert_frozen_policy(policies, "bj_nuts", "T0", dict(cfg),
                                    bad, "A")


def test_frozen_policy_track_b_scales_budget_but_pins_config():
    policies, cfg, bud = _fake_policies()
    scaled = {**bud, "n_keep": 4 * bud["n_keep"], "track": "B"}
    common.assert_frozen_policy(policies, "bj_nuts", "T0", dict(cfg),
                                scaled, "B")            # budget free on B
    with pytest.raises(RuntimeError, match="FROZEN-POLICY VIOLATION"):
        common.assert_frozen_policy(policies, "bj_nuts", "T0",
                                    {**cfg, "eps0": 1.0}, scaled, "B")


def test_missing_policy_raises():
    policies, _, _ = _fake_policies()
    with pytest.raises(KeyError):
        common.frozen_policy_for(policies, "bj_nuts", "T1")
    with pytest.raises(KeyError):
        common.frozen_policy_for(policies, "glnt", "T0")


# --------------------------------------------------------------------------- #
# REMC ladder + per-replica tempered target
# --------------------------------------------------------------------------- #
def test_geometric_ladder():
    from cgl.samplers.remc_pt import geometric_ladder

    betas = geometric_ladder(6, 0.01)
    assert betas.shape == (6,)
    assert betas[0] == 1.0
    assert np.isclose(betas[-1], 0.01)
    ratios = betas[1:] / betas[:-1]
    assert np.allclose(ratios, ratios[0]), "ladder must be geometric"
    assert np.all(np.diff(betas) < 0), "ladder must be decreasing"
    with pytest.raises(AssertionError):
        geometric_ladder(1, 0.5)
    with pytest.raises(AssertionError):
        geometric_ladder(4, 1.5)


def test_remc_per_replica_target_vs_handbuilt():
    """composed(Z) must equal log_prior(Z) + beta_r * log_like(Z) with the
    replica dim broadcast, on replica-stacked (R, C, dim) states."""
    from cgl.samplers.remc_pt import geometric_ladder, make_replica_logp
    from cgl.zoo import get_target

    target = get_target("t0_mix2")           # CPU-safe synthetic target
    R, C = 4, 5
    betas = geometric_ladder(R, 0.05)
    tempered, untempered, composed = make_replica_logp(target, betas)

    rng = np.random.default_rng(7)
    Z = rng.normal(scale=2.0, size=(R, C, target.dim)).astype(np.float32)

    got = np.asarray(composed(Z), dtype=np.float64)
    assert got.shape == (R, C)

    # hand-built reference through the zoo batch surface, replica by replica
    want = np.empty((R, C))
    for r in range(R):
        lp = np.asarray(target.log_prior_batch(Z[r]), dtype=np.float64)
        ll = np.asarray(target.log_like_batch(Z[r]), dtype=np.float64)
        want[r] = lp + betas[r] * ll
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)

    # the tempered/untempered faces flatten leading dims correctly
    t = np.asarray(tempered(Z), dtype=np.float64)
    u = np.asarray(untempered(Z), dtype=np.float64)
    assert t.shape == u.shape == (R, C)
    ll2 = np.asarray(target.log_like_batch(
        Z.reshape(-1, target.dim)), dtype=np.float64).reshape(R, C)
    np.testing.assert_allclose(t, ll2, rtol=1e-6)


# --------------------------------------------------------------------------- #
# P2c PT-reference helpers (metrics) + default-budget registry
# --------------------------------------------------------------------------- #
def test_pt_walker_temps_and_round_trips():
    """A hand-built accepted-adjacent-swap trace on R=3 slots must move one
    walker cold(0)->hot(2)->cold(0) for exactly one beta=1 round trip."""
    from cgl import metrics

    # pairs: col0 = swap slots(0,1), col1 = swap slots(1,2)
    acc = np.array([[True, False],   # occupant [0,1,2]->[1,0,2]: w0 -> slot1
                    [False, True],   #          [1,0,2]->[1,2,0]: w0 -> slot2 (hot)
                    [False, True],   #          [1,2,0]->[1,0,2]: w0 -> slot1
                    [True, False]])  #          [1,0,2]->[0,1,2]: w0 -> slot0 (cold)
    temps = metrics.pt_walker_temps_from_adjacent(acc)
    assert temps.shape == (4, 3)
    assert temps[:, 0].tolist() == [1, 2, 1, 0]        # walker 0 trajectory
    # every row is a valid permutation (walker<->slot bijection)
    for t in range(4):
        assert sorted(temps[t].tolist()) == [0, 1, 2]
    rt = metrics.count_pt_round_trips(temps)
    assert rt["total_round_trips"] == 1
    assert rt["n_walkers"] == 3
    assert rt["round_trips_per_walker"][0] == 1
    assert rt["n_walkers_reaching_hot"] >= 1


def test_pt_round_trips_no_swaps_is_zero():
    from cgl import metrics

    acc = np.zeros((10, 4), dtype=bool)                # R=5, never swap
    temps = metrics.pt_walker_temps_from_adjacent(acc)
    # occupancy frozen at the identity -> each walker pinned to its slot
    assert np.all(temps == np.arange(5)[None, :])
    rt = metrics.count_pt_round_trips(temps)
    assert rt["total_round_trips"] == 0
    assert rt["n_walkers_reaching_hot"] == 1           # walker 4 sits at hot end


def test_get_default_budget():
    from cgl.samplers import get_default_budget

    assert "n_keep" in get_default_budget("remc_pt")
    assert "n_steps" in get_default_budget("bj_mclmc")
    assert "n_eff" in get_default_budget("nautilus_runner")
    with pytest.raises(KeyError):
        get_default_budget("does_not_exist")


# --------------------------------------------------------------------------- #
# flow pullback logp
# --------------------------------------------------------------------------- #
def test_pullback_logp_2d_flow():
    """logp_u(u) = logp(T(u)) + log|det J_T(u)| checked against (a) explicit
    recomposition and (b) numerical Jacobians of the transform."""
    import jax
    import jax.numpy as jnp

    from cgl import flows
    from cgl.zoo import get_target

    target = get_target("t0_mix2_f64")        # f64 target (CPU x64 process)
    rng = np.random.default_rng(3)
    # near-Gaussian training blob offset from the origin (identity-ish flow)
    train = rng.normal(loc=[1.5, -0.5], scale=[0.7, 1.3], size=(600, 2))
    bundle = flows.fit_nsf(0, train, flow_layers=2, knots=5, interval=4.0,
                           nn_width=16, nn_depth=1, max_epochs=3,
                           batch_size=128)

    U = jnp.asarray(rng.normal(size=(9, 2)))
    x, ld = bundle.forward_batch(U)
    x, ld = np.asarray(x), np.asarray(ld)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(ld))

    # (a) pullback == target(x) + logdet
    pullback = bundle.make_pullback(target.log_prob_batch)
    got = np.asarray(pullback(U), dtype=np.float64)
    want = np.asarray(target.log_prob_batch(jnp.asarray(x)),
                      dtype=np.float64) + ld
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)

    # (b) logdet == slogdet of the full transform Jacobian (incl. the
    # destandardization affine), point by point
    def T(u):
        xx, _ = bundle.forward_batch(u[None, :])
        return xx[0]

    for i in range(4):
        J = np.asarray(jax.jacfwd(T)(U[i]), dtype=np.float64)
        sign, logdet = np.linalg.slogdet(J)
        assert sign > 0
        np.testing.assert_allclose(ld[i], logdet, rtol=1e-6, atol=1e-6)

    # (c) pushing HMC-style draws through matches the direct transform
    U_tc = np.asarray(rng.normal(size=(5, 3, 2)))
    Z = bundle.push_samples(U_tc)
    x2, _ = bundle.forward_batch(jnp.asarray(U_tc.reshape(15, 2)))
    np.testing.assert_allclose(Z.reshape(15, 2), np.asarray(x2), rtol=1e-6)


# --------------------------------------------------------------------------- #
# particle helpers
# --------------------------------------------------------------------------- #
def test_weight_ess_and_pseudo_chains():
    w = np.ones(100)
    assert np.isclose(common.weight_ess(w), 100.0)
    w2 = np.zeros(100)
    w2[0] = 1.0
    assert np.isclose(common.weight_ess(w2), 1.0)

    rng = np.random.default_rng(0)
    p = rng.normal(size=(1003, 4))
    chains = common.particles_to_chains(p, 8, rng)
    assert chains.shape == (125, 8, 4)
    # shuffled reshape preserves the particle population (up to trimming)
    flat = chains.reshape(-1, 4)
    assert flat.shape[0] == 1000


def test_flatten_leading():
    import jax.numpy as jnp

    def batch_fn(Z):
        return jnp.sum(Z ** 2, axis=-1)

    fn = common.flatten_leading(batch_fn)
    Z = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    got = np.asarray(fn(Z))
    want = (Z ** 2).sum(axis=-1)
    np.testing.assert_allclose(got, want)
