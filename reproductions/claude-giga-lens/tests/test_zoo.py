"""Zoo unit tests. CPU: T0 analytics, identity, registry, InitBundle PD.
GPU-marked: a ~30 s S0 baseline smoke on the 2-D mixture.

conftest forces CPU + x64 for the unit tests; T0 targets carry explicit
dtypes so f32 targets still build correctly under the x64 session.
"""
import os

import numpy as np
import pytest

from cgl.zoo import get_target, get_target_info, list_targets

_LOG2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_listing():
    infos = {t["name"]: t for t in list_targets()}
    expected = (["t0_mix2", "t0_mix2_f64", "t0_mix22", "t0_funnel10",
                 "t0_illcond46"]
                + [f"gu2022_sys{i:03d}" for i in range(12)]
                + ["foundry_marg46", "foundry_v3b74", "euclid_q1"])
    assert sorted(infos) == sorted(expected)
    assert not infos["euclid_q1"]["available"]
    assert infos["foundry_marg46"]["dtype"] == "float64"
    assert infos["t0_illcond46"]["dtype"] == "float64"
    assert infos["gu2022_sys003"]["dtype"] == "float32"
    avail = [t["name"] for t in list_targets(available_only=True)]
    assert "euclid_q1" not in avail and len(avail) == 19


def test_euclid_stub_raises():
    with pytest.raises(NotImplementedError):
        get_target("euclid_q1")
    with pytest.raises(KeyError):
        get_target_info("not_a_target")


# --------------------------------------------------------------------------- #
# T0 analytics (registry-built; CPU)
# --------------------------------------------------------------------------- #
def test_mixture_logz_and_moments_vs_numpy():
    t = get_target("t0_mix2")
    # independent numpy recompute of logZ
    s2, sp2, x0 = 0.5 ** 2, 5.0 ** 2, 2.0
    ev = (-0.5 * 2 * (_LOG2PI + np.log(s2 + sp2))
          - 0.5 * x0 ** 2 / (s2 + sp2))
    logZ_np = float(np.logaddexp(np.log(0.8) + ev, np.log(0.2) + ev))
    assert abs(t.reference.logZ - logZ_np) < 1e-12

    # log_prob at the dominant mode center vs manual density
    z = np.asarray(t.reference.mode_centers[0], dtype=np.float32)
    lp = float(t.log_prob(z))
    like = np.log(
        0.8 * np.exp(-0.5 * ((z[0] - 2.0) ** 2 + z[1] ** 2) / s2)
        + 0.2 * np.exp(-0.5 * ((z[0] + 2.0) ** 2 + z[1] ** 2) / s2)
    ) - 0.5 * 2 * (_LOG2PI + np.log(s2))
    prior = -0.5 * 2 * (_LOG2PI + np.log(sp2)) - 0.5 * (z ** 2).sum() / sp2
    assert abs(lp - (like + prior)) < 1e-4      # f32 target

    # exact sampler moments vs analytic truth
    X = t.reference.exact_sample_fn(0, 200_000)
    tr = t.reference.truth
    assert abs(X[:, 0].mean() - tr["mean"][0]) < 0.02
    assert abs(X[:, 0].var() / tr["var"][0] - 1.0) < 0.02
    frac_pos = float((X[:, 0] > 0).mean())
    assert abs(frac_pos - 0.8) < 0.01           # posterior mode weights exact


def test_funnel_logz_zero_and_moments():
    t = get_target("t0_funnel10")
    assert t.reference.logZ == 0.0
    X = t.reference.exact_sample_fn(1, 100_000)
    assert abs(X[:, 0].var() - 9.0) < 0.3
    assert abs(np.log(X[:, 1:] ** 2).mean() - (-1.2703628)) < 0.03


def test_illcond_logz_and_condition():
    t = get_target("t0_illcond46")
    from scipy.stats import norm
    lam = np.logspace(0.0, -14.0, 46)
    logZ_sp = float(np.sum(norm.logpdf(0.0, scale=np.sqrt(lam + 100.0))))
    assert abs(t.reference.logZ - logZ_sp) < 1e-9
    cond = t.reference.truth["cond_posterior"]
    assert 1e13 < cond < 1.1e14


@pytest.mark.parametrize("name", ["t0_mix2", "t0_mix22", "t0_funnel10",
                                  "t0_mix2_f64", "t0_illcond46"])
def test_identity_prior_plus_like_equals_prob(name):
    t = get_target(name)
    assert t.has_independent_loglike
    fd = np.float64 if t.dtype == "float64" else np.float32
    rng = np.random.default_rng(99)
    Z = (0.5 * rng.standard_normal((8, t.dim))).astype(fd)
    lp = np.asarray(t.log_prob_batch(Z), dtype=np.float64)
    pri = np.asarray(t.log_prior_batch(Z), dtype=np.float64)
    lik = np.asarray(t.log_like_batch(Z), dtype=np.float64)
    tol = 1e-10 if t.dtype == "float64" else 1e-5
    rel = np.max(np.abs(pri + lik - lp) / np.maximum(np.abs(lp), 1.0))
    assert rel <= tol, f"{name}: identity rel {rel:.3e} > {tol:g}"
    assert np.all(np.isfinite(lp))


def test_initbundle_floored_cov_pd():
    for name in ["t0_mix2", "t0_funnel10", "t0_illcond46"]:
        t = get_target(name)
        L = t.init.svi_scale_tril
        assert L is not None and L.shape == (t.dim, t.dim)
        cov = L @ L.T
        w = np.linalg.eigvalsh(cov)
        assert w.min() > 0, f"{name}: floored cov not PD"
        # the guard floor: min eigenvalue >= ~1e-10 * max
        assert w.min() >= 0.9e-10 * w.max()
        np.linalg.cholesky(cov)                  # must not raise
    # illcond: the floor MUST have engaged (posterior vars reach 1e-14)
    t = get_target("t0_illcond46")
    assert t.init.svi_n_floored > 0


def test_prior_transform_unit_cube():
    t = get_target("t0_mix2")
    u = np.random.default_rng(3).uniform(size=(64, 2))
    x = t.prior_transform(u)
    assert x.shape == (64, 2)
    # prior is N(0, 5^2): quantiles map median to ~0
    assert abs(t.prior_transform(np.full((1, 2), 0.5)).ravel()[0]) < 1e-6
    ll = t.log_like_x(x)
    assert ll.shape == (64,) and np.all(np.isfinite(ll))


def test_labels_and_mass_indices():
    t = get_target("t0_illcond46")
    assert len(t.labels) == 46 == t.dim
    assert all(m in t.labels for m in t.mass_labels)
    idx = t.mass_indices()
    assert idx == sorted(idx)
    ph = t.to_physical(np.zeros((3, 46)))
    assert set(ph) == set(t.mass_labels)


def test_reference_provenance_required():
    for info in list_targets(available_only=True):
        if not info["name"].startswith("t0_"):
            continue
        t = get_target(info["name"])
        assert t.reference is not None
        assert isinstance(t.reference.provenance, str)
        assert len(t.reference.provenance) > 20


# --------------------------------------------------------------------------- #
# metrics sanity on analytic samples (CPU)
# --------------------------------------------------------------------------- #
def test_metrics_mode_machinery_on_mixture():
    from cgl import metrics

    t = get_target("t0_mix2")
    X = t.reference.exact_sample_fn(7, 4000).reshape(500, 8, 2)
    assign = metrics.assign_modes(t.reference, Z=X)
    occ = metrics.mode_occupancy(assign, 2, t.reference.mode_weights)
    assert occ["recovery_rate"] == 1.0
    assert occ["max_abs_weight_error"] < 0.05
    rt = metrics.count_mode_round_trips(assign.reshape(500, 8))
    assert rt["total_round_trips"] > 0           # iid chains hop constantly
    assert rt["n_migrating_chains"] == 8

    diag = metrics.rank_diagnostics(X, t.labels, t.mass_labels)
    assert diag["summary"]["rhat_all"]["max"] < 1.02
    eff = metrics.efficiency(diag, n_grad=1000, n_logp=10, wall_s=2.0,
                             hardware="cpu-test")
    assert eff["ess_per_grad_mass"] > 0
    assert abs(eff["ess_per_sec_mass"] - eff["ess_mass_min"] / 2.0) < 1e-9


def test_rank_diagnostics_axis_convention_catches_stuck_chains():
    """REGRESSION (P2a): (T, C, dim) fed to arviz with draws/chains swapped
    hides stuck-chain pathologies (R-hat ~ 1 on chains stuck in different
    places). Two stuck chain groups MUST produce a large R-hat and tiny ESS."""
    from cgl.metrics import rank_diagnostics

    rng = np.random.default_rng(0)
    T, C = 500, 8
    x = rng.standard_normal((T, C, 1)) * 0.1
    x[:, : C // 2, 0] += 5.0          # half the chains stuck 50 sigma away
    d = rank_diagnostics(x, ["z00"], ["z00"])
    assert d["summary"]["rhat_all"]["max"] > 1.5, d["summary"]
    assert d["summary"]["ess_bulk_all"]["min"] < 0.05 * T * C
    # and well-mixed iid chains stay clean
    d2 = rank_diagnostics(rng.standard_normal((T, C, 1)), ["z00"], ["z00"])
    assert d2["summary"]["rhat_all"]["max"] < 1.02


def test_budget_ledger():
    from cgl.metrics import BudgetLedger

    led = BudgetLedger()
    led.add("map", n_grad=32000, note="250x128")
    led.add("hmc", n_grad=150000, n_logp=50000)
    led.add("hmc", n_grad=1000)
    assert led.n_grad == 183000
    assert led.n_logp == 50000
    d = led.as_dict()
    assert d["phases"]["hmc"]["n_grad"] == 151000
    assert "convention" in d


def test_io_roundtrip(tmp_path):
    from cgl.io import CellResult, load_cell_result, save_cell_result

    res = CellResult(
        sampler="s0_baseline", target="t0_mix2", seed=0, track="smoke",
        samples=np.random.default_rng(0).standard_normal(
            (50, 4, 2)).astype(np.float32),
        labels=["z00", "z01"], mass_labels=["z00"],
        diagnostics={"step_size": np.ones(60), "is_accepted":
                     np.ones((60, 4), dtype=bool)},
        budget={"n_grad_total": 123}, timing={"total_s": 1.0},
        config={"mode": "chees"}, env={"device": "cpu"},
        freeze_check={"checked": False}, metrics={"acceptance_rate": 1.0},
    )
    save_cell_result(res, root=tmp_path)
    back = load_cell_result("s0_baseline", "t0_mix2", 0, "smoke",
                            root=tmp_path)
    np.testing.assert_array_equal(back.samples, res.samples)
    assert back.budget["n_grad_total"] == 123
    assert set(back.diagnostics) == {"step_size", "is_accepted"}
    assert back.config_sha == res.config_sha


# --------------------------------------------------------------------------- #
# GPU: 30-second S0 smoke on the 2-D mixture
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("CGL_TEST_GPU") != "1",
                    reason="GPU tests require CGL_TEST_GPU=1")
def test_s0_baseline_smoke_mix2():
    from cgl.samplers.baseline_gigalens import run_cell

    # conftest runs the GPU session with x64 ON -> use the f64 mixture
    # (identical construction; the f32 variant is exercised on CPU above).
    t = get_target("t0_mix2_f64")
    budget = dict(n_chains=8, n_burn=100, n_keep=150, track="smoke")
    config = dict(n_map=32, map_steps=100, n_vi=32, svi_steps=300,
                  max_leapfrog=10)
    res = run_cell(t, seed=0, budget=budget, config=config)

    assert res.samples.shape == (150, 8, 2)
    assert np.all(np.isfinite(res.samples))
    assert res.budget["n_grad_total"] > 0
    assert res.budget["phases"]["map"]["n_grad"] == 32 * 100
    assert res.budget["phases"]["svi"]["n_grad"] == 300 * 32
    ds = res.metrics["diagnostics"]["summary"]
    assert ds["rhat_all"]["max"] < 2.0           # smoke, not statistics
    assert res.metrics["acceptance_rate"] > 0.2
    assert res.metrics["efficiency"]["ess_per_grad_mass"] > 0
    # both posterior modes are reachable from the SVI init in principle;
    # do not gate on it in a 150-draw smoke.
