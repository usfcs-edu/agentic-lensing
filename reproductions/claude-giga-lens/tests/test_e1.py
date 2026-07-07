"""CPU tests for the P1b E1 fit driver (cgl/e1.py): z-score / coverage /
SBC-rank machinery against hand-built cases, the two-pass kernel-fit logic on
tiny synthetics, the 40b-recalibration mock adaptation, and the 07 batcher
manifest (dry-run). No GPU, no gigalens (conftest forces CPU+x64)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cgl import e1

REPRO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# stats helpers
# --------------------------------------------------------------------------- #
def test_z_scores_hand():
    z = e1.z_scores([1.0, 2.0, 3.0], [0.5, 1.0, 0.0], [0.0, 4.0, 3.0])
    assert z[0] == pytest.approx(2.0)
    assert z[1] == pytest.approx(-2.0)
    assert np.isnan(z[2])                       # zero std -> nan, not inf


def test_coverage_flag_hand():
    draws = np.linspace(0.0, 1.0, 10001)        # ~U(0,1)
    assert e1.coverage_flag(draws, 0.5, 0.68)
    assert not e1.coverage_flag(draws, 0.01, 0.68)   # outside [0.16, 0.84]
    assert e1.coverage_flag(draws, 0.01, 0.99)


def test_thin_indices_properties():
    idx = e1.thin_indices(18000, 127)
    assert idx.size == 127
    assert len(np.unique(idx)) == 127           # no repeats when n >> m
    assert idx[0] >= 0 and idx[-1] < 18000
    assert np.all(np.diff(idx) > 0)
    # n <= m: identity
    np.testing.assert_array_equal(e1.thin_indices(5, 127), np.arange(5))


def test_sbc_rank_hand():
    draws = np.arange(127, dtype=float)         # exactly 127 draws, 0..126
    assert e1.sbc_rank(draws, truth=-1.0) == 0
    assert e1.sbc_rank(draws, truth=1000.0) == 127
    assert e1.sbc_rank(draws, truth=63.5) == 64
    # thinning is deterministic
    big = np.repeat(draws, 100)
    assert e1.sbc_rank(big, 63.5) == e1.sbc_rank(big, 63.5)


def test_rank_uniformity_chi2():
    rng = np.random.default_rng(0)
    uni = rng.integers(0, 128, size=6400)
    chi2_u, p_u, obs = e1.rank_uniformity_chi2(uni)
    assert p_u > 0.01
    assert sum(obs) == 6400
    # pathological: everything in the first bin
    chi2_b, p_b, _ = e1.rank_uniformity_chi2(np.zeros(64, dtype=int))
    assert p_b < 1e-6
    assert chi2_b > chi2_u


def test_rank_uniformity_bins_must_divide():
    with pytest.raises(AssertionError):
        e1.rank_uniformity_chi2(np.arange(10), n_use=126, n_bins=8)


def test_recalibrate_err_40b():
    rng = np.random.default_rng(1)
    model = np.full((40, 40), 3.0)
    err = np.full((40, 40), 0.2)
    img = model + rng.normal(0, 2.0 * 0.2, size=model.shape)  # 2x miscal
    sky = np.ones_like(model, dtype=bool)
    err_r, s, chi2b = e1.recalibrate_err_40b(img, model, err, sky)
    assert s == pytest.approx(2.0, rel=0.05)
    assert chi2b == pytest.approx(4.0, rel=0.1)
    # endpoint: per-pixel sky chi2 == 1 exactly, by construction
    chi2_after = np.mean(((img - model)[sky] / err_r[sky]) ** 2)
    assert chi2_after == pytest.approx(1.0, abs=1e-12)


def test_r_arc_maps():
    r_f = e1.r_arc_map("fine")
    assert r_f.shape == (208, 208)
    assert r_f.min() >= 0.02                    # center is off-grid (104.5)
    r_b = e1.r_arc_map("binned")
    assert r_b.shape == (104, 104)
    assert r_b[52, 52] == pytest.approx(0.0)    # binned center on-pixel
    r_n = e1.r_arc_map("native", frame=(2, 2))
    assert r_n.shape == (70, 70)
    # exactly-centered frame: min distance = half-diagonal of 0.5 px
    assert r_n.min() == pytest.approx(np.hypot(0.5, 0.5) * 0.12, rel=1e-6)


# --------------------------------------------------------------------------- #
# two-pass kernel-fit logic on tiny synthetics (native-scale machinery)
# --------------------------------------------------------------------------- #
def _toy_native_mock(rng, correlated=False):
    """Synthetic 9-frame native mock: known flat model + noise; the 'MAP
    model' handed to kernel_pass is the true model (pass-1 stand-in)."""
    from scipy.signal import fftconvolve

    model = np.full((70, 70), 5.0)
    nats, errs = [], []
    for _ in range(9):
        err = np.full((70, 70), 0.3)
        if correlated:
            w = rng.normal(0, 1.0, size=(74, 74))
            box = np.ones((3, 3)) / 3.0          # autocorr = the 2-D tent
            n = fftconvolve(w, box, mode="valid")[:70, :70]
        else:
            n = rng.normal(0, 1.0, size=(70, 70))
        nats.append(model + n * err)
        errs.append(err)
    return dict(
        native_img=np.stack(nats), native_err=np.stack(errs),
        native_model=np.stack([model] * 9),
        meta=dict(seed=-1), flat={}, truth_nested=None,
    ), [model] * 9


def test_kernel_pass_iid_native():
    rng = np.random.default_rng(2)
    mock, prods = _toy_native_mock(rng, correlated=False)
    kp = e1.kernel_pass(mock, "native", prods)
    c = (kp["rho_fit"].shape[0] - 1) // 2
    # iid noise -> near-delta kernel, tiny whitener, gate met
    assert abs(kp["rho_fit"][c, c + 1]) < 0.05
    assert kp["M"] <= 1
    assert kp["e_op"] <= 0.02
    assert kp["reg_lambda"] == 0.0
    assert kp["gate_le_0p05"]
    assert kp["model_subtracted"] is True


def test_kernel_pass_correlated_native():
    rng = np.random.default_rng(3)
    mock, prods = _toy_native_mock(rng, correlated=True)
    kp = e1.kernel_pass(mock, "native", prods)
    c = (kp["rho_fit"].shape[0] - 1) // 2
    # box-smoothed noise: rho(1) ~ 2/3 must be detected
    assert kp["rho_fit"][c, c + 1] > 0.35
    # near-singular tent spectrum -> the delta-regularization engages
    assert kp["reg_lambda"] > 0.0


def test_adaptive_s_floor_policy():
    delta = np.zeros((5, 5))
    delta[2, 2] = 1.0
    s_floor, ratio = e1.adaptive_s_floor(delta, grid=128)
    assert ratio == pytest.approx(1.0)
    assert s_floor == 0.05                      # inactive (ratio >> floor)
    tent1d = np.array([1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3])
    tent = np.outer(tent1d, tent1d)
    _, ratio_t = e1.adaptive_s_floor(tent / tent[2, 2], grid=128)
    # near-singular: exact zeros at omega=2pi/3 (off-grid at 128 -> ~1e-8)
    assert ratio_t < 1e-4


def test_build_product_whitener_regularizes_singular():
    tent1d = np.array([1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3])
    tent = np.outer(tent1d, tent1d)
    wh = e1.build_product_whitener(tent / tent[2, 2], m_grid=(2,),
                                   e_target=0.5, grid=128)
    assert wh["reg_lambda"] == e1.REG_LAMBDA
    c = (wh["rho_whitened"].shape[0] - 1) // 2
    assert wh["rho_whitened"][c, c] == pytest.approx(1.0)  # renormalized
    # regularized center: (1+lam)/(1+lam) = 1; off-center scaled by 1/(1+lam)
    assert wh["rho_whitened"][c, c + 1] == pytest.approx(
        (2 / 3) / (1 + e1.REG_LAMBDA))


# --------------------------------------------------------------------------- #
# batcher manifest + dry run
# --------------------------------------------------------------------------- #
def test_build_job_manifest_counts(tmp_path):
    kw = dict(fits_dir=tmp_path, skip_existing=False)
    assert len(e1.build_job_manifest("pilot", **kw)) == 5
    assert len(e1.build_job_manifest("e1a", **kw)) == 8
    assert len(e1.build_job_manifest("e1b", **kw)) == 26      # 24 + 2 ablation
    assert len(e1.build_job_manifest("e1c", **kw)) == 64
    assert len(e1.build_job_manifest("e1d", **kw)) == 48
    # 'all' dedups the 8 e1b-fine == e1c seeds 0-7
    assert len(e1.build_job_manifest("all", **kw)) == 8 + 26 + 64 + 48 - 8
    # every job is unique by output
    jobs = e1.build_job_manifest("all", **kw)
    assert len({j["out"] for j in jobs}) == len(jobs)


def test_build_job_manifest_resumable(tmp_path):
    jobs = e1.build_job_manifest("e1a", fits_dir=tmp_path, skip_existing=False)
    (tmp_path / Path(jobs[0]["out"]).name).write_bytes(b"x")
    left = e1.build_job_manifest("e1a", fits_dir=tmp_path, skip_existing=True)
    assert len(left) == 7
    assert jobs[0]["out"] not in {j["out"] for j in left}


def test_batcher_dry_run():
    rc = subprocess.run(
        ["bash", str(REPRO / "07_run_e1_batch.sh"), "e1a", "--dry-run"],
        capture_output=True, text=True, timeout=300)
    assert rc.returncode == 0, rc.stderr
    assert "DRY" in rc.stdout
    # dry-run lists jobs (8 e1a fits, unless outputs already exist)
    n_listed = sum(1 for ln in rc.stdout.splitlines()
                   if ln.startswith("JOB\t"))
    n_expected = len(e1.build_job_manifest("e1a"))
    assert n_listed == n_expected
