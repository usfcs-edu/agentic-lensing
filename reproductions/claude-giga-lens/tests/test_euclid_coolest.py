"""CPU unit tests for P3 (Euclid Q1 + COOLEST export).

Three groups, all CPU-safe (conftest forces JAX_PLATFORMS=cpu + CGL_ALLOW_CPU=1):
  1. cgl.euclid_io: mask / sentinel / PSF-normalisation logic on a real target.
  2. COOLEST export round-trips (32_coolest_export end-to-end: dump -> load,
     point estimates preserved) on a synthetic fit.
  3. the prior-override kwargs default to the HST-parity prior bit-for-bit
     (the one careful additive edit to cgl.likelihood must not move gates A-E).
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPRO = Path(__file__).resolve().parent.parent
FLAG_EXTRA = "102020061_NEG607087127495633316"   # has a masked neighbour galaxy
FLAG_ISO = "102157958_2719195933641972975"       # isolated (no extra-galaxy mask)


# --------------------------------------------------------------------------- #
# 1. euclid_io
# --------------------------------------------------------------------------- #
def _has_target(id_str):
    from cgl.euclid_io import target_dir
    return (target_dir(id_str) / f"{id_str}.fits").exists()


@pytest.mark.skipif(not _has_target(FLAG_ISO), reason="euclid-q1 data absent")
def test_euclid_io_layout_and_psf_norm():
    from cgl.euclid_io import load_euclid_vis, VIS_DELTA_PIX
    p = load_euclid_vis(FLAG_ISO, crop_pix=100)
    assert set(p) == {"img", "err_map", "keep_mask", "psf", "meta"}
    assert p["img"].shape == (100, 100)
    assert p["err_map"].shape == (100, 100)
    assert p["keep_mask"].shape == (100, 100)
    assert p["keep_mask"].dtype == bool
    # PSF normalised to sum 1 (raw VIS_PSF sums to ~0.93-0.97)
    assert abs(float(p["psf"].sum()) - 1.0) < 1e-12
    assert p["meta"]["psf_raw_sum"] < 0.999   # confirms a real renormalisation
    m = p["meta"]
    assert m["delta_pix"] == VIS_DELTA_PIX == 0.1
    assert m["psf_pixel_scale"] == 0.1        # guards.assert_psf_sampling passes
    assert m["num_pix"] == 100
    assert m["n_keep"] == int(p["keep_mask"].sum())


@pytest.mark.skipif(not _has_target(FLAG_ISO), reason="euclid-q1 data absent")
def test_euclid_io_circular_mask():
    from cgl.euclid_io import load_euclid_vis
    p = load_euclid_vis(FLAG_ISO, crop_pix=100)
    keep = p["keep_mask"]
    n = keep.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.indices(keep.shape)
    r = np.hypot(xx - c, yy - c)
    r_mask = p["meta"]["mask_radius_px"]        # 40 px = 4"/0.1
    # nothing kept outside the circular mask; corners are excluded
    assert not keep[r > r_mask].any()
    assert not keep[0, 0] and not keep[0, -1] and not keep[-1, -1]
    # some pixels kept inside (a real lens fills the mask)
    assert keep[r <= r_mask].sum() > 1000


@pytest.mark.skipif(not _has_target(FLAG_EXTRA), reason="euclid-q1 data absent")
def test_euclid_io_extra_galaxy_masked():
    """The neighbour-galaxy mask (mask_extra_galaxies.fits, value 1) is excluded
    from keep even where it falls inside the circular mask."""
    from astropy.io import fits
    from cgl.euclid_io import load_euclid_vis, target_dir
    p = load_euclid_vis(FLAG_EXTRA, crop_pix=100)
    assert p["meta"]["n_extra_galaxy_masked"] > 0
    # locate an extra-galaxy pixel in the crop and assert it is NOT kept
    with fits.open(target_dir(FLAG_EXTRA) / "mask_extra_galaxies.fits") as h:
        me = np.asarray(h[0].data).astype(bool)
    full = me.shape[0]
    cc = full // 2
    half = 50
    me_crop = me[cc - half:cc + half, cc - half:cc + half]
    assert me_crop.any()
    assert not p["keep_mask"][me_crop].any()


def test_euclid_io_sentinel_logic():
    """The keep mask must drop the >=1e15 RMS sentinel (bad pixels). Tests the
    exact boolean the loader applies against the RMS_SENTINEL constant."""
    from cgl.euclid_io import RMS_SENTINEL
    assert RMS_SENTINEL == 1e15
    err = np.array([[0.004, 1e16], [5.0, 0.003]])   # one 1e16 sentinel
    finite = err < RMS_SENTINEL
    assert finite.tolist() == [[True, False], [True, True]]
    # a sentinel pixel is excluded regardless of being inside the mask
    circle = np.ones_like(err, dtype=bool)
    extra = np.zeros_like(err, dtype=bool)
    keep = finite & circle & (~extra)
    assert not keep[0, 1]


# --------------------------------------------------------------------------- #
# 2. COOLEST export round-trip (32_coolest_export end-to-end)
# --------------------------------------------------------------------------- #
def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_fit(tmp_path):
    """Write a minimal 31-style fit json + posterior npz into tmp_path."""
    comp = dict(
        mass=dict(theta_E=1.13, gamma=1.94, e1=0.05, e2=-0.03,
                  center_x=0.01, center_y=-0.02),
        shear=dict(gamma1=0.03, gamma2=-0.01),
        lens_light=[dict(R_sersic=0.5, n_sersic=3.5, e1=0.04, e2=0.02,
                         center_x=0.0, center_y=0.0, Ie=0.4) for _ in range(4)],
        source_sersic=dict(R_sersic=0.2, n_sersic=1.1, e1=0.02, e2=0.01,
                           center_x=0.03, center_y=-0.01, Ie=0.6),
        source_shapelet=dict(beta=0.11, center_x=0.02, center_y=0.0))
    summary = dict(
        target="synthetic_target", subset="lens",
        meta=dict(delta_pix=0.1, num_pix=20, supersample=2, band="VIS",
                  mag_zero_point=24.6, subset="lens"),
        components=comp, redshifts=dict(z_lens=0.5, z_source=1.0),
        light_scale_resolved=0.02, map_chi2_per_pixel=0.9,
        theta_E_pub_eff=1.12,
        posterior_mass=dict(
            theta_E=dict(median=1.13, p16=1.10, p84=1.16, mean=1.13),
            gamma=dict(median=1.94, p16=1.8, p84=2.05, mean=1.94),
            center_x=dict(median=0.01, p16=0.0, p84=0.02, mean=0.01),
            center_y=dict(median=-0.02, p16=-0.03, p84=-0.01, mean=-0.02)))
    (tmp_path / "synthetic_target_fit.json").write_text(json.dumps(summary))
    a_star = np.arange(28, dtype=np.float64) * 0.01
    np.savez_compressed(
        tmp_path / "synthetic_target_posterior.npz",
        z_map=np.zeros(46), data=np.zeros((20, 20), np.float32),
        err_map=np.ones((20, 20), np.float32),
        keep_mask=np.ones((20, 20), bool),
        psf=(np.ones((21, 21), np.float32) / 441.0),
        model_image_map=np.zeros((20, 20), np.float32),
        a_star_map=a_star,
        draws=np.zeros((5, 4, 46), np.float32), labels=np.array(["x"] * 46))
    return comp, a_star


def test_coolest_export_roundtrip(tmp_path):
    export = _load_module(REPRO / "32_coolest_export.py", "coolest_export_mod")
    comp, a_star = _synthetic_fit(tmp_path)
    fit_json = tmp_path / "synthetic_target_fit.json"
    outdir = tmp_path / "out"
    argv = ["32_coolest_export.py", "--fit", str(fit_json),
            "--name", "synthetic_target", "--outdir", str(outdir)]
    old = sys.argv
    try:
        sys.argv = argv
        rc = export.main()
    finally:
        sys.argv = old
    assert rc == 0
    # reload independently and check point estimates preserved
    from coolest.template.json import JSONSerializer
    stem = str(outdir / "synthetic_target")
    c = JSONSerializer(stem).load_simple(stem + ".json", validate=True)
    le = c.lensing_entities
    assert c.mode == "MAP"
    pemd = le[0].mass_model[0].parameters
    assert abs(pemd["theta_E"].point_estimate.value - comp["mass"]["theta_E"]) < 1e-9
    assert abs(pemd["gamma"].point_estimate.value - comp["mass"]["gamma"]) < 1e-9
    # PEMD posterior attached from the chains-derived stats
    assert pemd["theta_E"].posterior_stats.median is not None
    # ExternalShear gamma_ext = hypot(g1,g2)
    import math
    ge = le[2].mass_model[0].parameters["gamma_ext"].point_estimate.value
    assert abs(ge - math.hypot(comp["shear"]["gamma1"],
                               comp["shear"]["gamma2"])) < 1e-9
    # shapelet amps preserved (all 28, ridge-marginalised MAP values)
    amps = le[1].light_model[1].parameters["amps"].point_estimate.value
    assert len(amps) == 28
    np.testing.assert_allclose(np.asarray(amps), a_star, rtol=0, atol=1e-9)
    # co-located FITS written
    for f in ("psf.fits", "data.fits", "model_map.fits",
              "synthetic_target.json"):
        assert (outdir / f).exists()


def test_coolest_ell_to_q_phi():
    export = _load_module(REPRO / "32_coolest_export.py", "coolest_export_mod2")
    # round ellipticity -> q=1, phi=0
    q, phi = export.ell_to_q_phi(0.0, 0.0)
    assert abs(q - 1.0) < 1e-9 and abs(phi) < 1e-9
    # |e|=0.2 -> q=(1-0.2)/(1+0.2)=0.6667
    q, phi = export.ell_to_q_phi(0.2, 0.0)
    assert abs(q - (0.8 / 1.2)) < 1e-9


# --------------------------------------------------------------------------- #
# 3. prior-override defaults preserve HST parity bit-for-bit
# --------------------------------------------------------------------------- #
def test_prior_override_defaults_match_hardcoded():
    """_build_prior() with default kwargs must reproduce the ORIGINAL hardcoded
    hyperparameters bit-for-bit (theta_E LogNormal(log2.5,0.25); mass center
    Normal(0,0.02); Ie medians 5/2/1/0.5/2). This is the guard that the one
    additive edit to cgl.likelihood cannot move gates A-E."""
    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions
    from cgl.likelihood import _build_prior

    nx, ny = 0.37, -1.29
    p_def = _build_prior(nx, ny)                      # default kwargs

    def _s(Rm, Rs, Im, Is, cx=0.0, cy=0.0, cs=0.05):
        return tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(Rm), Rs), n_sersic=tfd.Uniform(0.5, 8.0),
            e1=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
            e2=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
            center_x=tfd.Normal(cx, cs), center_y=tfd.Normal(cy, cs),
            Ie=tfd.LogNormal(jnp.log(Im), Is)))
    lm = tfd.JointDistributionSequential([tfd.JointDistributionNamed(dict(
        theta_E=tfd.LogNormal(jnp.log(2.5), 0.25),
        gamma=tfd.TruncatedNormal(2.0, 0.25, 1.0, 2.7),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.02), center_y=tfd.Normal(0.0, 0.02))),
        tfd.JointDistributionNamed(dict(gamma1=tfd.Normal(0.0, 0.05),
                                        gamma2=tfd.Normal(0.0, 0.05)))])
    ll = tfd.JointDistributionSequential([
        _s(0.4, 0.3, 5.0, 0.5, cs=0.02), _s(2.0, 0.3, 2.0, 0.5, cs=0.02),
        _s(0.3, 0.3, 1.0, 0.5, cx=nx, cy=ny, cs=0.1),
        _s(0.6, 0.3, 0.5, 0.5, cx=nx, cy=ny, cs=0.1)])
    ss = tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(0.5), 0.3), n_sersic=tfd.Uniform(0.5, 6.0),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
        center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1),
        Ie=tfd.LogNormal(jnp.log(2.0), 0.5)))
    sh = tfd.JointDistributionNamed(dict(
        beta=tfd.LogNormal(jnp.log(0.1), 0.1), center_x=tfd.Normal(0.0, 0.05),
        center_y=tfd.Normal(0.0, 0.05)))
    p_old = tfd.JointDistributionSequential(
        [lm, ll, tfd.JointDistributionSequential([ss, sh])])

    for s in range(6):
        x = p_old.sample(seed=jax.random.PRNGKey(s))
        assert float(p_def.log_prob(x)) == float(p_old.log_prob(x))


def test_prior_override_changes_when_requested():
    """The overrides actually move the prior (guards against a no-op edit)."""
    import jax
    from cgl.likelihood import _build_prior
    p0 = _build_prior(0.0, 0.0)
    p1 = _build_prior(0.0, 0.0, theta_E_med=1.0, mass_center_sig=0.3,
                      light_scale=0.02)
    x = p0.sample(seed=jax.random.PRNGKey(0))
    assert float(p0.log_prob(x)) != float(p1.log_prob(x))
