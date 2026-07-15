"""Fixtures and configuration for the lenstronomy physics-validation suite.

Sets float64 before ``jax.numpy`` is imported anywhere (project-standards §8),
registers markers, and provides the oracle cosmology pair and seeded *valid*
parameter draws used by the differential / metamorphic sweeps.
"""
import os

# Must be set before jax.numpy is first imported (the gigalens precision guard
# reads it at import time). project-standards §8.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

try:
    import jax
    jax.config.update("jax_enable_x64", True)
except Exception:  # pragma: no cover - jax always present in the runtime
    pass


# --- options & markers ------------------------------------------------------------
def pytest_addoption(parser):
    group = parser.getgroup("validation")
    group.addoption(
        "--plots", action="store_true", default=False,
        help="write lenstronomy-vs-gigalens comparison plots for every registered "
             "comparison (so passing tests can be eyeballed too; method-discipline §5)."
    )
    group.addoption(
        "--plot-dir", action="store", default=None,
        help="directory for comparison plots (default: tests/validation/_artifacts)."
    )


def pytest_configure(config):
    for name, desc in [
        ("physics", "validates simulator physics against an oracle"),
        ("cosmology", "cosmological-distance validation vs astropy"),
        ("multiplane", "recursive multi-lens-plane ray-shooting"),
        ("metamorphic", "oracle-free invariant / differential test"),
        ("redteam", "adversarial: naive misuse must fail loudly or be flagged (misuse_register.md)"),
        ("slow", "slower test (excluded by '-m \"not slow\"')"),
        ("pending", "written but skipped until the new API/trace exists"),
    ]:
        config.addinivalue_line("markers", f"{name}: {desc}")


# --- plot artifacts (NOT the pass/fail gate; see plotting.py) ---------------------
_REPORT_KEY = pytest.StashKey[dict]()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Stash each phase's outcome so the plotter fixture can render on failure."""
    outcome = yield
    rep = outcome.get_result()
    item.stash.setdefault(_REPORT_KEY, {})[rep.when] = rep.outcome


@pytest.fixture
def plotter(request):
    """A ComparisonPlotter for the test. Renders iff --plots is set OR the test failed.

    Tests register comparisons (``plotter.image(...)`` / ``plotter.curve(...)``)
    unconditionally; rendering is decided here, so normal numeric runs stay fast and
    matplotlib stays optional.
    """
    from .plotting import ComparisonPlotter

    plot_dir = request.config.getoption("--plot-dir")
    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(__file__), "_artifacts")
    p = ComparisonPlotter(name=request.node.name, plot_dir=plot_dir)
    yield p

    want_plots = bool(request.config.getoption("--plots"))
    reports = request.node.stash.get(_REPORT_KEY, {})
    failed = any(v == "failed" for v in reports.values())
    if want_plots or failed:
        try:
            paths = p.render()
            if paths:
                term = request.config.pluginmanager.get_plugin("terminalreporter")
                if term is not None:
                    term.write_line(f"\n[validation] wrote {len(paths)} plot(s) to "
                                    f"{os.path.dirname(str(paths[0]))}")
        except Exception as exc:  # plotting must never mask the real result
            import warnings
            warnings.warn(f"comparison plotting failed: {exc!r}")


# --- seeding (project-standards §5: seeds recorded) -------------------------------
SEED = 20260623


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


# --- oracle availability ----------------------------------------------------------
@pytest.fixture(scope="session")
def lenstronomy():
    return pytest.importorskip("lenstronomy")


@pytest.fixture(scope="session")
def astropy_cosmo_cls():
    cosmo = pytest.importorskip("astropy.cosmology")
    return cosmo


# --- cosmology oracle pair --------------------------------------------------------
# A flat wCDM with explicit parameters, expressed identically in gigalens and astropy
# so distance comparisons isolate the integrator/model, not a parameter mismatch.
FLAT_WCDM = dict(H0=70.0, Om0=0.3, w0=-1.0)  # k=0 (flat)


@pytest.fixture
def gigalens_cosmo():
    """gigalens JAX wCDM at FLAT_WCDM, lens at z=0.5, ref source z=10."""
    from gigalens.jax.cosmo import wCDM_Cosmo
    return wCDM_Cosmo(z_lens=0.5, z_source_ref=10.0)


@pytest.fixture
def astropy_cosmo(astropy_cosmo_cls):
    """astropy wCDM matching FLAT_WCDM, WITH radiation matched to gigalens.

    gigalens' efunc always includes a radiation term Or0 = 2.469e-5 h^-2 (1+0.2271
    Neff); astropy's FlatwCDM defaults to Tcmb0=0 (no radiation). Verified in Phase-0
    diagnostics: with Tcmb0=0 the distance disagreement grows to ~2.6e-4 at z=4 (purely
    the missing radiation), while Tcmb0=2.7255/Neff=3.046 closes it to ~1e-6. So we
    match radiation here — the test then probes the model+integrator, not a radiation
    bookkeeping difference. (gigalens' Or0 is an ~1e-6-accurate approximation to the
    photon+neutrino density; that residual is far inside COSMO_DISTANCE_RTOL.)
    """
    from astropy.cosmology import FlatwCDM
    import astropy.units as u
    return FlatwCDM(H0=FLAT_WCDM["H0"] * u.km / u.s / u.Mpc,
                    Om0=FLAT_WCDM["Om0"], w0=FLAT_WCDM["w0"],
                    Tcmb0=2.7255, Neff=3.046)


# --- seeded valid parameter draws (differential sweep inputs) ---------------------
@pytest.fixture
def epl_params(rng):
    """A list of physically-valid EPL parameter dicts for sweep-style parity tests."""
    out = []
    for _ in range(8):
        out.append(dict(
            theta_E=float(rng.uniform(0.5, 2.0)),
            gamma=float(rng.uniform(1.6, 2.4)),
            e1=float(rng.uniform(-0.3, 0.3)),
            e2=float(rng.uniform(-0.3, 0.3)),
            center_x=float(rng.uniform(-0.2, 0.2)),
            center_y=float(rng.uniform(-0.2, 0.2)),
        ))
    return out


@pytest.fixture
def grid():
    """A modest float64 evaluation grid (arcsec), away from exact profile centers."""
    xs = np.linspace(-3.0, 3.0, 60, dtype=np.float64)
    X, Y = np.meshgrid(xs, xs)
    # nudge off the origin so center-singular profiles are not evaluated AT 0
    return X + 1e-3, Y - 1e-3
