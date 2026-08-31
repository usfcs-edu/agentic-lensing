import json
import math

import pytest

from lensmark import coords
from tests.conftest import FIXTURES

CASES = json.load(open(FIXTURES / "coords_cases.json"))
TOL = 1e-9


@pytest.mark.parametrize("c", CASES["cases"], ids=[c["name"] for c in CASES["cases"]])
def test_fixture_cases(c):
    W, H, cut = c["W"], c["H"], c["cutout"]
    x, y = coords.uv_to_px(c["u"], c["v"], W, H)
    assert abs(x - c["px"][0]) < 1e-6 and abs(y - c["px"][1]) < 1e-6
    dE, dN = coords.uv_to_dEdN(c["u"], c["v"], W, H, cut, c["north_up"], c["east_left"])
    assert abs(dE - c["dEdN"][0]) < 1e-6 and abs(dN - c["dEdN"][1]) < 1e-6
    if "rpa" in c:
        r, pa = coords.dEdN_to_rpa(dE, dN)
        assert abs(r - c["rpa"][0]) < 1e-6 and abs(pa - c["rpa"][1]) < 1e-6
    if "fits" in c:
        fx, fy = coords.uv_to_fits(c["u"], c["v"], W, H, c["array_origin"])
        assert abs(fx - c["fits"][0]) < 1e-6 and abs(fy - c["fits"][1]) < 1e-6


@pytest.mark.parametrize("c", CASES["arcsec_to_px"])
def test_arcsec_to_px(c):
    assert abs(coords.arcsec_to_px(c["a"], c["W"], c["cutout"]) - c["px"]) < 1e-9


@pytest.mark.parametrize("origin", ["upper", "lower"])
@pytest.mark.parametrize("east_left", [True, False])
def test_round_trips(origin, east_left):
    W, H, cut = 410, 300, 16.0
    for i in range(0, 11):
        for j in range(0, 11):
            u, v = i / 10, j / 10
            assert coords.dist_uv(coords.px_to_uv(*coords.uv_to_px(u, v, W, H), W, H), (u, v)) < TOL
            assert coords.dist_uv(coords.fits_to_uv(*coords.uv_to_fits(u, v, W, H, origin), W, H, origin), (u, v)) < TOL
            dE, dN = coords.uv_to_dEdN(u, v, W, H, cut, True, east_left)
            assert coords.dist_uv(coords.dEdN_to_uv(dE, dN, W, H, cut, True, east_left), (u, v)) < TOL
            r, pa = coords.dEdN_to_rpa(dE, dN)
            dE2, dN2 = coords.rpa_to_dEdN(r, pa)
            assert abs(dE2 - dE) < TOL and abs(dN2 - dN) < TOL


def test_flip_applied_exactly_once():
    # upper: v=0 (top row of the PNG) is the LAST FITS row; lower: v=0 is FITS row 1
    assert coords.uv_to_fits(0.0, 0.0, 10, 10, "upper")[1] == pytest.approx(10.5)
    assert coords.uv_to_fits(0.0, 0.0, 10, 10, "lower")[1] == pytest.approx(0.5)


def test_screen_angle_convention():
    assert coords.screen_angle_deg(0, 0, 1, 0) == pytest.approx(0.0)     # right
    assert coords.screen_angle_deg(0, 0, 0, -1) == pytest.approx(90.0)   # up (y down on screen)
    assert coords.screen_angle_deg(0, 0, -1, 0) == pytest.approx(180.0)
    assert coords.screen_angle_deg(0, 0, 0, 1) == pytest.approx(270.0)


def test_polar_about_arbitrary_centre():
    r, pa = coords.uv_to_rpa(0.25, 0.25, 400, 400, 16.0, cu=0.25, cv=0.5)
    assert r == pytest.approx(4.0) and pa == pytest.approx(0.0)
    assert coords.dist_arcsec((0.25, 0.5), (0.5, 0.5), 400, 400, 16.0) == pytest.approx(4.0)
