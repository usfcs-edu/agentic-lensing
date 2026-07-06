"""Vendor bootstrap + artifact-path sanity."""
import sys

from cgl import paths


def test_expected_artifacts_exist():
    for p in [paths.CUTOUT_V2D, paths.CUTOUT_V3, paths.CUTOUT_V3B,
              paths.MODEL_MAP_V3COLD, paths.MODEL_MAP_V3B_COLD,
              paths.MAP_MARG_PD, paths.HESS_MARG_PD, paths.HMC_V13_V3B]:
        assert p.exists(), f"missing foundry-i artifact: {p}"


def test_bootstrap_vendor_ref_and_shadowing():
    paths.bootstrap_vendor()
    assert sys.path[0] == str(paths.VENDOR_SRC)
    import gigalens
    assert str(paths.VENDOR_SRC) in gigalens.__file__, (
        f"gigalens resolved outside the campaign vendor tree: {gigalens.__file__}"
    )


def test_load_product_layout():
    d = paths.load_product(paths.CUTOUT_V2D)
    assert d["img"].shape == d["err_map"].shape == d["keep_mask"].shape
    assert d["psf"].ndim == 2
    assert "meta" in d and isinstance(d["meta"], dict)
