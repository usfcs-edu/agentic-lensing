"""Guard fences raise (never warn) on their incident patterns."""
import numpy as np
import pytest

from cgl2 import guards


def test_psf_sampling():
    guards.assert_psf_sampling(0.13, 0.13)                # at delta_pix
    guards.assert_psf_sampling(0.065, 0.13, supersample=2)
    with pytest.raises(RuntimeError):
        guards.assert_psf_sampling(0.065, 0.13)           # the ss2 trap


def test_whitener_bundle_gate():
    ok = dict(h_taps=np.ones((1, 1)), keep_mask=np.ones((4, 4), bool),
              e_op=0.019)
    guards.assert_whitener_bundle(ok)
    with pytest.raises(RuntimeError):
        guards.assert_whitener_bundle(dict(ok, e_op=0.021))
    with pytest.raises(RuntimeError):
        guards.assert_whitener_bundle(dict(e_op=0.01, keep_mask=ok["keep_mask"]))


def test_scene_config_certified(monkeypatch):
    monkeypatch.delenv("CGL2_UNCERTIFIED_OK", raising=False)
    guards.assert_scene_config_certified(["EPL", "Shear"],
                                         ["SersicEllipse", "Shapelets"])
    with pytest.raises(RuntimeError):
        guards.assert_scene_config_certified(["BPL"], ["SersicEllipse"])
    with pytest.raises(RuntimeError):
        guards.assert_scene_config_certified(["EPL"], ["Shapelets"],
                                             supersample=4)
    monkeypatch.setenv("CGL2_UNCERTIFIED_OK", "1")
    guards.assert_scene_config_certified(["BPL"], ["SersicEllipse"])


def test_where_mask_semantics():
    guards.require_where_mask_semantics()


def test_require_x64():
    guards.require_x64()  # conftest enables x64 before any jax import


def test_make_whitener_bundle_validation():
    from cgl2.correlated import make_whitener_bundle

    b = make_whitener_bundle(np.ones((1, 1)), np.ones((4, 4)),
                             np.ones((4, 4), bool), e_op=0.0)
    assert b["kernel_hash"]
    with pytest.raises(ValueError):
        make_whitener_bundle(np.ones((2, 2)), np.ones((4, 4)),
                             np.ones((4, 4), bool), e_op=0.0)  # even taps
    with pytest.raises(ValueError):
        make_whitener_bundle(np.ones((1, 1)), np.ones((4, 4)),
                             np.zeros((4, 4), bool), e_op=0.0)  # empty keep
    with pytest.raises(ValueError):
        make_whitener_bundle(np.ones((1, 1)), np.full((4, 4), np.nan),
                             np.ones((4, 4), bool), e_op=0.0)
