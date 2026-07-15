"""param_map round-trips + convention reconciliations (pure numpy)."""
import numpy as np
import pytest

from cgl2 import param_map as pm


def test_label_bijection():
    pm.audit_roundtrip()
    assert len(pm.OLD_LABELS_46) == 46
    assert len(set(pm.SCENE_NAMES_46)) == 46


def test_known_mappings():
    assert pm.old_to_scene("mass.theta_E") == "planes/0/mass/0/theta_E"
    assert pm.old_to_scene("shear.gamma1") == "planes/0/mass/1/gamma1"
    assert pm.old_to_scene("LL2.center_x") == "planes/0/light/2/center_x"
    assert pm.old_to_scene("srcS.Ie") == "planes/1/light/0/Ie"
    assert pm.old_to_scene("srcShp.beta") == "planes/1/light/1/beta"
    assert pm.scene_to_old("planes/1/light/1/beta") == "srcShp.beta"
    with pytest.raises(KeyError):
        pm.old_to_scene("nope.theta_E")
    with pytest.raises(KeyError):
        pm.scene_to_old("planes/2/light/0/beta")


def test_value_roundtrip_identity_and_ie_scale():
    rng = np.random.default_rng(1)
    labeled = {lab: float(v) for lab, v in
               zip(pm.OLD_LABELS_46, rng.uniform(0.1, 2.0, 46))}
    for ie_scale in (1.0, 1.0 / 0.0169):
        uniq = pm.unique_from_old(labeled, ie_scale=ie_scale)
        back = pm.old_from_unique(uniq, ie_scale=ie_scale)
        assert max(abs(back[k] - labeled[k]) for k in labeled) < 1e-14
        # Ie leaves actually scaled in scene space
        assert np.isclose(uniq["planes/1/light/0/Ie"],
                          labeled["srcS.Ie"] * ie_scale)


def test_vec_roundtrip():
    rng = np.random.default_rng(2)
    v = rng.standard_normal(46)
    lab = pm.labeled_from_vec46(v)
    v2 = pm.vec46_from_labeled(lab)
    assert np.array_equal(v, v2)
    with pytest.raises(ValueError):
        pm.labeled_from_vec46(np.zeros(45))


def test_grad_chain_rule_ie():
    """x_scene = s*x_old on Ie => dL/dx_old = s * dL/dx_scene."""
    rng = np.random.default_rng(3)
    g_scene = {pm.old_to_scene(lab): rng.standard_normal()
               for lab in pm.OLD_LABELS_46}
    s = 1.0 / 0.0169
    g_old = pm.grad_old_from_scene(g_scene, ie_scale=s)
    for k, lab in enumerate(pm.OLD_LABELS_46):
        expect = g_scene[pm.old_to_scene(lab)]
        if lab.endswith(".Ie"):
            expect *= s
        assert np.isclose(g_old[k], expect, rtol=0, atol=0)
