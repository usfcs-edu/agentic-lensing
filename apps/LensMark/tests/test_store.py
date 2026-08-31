import json

import pytest

from lensmark.model import Arrow, LensMarkFile, MaskCircle
from lensmark.store import Campaign, diff_events


def test_list_ids_and_new_file(nine):
    c = Campaign(nine)
    assert c.list_ids() == [f"deck-{i:02d}" for i in range(1, 10)]
    f = c.new_file("deck-01")
    assert f.image.width == f.image.height
    assert f.image.cutout_arcsec == 16.0 and f.image.scale_source == "assumed"
    assert f.image.pixel_scale_arcsec == pytest.approx(16.0 / f.image.width)
    assert f.system.rank == 91 and f.system.theta_e.value_arcsec == 1.5
    assert len(f.image.sha256) == 64
    assert not c.exists("deck-01") and c.load("deck-01") is None


def test_save_load_and_log(nine):
    c = Campaign(nine)
    f = c.new_file("deck-02")
    f.items.append(Arrow(id="ann-arrow-001", tail=[0.4, 0.7], head=[0.45, 0.6], color="cyan", label="arc"))
    c.save("deck-02", f, actor="test")
    assert c.json_path("deck-02").exists() and c.log_path("deck-02").exists()
    g = c.load("deck-02")
    assert g.items[0].label == "arc" and g.provenance.log == "deck-02.lensmark.log.jsonl"
    g.items.append(MaskCircle(id="ann-mask-001", center=[0.1, 0.1], radius_arcsec=0.5, kind="star"))
    g.items[0].label = "tight arc"
    g.system.description = "hi"
    c.save("deck-02", g, actor="test")
    ops = [(e["op"], e["item_id"]) for e in c.read_log("deck-02")]
    assert ("add", "ann-arrow-001") in ops
    assert ("update", "ann-arrow-001") in ops and ("add", "ann-mask-001") in ops and ("update", "$system") in ops
    assert all(e["actor"] == "test" and "ts" in e for e in c.read_log("deck-02"))
    # derived files: annot missing -> stale; manifest summarises
    assert c.annot_stale("deck-02")
    rows = c.write_manifest()
    row = next(r for r in rows if r["id"] == "deck-02")
    assert row["has_json"] and row["n_items"] == 2 and row["annot_stale"] and row["by_status"] == {"accepted": 2}
    assert json.load(open(c.manifest_path))["images"][1]["id"] == "deck-02"


def test_atomic_write_leaves_no_tmp(nine):
    c = Campaign(nine)
    c.save("deck-03", c.new_file("deck-03"))
    assert not list(nine.glob("*.tmp"))


def test_id_mismatch_rejected(nine):
    c = Campaign(nine)
    with pytest.raises(ValueError):
        c.save("deck-04", c.new_file("deck-05"))


def test_diff_events_delete():
    from lensmark.model import ImageMeta
    img = ImageMeta(file="x.png", sha256="0" * 64, width=10, height=10, cutout_arcsec=1.0, pixel_scale_arcsec=0.1)
    a = LensMarkFile(id="x", image=img, items=[Arrow(id="a1", tail=[0, 0], head=[1, 1])])
    b = LensMarkFile(id="x", image=img, items=[])
    assert [(e["op"], e["item_id"]) for e in diff_events(a, b)] == [("delete", "a1")]
    assert [e["op"] for e in diff_events(None, b)][0] == "create"


def test_overrides_and_ignores_derived_files(nine):
    (nine / "deck-01.annot.png").write_bytes(b"x")
    (nine / "deck-01.mask.png").write_bytes(b"x")
    c = Campaign(nine)
    assert "deck-01.annot" not in c.list_ids() and len(c.list_ids()) == 9
    cfg = c.config
    cfg["overrides"]["deck-01"]["cutout_arcsec"] = 20.0
    c.save_config(cfg)
    c2 = Campaign(nine)
    f = c2.new_file("deck-01")
    assert f.image.cutout_arcsec == 20.0 and f.image.scale_source == "override"
