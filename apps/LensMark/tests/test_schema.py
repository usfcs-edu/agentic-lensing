import json

import pytest
from pydantic import ValidationError

from lensmark import config
from lensmark.model import (Arrow, EinsteinRing, ImageMeta, LensMarkFile, MaskCircle, Proposal, RenderInfo,
                            TextNote, parse_item)

SHA = "0" * 64


def image(W=410, H=410, cut=16.0):
    return ImageMeta(file="x.png", sha256=SHA, width=W, height=H, cutout_arcsec=cut, pixel_scale_arcsec=cut / W)


def sample() -> LensMarkFile:
    f = LensMarkFile(id="deck-01", image=image())
    f.items.append(Arrow(id="ann-arrow-001", tail=[0.41, 0.70], head=[0.446, 0.586], color="cyan", label="tight arc"))
    f.items.append(Arrow(id="ann-arrow-002", tail=[0.76, 0.494], head=[0.586, 0.494], color="green", label="deflector"))
    f.items.append(MaskCircle(id="ann-mask-001", center=[0.043, 0.352], radius_arcsec=1.28, kind="galaxy"))
    f.items.append(MaskCircle(id="ann-mask-002", center=[0.64, 0.113], radius_arcsec=0.42, kind="star"))
    f.items.append(EinsteinRing(id="ann-ring-001", center=[0.514, 0.494], theta_e_arcsec=1.5, center_ref="ann-arrow-002",
                                label="θ_E ≈ 1.5″"))
    f.items.append(TextNote(id="ann-text-001", pos=[0.06, 0.94], text="seeing 1.1″"))
    f.system.description = "The cyan arrow marks a tight arc around the green deflector."
    return f


def test_round_trip_is_lossless():
    f = sample()
    text = f.to_json()
    g = LensMarkFile.from_json(text)
    assert g == f
    assert g.to_json() == text
    d = json.loads(text)
    assert d["schema_version"] == "lensmark/1.0" and d["items"][2]["type"] == "mask_circle"
    assert d["style_defaults"] == config.STYLE_DEFAULTS and d["palette"] == config.PALETTE_VERSION


def test_extra_keys_are_rejected():
    d = sample().to_dict()
    d["items"][0]["smuggled"] = 1
    with pytest.raises(ValidationError):
        LensMarkFile.model_validate(d)
    d = sample().to_dict()
    d["surprise"] = True
    with pytest.raises(ValidationError):
        LensMarkFile.model_validate(d)


def test_duplicate_ids_rejected():
    f = sample()
    f.items.append(TextNote(id="ann-text-001", pos=[0.1, 0.1], text="dup"))
    with pytest.raises(ValidationError):
        LensMarkFile.model_validate(f.to_dict())


def test_pixel_scale_must_match_cutout_over_width():
    with pytest.raises(ValidationError):
        ImageMeta(file="x.png", sha256=SHA, width=410, height=410, cutout_arcsec=16.0, pixel_scale_arcsec=0.05)


def test_unknown_type_and_colour_rejected():
    with pytest.raises(ValueError):
        parse_item({"type": "blob", "id": "x"})
    with pytest.raises(ValidationError):
        Arrow(id="a", tail=[0, 0], head=[1, 1], color="purple")
    with pytest.raises(ValidationError):
        MaskCircle(id="m", center=[0.5, 0.5], radius_arcsec=0, kind="galaxy")


def test_defaults_per_type():
    assert MaskCircle(id="m", center=[0.5, 0.5], radius_arcsec=1, kind="star").color == "mask_red"
    assert EinsteinRing(id="r", center=[0.5, 0.5], theta_e_arcsec=1).color == "ring_white"
    assert Arrow(id="a", tail=[0, 0], head=[1, 1]).status == "accepted"
    assert Arrow(id="a", tail=[0, 0], head=[1, 1]).created_by.kind == "human"


def test_next_id_and_lookup():
    f = sample()
    assert f.next_id("arrow") == "ann-arrow-003"
    assert f.next_id("mask_circle") == "ann-mask-003"
    assert f.next_id("einstein_ring") == "ann-ring-002"
    assert f.item("ann-ring-001").theta_e_arcsec == 1.5
    assert f.item("nope") is None


def test_lint_flags_dangling_colour_words_and_refs():
    f = sample()
    f.system.description = "The magenta arrow marks a knot; the cyan arrow an arc."
    f.items[4].center_ref = "ann-arrow-999"
    warns = f.lint()
    assert any("magenta" in w for w in warns)
    assert not any("cyan" in w for w in warns)
    assert any("center_ref" in w for w in warns)


def test_content_sha_ignores_render_block():
    f = sample()
    s1 = f.content_sha256()
    f.render = RenderInfo(renderer="x", output="deck-01.annot.png", of_json_sha256=s1)
    f = LensMarkFile.model_validate(f.to_dict())
    assert f.content_sha256() == s1
    f.items[0].label = "changed"
    assert f.content_sha256() != s1


def test_proposal_document_is_lenient():
    p = Proposal.model_validate({"system": {"verdict": "likely_lens", "description": "x", "bonus": 1},
                                 "items": [{"type": "arrow", "head": [0.5, 0.5], "tail": [0.6, 0.6], "extra": "ignored"}]})
    assert p.items[0].head == [0.5, 0.5] and p.system.verdict == "likely_lens"


def test_proposal_json_schema_is_flat_and_closed():
    schema = json.load(open(config.SCHEMA_DIR / "lensmark-proposal-1.0.schema.json"))
    assert '"$ref"' not in json.dumps(schema)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["items"]["items"]["additionalProperties"] is False
