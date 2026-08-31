import pytest

from lensmark.model import CreatedBy, ImageMeta, LensMarkFile, Proposal
from lensmark.validate import nearest_palette, validate_proposal

IMG = ImageMeta(file="x.png", sha256="0" * 64, width=400, height=400, cutout_arcsec=16.0, pixel_scale_arcsec=0.04)
BY = CreatedBy(kind="claude", model="claude-opus-5", effort="xhigh", run_id="run-test")


def prop(items, system=None):
    return Proposal.model_validate({"system": system or {"verdict": "likely_lens", "description": "x"}, "items": items})


def test_happy_path_mints_ids_and_defaults():
    f = LensMarkFile(id="x", image=IMG)
    res = validate_proposal(prop([
        {"type": "arrow", "head": [0.5, 0.5], "tail": [0.7, 0.5], "label": "deflector", "color": "cyan"},
        {"type": "arrow", "head": [0.4, 0.6], "tail": [0.3, 0.8], "label": "tight arc"},
        {"type": "mask_circle", "center": [0.1, 0.1], "radius_arcsec": 1.0, "kind": "galaxy"},
        {"type": "einstein_ring", "theta_e_arcsec": 1.5},
        {"type": "text", "pos": [0.05, 0.95], "text": "note"},
    ]), f, BY)
    ids = [it.id for it in res.items]
    assert ids == ["ann-arrow-001", "ann-arrow-002", "ann-mask-001", "ann-ring-001", "ann-text-001"]
    defl, arc, mask, ring, note = res.items
    assert defl.color == "green" and any(r["why"] == "deflector_is_green" for r in res.repairs)
    assert arc.color != "green" and any(r["why"] == "auto_palette" for r in res.repairs)
    assert ring.center == [0.5, 0.5] and any(r["why"] == "default_center" for r in res.repairs)  # deflector head
    assert all(it.status == "proposed" and it.created_by.run_id == "run-test" for it in res.items)
    assert res.n_invalid == 0


def test_clamp_and_out_of_bounds_and_degenerate():
    f = LensMarkFile(id="x", image=IMG)
    res = validate_proposal(prop([
        {"type": "arrow", "head": [1.02, 0.5], "tail": [0.9, 0.5]},                # clamped
        {"type": "arrow", "head": [1.5, 0.5], "tail": [0.9, 0.5]},                 # out of bounds -> invalid
        {"type": "arrow", "head": [0.5, 0.5], "tail": [0.505, 0.5]},               # degenerate
        {"type": "mask_circle", "center": [0.5, 0.5], "radius_arcsec": 40, "kind": "star"},   # bad radius
        {"type": "mask_circle", "center": [0.5, 0.5], "radius_arcsec": 0.5},       # missing kind -> galaxy
        {"type": "einstein_ring", "center": [0.5, 0.5], "theta_e_arcsec": 12},     # implausible
        {"type": "blob"},
    ]), f, BY)
    by = {it.id: it for it in res.items}
    assert by["ann-arrow-001"].head == [1.0, 0.5] and by["ann-arrow-001"].status == "proposed"
    assert by["ann-arrow-002"].status == "invalid" and by["ann-arrow-002"].invalid_reason == "out_of_bounds"
    assert by["ann-arrow-003"].invalid_reason == "degenerate_arrow"
    assert by["ann-mask-001"].invalid_reason == "bad_radius"
    assert by["ann-mask-002"].kind == "galaxy" and by["ann-mask-002"].status == "proposed"
    assert by["ann-ring-001"].invalid_reason == "theta_e_implausible"
    assert any(i["reason"].startswith("unknown_type") for i in res.invalid)
    assert res.n_invalid == 5 and res.n_repaired >= 2


def test_missing_tail_is_synthesized_outside():
    f = LensMarkFile(id="x", image=IMG)
    res = validate_proposal(prop([{"type": "arrow", "head": [0.6, 0.4]}]), f, BY)
    a = res.items[0]
    assert a.status == "proposed"
    # tail lies further from the centre than the head
    assert (a.tail[0] - 0.5) ** 2 + (a.tail[1] - 0.5) ** 2 > (a.head[0] - 0.5) ** 2 + (a.head[1] - 0.5) ** 2


def test_mask_cap_flags_not_drops():
    f = LensMarkFile(id="x", image=IMG)
    items = [{"type": "mask_circle", "center": [i / 20, 0.5], "radius_arcsec": 0.5, "kind": "star"} for i in range(15)]
    res = validate_proposal(prop(items), f, BY, mask_cap=12)
    assert len(res.items) == 15
    assert sum(1 for it in res.items if it.invalid_reason == "over_cap") == 3


def test_colour_snapping():
    assert nearest_palette("#ff00ff", allowed=("magenta", "cyan")) == "magenta"
    assert nearest_palette("purple", allowed=("magenta", "cyan", "green")) == "magenta"
    assert nearest_palette("grey", allowed=("gray", "white")) == "gray"
    assert nearest_palette("nonsense", allowed=("cyan", "gray")) == "cyan"


def test_ids_do_not_collide_with_existing_items():
    from lensmark.model import Arrow
    f = LensMarkFile(id="x", image=IMG, items=[Arrow(id="ann-arrow-001", tail=[0, 0], head=[0.5, 0.5])])
    res = validate_proposal(prop([{"type": "arrow", "head": [0.2, 0.2], "tail": [0.1, 0.1]}]), f, BY)
    assert res.items[0].id == "ann-arrow-002"
