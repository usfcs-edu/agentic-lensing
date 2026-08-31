import hashlib
import json
from pathlib import Path
import math

import numpy as np
import pytest
from PIL import Image

from lensmark.critique import submit_critique
from lensmark.exports import cli_export, exportable_items, run_export
from lensmark.exports.coco import CATEGORIES, to_coco
from lensmark.exports.ds9 import HEADER, export_ds9, parse_ds9, to_ds9
from lensmark.exports.fewshot import example_markdown, select_fewshot
from lensmark.exports.masks import mask_array, mask_image
from lensmark.model import Critique, CritiquePanel, RenderInfo, Wcs
from lensmark.store import Campaign
from synth_campaign import accepted_file, arrow, mask, note, ring, seed_run

W = H = 403

def _jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))



def deck01(c: Campaign):
    return accepted_file(c, "deck-01", [
        arrow("ann-arrow-001", [0.5, 0.5], [0.7, 0.5], "deflector", "green"),
        arrow("ann-arrow-002", [0.446, 0.586], [0.41, 0.70], "tight arc", "cyan", status="rejected"),
        mask("ann-mask-001", [0.043, 0.352], 1.28, "galaxy"),
        mask("ann-mask-002", [0.64, 0.113], 0.42, "star", status="edited"),
        mask("ann-mask-003", [0.9, 0.9], 0.5, "galaxy", status="proposed"),
        ring("ann-ring-001", [0.514, 0.494], 1.5, center_ref="ann-arrow-001", label="theta_E ~ 1.5\""),
        note("ann-text-001", [0.06, 0.94], "seeing 1.1\""),
    ], verdict="likely_lens", rank=91, description="The cyan arrow marks a tight arc.", theta_e=1.5)


# ----------------------------------------------------------------------------- COCO
def test_coco_document(nine):
    c = Campaign(nine)
    f1 = deck01(c)
    f2 = accepted_file(c, "deck-02", [arrow("ann-arrow-001", [0.3, 0.3], [0.2, 0.1], "arc", "magenta")], verdict="possible", rank=89)
    doc = to_coco([f1, f2])
    assert [im["id"] for im in doc["images"]] == [1, 2] and doc["images"][0]["lensmark_id"] == "deck-01"
    assert doc["images"][0]["attributes"]["theta_e_arcsec"] == 1.5 and doc["images"][0]["width"] == W
    assert {cat["name"] for cat in doc["categories"]} == {"arrow", "field_galaxy_mask", "star_mask", "artifact_mask", "einstein_ring", "text"}
    assert doc["categories"] is CATEGORIES or doc["categories"] == CATEGORIES
    anns = doc["annotations"]
    ids = [a["id"] for a in anns]
    assert len(ids) == len(set(ids)) == 6                       # 5 accepted|edited on deck-01 + 1 on deck-02
    exported = {a["attributes"]["lensmark_item_id"] for a in anns if a["image_id"] == 1}
    assert exported == {"ann-arrow-001", "ann-mask-001", "ann-mask-002", "ann-ring-001", "ann-text-001"}
    for a in anns:
        x, y, w, h = a["bbox"]
        assert 0 <= x <= W and 0 <= y <= H and w >= 0 and h >= 0 and x + w <= W + 1e-6 and y + h <= H + 1e-6
        assert a["attributes"]["created_by"]["kind"] == "human" and "color" in a["attributes"]
    arr = next(a for a in anns if a["attributes"]["lensmark_item_id"] == "ann-arrow-001")
    assert arr["category_id"] == 1 and len(arr["keypoints"]) == 6 and arr["num_keypoints"] == 2
    assert arr["keypoints"][3:5] == pytest.approx([0.5 * W, 0.5 * H], abs=0.01)
    gal = next(a for a in anns if a["attributes"]["lensmark_item_id"] == "ann-mask-001")
    r_px = 1.28 / (16 / W)
    assert gal["category_id"] == 2 and gal["area"] == pytest.approx(math.pi * r_px ** 2, rel=1e-3)
    assert len(gal["segmentation"][0]) == 128 and gal["bbox"][0] == 0.0     # clipped at the left edge
    assert gal["attributes"]["radius_arcsec"] == 1.28 and gal["attributes"]["kind"] == "galaxy"
    star = next(a for a in anns if a["attributes"]["lensmark_item_id"] == "ann-mask-002")
    assert star["category_id"] == 3 and star["attributes"]["status"] == "edited"
    rng = next(a for a in anns if a["attributes"]["lensmark_item_id"] == "ann-ring-001")
    assert rng["category_id"] == 5 and rng["attributes"]["theta_e_arcsec"] == 1.5
    assert rng["attributes"]["center_uv"] == [0.5, 0.5]                     # follows center_ref to the deflector head
    txt = next(a for a in anns if a["attributes"]["lensmark_item_id"] == "ann-text-001")
    assert txt["category_id"] == 6 and txt["attributes"]["text"] == "seeing 1.1\""
    paths = run_export(c, "coco")
    assert paths == [c.exports_dir / "coco" / "instances.json"]
    assert _jload(paths[0])["annotations"][0]["id"] == 1


# ----------------------------------------------------------------------------- DS9
def test_ds9_image_coords_and_round_trip(nine):
    c = Campaign(nine)
    f = deck01(c)
    f.items.append(mask("ann-mask-004", [0.25, 0.25], 1.0, "galaxy"))
    text = to_ds9(f)
    lines = text.splitlines()
    assert lines[0] == HEADER and lines[1].startswith("# lensmark deck-01") and lines[2].startswith("global ")
    assert lines[3] == "image"
    regs = parse_ds9(text)
    exported = {it.id for it in exportable_items(f)}
    assert {r["id"] for r in regs} == exported and len(regs) == len(exported)
    known = next(r for r in regs if r["id"] == "ann-mask-004")
    # 1-based, y up: x = u*W + 0.5, y = (1-v)*H + 0.5 ; radius in display px
    assert known["shape"] == "circle" and known["coordsys"] == "image"
    assert known["coords"] == pytest.approx([0.25 * W + 0.5, 0.75 * H + 0.5, 1.0 / (16 / W)], abs=1e-3)
    assert known["color"] == "red" and known["props"]["dashlist"] == "8 3" and known["props"]["dash"] == "1"
    assert {"mask", "galaxy"} <= set(known["tags"])
    star = next(r for r in regs if r["id"] == "ann-mask-002")
    assert star["props"]["dashlist"] == "2 6" and "star" in star["tags"]
    rng = next(r for r in regs if r["id"] == "ann-ring-001")
    assert rng["props"]["dashlist"] == "1 3" and rng["color"] == "white" and "einstein_ring" in rng["tags"]
    assert rng["coords"][:2] == pytest.approx([0.5 * W + 0.5, 0.5 * H + 0.5], abs=1e-3)   # center_ref -> deflector head
    arr = next(r for r in regs if r["id"] == "ann-arrow-001")
    assert arr["shape"] == "line" and arr["props"]["line"] == "0 1" and arr["color"] == "green" and arr["text"] == "deflector"
    assert arr["coords"] == pytest.approx([0.7 * W + 0.5, 0.5 * H + 0.5, 0.5 * W + 0.5, 0.5 * H + 0.5], abs=1e-3)
    txt = next(r for r in regs if r["id"] == "ann-text-001")
    assert txt["shape"] == "text" and txt["text"] == "seeing 1.1\""
    assert "ann-arrow-002" not in text and "ann-mask-003" not in text    # rejected / proposed never exported
    paths = export_ds9(c)
    assert paths == [c.exports_dir / "ds9" / "deck-01.reg"] and paths[0].read_text().startswith(HEADER)


def test_ds9_fk5_when_wcs(nine):
    c = Campaign(nine)
    f = accepted_file(c, "deck-03", [mask("ann-mask-001", [0.5, 0.5], 1.0, "galaxy"), mask("ann-mask-002", [0.25, 0.5], 1.0, "star")])
    f.image.wcs = Wcs(ra_deg=150.0, dec_deg=2.0)
    text = to_ds9(f)
    assert "\nfk5\n" in text and "fk5 from image.wcs" in text
    regs = {r["id"]: r for r in parse_ds9(text)}
    assert regs["ann-mask-001"]["coordsys"] == "fk5"
    assert regs["ann-mask-001"]["coords"] == pytest.approx([150.0, 2.0, 1.0], abs=1e-6)
    dE = (0.5 - 0.25) * 16.0                                         # east_left: u < 0.5 is east
    assert regs["ann-mask-002"]["coords"][0] == pytest.approx(150.0 + dE / 3600 / math.cos(math.radians(2.0)), abs=1e-6)
    assert regs["ann-mask-002"]["coords"][1] == pytest.approx(2.0, abs=1e-6)
    assert 'circle(150.0000000,2.0000000,1.000")' in text


def test_parse_ds9_handles_other_syntax():
    regs = parse_ds9('# Region file format: DS9 version 4.1\nphysical\ncircle(10,20,3) # color=#FF9500 tag={id:x-1} tag={mask}\n'
                     '-line(1,2,3,4) # line=0 1 text={a b}\n# text(5,6) text={hello}\nbox(1,2,3,4,0)\n')
    assert regs[0]["coordsys"] == "physical" and regs[0]["color"] == "#FF9500" and regs[0]["id"] == "x-1"
    assert regs[1]["shape"] == "line" and regs[1]["include"] is False and regs[1]["text"] == "a b" and regs[1]["props"]["line"] == "0 1"
    assert regs[2]["shape"] == "text" and regs[2]["text"] == "hello" and regs[3]["shape"] == "box"


# ----------------------------------------------------------------------------- masks
def _expected_px(radii_arcsec, scale):
    return sum(math.pi * (r / scale) ** 2 for r in radii_arcsec)


def test_masks_native_and_display_scale(nine):
    c = Campaign(nine)
    items = [mask("ann-mask-001", [0.3, 0.3], 1.5, "galaxy"), mask("ann-mask-002", [0.7, 0.7], 2.0, "star"),
             mask("ann-mask-003", [0.5, 0.1], 0.8, "artifact", status="rejected"),
             arrow("ann-arrow-001", [0.5, 0.5], [0.7, 0.5], "deflector", "green")]
    f = accepted_file(c, "deck-04", items, native_scale=0.05)              # 16" / 0.05 = 320 px
    im = mask_image(f)
    assert im.mode == "L" and im.size == (320, 320)
    a = np.asarray(im)
    assert set(np.unique(a)) <= {0, 255}
    assert (a == 255).sum() == pytest.approx(_expected_px([1.5, 2.0], 0.05), rel=0.03)
    # display scale when no native scale is recorded
    g = accepted_file(c, "deck-05", [mask("ann-mask-001", [0.3, 0.3], 1.5, "galaxy"), mask("ann-mask-002", [0.7, 0.7], 2.0, "star")])
    assert g.image.native_pixel_scale_arcsec is None
    b = mask_array(g)
    assert b.shape == (H, W)
    assert b.sum() == pytest.approx(_expected_px([1.5, 2.0], 16 / W), rel=0.03)
    paths = run_export(c, "masks")
    assert [p.name for p in paths] == ["deck-04.mask.png", "deck-05.mask.png"]
    with Image.open(paths[0]) as im2:
        assert im2.size == (320, 320) and im2.mode == "L"
    assert run_export(c, "masks", ids=["deck-01"]) == []                   # no JSON -> nothing


# ----------------------------------------------------------------------------- few-shot
def _seed_fewshot(c: Campaign):
    specs = {"deck-01": (91, "likely_lens"), "deck-02": (89, "possible"), "deck-03": (68, "likely_lens"),
             "deck-04": (95, "possible"), "deck-05": (60, "likely_lens")}
    for iid, (rank, verdict) in specs.items():
        accepted_file(c, iid, [arrow("ann-arrow-001", [0.5, 0.5], [0.7, 0.5], "deflector", "green"),
                               arrow("ann-arrow-002", [0.4, 0.6], [0.3, 0.75], "arc", "cyan"),
                               mask("ann-mask-001", [0.1, 0.1], 1.0, "galaxy"),
                               ring("ann-ring-001", [0.5, 0.5], 1.5, center_ref="ann-arrow-001")],
                      verdict=verdict, rank=rank, description=f"The cyan arrow marks an arc on {iid}.", theta_e=1.5)
    # deck-06 still has a proposed item -> not finished; deck-07 has no JSON at all
    seed_run(c, "deck-06", "r6", [mask("ann-mask-001", [0.2, 0.2], 1.0)],
             human_items=[arrow("ann-arrow-001", [0.5, 0.5], [0.7, 0.5], "deflector", "green")])
    # deck-08 has only rejected items
    accepted_file(c, "deck-08", [arrow("ann-arrow-001", [0.5, 0.5], [0.7, 0.5], "x", "green", status="rejected")])


def test_fewshot_selection_is_stratified_and_ordered(nine):
    c = Campaign(nine)
    _seed_fewshot(c)
    # rank order: 05(60,L) 03(68,L) 02(89,P) 01(91,L) 04(95,P); round-robin over verdict -> L,P,L,P ...
    assert [f.id for f in select_fewshot(c, k=2)] == ["deck-05", "deck-02"]
    assert [f.id for f in select_fewshot(c, k=4)] == ["deck-05", "deck-03", "deck-02", "deck-04"]
    assert [f.id for f in select_fewshot(c, k=10)] == ["deck-05", "deck-03", "deck-02", "deck-01", "deck-04"]


def test_fewshot_bundle_is_content_addressed(nine, tmp_path):
    c = Campaign(nine)
    _seed_fewshot(c)
    # a fresh (non-stale) annotated PNG for deck-05 -> copied; the others have none
    f5 = c.load("deck-05")
    Image.new("RGB", (W, H)).save(c.annot_path("deck-05"))
    f5.render = RenderInfo(renderer="test", output="deck-05.annot.png", of_json_sha256=f5.content_sha256())
    c.save("deck-05", f5, touch_modified=False)
    assert not c.annot_stale("deck-05")
    paths = run_export(c, "fewshot", k=4)
    out = c.exports_dir / "fewshot"
    manifest = _jload(out / "manifest.json")
    assert manifest["schema_version"] == "lensmark-fewshot/1.0" and manifest["k"] == 4 and len(manifest["examples"]) == 4
    assert [e["id"] for e in manifest["examples"]] == ["deck-05", "deck-03", "deck-02", "deck-04"]
    ex = manifest["examples"][0]
    assert ex == {"id": "deck-05", "png": "001-deck-05.png", "annot": "001-deck-05.annot.png", "json": "001-deck-05.lensmark.json",
                  "md": "001-deck-05.md", "rank": 60, "verdict": "likely_lens"}
    assert manifest["examples"][1]["annot"] is None
    for e in manifest["examples"]:
        for key in ("png", "json", "md"):
            assert (out / e[key]).exists() and (out / e[key]) in paths
    assert (out / "001-deck-05.annot.png").exists() and not (out / "002-deck-03.annot.png").exists()
    sha = (out / "prompt.sha256").read_text().strip()
    assert sha == manifest["prompt_sha256"]
    h = hashlib.sha256()
    for e in manifest["examples"]:
        h.update((out / e["json"]).read_bytes())
        h.update((out / e["md"]).read_bytes())
    assert h.hexdigest() == sha
    md = (out / "001-deck-05.md").read_text()
    assert md.startswith("# deck-05 - rank 60, likely_lens, theta_E ~ 1.50") and "- cyan arrow: arc" in md and "- green arrow: deflector" in md
    assert "Masks: 1 galaxy (dashed)" in md and "Einstein ring: theta_E = 1.50" in md
    assert _jload(out / "001-deck-05.lensmark.json")["id"] == "deck-05"
    # second build elsewhere -> identical sha (content addressed)
    run_export(c, "fewshot", k=4, out=tmp_path / "again")
    assert (tmp_path / "again" / "prompt.sha256").read_text().strip() == sha
    # a different k that changes the set -> different sha; stale numbered files are cleared
    run_export(c, "fewshot", k=2)
    assert (out / "prompt.sha256").read_text().strip() != sha and not (out / "003-deck-02.md").exists()


def test_fewshot_require_flag_uses_latest_critique(nine):
    c = Campaign(nine)
    _seed_fewshot(c)

    def crit(iid, flag, at):
        return Critique(image_id=iid, run_id="r0", reviewer="xhuang", reviewed_at=at, panel=CritiquePanel(would_use_as_fewshot=flag))

    submit_critique(c, crit("deck-05", True, "2026-08-30T10:00:00Z"))
    submit_critique(c, crit("deck-03", True, "2026-08-30T10:00:00Z"))
    submit_critique(c, Critique(image_id="deck-03", run_id="r1", reviewer="xhuang", reviewed_at="2026-08-30T11:00:00Z",
                                panel=CritiquePanel(would_use_as_fewshot=False)))
    submit_critique(c, crit("deck-02", False, "2026-08-30T10:00:00Z"))
    assert [f.id for f in select_fewshot(c, k=6, require_flag=True)] == ["deck-05"]
    assert len(select_fewshot(c, k=6, require_flag=False)) == 5
    assert cli_export(str(nine), "fewshot", k=6, require_flag=True) == 0
    assert cli_export(str(nine), "coco", ids=["deck-07"]) == 1              # nothing to export


def test_example_markdown_without_items_sections(nine):
    c = Campaign(nine)
    f = accepted_file(c, "deck-09", [note("ann-text-001", [0.1, 0.9], "seeing 1.1\"")], description="")
    md = example_markdown(f)
    assert md.startswith("# deck-09 - rank 40") and "(no description)" in md and "Note: \"seeing 1.1\"\"" in md and "Arrows:" not in md
