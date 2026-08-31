"""Golden + behavioural tests for the canonical PIL renderer (``lensmark/render``).

(a) byte-level goldens of the nine hand-authored examples (``--update-golden`` rewrites
    tests/golden/sha256.json and the reference PNGs; a mismatch writes <id>.diff.png and reports
    max |dRGB| / % pixels within 8 so a Pillow bump is diagnosable), (b) render-twice identity,
(c) scale invariance, (d) colour-mask geometry asserts on a synthetic black base,
(e)-(g) render_to_file pinning / staleness, sha mismatch, ``lensmark render --check``.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import PIL
import pytest
from PIL import Image, ImageChops

from lensmark import config
from lensmark.model import Arrow, EinsteinRing, ImageMeta, LensMarkFile, MaskCircle, TextNote
from lensmark.render import cli_render, is_stale, label_boxes, png_bytes, render_image, render_png_bytes, render_to_file
from lensmark.render import primitives as P
from lensmark.render.draw import load_base
from lensmark.store import Campaign

GOLDEN = Path(__file__).resolve().parent / "golden"
SHA_FILE = GOLDEN / "sha256.json"
IDS = [f"deck-{i:02d}" for i in range(1, 10)]
W = 400                                       # synthetic base size
M = W                                         # min(W, H) for the synthetic base


# ----------------------------------------------------------------------------- helpers
def _render_bytes(campaign: Campaign, image_id: str) -> bytes:
    file = campaign.load(image_id)
    assert file is not None
    return png_bytes(render_image(file, load_base(campaign, file)))


def _diff_stats(a: Image.Image, b: Image.Image) -> tuple[int, float]:
    da = np.asarray(a.convert("RGB")).astype(int)
    db = np.asarray(b.convert("RGB")).astype(int)
    d = np.abs(da - db).max(axis=-1)
    return int(d.max()), float((d <= 8).mean() * 100.0)


def _black() -> Image.Image:
    return Image.new("RGB", (W, W), (0, 0, 0))


def _synthetic(items, *, legend=False) -> LensMarkFile:
    img = ImageMeta(file="synthetic.png", sha256="0" * 64, width=W, height=W, cutout_arcsec=16.0,
                    pixel_scale_arcsec=16.0 / W)
    f = LensMarkFile(id="synthetic", image=img, items=items)
    f.legend.show = legend
    return f


def _mask(img: Image.Image, colour, box=None, tol: int = 0):
    """Boolean (y, x) mask of pixels within ``tol`` of ``colour`` (reproductions/lensjudge/tests/test_golden_annotate.py:81)."""
    a = np.asarray(img.convert("RGB")).astype(int)
    m = np.all(np.abs(a - np.array(colour)) <= tol, axis=-1)
    if box is not None:
        x0, y0, x1, y1 = box
        sub = np.zeros_like(m)
        sub[y0:y1, x0:x1] = m[y0:y1, x0:x1]
        m = sub
    return m


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    assert len(xs) > 0, "no pixels of that colour"
    return float(xs.mean()), float(ys.mean())


def _n_components(mask) -> int:
    """8-connected components of a boolean mask (pure Python BFS; the masks are small)."""
    H, Wd = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    n = 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        n += 1
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        while q:
            y, x = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < Wd and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        q.append((yy, xx))
    return n


def _annulus(cx: float, cy: float, r: float, half_w: float):
    yy, xx = np.mgrid[0:W, 0:W]
    d = np.hypot(xx - cx, yy - cy)
    return (d >= r - half_w) & (d <= r + half_w)


def _bright(img: Image.Image, thresh: int = 60):
    return np.asarray(img.convert("RGB")).max(axis=-1) > thresh


def _rgb(name: str):
    return config.palette_rgb(name)


ST = config.STYLE_DEFAULTS


# ----------------------------------------------------------------------------- (a) goldens
def test_golden_sha256(nine_full, update_golden):
    c = Campaign(nine_full)
    got = {i: _render_bytes(c, i) for i in IDS}
    shas = {i: hashlib.sha256(b).hexdigest() for i, b in got.items()}
    if update_golden:
        GOLDEN.mkdir(exist_ok=True)
        SHA_FILE.write_text(json.dumps({"renderer": config.RENDERER_VERSION, "pillow": PIL.__version__,
                                        "sha256": shas}, indent=2) + "\n", encoding="utf-8")
        for i, b in got.items():
            (GOLDEN / f"{i}.annot.png").write_bytes(b)
        pytest.skip("golden shas + reference PNGs rewritten")
    assert SHA_FILE.exists(), "no tests/golden/sha256.json - run: pytest tests/test_render_golden.py --update-golden"
    want = json.loads(SHA_FILE.read_text(encoding="utf-8"))["sha256"]
    bad = []
    for i in IDS:
        if shas[i] == want.get(i):
            continue
        ref = GOLDEN / f"{i}.annot.png"
        if ref.exists():
            a = Image.open(io.BytesIO(got[i]))
            b = Image.open(ref)
            mx, pct = _diff_stats(a, b)
            ImageChops.difference(a.convert("RGB"), b.convert("RGB")).save(GOLDEN / f"{i}.diff.png")
            msg = f"{i}: sha mismatch - max |dRGB| = {mx}, {pct:.2f}% of pixels within 8 -> tests/golden/{i}.diff.png"
        else:
            msg = f"{i}: sha mismatch (no reference PNG to diff against)"
        print(msg)
        bad.append(msg)
    assert not bad, "\n".join(bad) + (f"\nPillow {PIL.__version__}, renderer {config.RENDERER_VERSION}; "
                                       "if the change is intended: pytest tests/test_render_golden.py --update-golden")


# ----------------------------------------------------------------------------- (b) determinism
def test_render_twice_identical(nine_full):
    c = Campaign(nine_full)
    for i in ("deck-01", "deck-07"):
        assert _render_bytes(c, i) == _render_bytes(c, i)
        assert render_png_bytes(c, i) == _render_bytes(c, i)
    assert not list(nine_full.glob("*.annot.png")), "render_png_bytes must not write files"


def test_png_has_no_metadata(nine_full):
    c = Campaign(nine_full)
    im = Image.open(io.BytesIO(_render_bytes(c, "deck-02")))
    assert im.format == "PNG" and im.mode == "RGB" and im.size == (403, 403)
    assert not im.text and "icc_profile" not in im.info and "dpi" not in im.info


# ----------------------------------------------------------------------------- (c) scale invariance
def test_scale_invariance(nine_full):
    c = Campaign(nine_full)
    file = c.load("deck-04")
    base = load_base(c, file)
    one = render_image(file, base, scale=1.0)
    two = render_image(file, base, scale=2.0)
    assert two.size == (806, 806)
    down = two.resize(one.size, Image.LANCZOS)
    d = np.abs(np.asarray(one).astype(int) - np.asarray(down).astype(int))
    assert d.mean() < 8.0, f"mean |dRGB| {d.mean():.2f}"
    boxes1 = {b["id"]: b["box"] for b in label_boxes(file, scale=1.0)}
    boxes2 = {b["id"]: b["box"] for b in label_boxes(file, scale=2.0)}
    for k, b1 in boxes1.items():
        assert all(abs(v1 * 2 - v2) < 3.0 for v1, v2 in zip(b1, boxes2[k])), k


# ----------------------------------------------------------------------------- (d) synthetic geometry
@pytest.mark.parametrize("tail,head,color", [
    ((0.20, 0.80), (0.50, 0.50), "cyan"),       # diagonal, pointing up-right
    ((0.10, 0.30), (0.40, 0.30), "magenta"),    # horizontal, pointing right
    ((0.70, 0.20), (0.70, 0.60), "green"),      # vertical, pointing down
])
def test_arrow_tip_and_head_centroid(tail, head, color):
    f = _synthetic([Arrow(id="a", tail=list(tail), head=list(head), color=color, label=None)])
    im = render_image(f, _black())
    st = ST["arrow"]
    tx, ty, hx, hy = tail[0] * W, tail[1] * W, head[0] * W, head[1] * W
    dx, dy = P.unit(hx - tx, hy - ty)
    tip_gap, head_len = st["tip_gap"] * M, st["head_len"] * M
    apex = (hx - dx * tip_gap, hy - dy * tip_gap)
    m = _mask(im, _rgb(color), tol=70)          # LANCZOS ringing: only flat interiors are exactly the palette colour
    ys, xs = np.nonzero(m)
    proj = (xs - tx) * dx + (ys - ty) * dy
    tip_i = int(np.argmax(proj))
    assert math.hypot(xs[tip_i] - apex[0], ys[tip_i] - apex[1]) <= 2.5, "arrow tip is not tip_gap short of the head"
    # head triangle only (past the base), centroid = apex - 2/3 head_len along the direction
    base_proj = ((apex[0] - dx * head_len) - tx) * dx + ((apex[1] - dy * head_len) - ty) * dy
    sel = proj > base_proj + 1.0
    cx, cy = float(xs[sel].mean()), float(ys[sel].mean())
    ex, ey = apex[0] - dx * head_len * 2.0 / 3.0, apex[1] - dy * head_len * 2.0 / 3.0
    assert math.hypot(cx - ex, cy - ey) <= 2.5, f"head centroid ({cx:.1f},{cy:.1f}) vs ({ex:.1f},{ey:.1f})"
    # nothing beyond the feature point itself
    assert proj.max() <= ((hx - tx) * dx + (hy - ty) * dy) + 0.5


def test_ring_dot_count_and_theta_label():
    f = _synthetic([EinsteinRing(id="r", center=[0.5, 0.5], theta_e_arcsec=2.0)])
    im = render_image(f, _black())
    r = 2.0 / f.image.pixel_scale_arcsec
    st = ST["einstein_ring"]
    spacing = (2.0 + st["gap_mult"]) * st["dot_r"] * M
    expected = P.dot_count(r, spacing)
    ring = _bright(im, 40) & _annulus(W / 2, W / 2, r, 3.0)
    n = _n_components(ring)
    assert abs(n - expected) <= 0.15 * expected, f"{n} dots, expected ~{expected}"
    # the θ label sits below-right of the ring, inside the image, and is drawn in white
    (box,) = [b["box"] for b in label_boxes(f) if b["kind"] == "theta"]
    assert box[0] > W / 2 + r * 0.3 and box[1] > W / 2 + r * 0.3 and box[2] <= W and box[3] <= W
    assert _mask(im, (255, 255, 255), tol=12, box=tuple(int(v) for v in box)).sum() > 20


def test_galaxy_dashes_vs_star_dots():
    f = _synthetic([MaskCircle(id="g", center=[0.25, 0.25], radius_arcsec=1.0, kind="galaxy"),
                    MaskCircle(id="s", center=[0.75, 0.75], radius_arcsec=1.0, kind="star")])
    im = render_image(f, _black())
    r = 1.0 / f.image.pixel_scale_arcsec
    red = _bright(im, 60)
    n_dash = _n_components(red & _annulus(0.25 * W, 0.25 * W, r, 4.0))
    n_dot = _n_components(red & _annulus(0.75 * W, 0.75 * W, r, 4.0))
    exp_dash = max(1, round(2 * math.pi * r / ((ST["mask_galaxy"]["dash_len"] + ST["mask_galaxy"]["gap_len"]) * M)))
    exp_dot = P.dot_count(r, (2.0 + ST["mask_star"]["gap_mult"]) * ST["mask_star"]["dot_r"] * M)
    assert abs(n_dash - exp_dash) <= 1, f"{n_dash} dashes, expected {exp_dash}"
    assert abs(n_dot - exp_dot) <= 0.15 * exp_dot, f"{n_dot} dots, expected ~{exp_dot}"
    assert n_dot > 2 * n_dash
    # both are mask_red (red-dominant everywhere); the dash phase starts at screen angle 0 (a dash covers the rightmost point)
    assert red[int(0.25 * W), int(0.25 * W + r)]
    a = np.asarray(im).astype(int)
    assert np.all(a[red][:, 0] >= a[red][:, 1]) and np.all(a[red][:, 0] >= a[red][:, 2])
    assert _mask(im, _rgb("mask_red"), tol=12).sum() > 30


def test_labels_inside_image_and_clamped():
    f = _synthetic([
        Arrow(id="right", tail=[0.97, 0.50], head=[0.80, 0.50], color="cyan", label="a long label at the right edge"),
        Arrow(id="top", tail=[0.50, 0.02], head=[0.50, 0.30], color="magenta", label="top"),
        Arrow(id="left", tail=[0.03, 0.85], head=[0.30, 0.70], color="yellow", label="another long label near the left"),
        Arrow(id="headside", tail=[0.55, 0.55], head=[0.60, 0.60], color="green", label="head", label_anchor="head"),
        Arrow(id="nudged", tail=[0.60, 0.20], head=[0.75, 0.25], color="orange", label="nudged", label_offset=[0.0, -0.05]),
        TextNote(id="note", pos=[0.99, 0.99], text="seeing 1.1″", color="white"),
        EinsteinRing(id="ring", center=[0.9, 0.9], theta_e_arcsec=1.0),
    ], legend=True)
    boxes = label_boxes(f)
    assert {b["id"] for b in boxes if b["kind"] == "label"} == {"right", "top", "left", "headside", "nudged"}
    for b in boxes:
        x0, y0, x1, y1 = b["box"]
        assert x0 >= 3.9 and y0 >= 3.9 and x1 <= W - 3.9 and y1 <= W - 3.9, (b["id"], b["kind"], b["box"])
    by_id = {b["id"]: b for b in boxes if b["id"]}
    assert by_id["headside"]["anchor"] == "head" and by_id["right"]["anchor"] in ("tail", "tail_side")
    im = render_image(f, _black())
    for color in ("cyan", "magenta", "yellow", "green", "orange"):
        assert _mask(im, _rgb(color)).sum() > 0, color
    assert (b for b in boxes if b["kind"] == "legend"), "legend present"


def test_status_visibility_and_include_proposed():
    f = _synthetic([
        Arrow(id="ok", tail=[0.2, 0.2], head=[0.4, 0.4], color="cyan", label=None, status="accepted"),
        Arrow(id="rej", tail=[0.8, 0.2], head=[0.6, 0.4], color="orange", label=None, status="rejected"),
        Arrow(id="inv", tail=[0.2, 0.8], head=[0.4, 0.6], color="gray", label=None, status="invalid", invalid_reason="x"),
        Arrow(id="prop", tail=[0.8, 0.8], head=[0.6, 0.6], color="yellow", label=None, status="proposed"),
        MaskCircle(id="mrej", center=[0.5, 0.9], radius_arcsec=0.8, kind="star", status="rejected"),
    ])
    im = render_image(f, _black())
    assert _mask(im, _rgb("cyan")).sum() > 0
    assert _mask(im, _rgb("yellow")).sum() > 0, "proposed items are drawn by the canonical render"
    assert _mask(im, _rgb("orange"), tol=40).sum() == 0, "rejected items draw nothing"
    assert _mask(im, _rgb("gray"), tol=40).sum() == 0, "invalid items draw nothing"
    assert _mask(im, _rgb("mask_red"), tol=60).sum() == 0
    im2 = render_image(f, _black(), include_proposed=False)
    assert _mask(im2, _rgb("yellow"), tol=40).sum() == 0
    assert _mask(im2, _rgb("cyan")).sum() > 0


def test_legend_auto_corner_avoids_items():
    items = [Arrow(id=f"a{k}", tail=[0.05 + 0.03 * k, 0.05], head=[0.15 + 0.03 * k, 0.20], color="cyan", label=f"arrow {k}")
             for k in range(3)]
    f = _synthetic(items, legend=True)
    (lg,) = [b for b in label_boxes(f) if b["kind"] == "legend"]
    assert lg["anchor"] != "top_left"
    f.legend.position = "bottom_right"
    (lg2,) = [b for b in label_boxes(f) if b["kind"] == "legend"]
    assert lg2["anchor"] == "bottom_right" and lg2["box"][0] > W / 2 and lg2["box"][1] > W / 2
    assert "→ arrow 0" in lg2["text"]
    im = render_image(f, _black())
    x0, y0, x1, y1 = (int(v) for v in lg2["box"])
    assert _mask(im, _rgb("cyan"), box=(x0, y0, x1, y1), tol=90).sum() > 0, "legend rows are coloured like the item"


def test_primitives_conventions():
    assert P.approach_angle(1.0, 0.0) == 0.0            # feature to the right -> approach from the right
    assert P.approach_angle(0.0, -1.0) == 90.0          # feature above -> approach from above
    assert P.approach_angle(-1.0, 0.0) == 180.0
    assert P.approach_angle(0.0, 0.0) == 225.0
    assert P.legend_corner([(0.1, 0.1), (0.9, 0.9)]) == "top_right"
    assert P.legend_corner([]) == "top_left"
    assert P.theta_parts("θ_E ≈ 1.5″")[1] == ("E", 0.62, 0.28) and P.theta_parts("2.09″ (alt)") == [("2.09″ (alt)", 1.0, 0.0)]
    pts = P.circle_points(0, 0, 10)
    assert pts[0] == pytest.approx((10.0, 0.0)) and pts[1][1] > 0      # clockwise on screen: y grows first
    assert P.clamp_center((0, 0), 20, 10, 100, 100, margin=4) == (14.0, 9.0)
    with pytest.raises(FileNotFoundError):
        P.load_font("Arial.ttf", 12)


# ----------------------------------------------------------------------------- (e) (f) (g) campaign I/O
def test_render_to_file_pins_json_and_staleness(nine_full):
    c = Campaign(nine_full)
    assert is_stale(c, "deck-01")
    out = render_to_file(c, "deck-01")
    assert out == c.annot_path("deck-01") and out.exists() and not list(nine_full.glob("*.tmp"))
    file = c.load("deck-01")
    assert file.render is not None and file.render.of_json_sha256 == file.content_sha256()
    assert file.render.renderer == config.RENDERER_VERSION and file.render.output == "deck-01.annot.png"
    assert not is_stale(c, "deck-01") and not c.annot_stale("deck-01")
    assert file.modified == "2026-08-30T12:00:00Z", "touch_modified=False"
    assert not any(e["op"] != "create" for e in c.read_log("deck-01")), "re-saving the render block logs no item events"
    # editing the JSON afterwards makes the PNG stale again
    file.system.description += " (edited)"
    c.save("deck-01", file, actor="test")
    assert is_stale(c, "deck-01")
    # --out / --scale never touch the JSON
    p = render_to_file(c, "deck-02", scale=2.0, out=nine_full / "exports")
    assert p.exists() and Image.open(p).size == (806, 806) and c.load("deck-02").render is None


def test_sha_mismatch_raises(nine_full):
    c = Campaign(nine_full)
    file = c.load("deck-03")
    file.image.sha256 = "f" * 64
    c.save("deck-03", file, actor="test")
    with pytest.raises(ValueError, match="sha256"):
        render_to_file(c, "deck-03")
    with pytest.raises(ValueError):
        render_png_bytes(c, "deck-03")
    assert not c.annot_path("deck-03").exists()


def test_cli_render_check(nine_full, capsys):
    assert cli_render(str(nine_full), check=True) == 1
    assert "STALE" in capsys.readouterr().out
    assert cli_render(str(nine_full)) == 0
    assert len(list(nine_full.glob("*.annot.png"))) == 9
    assert cli_render(str(nine_full), check=True) == 0
    assert cli_render(str(nine_full), image_id="deck-05", check=True) == 0
    # a hand edit of one JSON -> --check fails again for that id only
    p = nine_full / "deck-05.lensmark.json"
    p.write_text(p.read_text(encoding="utf-8").replace('"likely_lens"', '"possible"'), encoding="utf-8")
    assert cli_render(str(nine_full), check=True) == 1
    assert "1 stale: deck-05" in capsys.readouterr().out
    assert cli_render(str(nine_full), image_id="deck-05") == 0 and cli_render(str(nine_full), check=True) == 0
