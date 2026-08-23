"""Stage 8d: evidence batches for the ADVOCATE (v2 verifier, stage 1 of 3).

Replaces 08c's "top 350 by inspector confidence" cut, whose threshold sat on top of the
band where the recovered known lenses landed (confidence 15-30) and which routed the
inspector's own claim text to every verifier. Items carry ONLY {id, image, layout, context}:
no claim_center / claim_quadrant / claimed_evidence, so the advocate locates evidence from
pixels and nothing the inspector said can anchor or leak.

  --select top350      the 08c set (inspector confidence, TOP_N) - for the like-for-like regrade
  --select all_flagged every flagged candidate (2,024)
  --select ids-file    --ids-file path (CSV with an `id` column, or one id per line; # comments)
  --include-ctl        the ids of data/verify/*_ctl*.json (the COWLS controls whose verdicts the
                       v1 export dropped)
  --known-flagged      VALIDATION ONLY: the flagged-but-never-verified catalogued lenses
                       (results/known_lens_recovery.csv, sep <= 2"). Selecting items because
                       they are known is fine for an evaluation copy and must never become a
                       production selection rule.
  --context            also render <id>_ctx20.jpg (20" deep + colour) through util.RemoteImage;
                       needs network and ~30 min for 362 items; off by default

Output: data/verify/views/full/<id>.jpg (footer-cropped copy), data/verify/views/ctx20/<id>.jpg
(with --context), data/verify/advocate_e<n>.json, data/evidence_index.json
"""

import collections
import glob
import json
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import util  # noqa: E402

BASE = util.BASE
VDIR = f"{BASE}/data/verify"
VIEWS = f"{VDIR}/views"
PER = int(os.environ.get("PER_VERIFY", "12"))
CRITICS = ("artifact", "geometry", "morphology")
MIN_FINITE = 0.55        # 05_fetch_cutouts.MIN_FINITE: a channel below it was dropped by the render


def view_path(kind, cid):
    """data/verify/views/<kind>/<id>.jpg — one sub-directory per view kind (full, ctx20,
    geometry, morphology) so a critic's job lists a directory that holds ONLY its own view
    kind; the full composite never sits beside a cropped view."""
    d = f"{VIEWS}/{kind}"
    os.makedirs(d, exist_ok=True)
    return f"{d}/{cid}.jpg"

# composite geometry (util.render_cutout with px=240, gap=8, label strip 18, footer 22)
PX, GAP, TH, FOOT_H = 240, 8, 18, 22
COMPOSITE_W, COMPOSITE_H = 3 * PX + 4 * GAP, 2 * (PX + TH) + 3 * GAP + FOOT_H   # 752 x 562
FOOTER_Y = COMPOSITE_H - FOOT_H                                                 # 540
PANEL_RC = {"a": (0, 0), "b": (0, 1), "c": (0, 2), "d": (1, 0), "e": (1, 1), "f": (1, 2)}


def panel_box(panel, with_label=True):
    """Pixel box (x0, y0, x1, y1) of one panel in the composite; with_label keeps the
    18-px caption strip above it (the caption names the band and stretch, never the object)."""
    r, c = PANEL_RC[panel]
    x0 = GAP + c * (PX + GAP)
    y0 = GAP + r * (PX + TH + GAP) + TH
    return (x0, y0 - (TH if with_label else 0), x0 + PX, y0 + PX)


def footer_cropped(src, dst):
    """Copy the composite without its footer (id, RA/Dec, r-mag, type, programme)."""
    from PIL import Image
    im = Image.open(src).convert("RGB")
    im.crop((0, 0, im.width, min(FOOTER_Y, im.height))).save(dst, quality=92)
    return dst


def load_jsonl(pattern):
    """[(file, lineno, dict)] for every parseable line of every file matching pattern
    (the `*_ctl*` files match too — the v1 export that dropped them was the bug)."""
    out = []
    for f in sorted(glob.glob(pattern)):
        for i, line in enumerate(open(f), start=1):
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if isinstance(d, dict):
                out.append((f, i, d))
    return out


def _present(row, ch):
    """A channel is in the composite when it was observed AND its cutout passed the finite
    gate (05_fetch_cutouts drops a channel with finite < MIN_FINITE, which is what flips the
    render to the gray layout — `sw_obs`/`lw_obs` alone over-count colour layouts)."""
    get = row.get if isinstance(row, dict) else (lambda k, d=None: row[k] if k in row.index else d)
    obs = get(f"{ch}_obs")
    if not (isinstance(obs, str) and obs):
        return False
    fin = get(f"finite_{ch}")
    try:
        fin = float(fin)
    except (TypeError, ValueError):
        return True                       # no finite column: observed is the best we know
    return not (fin == fin) or fin >= MIN_FINITE


def layout_of(row):
    """'color' when both NIRCam channels are in the composite (observed and above the finite
    gate), else 'gray' (util.render_cutout then puts a radial-profile subtraction in slot
    (c) and a normal-stretch zoom in slot (e))."""
    return "color" if _present(row, "sw") and _present(row, "lw") else "gray"


def write_jobs(prefix, items, per, persona, extra=None):
    """Chunk items into data/verify/<prefix><n>.json (08c's shape: persona, batch, items)."""
    os.makedirs(VDIR, exist_ok=True)
    jobs = []
    for b in range(math.ceil(len(items) / per)):
        name = f"{prefix}{b}"
        payload = {"persona": persona, "batch": name, "items": items[b * per:(b + 1) * per]}
        if extra:
            payload.update(extra)
        with open(f"{VDIR}/{name}.json", "w") as fh:
            json.dump(payload, fh, indent=1)
        jobs.append(name)
    return jobs


TOP_N = int(os.environ.get("TOP_N", "350"))


def verified_ids():
    have = set()
    for _, _, d in load_jsonl(f"{BASE}/results/verdicts/verify_*.jsonl"):
        if "id" in d:
            have.add(str(d["id"]))
    return have


def ctl_ids():
    ids = []
    for f in sorted(glob.glob(f"{VDIR}/*_ctl*.json")):
        try:
            for it in json.load(open(f)).get("items", []):
                ids.append(str(it["id"]))
        except Exception:
            continue
    return ids


def read_ids_file(path):
    ids = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line.split(",")[0].strip())
    if ids and ids[0] in ("id", "candidate_id", "name"):
        ids = ids[1:]
    return ids


def context_image(row, dst, arcsec=20.0, out_px=640):
    """20" deep + colour pair from the SCI mosaics (util.RemoteImage.cutout), written as
    one two-panel JPG. Returns dst or None (no coverage / fetch error)."""
    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    bands = {}
    for ch in ("sw", "lw"):
        obs = row.get(f"{ch}_obs")
        if not (isinstance(obs, str) and obs):
            continue
        try:
            with util.RemoteImage(util.product_urls(obs)) as ri:
                im, frac = ri.cutout(float(row["ra"]), float(row["dec"]), out_px=out_px, arcsec=arcsec)
            if im is not None and frac > 0.2:
                bands[ch] = im
        except Exception as e:   # noqa: BLE001
            print(f"    ctx20 {row['id']} {ch}: {type(e).__name__}: {e}", flush=True)
    if not bands:
        return None
    base = bands.get("sw", bands.get("lw"))
    panels = [util._gray_panel(util.asinh_stretch(base, soft=0.7, cap=30.0), out_px // 2)]
    if "sw" in bands and "lw" in bands:
        sm = lambda z: gaussian_filter(np.nan_to_num(z, nan=0.0), 1.1)   # noqa: E731
        rr = util.asinh_stretch(sm(bands["lw"]), soft=1.2, cap=120.0)
        bb = util.asinh_stretch(sm(bands["sw"]), soft=1.2, cap=120.0)
        panels.append(util._rgb_panel(rr, (rr + bb) / 2.0, bb, out_px // 2))
    w = sum(p.width for p in panels) + GAP * (len(panels) + 1)
    canvas = Image.new("RGB", (w, panels[0].height + 2 * GAP), (12, 12, 16))
    x = GAP
    for p in panels:
        canvas.paste(p, (x, GAP))
        x += p.width + GAP
    canvas.save(dst, quality=92)
    return dst


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--select", choices=("top350", "all_flagged", "ids-file"), default="top350")
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--include-ctl", action="store_true")
    ap.add_argument("--known-flagged", action="store_true", help="VALIDATION ONLY (see module doc)")
    ap.add_argument("--context", action="store_true")
    ap.add_argument("--per", type=int, default=PER)
    ap.add_argument("--prefix", default="advocate_e")
    a = ap.parse_args()

    ins = pd.read_csv(f"{BASE}/results/inspections.csv")
    ins["id"] = ins["id"].astype(str)
    ins = ins[ins["status"].astype(str) == "ok"].set_index("id", drop=False)
    fl = ins[ins["flagged"].fillna(False).astype(bool)].copy()
    fl["confidence"] = pd.to_numeric(fl["confidence"], errors="coerce").fillna(0)

    if a.select == "top350":
        ids = fl.sort_values("confidence", ascending=False).head(a.top_n)["id"].tolist()
    elif a.select == "all_flagged":
        ids = fl.sort_values("confidence", ascending=False)["id"].tolist()
    else:
        if not a.ids_file:
            ap.error("--select ids-file needs --ids-file")
        ids = read_ids_file(a.ids_file)
    print(f"select={a.select}: {len(ids)} ids", flush=True)
    if a.include_ctl:
        extra = ctl_ids()
        ids += extra
        print(f"--include-ctl: +{len(extra)} control ids from {VDIR}/*_ctl*.json", flush=True)
    if a.known_flagged:
        kl = pd.read_csv(f"{BASE}/results/known_lens_recovery.csv")
        kl["cutout_id"] = kl["cutout_id"].astype(str)
        kl = kl[kl["flagged"].fillna(False).astype(bool) & (pd.to_numeric(kl["sep_arcsec"], errors="coerce") <= 2.0)]
        have = verified_ids()
        extra = [i for i in kl["cutout_id"].tolist() if i not in have]
        ids += extra
        print(f"--known-flagged (VALIDATION ONLY, never a production selection rule): "
              f"+{len(extra)} flagged-never-verified catalogued lenses", flush=True)
    seen, order = set(), []
    for i in ids:
        if i not in seen and i in ins.index:
            seen.add(i); order.append(i)
    missing = [i for i in ids if i not in ins.index]
    if missing:
        print(f"  {len(missing)} ids not in inspections.csv (status ok) skipped: {missing[:5]}...", flush=True)

    os.makedirs(VIEWS, exist_ok=True)
    items, n_ctx = [], 0
    for i in order:
        row = ins.loc[i]
        src = os.path.join(BASE, str(row["png"]))
        if not os.path.exists(src):
            print(f"  {i}: missing {src}", flush=True)
            continue
        full = footer_cropped(src, view_path("full", i))
        ctx = None
        if a.context:
            ctx = context_image(row.to_dict(), view_path("ctx20", i))
            n_ctx += ctx is not None
        items.append({"id": i, "image": full, "layout": layout_of(row.to_dict()), "context": ctx})
    jobs = write_jobs(a.prefix, items, a.per, "advocate")
    with open(f"{BASE}/data/evidence_index.json", "w") as fh:
        json.dump({"jobs": jobs, "n_items": len(items), "per_batch": a.per, "select": a.select,
                   "include_ctl": a.include_ctl, "known_flagged": a.known_flagged,
                   "context": a.context, "n_context": n_ctx, "base": BASE}, fh, indent=1)
    print(f"\nadvocate jobs: {len(jobs)} agents over {len(items)} items"
          + (f" ({n_ctx} with ctx20)" if a.context else ""), flush=True)
    print(json.dumps(jobs), flush=True)
    print("EVIDENCE_DONE", flush=True)


if __name__ == "__main__":
    main()
