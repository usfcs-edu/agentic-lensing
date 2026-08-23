"""Stage 8f: arbitrator batches (v2 verifier, stage 3 of 3).

Joins the advocate record with the critics' records for every item where at least one
critic NAMED an alternative (no_opinion false, alternative not null); items whose critics
all abstained or found nothing need no ruling (aggregate_v2 scores them from the advocate
alone). The arbitrator sees the full composite (footer removed) plus all texts.

Output: data/verify/arbitrator_a<n>.json, data/arbitrator_index.json
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



def records(pattern, persona_set):
    out = collections.defaultdict(dict)
    for _, _, d in load_jsonl(pattern):
        p = str(d.get("persona", "")).lower()
        if p in persona_set and "id" in d:
            out[str(d["id"])].setdefault(p, d)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per", type=int, default=PER)
    a = ap.parse_args()

    adv = records(f"{BASE}/results/verdicts/verify_advocate_*.jsonl", {"advocate"})
    crit = records(f"{BASE}/results/verdicts/verify_*.jsonl", set(CRITICS))
    have = {i for i, r in records(f"{BASE}/results/verdicts/verify_arbitrator_*.jsonl", {"arbitrator"}).items()}
    ins = pd.read_csv(f"{BASE}/results/inspections.csv")
    ins["id"] = ins["id"].astype(str)
    ins = ins.set_index("id", drop=False)

    items, n_unnamed = [], 0
    for cid, a_rec in adv.items():
        critics = [c for c in crit.get(cid, {}).values() if "refutation_strength" in c]
        named = [c for c in critics if not c.get("no_opinion") and c.get("alternative")]
        if not named:
            n_unnamed += 1
            continue
        if cid in have:
            continue
        full = view_path("full", cid)
        if not os.path.exists(full):
            if cid not in ins.index:
                continue
            src = os.path.join(BASE, str(ins.loc[cid, "png"]))
            if not os.path.exists(src):
                continue
            footer_cropped(src, full)
        layout = layout_of(ins.loc[cid].to_dict()) if cid in ins.index else "color"
        items.append({"id": cid, "image": full, "layout": layout,
                      "advocate": a_rec["advocate"], "critics": critics})
    jobs = write_jobs("arbitrator_a", items, a.per, "arbitrator")
    with open(f"{BASE}/data/arbitrator_index.json", "w") as fh:
        json.dump({"jobs": jobs, "per_batch": a.per, "n_items": len(items),
                   "n_no_named_alternative": n_unnamed, "base": BASE}, fh, indent=1)
    print(f"arbitrator jobs: {len(jobs)} agents over {len(items)} items "
          f"({n_unnamed} items had no named alternative)", flush=True)
    print(json.dumps(jobs), flush=True)
    print("ARBITRATOR_DONE", flush=True)


if __name__ == "__main__":
    main()
