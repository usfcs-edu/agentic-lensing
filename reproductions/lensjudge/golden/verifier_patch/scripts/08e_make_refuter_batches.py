"""Stage 8e: critic batches from the advocate's located evidence (v2 verifier, stage 2 of 3).

Reads results/verdicts/verify_advocate_*.jsonl, keeps items with p_evidence >= tau0
(scripts/thresholds_v2.json › provisional.tau0, or --tau0), renders each critic's OWN view
from the composite with PIL crops (no new fetch), and writes one job per critic role. The
critics receive the advocate's numbered evidence_items and its scale_class ONLY: no
p_evidence, no criteria scores, no inspector text.

Views (composite geometry from util.render_cutout; footer always removed; the panel sets are
those of lensjudge's prompts/personas/jwst_v1/panel_gloss.json, emitted by the generator):
  artifact    the full composite (the only critic shown panel (f), the subtraction)
  geometry    panels (b) deep 10", (d) deep 3.5", (e) 3.5" zoom, upscaled 2x, + ctx20 when 08d
              rendered it; never (f)
  morphology  colour layout: (c) colour 10", (d), (e) upscaled 2x; gray layout: (a) normal 10",
              (d), (e), because util.render_cutout puts a 10" radial-profile SUBTRACTION in
              slot (c) when one channel is missing - no subtraction panel ever reaches this critic
Each role's views live in their own directory (data/verify/views/<role>/<id>.jpg); the full
composite is under views/full/ and is listed only in the artifact and arbitrator jobs.

Output: data/verify/views/{geometry,morphology}/<id>.jpg,
data/verify/{artifact,geometry,morphology}_v<n>.json, data/refuter_index.json
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


GEOMETRY_PANELS = ('b', 'd', 'e')
MORPHOLOGY_PANELS = {'color': ('c', 'd', 'e'), 'gray': ('a', 'd', 'e')}
UPSCALE = 2


def strip(composite_path, panels, dst, ctx_path=None):
    """Paste the named panels (each upscaled UPSCALE x, caption strip kept) side by side,
    the ctx20 pair (if any) on a second row."""
    from PIL import Image
    im = Image.open(composite_path).convert("RGB")
    tiles = []
    for p in panels:
        x0, y0, x1, y1 = panel_box(p, with_label=True)
        t = im.crop((x0, y0, x1, y1))
        tiles.append(t.resize((t.width * UPSCALE, t.height * UPSCALE), Image.LANCZOS))
    row_w = sum(t.width for t in tiles) + GAP * (len(tiles) + 1)
    row_h = max(t.height for t in tiles) + 2 * GAP
    ctx = Image.open(ctx_path).convert("RGB") if ctx_path and os.path.exists(ctx_path) else None
    W = max(row_w, ctx.width if ctx else 0)
    H = row_h + (ctx.height + GAP if ctx else 0)
    canvas = Image.new("RGB", (W, H), (12, 12, 16))
    x = GAP
    for t in tiles:
        canvas.paste(t, (x, GAP))
        x += t.width + GAP
    if ctx:
        canvas.paste(ctx, (0, row_h))
    canvas.save(dst, quality=92)
    return dst


def advocate_records():
    """{id: advocate record} from every verify_advocate_*.jsonl (first record per id wins,
    like 09's drop_duplicates keep='first')."""
    recs = {}
    for _, _, d in load_jsonl(f"{BASE}/results/verdicts/verify_advocate_*.jsonl"):
        if str(d.get("persona", "")).lower() != "advocate" or "id" not in d:
            continue
        if not isinstance(d.get("items"), list):
            continue
        try:
            float(d.get("p_evidence"))
        except (TypeError, ValueError):
            continue
        recs.setdefault(str(d["id"]), d)
    return recs


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tau0", type=float, default=None)
    ap.add_argument("--thresholds", default=os.path.join(os.path.dirname(__file__), "thresholds_v2.json"))
    ap.add_argument("--per", type=int, default=PER)
    ap.add_argument("--roles", nargs="*", default=list(CRITICS))
    a = ap.parse_args()
    tau0 = a.tau0
    if tau0 is None:
        thr = json.load(open(a.thresholds))
        tau0 = float((thr.get("provisional") or {}).get("tau0", 0.15))

    ins = pd.read_csv(f"{BASE}/results/inspections.csv")
    ins["id"] = ins["id"].astype(str)
    ins = ins.set_index("id", drop=False)
    recs = advocate_records()
    print(f"advocate records: {len(recs)}; tau0={tau0}", flush=True)

    # what each candidate already has (so a re-run only emits the missing critic votes)
    have = collections.defaultdict(set)
    for _, _, d in load_jsonl(f"{BASE}/results/verdicts/verify_*.jsonl"):
        p = str(d.get("persona", "")).lower()
        if p in CRITICS and "refutation_strength" in d:
            have[str(d.get("id"))].add(p)

    os.makedirs(VIEWS, exist_ok=True)
    per_role = {r: [] for r in a.roles}
    n_gated = 0
    for cid, rec in recs.items():
        if float(rec["p_evidence"]) < tau0:
            n_gated += 1
            continue
        if cid not in ins.index:
            print(f"  {cid}: not in inspections.csv, skipped", flush=True)
            continue
        row = ins.loc[cid].to_dict()
        layout = layout_of(row)
        full = view_path("full", cid)
        if not os.path.exists(full):
            src = os.path.join(BASE, str(row["png"]))
            if not os.path.exists(src):
                print(f"  {cid}: missing {src}", flush=True)
                continue
            footer_cropped(src, full)
        ctx = f"{VIEWS}/ctx20/{cid}.jpg"
        views = {"artifact": full,
                 "geometry": strip(full, GEOMETRY_PANELS, view_path("geometry", cid),
                                   ctx_path=ctx if os.path.exists(ctx) else None),
                 "morphology": strip(full, MORPHOLOGY_PANELS[layout], view_path("morphology", cid))}
        payload = {"id": cid, "layout": layout,
                   "evidence_items": rec["items"],
                   "advocate_scale_class": rec.get("scale_class")}
        for role in a.roles:
            if role in have.get(cid, set()):
                continue
            per_role[role].append({**payload, "image": views[role]})
    jobs = []
    for role in a.roles:
        # the critics' jobs carry the items + scale class only: tau0 is recorded in the index
        j = write_jobs(f"{role}_v", per_role[role], a.per, role)
        jobs += j
        print(f"  {role}: {len(per_role[role])} items -> {len(j)} jobs", flush=True)
    with open(f"{BASE}/data/refuter_index.json", "w") as fh:
        json.dump({"jobs": jobs, "tau0": tau0, "per_batch": a.per, "n_advocate": len(recs),
                   "n_below_tau0": n_gated, "base": BASE}, fh, indent=1)
    print(f"\ncritic jobs: {len(jobs)} agents ({n_gated} items below tau0 skip the critics)", flush=True)
    print(json.dumps(jobs), flush=True)
    print("REFUTER_DONE", flush=True)


if __name__ == "__main__":
    main()
