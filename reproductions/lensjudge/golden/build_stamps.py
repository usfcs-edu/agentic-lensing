#!/usr/bin/env python3
"""golden/build_stamps.py — FITS stamps + byte-faithful v1 composites for the golden frame.

  python lensjudge/golden/build_stamps.py --top100 --limit 12 \
         --check-against ~/sync/research/jwst-strong-lens-search/top100_clean
  python lensjudge/golden/build_stamps.py --frame lensjudge/golden/frame.csv

For every frame row this streams the NIRCam SW and LW mosaics (public MAST S3 mirror, no
auth, HTTP range reads — nothing is downloaded whole) and writes, under
golden/stamps/<candidate_id>/:

  <id>_SW_10as.fits, <id>_LW_10as.fits   320 px at 0.03125"/px, the run's inspection grid
  <id>_SW_20as.fits, <id>_LW_20as.fits   640 px context at the same scale
  <id>_v1.jpg                            the 6-panel 752x562 composite, footer intact

The composite is rendered by the VENDORED run code (common/jwst_fetch.render_cutout @
util.py 4f81493) from the freshly fetched 10" arrays with the run's own MIN_FINITE gate,
so for the top-100 it reproduces J/top100_clean/<id>.jpg byte for byte; `--check-against`
measures that per system (bytes, then decoded pixels) into golden/stamps_check.csv.
Why FITS too: the run kept only JPEGs; Xiaosheng asked that the pixels be kept as FITS with a
stretch tuned for a human grader, and Campaign 2's human-eye v2 render needs the pixels.

Provenance: golden/stamps_manifest.csv (tracked, SHA-pinned via _util.pin) — one row per
written file with obs id, source URL, finite fraction and sha256; one row per composite
(channel=COMPOSITE: `filter` = the bands rendered, "F150W+F277W" for the colour layout or a
single band for gray; `finite_fraction` = the base channel's, SW if present else LW;
out_px = 240, the panel size). A channel whose mosaic exists but misses the target
(off-image) gets a row with an empty path and finite 0.0. Idempotent: a candidate whose manifest rows all exist on disk with
matching sha is skipped (`--force` re-fetches). Work is grouped by (sw_obs, lw_obs) as the
run did — opening a remote mosaic costs seconds, each cutout well under one — with 3
polite workers by default. Transient failures are retried inside jwst_fetch; a candidate
that still fails gets no manifest rows (so it re-runs next time) and a line in
golden/stamps/_failures.csv (gitignored).

Fields the footer prints (ra, dec, mag_r, type, proposal, filters) are taken from
J/data/targets.parquet — the run's own inputs — joined on candidate_id == id; frame
ra_deg/dec_deg are cross-checked against them (they are the same float64 for the top-100).
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# bootstrap: put reproductions/ on the path so `import lensjudge` works when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from lensjudge.common import jwst_fetch as jf  # noqa: E402
from lensjudge.golden import _util  # noqa: E402

STAMPS = _util.HERE / "stamps"
MANIFEST = _util.HERE / "stamps_manifest.csv"
CHECK = _util.HERE / "stamps_check.csv"
MASTER = _util.JWST_REPO / "results" / "JWST_top100_master.csv"
TARGETS = _util.JWST_REPO / "data" / "targets.parquet"
TOP100_CLEAN = _util.JWST_REPO / "top100_clean"

SCALES = ((10.0, 320), (20.0, 640))       # (arcsec, out_px): inspection grid + context
MANIFEST_COLS = ["candidate_id", "channel", "filter", "obs_id", "url", "arcsec", "out_px",
                 "finite_fraction", "path", "sha256", "fetched_at"]
CHECK_COLS = ["candidate_id", "ref_path", "ours_path", "bytes_identical", "max_abs_diff",
              "frac_pixels_differ", "ref_bytes", "ours_bytes", "note"]
# targets.parquet fields the run fed to the renderer/footer; carried on every frame row
_TARGET_COLS = ["ra", "dec", "mag_r", "type", "sw_obs", "sw_filter", "lw_obs", "lw_filter",
                "proposal"]
_MAX_POS_MISMATCH_ARCSEC = 0.05


# ------------------------------------------------------------------ frame loading

def _load_targets() -> pd.DataFrame:
    t = pd.read_parquet(TARGETS, columns=_TARGET_COLS + ["id"])
    t["id"] = t["id"].astype(str)
    return t.set_index("id")


def load_top100() -> pd.DataFrame:
    """The run's top-100 in rank order, joined to targets.parquet for the fetch fields."""
    m = pd.read_csv(MASTER)
    m["candidate_id"] = m["candidate_id"].astype(str)
    fr = m[["rank", "candidate_id", "ra_deg", "dec_deg"]].copy()
    return _join_targets(fr)


def load_frame(path: Path) -> pd.DataFrame:
    """The contract frame.csv (pinned). Only candidate_id, ra_deg, dec_deg are needed; the
    fetch/footer fields come from targets.parquet exactly as they did for the run."""
    try:
        fr = _util.read_pinned(path)
    except FileNotFoundError as e:         # an unpinned CSV (hand-made subset) still works
        print(f"WARN: {e}; reading {path} without pin verification", flush=True)
        fr = pd.read_csv(path)
    fr["candidate_id"] = fr["candidate_id"].astype(str)
    keep = [c for c in ("unit_id", "candidate_id", "ra_deg", "dec_deg", "rank_top100", "layout") if c in fr]
    return _join_targets(fr[keep].copy())


def _join_targets(fr: pd.DataFrame) -> pd.DataFrame:
    t = _load_targets()
    missing = [c for c in fr["candidate_id"] if c not in t.index]
    if missing:
        print(f"WARN: {len(missing)} frame ids absent from targets.parquet (will fail): "
              f"{missing[:5]}", flush=True)
    j = fr.join(t, on="candidate_id", how="left")
    have = j["ra"].notna()
    if "ra_deg" in j and have.any():
        cosd = np.cos(np.radians(j.loc[have, "dec"]))
        sep = 3600.0 * np.hypot((j.loc[have, "ra_deg"] - j.loc[have, "ra"]) * cosd,
                                j.loc[have, "dec_deg"] - j.loc[have, "dec"])
        bad = sep > _MAX_POS_MISMATCH_ARCSEC
        if bad.any():
            print(f"WARN: {int(bad.sum())} rows where frame ra_deg/dec_deg differ from "
                  f"targets.parquet by >{_MAX_POS_MISMATCH_ARCSEC}\" (max {sep.max():.3f}\"); "
                  f"using targets.parquet (the run's inputs)", flush=True)
    return j.reset_index(drop=True)


# ------------------------------------------------------------------ file naming

def stamp_paths(cid: str, stamps_dir: Path = STAMPS) -> dict:
    """{(channel, arcsec): fits path, 'COMPOSITE': jpg path} for one candidate."""
    d = stamps_dir / _util.safe_name(cid)
    out = {(ch, arc): d / f"{cid}_{ch}_{int(arc)}as.fits"
           for ch in jf.CHANNELS for arc, _ in SCALES}
    out["COMPOSITE"] = d / f"{cid}_v1.jpg"
    return out


def _rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(_util.LENSJUDGE))


def _abs(rel: str) -> Path:
    return _util.LENSJUDGE / rel


# ------------------------------------------------------------------ manifest bookkeeping

class Manifest:
    """In-memory manifest keyed by candidate; pinned to disk after every group so an
    interrupted run resumes without re-fetching what it already wrote."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.rows: dict[str, list[dict]] = {}
        if self.path.exists():
            try:
                df = _util.read_pinned(self.path, dtype=str, keep_default_na=False)
            except (FileNotFoundError, ValueError) as e:
                print(f"WARN: existing manifest not trusted ({e}); starting fresh", flush=True)
                df = pd.DataFrame(columns=MANIFEST_COLS)
            for cid, g in df.groupby("candidate_id", sort=False):
                self.rows[str(cid)] = g[MANIFEST_COLS].to_dict("records")

    def is_done(self, cid: str) -> bool:
        """All manifest rows for cid point at files that exist with the recorded sha
        (rows with an empty path mark a deterministic off-image channel and count as
        satisfied). A composite row is required."""
        rows = self.rows.get(cid)
        if not rows or not any(r["channel"] == "COMPOSITE" for r in rows):
            return False
        for r in rows:
            if not r["path"]:
                continue
            p = _abs(r["path"])
            if not p.exists() or _util.sha_file(p, n=0) != r["sha256"]:
                return False
        return True

    def replace(self, cid: str, rows: list[dict]) -> None:
        with self.lock:
            self.rows[cid] = rows

    def drop(self, cid: str) -> None:
        with self.lock:
            self.rows.pop(cid, None)

    def to_frame(self, order: list[str] | None = None) -> pd.DataFrame:
        cids = list(self.rows)
        if order:
            seen = set(order)
            cids = [c for c in order if c in self.rows] + [c for c in cids if c not in seen]
        recs = [r for c in cids for r in self.rows[c]]
        df = pd.DataFrame(recs, columns=MANIFEST_COLS)
        for c in ("arcsec", "finite_fraction"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["out_px"] = pd.to_numeric(df["out_px"], errors="coerce").astype("Int64")
        return df

    def pin(self, order: list[str] | None = None) -> str:
        with self.lock:
            return _util.pin(self.to_frame(order), self.path)


def _row(cid, channel, filt, obs_id, url, arcsec, out_px, finite, path, fetched_at) -> dict:
    return {"candidate_id": cid, "channel": channel,
            "filter": "" if filt is None or filt != filt else str(filt),
            "obs_id": "" if obs_id is None or obs_id != obs_id else str(obs_id),
            "url": url or "", "arcsec": float(arcsec), "out_px": int(out_px),
            "finite_fraction": round(float(finite), 4),
            "path": _rel(path) if path else "",
            "sha256": _util.sha_file(path, n=0) if path else "",
            "fetched_at": fetched_at}


# ------------------------------------------------------------------ the work

def process_candidate(t, images: dict, stamps_dir: Path, fetched_at: str) -> list[dict]:
    """Fetch both scales for one target from open mosaics, write FITS + composite, return
    the manifest rows. Raises on transient failure (caller records and moves on)."""
    cid = str(t.candidate_id)
    paths = stamp_paths(cid, stamps_dir)
    obs = {"SW": t.sw_obs, "LW": t.lw_obs}
    filt = {"SW": t.sw_filter, "LW": t.lw_filter}
    rows, ten = [], None
    for arcsec, out_px in SCALES:
        ch = jf.cutout_channels(images, float(t.ra), float(t.dec), arcsec=arcsec, out_px=out_px)
        if arcsec == jf.CUT_ARCSEC:
            ten = ch
        for name in jf.CHANNELS:
            if images.get(name) is None:       # no observation in this channel: no row
                continue
            arr, ff = ch[name]
            url = images[name].url
            if arr is None:                    # off-image: deterministic, record empty row
                rows.append(_row(cid, name, filt[name], obs[name], url, arcsec, out_px,
                                 0.0, None, fetched_at))
                continue
            hdr = jf.stamp_header(cid, t.ra, t.dec, name, filt[name], obs[name], url,
                                  arcsec, out_px, ff, fetched_at)
            p = jf.write_stamp_fits(paths[(name, arcsec)], arr, hdr)
            rows.append(_row(cid, name, filt[name], obs[name], url, arcsec, out_px, ff,
                             p, fetched_at))

    gated = jf.gate_min_finite(ten)
    if gated["SW"] is None and gated["LW"] is None:
        raise RuntimeError("no_coverage: both channels absent/off-image/below MIN_FINITE")
    meta = dict(id=cid, ra=float(t.ra), dec=float(t.dec), mag_r=float(t.mag_r),
                type=t.type, proposal=t.proposal, sw_filter=t.sw_filter, lw_filter=t.lw_filter)
    img = jf.render_v1_composite(gated["SW"], gated["LW"], meta)
    p = jf.save_composite(img, paths["COMPOSITE"])
    used = [filt[n] for n in jf.CHANNELS if gated[n] is not None]   # colour iff both
    base = "SW" if gated["SW"] is not None else "LW"
    rows.append(_row(cid, "COMPOSITE", "+".join(str(f) for f in used), "", "",
                     jf.CUT_ARCSEC, 240, ten[base][1], p, fetched_at))
    return rows


def run(frame: pd.DataFrame, *, stamps_dir: Path, manifest: Manifest, workers: int = 3,
        force: bool = False, failures_path: Path | None = None) -> dict:
    """Group by (sw_obs, lw_obs), fetch with a small thread pool, pin the manifest after
    every group. Returns a summary dict (counts + timing + failure list)."""
    order = list(frame["candidate_id"])
    if not force:
        skip = frame["candidate_id"].map(manifest.is_done)
        if skip.any():
            print(f"skipping {int(skip.sum())} candidates already complete (use --force)", flush=True)
        frame = frame[~skip]
    todo = frame[frame["ra"].notna()]
    missing = frame[frame["ra"].isna()]
    fails = [(str(c), "not_in_targets") for c in missing["candidate_id"]]
    groups = [(k, g) for k, g in todo.groupby(["sw_obs", "lw_obs"], dropna=False, sort=False)]
    groups.sort(key=lambda kv: -len(kv[1]))
    print(f"to fetch: {len(todo)} candidates in {len(groups)} mosaic groups, {workers} workers",
          flush=True)
    lock = threading.Lock()
    state = {"n": 0, "ok": 0, "fail": 0, "t0": time.time()}
    total = len(todo)

    def process_group(args):
        (sw_obs, lw_obs), g = args
        images = {"SW": None, "LW": None}
        try:
            try:
                images["SW"] = jf.open_image(sw_obs)
                images["LW"] = jf.open_image(lw_obs)
            except Exception as e:  # noqa: BLE001  (a mosaic we could not open: retry later)
                with lock:
                    for cid in g["candidate_id"]:
                        fails.append((str(cid), f"open_failed:{e}"))
                        manifest.drop(str(cid))
                    state["n"] += len(g)
                    state["fail"] += len(g)
                return
            # sort by detector row so neighbouring targets reuse cached blocks (as the run)
            ref = images["SW"] or images["LW"]
            if ref is not None and len(g) > 1:
                ys = []
                for t in g.itertuples():
                    try:
                        ys.append(ref._jacobian(t.ra, t.dec)[1])
                    except Exception:  # noqa: BLE001
                        ys.append(0.0)
                g = g.assign(_y=ys).sort_values("_y")
            for t in g.itertuples():
                cid = str(t.candidate_id)
                fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                try:
                    rows = process_candidate(t, images, stamps_dir, fetched_at)
                    manifest.replace(cid, rows)
                    with lock:
                        state["ok"] += 1
                except Exception as e:  # noqa: BLE001
                    manifest.drop(cid)
                    with lock:
                        fails.append((cid, f"{type(e).__name__}:{e}"))
                        state["fail"] += 1
                with lock:
                    state["n"] += 1
                    n = state["n"]
                time.sleep(0.2)           # polite: do not hammer the mirror from 3 threads
                if n % 10 == 0 or n == total:
                    el = time.time() - state["t0"]
                    rate = n / max(el, 1e-6)
                    print(f"{n}/{total}  ok={state['ok']} fail={state['fail']}  "
                          f"{el/60:.1f}min  eta {(total-n)/max(rate,1e-9)/60:.1f}min", flush=True)
        finally:
            for im in images.values():
                if im is not None:
                    im.close()
            manifest.pin(order)

    if groups:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(process_group, groups))
    sha = manifest.pin(order)
    if failures_path is not None:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(fails, columns=["candidate_id", "reason"]).to_csv(failures_path, index=False)
    el = (time.time() - state["t0"]) / 60
    print(f"STAMPS_DONE fetched={state['n']} ok={state['ok']} fail={state['fail']} "
          f"{el:.1f}min  manifest sha {sha}", flush=True)
    for cid, why in fails:
        print(f"  FAIL {cid}: {why}", flush=True)
    return {"n": state["n"], "ok": state["ok"], "fail": state["fail"], "minutes": el,
            "fails": fails, "manifest_sha": sha}


# ------------------------------------------------------------------ the pixel check

def compare_jpeg(ours: Path, ref: Path) -> dict:
    """Bytes first; if they differ, decode both and report the pixel-level distance."""
    a, b = Path(ours).read_bytes(), Path(ref).read_bytes()
    out = {"bytes_identical": a == b, "ref_bytes": len(b), "ours_bytes": len(a),
           "max_abs_diff": 0, "frac_pixels_differ": 0.0, "note": ""}
    if a == b:
        return out
    A = np.asarray(Image.open(ours).convert("RGB")).astype(np.int16)
    B = np.asarray(Image.open(ref).convert("RGB")).astype(np.int16)
    if A.shape != B.shape:
        out.update(max_abs_diff=255, frac_pixels_differ=1.0,
                   note=f"shape {A.shape} vs {B.shape}")
        return out
    d = np.abs(A - B)
    out["max_abs_diff"] = int(d.max())
    out["frac_pixels_differ"] = float((d.max(axis=2) > 0).mean())
    out["note"] = "pixel-identical, re-encode differs" if out["max_abs_diff"] == 0 else ""
    return out


def check_against(frame: pd.DataFrame, ref_dir: Path, *, stamps_dir: Path = STAMPS,
                  out_path: Path = CHECK) -> pd.DataFrame:
    recs = []
    for cid in frame["candidate_id"].astype(str):
        ours = stamp_paths(cid, stamps_dir)["COMPOSITE"]
        ref = Path(ref_dir) / f"{cid}.jpg"
        rec = {"candidate_id": cid, "ref_path": str(ref), "ours_path": _rel(ours) if ours.exists() else ""}
        if not ref.exists():
            rec.update(bytes_identical=False, max_abs_diff=np.nan, frac_pixels_differ=np.nan,
                       ref_bytes=0, ours_bytes=0, note="no reference")
        elif not ours.exists():
            rec.update(bytes_identical=False, max_abs_diff=np.nan, frac_pixels_differ=np.nan,
                       ref_bytes=ref.stat().st_size, ours_bytes=0, note="not rendered")
        else:
            rec.update(compare_jpeg(ours, ref))
        recs.append(rec)
    df = pd.DataFrame(recs, columns=CHECK_COLS)
    sha = _util.pin(df, out_path)
    cmp_ = df[df["ours_bytes"] > 0]
    n_b = int(cmp_["bytes_identical"].sum())
    n_p = int((cmp_["max_abs_diff"] == 0).sum())
    print(f"CHECK {len(cmp_)} compared: byte-identical {n_b}, pixel-identical {n_p}, "
          f"max_abs_diff max {cmp_['max_abs_diff'].max() if len(cmp_) else 'n/a'}; "
          f"{int((df['note'] != '').sum())} notes  -> {out_path} ({sha})", flush=True)
    for r in df[(df["ours_bytes"] > 0) & ~df["bytes_identical"]].itertuples():
        print(f"  DIFF {r.candidate_id}: max_abs {r.max_abs_diff} frac {r.frac_pixels_differ:.5f} "
              f"{r.note}", flush=True)
    return df


def rendered_layouts(manifest: "Manifest") -> pd.Series:
    """candidate_id -> layout actually rendered, read off the COMPOSITE rows: a '+' in
    `filter` means both channels survived the MIN_FINITE gate (colour); otherwise the single
    channel used is the one whose 10" cutout (the composite's input — the 20" context stamp
    does not decide the layout) passed the gate."""
    m = manifest.to_frame()
    comp = m[m["channel"] == "COMPOSITE"].set_index("candidate_id")
    chan = m[(m["channel"] != "COMPOSITE") & (m["arcsec"].astype(float) == jf.CUT_ARCSEC)]
    ok = chan[(chan["path"].fillna("") != "") & (chan["finite_fraction"].astype(float) >= jf.MIN_FINITE)]
    has_sw = set(ok.loc[ok["channel"] == "SW", "candidate_id"])
    out = {}
    for cid, r in comp.iterrows():
        if "+" in str(r["filter"]):
            out[cid] = "color"
        else:
            out[cid] = "gray_sw_only" if cid in has_sw else "gray_lw_only"
    return pd.Series(out, dtype=object)


def check_layout(frame: pd.DataFrame, manifest: "Manifest") -> pd.DataFrame:
    """frame.layout (build_frame's obs-presence + run finite-fraction rule) must equal what
    was rendered; a mismatch means the frame's layout column — which the kit key, the repeat
    picker and the few-shot eligibility all read — is wrong. Returns the mismatching rows."""
    if "layout" not in frame:
        return pd.DataFrame()
    got = rendered_layouts(manifest)
    f = frame[["candidate_id", "layout"]].copy()
    f["rendered"] = f["candidate_id"].map(got)
    bad = f[f["rendered"].notna() & (f["rendered"] != f["layout"])]
    if len(bad):
        print(f"LAYOUT MISMATCH frame vs rendered for {len(bad)} candidate(s) — rebuild the frame "
              f"(build_frame.derive_layout reads J/data/manifest.csv finite fractions):", flush=True)
        for r in bad.itertuples():
            print(f"  {r.candidate_id}: frame {r.layout} != rendered {r.rendered}", flush=True)
    else:
        print(f"LAYOUT OK: frame.layout == rendered layout for {int(f['rendered'].notna().sum())} candidates",
              flush=True)
    return bad


# ------------------------------------------------------------------ CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frame", type=Path, help="contract frame.csv (pinned)")
    src.add_argument("--top100", action="store_true",
                     help="the run's JWST_top100_master.csv joined to targets.parquet")
    ap.add_argument("--limit", type=int, default=None, help="first N frame rows")
    ap.add_argument("--ids", default=None, help="comma-separated candidate ids to restrict to")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="re-fetch even when files+sha match")
    ap.add_argument("--check-against", type=Path, default=None,
                    help="dir of reference <id>.jpg (e.g. J/top100_clean) to pixel-compare")
    ap.add_argument("--check-only", action="store_true", help="skip fetching; only compare")
    ap.add_argument("--stamps-dir", type=Path, default=STAMPS)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    args = ap.parse_args(argv)

    frame = load_top100() if args.top100 else load_frame(args.frame)
    if args.ids:
        want = set(args.ids.split(","))
        frame = frame[frame["candidate_id"].isin(want)]
    if args.limit:
        frame = frame.head(args.limit)
    print(f"frame: {len(frame)} candidates "
          f"(no SW obs: {int(frame['sw_obs'].isna().sum())}, no LW obs: {int(frame['lw_obs'].isna().sum())})",
          flush=True)

    manifest = Manifest(args.manifest)
    if not args.check_only:
        run(frame, stamps_dir=args.stamps_dir, manifest=manifest, workers=args.workers,
            force=args.force, failures_path=args.stamps_dir / "_failures.csv")
    check_layout(frame, manifest)
    if args.check_against is not None:
        check_against(frame, args.check_against, stamps_dir=args.stamps_dir)


if __name__ == "__main__":
    main()
