"""golden/build_kit.py — the blind grading kit, its answer key, and the hidden-repeat update.

A kit is what the grader receives: `kits/<kit_id>/` with footer-cropped JPEGs named by an
opaque item id, a single-file `grade.html` (the manifest embedded), a CSV fallback sheet, a
stdlib server, and a README that supersedes the earlier `top100_clean_scrambled/` protocol.
The *key* — which item is which unit — is written ONLY to `keys/<kit_id>_key.csv` (tracked,
never shipped). The whole point is that nothing the grader sees can be joined back to a
rank, a coordinate, a literature flag or the pipeline's own verdict; `assert_blind` greps the
shipped HTML for every such string before anything is written, and `assert_no_collision`
checks that no served JPEG is byte- (or size-) identical to a file in a set the grader
already holds (`J/top100_clean_scrambled/` ships its own answer key — a byte match would
join an item to a rank by `md5`).

Conventions follow J/scripts/22_clean_sets.py for the geometry: the six-panel render is
752x562 and its footer (id, RA/Dec, magnitude, programme) sits below y=540, so the crop is
(0,0,752,540). The JPEG is re-encoded at quality 92 (NOT the scrambled set's 93 — see above).
ONE seeded permutation of the whole frame (no blocks, so the order carries no stratum
information), seed 20260822 (contract). Item ids are a seeded sample WITHOUT replacement
from 001..999 in presentation order — never a running count, so a repeat's id (drawn later
from the unused remainder) looks exactly like every other id; `presentation_index` lives in
the key. Every served file gets one fixed mtime (a zip preserves mtimes, and "Date
Modified" would otherwise separate the files added by --add-repeats).

Modes:

  build          frame -> kit + key (pass 1 only).
  --add-repeats  run after session 1 is ingested: pick `--n` units stratified on the
                 OBSERVED pass-1 score (balanced as far as the marginals allow; never
                 prior_exposure==2, lit_known, gray layout, or a system with >1 unit), give
                 each a NEW item id from the unused remainder and a byte-identical copy of
                 its JPEG, and insert it at a uniformly random position >= original+min_gap
                 in the not-yet-graded tail (append when the tail is too short). Key,
                 manifest, grade.html and sheet are rewritten; the graded prefix is
                 untouched, so the tool resumes exactly where the grader stopped. Nothing
                 in the shipped kit marks a repeat: the tool shows only "k / N".
  --drop-units   the pre-specified pilot fallback (plan Phase 2: session-1 mean > 75 s ->
                 U_tail/N_unflagged cut to 15/10): removes UNGRADED items of the named
                 strata from the queue as a new manifest version, keeping every graded one
                 plus a deterministic core (lowest `_util.hash01(unit_id, "pilot_core")`
                 first) up to the quota. The key stays pinned for every item that remains;
                 removed rows go to `keys/<kit>_dropped.csv`. The kit is never rebuilt.

Every manifest version is appended to `keys/<kit_id>_manifests.jsonl` together with the
pin sha of the key it was written with, so collect.py can validate the `manifest_sha` of
events recorded under any version and refuse a key that was regenerated underneath them.
`--force` (pre-campaign only) deletes the key AND that history — a rebuilt kit starts from
nothing, so no old event can ever be joined to a new key.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lensjudge.golden import _util, schema  # noqa: E402

TOOL_DIR = _util.HERE / "tool"
TEMPLATE = TOOL_DIR / "grade_template.html"
SERVE_PY = TOOL_DIR / "serve.py"
KITS_ROOT = _util.HERE / "kits"
KEYS_DIR = _util.HERE / "keys"
KIT_SEED = 20260822
RENDER_VERSION = "jwst_v1"
PANEL_W, PANEL_H = 752, 562          # the run's six-panel composite
FOOTER_Y = 540                       # footer strip (id, RA/Dec, mag, programme) starts here
# quality 92, deliberately NOT the 93 of 22_clean_sets.py: the grader holds that earlier
# scrambled set (with its key), and a byte-identical file would join item -> rank by md5.
JPEG_KW = dict(quality=92, optimize=True)
PLACEHOLDER = "__KIT_MANIFEST_JSON__"
ID_SPACE = range(1, 1000)            # item ids: 3-digit zero-padded, drawn without replacement
ITEM_MTIME = 1767225600              # 2026-01-01T00:00:00Z: one mtime for every served file
# sets the grader may already hold; a served JPEG must not byte-match any file in them.
# golden/kits_truth/ + kits_truth_v2r/ (the Part-2 truth-eval JPEGs a model has scored; skipped when absent)
# is included so no truth JPEG can ever be served to the grader byte-for-byte
DEFAULT_COLLISION_DIRS = (_util.JWST_REPO / "top100_clean_scrambled", _util.JWST_REPO / "top100_clean",
                          _util.HERE / "kits_truth", _util.HERE / "kits_truth_v2r")
# the pilot fallback quotas (plan Phase 2, pre-specified): stratum -> items kept
PILOT_DROP_KEEP = {"U_tail": 15, "N_unflagged": 10}

# Frame columns the kit needs (everything else in the frame is deliberately not read).
FRAME_COLS = ("unit_id", "candidate_id", "layout", "stratum", "system_id", "prior_exposure",
              "lit_known")

# Legend shown to the grader. Absolute A/B/C/D wording lifted from
# prompts/rubric_imaging.md:51-58 ("Grades (the group's A/B/C/D scale)") and re-worded as
# 4/3/2/1; the machine-only sentence ("Set `contaminant`...") is dropped. Wording is to be
# approved by the grader before session 1 (plan, Phase 1 pre-flight) — edit here only.
LEGEND = {
    "title": "Legend — Huang visual-inspection scale",
    "banner": "This is the Huang visual-inspection scale (4=A … 1=D), not the pipeline pass-count",
    "scale": [
        {"score": 4, "letter": "A", "text":
         "Almost certainly a lens: a clear tangential arc, counter-image or ring around a red "
         "galaxy, with strong evidence from at least two of curvature, counter-images and "
         "arc morphology, and no decisive contaminant."},
        {"score": 3, "letter": "B", "text":
         "Probable lens: an arc-like blue feature at the right separation, but missing a clean "
         "counter-image or partly ambiguous."},
        {"score": 2, "letter": "C", "text":
         "Possible lens: a blue neighbour at a plausible separation but weak, round, or without "
         "clear curvature; could be a chance projection or a star-forming companion."},
        {"score": 1, "letter": "D", "text":
         "Not a lens: spiral arms, a ring galaxy, a merger or tidal tails, a bright star with a "
         "diffraction halo or spikes, a cosmic ray or satellite trail, an isolated galaxy, or "
         "pure noise."},
    ],
    "confidence_question": "How sure are you of this score?",
}
CONFIDENCE = {"keys": list(_util.CONF_LEVELS),
              "labels": ["low — could easily be a different score",
                         "medium",
                         "high — would defend this score"]}

# Words that must never appear in the shipped HTML (case-insensitive), on top of every
# id / coordinate / sha value from the frame and key.
FORBIDDEN_WORDS = ("candidate", "rank", "stratum", "ra_deg", "dec_deg", "pipe_", "cowls",
                   "known_", "prior_exposure", "lit_known", "system_id", "unit_id", "repeat",
                   "alias", "passcount", "render_sha", "sha256", "desi", "top100")


# ------------------------------------------------------------------ small helpers
def _read_frame(path: Path) -> pd.DataFrame:
    df = _util.read_pinned(path, dtype={"unit_id": str, "candidate_id": str, "layout": str,
                                        "stratum": str}, keep_default_na=False,
                           na_values=[""])
    missing = [c for c in FRAME_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: frame is missing columns {missing}")
    df["unit_id"] = df["unit_id"].astype(str)
    df["candidate_id"] = df["candidate_id"].astype(str)
    if df["unit_id"].duplicated().any():
        raise ValueError("frame has duplicate unit_id")
    if df["candidate_id"].duplicated().any():
        raise ValueError("frame has duplicate candidate_id")
    # bools may arrive as 'True'/'False' text or 0/1 — normalise, but never guess blanks
    lk = df["lit_known"]
    if lk.dtype != bool:
        df["lit_known"] = lk.astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False})
        if df["lit_known"].isna().any():
            raise ValueError("frame.lit_known has values outside {True,False,0,1}")
    df["prior_exposure"] = df["prior_exposure"].astype(int)
    return df


def find_source(cid: str, source_dir: Path, fallback_dir: Path | None) -> Path | None:
    """stamps/<cid>/<cid>_v1.jpg (WP-A composite), else <fallback>/<cid>.jpg (the run's own)."""
    p = Path(source_dir) / cid / f"{cid}_v1.jpg"
    if p.exists():
        return p
    if fallback_dir is not None:
        q = Path(fallback_dir) / f"{cid}.jpg"
        if q.exists():
            return q
    return None


def crop_footer(src: Path, dst: Path) -> str:
    """Write the footer-cropped JPEG (752x540) and return sha16 of the bytes as served."""
    im = Image.open(src).convert("RGB")
    if im.width != PANEL_W or im.height < FOOTER_Y:
        raise ValueError(f"{src}: expected a {PANEL_W}x{PANEL_H} composite, got {im.size}")
    im = im.crop((0, 0, im.width, min(FOOTER_Y, im.height)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, **JPEG_KW)
    return _util.sha_file(dst)


def make_manifest(kit_id: str, kit_version: int, grader_id: str, item_ids: list[str],
                  built_at: str | None = None) -> dict:
    """The manifest the tool embeds: opaque items only. manifest_sha = sha_json of the dict
    without the sha field (so the sha covers items, version, legend and build time)."""
    m = {
        "kit_id": kit_id,
        "kit_version": int(kit_version),
        "grader_id": grader_id,
        "grade_scale": _util.GRADE_SCALE,
        "render_version": RENDER_VERSION,
        "confidence": CONFIDENCE,
        "legend": LEGEND,
        "items": [{"item_id": i, "image": f"items/{i}.jpg"} for i in item_ids],
        "n_items": len(item_ids),
        "built_at": built_at or schema.utc_now_iso(),
    }
    m["manifest_sha"] = _util.sha_json(m)
    return m


def manifest_sha_of(m: dict) -> str:
    """Recompute a manifest's sha (what collect.py uses to validate events)."""
    return _util.sha_json({k: v for k, v in m.items() if k != "manifest_sha"})


def render_html(manifest: dict, template: Path = TEMPLATE) -> str:
    text = template.read_text()
    if PLACEHOLDER not in text:
        raise ValueError(f"{template} lacks {PLACEHOLDER}")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)      # ship no commentary
    js = json.dumps(manifest, indent=1, ensure_ascii=False).replace("</", "<\\/")
    return text.replace(PLACEHOLDER, js)


def _coord_strings(v) -> set[str]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return set()
    if not np.isfinite(f):
        return set()
    return {f"{f:.{d}f}" for d in (2, 3, 4, 5, 6)} | {repr(f)}


def assert_blind(html: str, frame: pd.DataFrame, key: pd.DataFrame,
                 extra: tuple[str, ...] = ()) -> None:
    """Raise if the shipped HTML contains any identifying string: candidate/alias/unit ids,
    known-lens names, any coordinate rendering, any render sha, the key's pin sha, or one of
    FORBIDDEN_WORDS."""
    low = html.lower()
    bad: list[str] = []
    vals: set[str] = set()
    for c in ("candidate_id", "unit_id", "known_lens_name", "cowls_ranking"):
        if c in frame.columns:
            vals |= {str(v) for v in frame[c] if str(v).strip() and str(v) != "nan"}
    if "alias_ids" in frame.columns:
        for v in frame["alias_ids"]:
            vals |= {a for a in str(v).split("|") if a.strip() and a != "nan"}
    for c in ("ra_deg", "dec_deg"):
        if c in frame.columns:
            for v in frame[c]:
                vals |= _coord_strings(v)
    vals |= {str(v) for v in key["render_sha"] if str(v).strip()}
    vals |= {str(v) for v in extra if str(v).strip()}
    for v in sorted(vals):
        if v.lower() in low:
            bad.append(v)
    for w in FORBIDDEN_WORDS:
        if w in low:
            bad.append(f"word:{w}")
    if bad:
        raise AssertionError(f"grade.html is NOT blind; found {bad[:12]}"
                             f"{' ...' if len(bad) > 12 else ''}")


def draw_item_ids(rng: np.random.Generator, n: int, used=()) -> list[str]:
    """n item ids drawn without replacement from ID_SPACE minus `used`, in draw order (NOT
    sorted: sorting would make the id grow with presentation position, and a later draw
    for repeats would then stand out as the ids that break the monotone run)."""
    pool = np.array(sorted(set(ID_SPACE) - {int(u) for u in used}), dtype=int)
    if n > len(pool):
        raise ValueError(f"item-id space exhausted: need {n}, {len(pool)} unused 3-digit ids left")
    return [f"{int(i):03d}" for i in rng.choice(pool, size=n, replace=False)]


def _sha256_and_size(path: Path) -> tuple[str, int]:
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size


def assert_no_collision(items_dir: Path, other_dirs) -> dict:
    """Raise if any served JPEG under `items_dir` is byte-identical to a *.jpg in one of
    `other_dirs` (sets the grader already holds, with their own keys); print a warning for
    size-only coincidences (a weaker join, and a chance one is possible at ~10^5 byte sizes).
    Directories that do not exist are skipped. Returns the per-dir counts."""
    ours = {p.name: _sha256_and_size(p) for p in sorted(Path(items_dir).glob("*.jpg"))}
    report = {}
    for d in other_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        theirs = {}
        for q in sorted(d.glob("*.jpg")):
            sha, size = _sha256_and_size(q)
            theirs.setdefault(sha, []).append(q.name)
        sizes = {q.stat().st_size for q in d.glob("*.jpg")}
        byte_hits = [(n, theirs[s][0]) for n, (s, _) in ours.items() if s in theirs]
        size_hits = [n for n, (_, z) in ours.items() if z in sizes]
        report[str(d)] = {"n_theirs": sum(len(v) for v in theirs.values()),
                          "byte_hits": len(byte_hits), "size_hits": len(size_hits)}
        if byte_hits:
            raise AssertionError(f"{len(byte_hits)} served JPEG(s) are byte-identical to files in "
                                 f"{d} (e.g. items/{byte_hits[0][0]} == {byte_hits[0][1]}); the "
                                 f"grader holds that set and its key")
        if size_hits:
            print(f"WARNING: {len(size_hits)} served JPEG(s) share a byte size with a file in {d} "
                  f"(chance-level is ~{len(ours) * len(sizes) / 6e4:.1f}; a byte match would have raised)",
                  flush=True)
        print(f"collision check vs {d}: {report[str(d)]['n_theirs']} files, 0 byte matches, "
              f"{len(size_hits)} size coincidences", flush=True)
    return report


def freeze_mtimes(items_dir: Path, mtime: int = ITEM_MTIME) -> int:
    """One fixed mtime for every served file: a zip preserves mtimes, and the files written by
    --add-repeats would otherwise sort apart under 'Date Modified'."""
    n = 0
    for q in Path(items_dir).glob("*.jpg"):
        os.utime(q, (mtime, mtime))
        n += 1
    return n


def _readme(kit_id: str, grader_id: str, n_items: int, built_at: str, kit_version: int) -> str:
    lines = [
        f"GOLDEN KIT {kit_id} — blind grading, one pass, grade + confidence only",
        "=" * 78,
        f"Grader: {grader_id}.  Items: {n_items}.  Kit version {kit_version}, built {built_at}.",
        "",
        "WHAT TO DO",
        "1. Open grade.html in a browser (double-click it; Safari, Chrome or Firefox all work).",
        "   Optional: run `python3 serve.py --port 8765` in this folder and open",
        "   http://localhost:8765/grade.html — the server then also keeps a copy of every",
        "   answer in records/ as you go.",
        "2. For each image press a SCORE key and a CONFIDENCE key, then Enter (or Space).",
        "   SCORE — the Huang visual-inspection scale, 4 = A ... 1 = D.",
        "   (This is NOT the pipeline pass-count letter.)",
    ]
    for row in LEGEND["scale"]:
        lines.append(f"     {row['score']} = {row['letter']}  {row['text']}")
    lines += [
        f"   CONFIDENCE — {LEGEND['confidence_question']}  (how sure you are of the SCORE,",
        "   not the probability that the object is a lens.)",
    ]
    for k, lab in zip(CONFIDENCE["keys"], CONFIDENCE["labels"]):
        lines.append(f"     {k} = {lab}")
    lines += [
        "   F flags an item (e.g. a rendering problem). No notes or comments are taken —",
        "   by design: the record is score + confidence, nothing else.",
        "   Backspace goes back one item to re-grade it. Z toggles 2x zoom. ? lists the keys.",
        "3. Grade in ONE pass, in the order shown, without consulting the annotated document,",
        "   the contact sheet or any catalogue. Some images may appear more than once; grade",
        "   each as you see it. You can stop and resume at any time — the page remembers",
        "   where you were. Every 20 items (and at the end) it downloads a file",
        "   events_<kit>_<session>.jsonl: keep every such file and send them all back.",
        "   To continue on another computer (or if you switch between double-clicking",
        "   grade.html and the serve.py address — the browser keeps separate memories for",
        "   the two), open grade.html there and click \"Import events...\" with the most",
        "   recent downloaded file.",
        "",
        "FALLBACK (no browser): grading_sheet.csv lists every item id; open items/NNN.jpg and",
        "fill score_1_4 (1-4) and confidence_lmh (L/M/H). Timing will be missing.",
        "",
        "THIS KIT SUPERSEDES top100_clean_scrambled/ AND ITS README. That earlier set asked",
        "for a lens yes/no, the arc position and an Einstein-radius estimate; this kit asks",
        "ONLY for the 1-4 score and L/M/H confidence. Please do not grade the old set, and",
        "please DELETE the top100_clean_scrambled/ folder (including its key CSV) before you",
        "start: nothing in this kit corresponds to it file-for-file, and keeping it around",
        "only invites comparisons that would bias the grading.",
        "",
        "Images: JWST NIRCam six-panel cutouts (north up, east left; yellow ticks mark the",
        "catalogued galaxy; white bar = 1\"), rendered exactly as before but with the footer",
        "strip removed so that no identifier is visible.",
        "",
    ]
    return "\n".join(lines)


def write_kit_files(kit_dir: Path, manifest: dict, frame: pd.DataFrame, key: pd.DataFrame,
                    keys_dir: Path, key_sha: str, collision_dirs=DEFAULT_COLLISION_DIRS) -> None:
    """grade.html, kit_manifest.json, grading_sheet.csv, README.txt, serve.py, and the
    manifest-history line (manifest + the pin sha of the key it was written with). Asserts
    blindness of the HTML and byte-distinctness of the served JPEGs before writing anything;
    freezes every served file's mtime."""
    html = render_html(manifest)
    assert_blind(html, frame, key, extra=(key_sha,))
    assert_no_collision(kit_dir / "items", collision_dirs)
    freeze_mtimes(kit_dir / "items")
    (kit_dir / "grade.html").write_text(html)
    (kit_dir / "kit_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    # fallback sheet: sorted by item id (NOT presentation order — the order of ids would
    # otherwise expose where later items were inserted)
    ids = sorted(i["item_id"] for i in manifest["items"])
    pd.DataFrame({"item_id": ids, "score_1_4": "", "confidence_lmh": ""}).to_csv(
        kit_dir / "grading_sheet.csv", index=False)
    (kit_dir / "README.txt").write_text(_readme(
        manifest["kit_id"], manifest["grader_id"], manifest["n_items"], manifest["built_at"],
        manifest["kit_version"]))
    shutil.copyfile(SERVE_PY, kit_dir / "serve.py")
    keys_dir.mkdir(parents=True, exist_ok=True)
    with open(keys_dir / f"{manifest['kit_id']}_manifests.jsonl", "a") as f:
        f.write(json.dumps({**manifest, "key_sha": key_sha}, sort_keys=True) + "\n")


def load_manifest_history(keys_dir: Path, kit_id: str) -> list[dict]:
    p = Path(keys_dir) / f"{kit_id}_manifests.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            m = json.loads(line)
            key_sha = m.pop("key_sha", None)      # history-only field, not part of the manifest
            if manifest_sha_of(m) != m["manifest_sha"]:
                raise ValueError(f"{p}: manifest_sha does not recompute for version "
                                 f"{m.get('kit_version')}")
            m["key_sha"] = key_sha
            out.append(m)
    return out


def key_pin_sha(keys_dir: Path, kit_id: str) -> str:
    """The current pin of keys/<kit>_key.csv (its .sha sidecar)."""
    return (Path(keys_dir) / f"{kit_id}_key.csv.sha").read_text().strip()


# ----------------------------------------------------------------------- build
def _campaign_files(keys_dir: Path, kit_id: str) -> list[Path]:
    """Everything a kit leaves under keys/: the key, its pin, the manifest history, the
    dropped-units sidecar. `--force` removes all of them so a rebuilt kit cannot inherit an
    old manifest version (an old event would otherwise validate and be joined to the NEW key:
    silent unit misattribution)."""
    keys_dir = Path(keys_dir)
    return [keys_dir / f"{kit_id}_key.csv", keys_dir / f"{kit_id}_key.csv.sha",
            keys_dir / f"{kit_id}_manifests.jsonl", keys_dir / f"{kit_id}_dropped.csv",
            keys_dir / f"{kit_id}_dropped.csv.sha"]


def build(frame_path: Path, kit_id: str, grader_id: str, seed: int, source_dir: Path,
          fallback_dir: Path | None, kits_root: Path = KITS_ROOT, keys_dir: Path = KEYS_DIR,
          force: bool = False, collision_dirs=DEFAULT_COLLISION_DIRS) -> tuple[Path, Path]:
    frame = _read_frame(frame_path)
    kit_dir = Path(kits_root) / kit_id
    key_path = Path(keys_dir) / f"{kit_id}_key.csv"
    if (key_path.exists() or (kit_dir.exists() and any(kit_dir.iterdir()))) and not force:
        raise FileExistsError(f"{kit_id} already exists ({key_path} / {kit_dir}); a key under "
                              f"a running campaign must not be regenerated — pass --force")
    if force:
        if kit_dir.exists():
            shutil.rmtree(kit_dir)
        gone = [str(q) for q in _campaign_files(keys_dir, kit_id) if q.exists()]
        for q in _campaign_files(keys_dir, kit_id):
            if q.exists():
                q.unlink()
        if gone:
            print(f"FORCE: removed the previous key/history of {kit_id}: {gone} — events recorded "
                  f"under the old kit can no longer be collected (by design)", flush=True)
    # every source must exist before a single byte is written
    src = {r.candidate_id: find_source(r.candidate_id, source_dir, fallback_dir)
           for r in frame.itertuples()}
    missing = sorted(c for c, p in src.items() if p is None)
    if missing:
        raise FileNotFoundError(f"{len(missing)} composite(s) not found under {source_dir}"
                                f"{f' or {fallback_dir}' if fallback_dir else ''}: {missing}")
    # ONE seeded permutation of the whole frame — no blocks, no stratum order; then the
    # item ids, a second seeded draw from the same generator (opaque, not a running count)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    item_ids = draw_item_ids(rng, len(frame))
    (kit_dir / "items").mkdir(parents=True, exist_ok=True)
    rows = []
    for pos, fi in enumerate(order, start=1):
        r = frame.iloc[int(fi)]
        item_id = item_ids[pos - 1]
        sha = crop_footer(src[r.candidate_id], kit_dir / "items" / f"{item_id}.jpg")
        rows.append({"kit_id": kit_id, "item_id": item_id, "presentation_index": pos,
                     "unit_id": r.unit_id, "candidate_id": r.candidate_id, "pass": 1,
                     "repeat_of_item": "", "render_sha": sha, "layout": r.layout,
                     "stratum": r.stratum})
    key = pd.DataFrame(rows, columns=list(schema.KEY_COLS))
    key_sha = _util.pin(key, key_path)
    manifest = make_manifest(kit_id, 1, grader_id, key["item_id"].tolist())
    write_kit_files(kit_dir, manifest, frame, key, keys_dir, key_sha, collision_dirs)
    print(f"kit {kit_id}: {len(key)} items -> {kit_dir}  key -> {key_path} (sha {key_sha})  "
          f"manifest_sha {manifest['manifest_sha']}  seed {seed}", flush=True)
    return kit_dir, key_path


# ----------------------------------------------------------------- add repeats
def allocate_balanced(avail: dict[int, int], n: int) -> dict[int, int]:
    """Split n over score levels as evenly as availability allows (water-filling)."""
    quota = {s: 0 for s in avail}
    left = n
    while left > 0:
        open_levels = [s for s in avail if quota[s] < avail[s]]
        if not open_levels:
            break
        # always top up the currently smallest quota that still has supply
        s = min(open_levels, key=lambda k: (quota[k], -k))
        quota[s] += 1
        left -= 1
    return quota


def pick_repeat_units(frame: pd.DataFrame, key: pd.DataFrame, grades: pd.DataFrame, kit_id: str,
                      n: int, rng: np.random.Generator, min_per_level: int = 8) -> list[str]:
    """Units eligible for a hidden repeat, stratified on observed pass-1 score."""
    sys_n = frame.groupby("system_id")["unit_id"].nunique()
    elig = frame[(frame["prior_exposure"] != 2) & (~frame["lit_known"].astype(bool))
                 & (frame["layout"] == "color") & (frame["system_id"].map(sys_n) == 1)]
    g1 = grades[(grades["kit_id"].astype(str) == kit_id) & (grades["pass"].astype(int) == 1)]
    already = set(key.loc[key["pass"] == 2, "unit_id"])
    pool = g1[g1["unit_id"].isin(set(elig["unit_id"])) & ~g1["unit_id"].isin(already)]
    by_level = {s: sorted(pool.loc[pool["score_1_4"].astype(int) == s, "unit_id"])
                for s in (4, 3, 2, 1)}
    avail = {s: len(v) for s, v in by_level.items()}
    quota = allocate_balanced(avail, n)
    got = sum(quota.values())
    if got < n:
        print(f"WARNING: only {got} eligible units for {n} repeats (avail {avail})", flush=True)
    for s in (4, 3, 2, 1):
        if quota[s] < min_per_level:     # short supply is exactly when this must be said
            print(f"WARNING: score {s}: {quota[s]} repeats < min_per_level {min_per_level} "
                  f"(avail {avail[s]}) — the QWK CI will be wider than the design target; "
                  f"consider staging the draw (--n 20 twice) after more items are graded", flush=True)
    chosen: list[str] = []
    for s in (4, 3, 2, 1):
        if quota[s]:
            chosen += [str(u) for u in rng.choice(by_level[s], quota[s], replace=False)]
    print(f"repeats: quota by score {quota} from availability {avail}", flush=True)
    return chosen


def add_repeats(kit_id: str, frame_path: Path, grades_path: Path, n: int, min_gap: int,
                seed: int, kits_root: Path = KITS_ROOT, keys_dir: Path = KEYS_DIR,
                min_per_level: int = 8, collision_dirs=DEFAULT_COLLISION_DIRS) -> Path:
    frame = _read_frame(frame_path)
    kit_dir = Path(kits_root) / kit_id
    key_path = Path(keys_dir) / f"{kit_id}_key.csv"
    key = schema.read_key(key_path)
    history = load_manifest_history(keys_dir, kit_id)
    if not history:
        raise FileNotFoundError(f"no manifest history for {kit_id} under {keys_dir}")
    cur = history[-1]
    if cur["manifest_sha"] != json.loads((kit_dir / "kit_manifest.json").read_text())["manifest_sha"]:
        raise ValueError("kit_manifest.json in the kit dir is not the latest recorded version")
    grades = _util.read_pinned(grades_path, dtype={"unit_id": str, "item_id": str,
                                                   "kit_id": str}, keep_default_na=False)
    if (key["pass"] == 2).any():
        print(f"NOTE: key already holds {(key['pass'] == 2).sum()} repeats; adding {n} more",
              flush=True)
    # the graded prefix (by presentation order) must stay untouched: repeats go in the tail
    queue = [it["item_id"] for it in cur["items"]]
    assert queue == key.sort_values("presentation_index")["item_id"].tolist(), \
        "key and manifest disagree on presentation order"
    graded_items = set(grades.loc[grades["kit_id"] == kit_id, "item_id"].astype(str))
    pos_of = {it: i + 1 for i, it in enumerate(queue)}
    graded_pos = sorted(pos_of[i] for i in graded_items if i in pos_of)
    if not graded_pos:
        raise ValueError("no graded items for this kit in the grades file — run collect first")
    tail_start = graded_pos[-1] + 1
    if graded_pos != list(range(1, tail_start)):
        print(f"WARNING: graded positions are not a contiguous prefix "
              f"({len(graded_pos)} graded, last at {graded_pos[-1]})", flush=True)
    rng = np.random.default_rng([int(seed), int(cur["kit_version"]) + 1])
    units = pick_repeat_units(frame, key, grades, kit_id, n, rng, min_per_level)
    if not units:
        raise ValueError("no eligible units to repeat")
    rng.shuffle(units)
    k1 = key[key["pass"] == 1].set_index("unit_id")
    used = set(key["item_id"]) | set(_dropped_item_ids(keys_dir, kit_id))
    new_ids = draw_item_ids(rng, len(units), used)    # opaque, from the unused remainder
    new_rows, forced = [], []
    for u, new_item in zip(units, new_ids):
        orig = k1.loc[u]
        orig_item = str(orig["item_id"])
        lo = max(pos_of[orig_item] + min_gap, tail_start)
        hi = len(queue) + 1                       # "append" position
        if lo <= hi:
            pos = int(rng.integers(lo, hi + 1))   # uniform over [lo, hi]
        else:                                     # tail too short: append (gap < min_gap)
            pos, lo = hi, hi
            forced.append(u)
        queue.insert(pos - 1, new_item)
        src, dst = kit_dir / "items" / f"{orig_item}.jpg", kit_dir / "items" / f"{new_item}.jpg"
        shutil.copyfile(src, dst)                 # byte-identical: NOT re-encoded
        sha = _util.sha_file(dst)
        assert sha == str(orig["render_sha"]) == _util.sha_file(src), "repeat bytes differ"
        new_rows.append({"kit_id": kit_id, "item_id": new_item, "presentation_index": -1,
                         "unit_id": u, "candidate_id": str(orig["candidate_id"]), "pass": 2,
                         "repeat_of_item": orig_item, "render_sha": sha,
                         "layout": str(orig["layout"]), "stratum": str(orig["stratum"])})
    key = pd.concat([key, pd.DataFrame(new_rows, columns=list(schema.KEY_COLS))],
                    ignore_index=True)
    new_pos = {it: i + 1 for i, it in enumerate(queue)}
    key["presentation_index"] = key["item_id"].map(new_pos).astype(int)
    key = key.sort_values("presentation_index").reset_index(drop=True)
    # invariants: graded prefix untouched; every non-forced repeat >= min_gap after its
    # original (later insertions can only push a repeat further from its original)
    for it in graded_items:
        assert new_pos[it] == pos_of[it], f"graded item {it} moved"
    for row in new_rows:
        if row["unit_id"] in forced:
            continue
        assert new_pos[row["item_id"]] - new_pos[row["repeat_of_item"]] >= min_gap, \
            f"gap rule violated for {row['item_id']}"
    if forced:
        print(f"WARNING: {len(forced)} repeat(s) appended with gap < {min_gap} (tail too short)",
              flush=True)
    key_sha = _util.pin(key, key_path)
    manifest = make_manifest(kit_id, int(cur["kit_version"]) + 1, cur["grader_id"], queue)
    write_kit_files(kit_dir, manifest, frame, key, keys_dir, key_sha, collision_dirs)
    gaps = [int(new_pos[r.item_id] - new_pos[r.repeat_of_item])
            for r in key[key["pass"] == 2].itertuples()]
    print(f"kit {kit_id} v{manifest['kit_version']}: +{len(new_rows)} repeats -> {len(queue)} "
          f"items; tail starts at {tail_start}; gaps min/median/max "
          f"{min(gaps)}/{int(np.median(gaps))}/{max(gaps)}; key sha {key_sha}; "
          f"manifest_sha {manifest['manifest_sha']}", flush=True)
    return key_path


# ------------------------------------------------------------------- drop units
def _dropped_path(keys_dir: Path, kit_id: str) -> Path:
    return Path(keys_dir) / f"{kit_id}_dropped.csv"


def _dropped_item_ids(keys_dir: Path, kit_id: str) -> list[str]:
    """Item ids retired by --drop-units (never reused: an old event could name them)."""
    q = _dropped_path(keys_dir, kit_id)
    if not q.exists():
        return []
    return _util.read_pinned(q, dtype=str, keep_default_na=False)["item_id"].tolist()


def pick_drop_units(key: pd.DataFrame, graded_units: set, keep: dict[str, int]) -> list[str]:
    """Units to retire: for every stratum in `keep`, the pass-1 units of that stratum that
    are NOT graded, minus a deterministic core of size max(0, keep - n_graded) chosen by the
    lowest `hash01(unit_id, "pilot_core")` — a rule fixed before the campaign, so the cut
    cannot be improvised on what the images look like. Graded units are always kept."""
    k1 = key[key["pass"] == 1]
    drop: list[str] = []
    for stratum, n_keep in keep.items():
        units = sorted(set(k1.loc[k1["stratum"].astype(str) == stratum, "unit_id"]))
        graded = [u for u in units if u in graded_units]
        ungraded = sorted((u for u in units if u not in graded_units),
                          key=lambda u: (_util.hash01(u, "pilot_core"), u))
        n_core = max(0, int(n_keep) - len(graded))
        drop += ungraded[n_core:]
        print(f"drop-units {stratum}: {len(units)} in kit, {len(graded)} graded (kept), core "
              f"{min(n_core, len(ungraded))} kept, {len(ungraded) - min(n_core, len(ungraded))} dropped "
              f"(quota {n_keep})", flush=True)
    return drop


def drop_units(kit_id: str, frame_path: Path, grades_path: Path, keep: dict[str, int] | None = None,
               kits_root: Path = KITS_ROOT, keys_dir: Path = KEYS_DIR,
               collision_dirs=DEFAULT_COLLISION_DIRS) -> Path:
    """The pilot fallback as a new manifest version: retire ungraded items of the strata in
    `keep` (default PILOT_DROP_KEEP) without touching any graded item or regenerating the key.
    Retired key rows go to keys/<kit>_dropped.csv (pinned); their JPEGs are deleted from the
    kit; their ids are never reused."""
    keep = dict(PILOT_DROP_KEEP if keep is None else keep)
    frame = _read_frame(frame_path)
    kit_dir = Path(kits_root) / kit_id
    key_path = Path(keys_dir) / f"{kit_id}_key.csv"
    key = schema.read_key(key_path)
    history = load_manifest_history(keys_dir, kit_id)
    if not history:
        raise FileNotFoundError(f"no manifest history for {kit_id} under {keys_dir}")
    cur = history[-1]
    if cur["manifest_sha"] != json.loads((kit_dir / "kit_manifest.json").read_text())["manifest_sha"]:
        raise ValueError("kit_manifest.json in the kit dir is not the latest recorded version")
    grades = _util.read_pinned(grades_path, dtype={"unit_id": str, "item_id": str, "kit_id": str},
                               keep_default_na=False)
    g = grades[grades["kit_id"].astype(str) == kit_id]
    if g.empty:
        raise ValueError("no graded items for this kit in the grades file — run collect first")
    graded_items = set(g["item_id"].astype(str))
    graded_units = set(g["unit_id"].astype(str))
    units = pick_drop_units(key, graded_units, keep)
    if not units:
        print("drop-units: nothing to drop (quotas already met by graded items)", flush=True)
        return key_path
    gone = key[key["unit_id"].isin(units)]
    assert not gone["item_id"].isin(graded_items).any(), "refusing to drop a graded item"
    assert (gone["pass"] == 1).all(), "a repeat of an ungraded unit cannot exist"
    queue = [it["item_id"] for it in cur["items"]]
    pos_of = {it: i + 1 for i, it in enumerate(queue)}
    gone_items = set(gone["item_id"])
    queue = [it for it in queue if it not in gone_items]
    new_pos = {it: i + 1 for i, it in enumerate(queue)}
    moved = [it for it in graded_items if it in pos_of and new_pos.get(it) != pos_of[it]]
    if moved:
        print(f"WARNING: {len(moved)} graded item(s) change position (ungraded holes before them); "
              f"their events stay valid under the manifest version they were recorded with", flush=True)
    for it in gone_items:
        (kit_dir / "items" / f"{it}.jpg").unlink(missing_ok=True)
    keep_key = key[~key["unit_id"].isin(units)].copy()
    keep_key["presentation_index"] = keep_key["item_id"].map(new_pos).astype(int)
    keep_key = keep_key.sort_values("presentation_index").reset_index(drop=True)
    # dropped sidecar: append (a second --drop-units must not lose the first)
    dropped = gone.copy()
    dropped["dropped_in_version"] = int(cur["kit_version"]) + 1
    dp = _dropped_path(keys_dir, kit_id)
    if dp.exists():
        dropped = pd.concat([_util.read_pinned(dp, dtype=str, keep_default_na=False), dropped],
                            ignore_index=True)
    _util.pin(dropped, dp)
    key_sha = _util.pin(keep_key, key_path)
    manifest = make_manifest(kit_id, int(cur["kit_version"]) + 1, cur["grader_id"], queue)
    write_kit_files(kit_dir, manifest, frame, keep_key, keys_dir, key_sha, collision_dirs)
    print(f"kit {kit_id} v{manifest['kit_version']}: -{len(gone)} items -> {len(queue)} items; "
          f"dropped {sorted(units)} -> {dp}; key sha {key_sha}; manifest_sha {manifest['manifest_sha']}",
          flush=True)
    return key_path


# ------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frame", type=Path, default=_util.HERE / "frame.csv")
    ap.add_argument("--kit-id", required=True)
    ap.add_argument("--grader-id", default="XH")
    ap.add_argument("--seed", type=int, default=KIT_SEED)
    ap.add_argument("--source-dir", type=Path, default=_util.HERE / "stamps")
    ap.add_argument("--fallback-dir", type=Path, default=None,
                    help="e.g. $JWST_REPO/top100_clean (the run's own v1 JPEGs; read-only)")
    ap.add_argument("--kits-root", type=Path, default=KITS_ROOT)
    ap.add_argument("--keys-dir", type=Path, default=KEYS_DIR)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing kit AND delete its key/manifest history (pre-campaign only)")
    ap.add_argument("--collision-dir", type=Path, action="append", default=None,
                    help="dir of *.jpg the grader already holds; served bytes must not match "
                         "(default: $JWST_REPO/top100_clean_scrambled, top100_clean, golden/kits_truth and golden/kits_truth_v2r)")
    ap.add_argument("--add-repeats", action="store_true")
    ap.add_argument("--drop-units", action="store_true",
                    help="pilot fallback: retire ungraded U_tail/N_unflagged items to --keep quotas")
    ap.add_argument("--keep", default=None,
                    help="--drop-units quotas as stratum=n,...; default U_tail=15,N_unflagged=10")
    ap.add_argument("--grades", type=Path, default=_util.HERE / "golden_grades.csv")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-gap", type=int, default=60)
    ap.add_argument("--min-per-level", type=int, default=8)
    a = ap.parse_args(argv)
    if a.add_repeats and a.drop_units:
        ap.error("--add-repeats and --drop-units are separate manifest versions; run them one at a time")
    if a.add_repeats:
        add_repeats(a.kit_id, a.frame, a.grades, a.n, a.min_gap, a.seed, a.kits_root,
                    a.keys_dir, a.min_per_level)
    elif a.drop_units:
        keep = None
        if a.keep:
            keep = {k.strip(): int(v) for k, v in (kv.split("=") for kv in a.keep.split(","))}
        drop_units(a.kit_id, a.frame, a.grades, keep, a.kits_root, a.keys_dir)
    else:
        dirs = tuple(a.collision_dir) if a.collision_dir else DEFAULT_COLLISION_DIRS
        build(a.frame, a.kit_id, a.grader_id, a.seed, a.source_dir, a.fallback_dir,
              a.kits_root, a.keys_dir, a.force, dirs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
