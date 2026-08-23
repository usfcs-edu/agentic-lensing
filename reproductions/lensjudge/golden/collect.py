"""golden/collect.py — tool events (+ optional paper sheet) ⋈ key -> golden_grades / golden_labels.

The grading tool emits one JSON line per commit (records/events_<kit>_<session>.jsonl; the
auto-downloads overlap, so the same event may appear in several files). This script is the
ONLY path from those events to the dataset:

  1. parse every events file through `schema.events_from_jsonl` — an event with any key
     beyond the 14 contract keys aborts the run (no smuggled free text, ever);
  2. check each event against the kit: its `manifest_sha` must be one of the kit's recorded
     manifest versions (keys/<kit>_manifests.jsonl; a kit gets a new version when hidden
     repeats are added or pilot units dropped), its item must exist in that version at that
     presentation index, and the latest version must have been written with the key that is
     on disk now (its recorded `key_sha` == the key's pin — a key regenerated underneath a
     campaign is refused, never silently joined); events recorded under a superseded version
     AFTER a newer one was built are reported (the grader kept an old grade.html open);
  3. join on item_id to the key (keys/<kit>_key.csv) — this is where an item becomes a
     (unit_id, pass): pass 2 rows are the byte-identical repeats;
  4. one row per (unit_id, pass), LAST REVISION WINS (Backspace re-grades emit revision+1;
     ties broken by timestamp), `revision_count` = number of commits recorded for it;
  5. golden_grades.csv = the new rows merged into the existing file (re-runs are idempotent:
     same inputs -> byte-identical CSV; rows for (kit, unit, pass) not covered by the given
     events are kept); golden_labels.csv = one row per unit, canonical label = pass 1,
     `pass2_*` and `label_stable` filled when a repeat was graded. Both tables are built in
     memory first and pinned together at the end, so a failure leaves neither half-updated.

Every output row is validated by `schema.GradeRecord` / `schema.GoldenLabel` (strict; the
human record never touches `common.schemas.ImageGrade`). Letters via `_util.score_to_letter`,
confidence01 via `_util.CONF_TO_01`.

    python golden/collect.py --kit-id jwst_v1_xh --events golden/records/events_*.jsonl
    python golden/collect.py --kit-id jwst_v1_xh --events ... --sheet kit/grading_sheet.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lensjudge.golden import _util, schema  # noqa: E402
from lensjudge.golden.build_kit import (KEYS_DIR, KITS_ROOT, RENDER_VERSION,  # noqa: E402
                                        key_pin_sha, load_manifest_history)

GRADES_PATH = _util.HERE / "golden_grades.csv"
LABELS_PATH = _util.HERE / "golden_labels.csv"
_MERGE_KEY = ["kit_id", "unit_id", "pass"]


# ---------------------------------------------------------------- loading
def load_events(paths: list[Path], kit_id: str) -> list[schema.GradeEvent]:
    """All events for `kit_id` across files, exact duplicates collapsed; any extra key or
    conflicting duplicate aborts."""
    seen: dict[tuple, schema.GradeEvent] = {}
    n_other = 0
    for p in paths:
        for ev in schema.events_from_jsonl(p):
            if ev.kit_id != kit_id:
                n_other += 1
                continue
            k = (ev.session_id, ev.item_id, ev.revision)
            if k in seen:
                a, b = seen[k].model_dump(), ev.model_dump()
                if a != b:
                    raise ValueError(f"conflicting duplicates for session={k[0]} item={k[1]} "
                                     f"revision={k[2]} (in {p})")
                continue
            seen[k] = ev
    if n_other:
        print(f"NOTE: skipped {n_other} event(s) belonging to other kits", flush=True)
    return list(seen.values())


def validate_against_kit(events: list[schema.GradeEvent], manifests: list[dict],
                         key: pd.DataFrame, key_sha: str | None = None) -> int:
    """manifest_sha known, item in that version at that index, grader matches, item in key;
    the latest manifest's recorded key_sha must equal `key_sha` (the key on disk) when both
    are known. Returns the number of events recorded under a superseded manifest version
    after the newer version was built (printed as a warning: the grader kept an old
    grade.html open, so those presentation indices are the OLD queue's)."""
    by_sha = {m["manifest_sha"]: m for m in manifests}
    latest = manifests[-1]
    if key_sha is not None and latest.get("key_sha") and latest["key_sha"] != key_sha:
        raise ValueError(f"the key on disk (pin {key_sha}) is not the key manifest "
                         f"v{latest.get('kit_version')} was written with ({latest['key_sha']}); "
                         f"a regenerated key cannot be joined to recorded events")
    key_items = set(key["item_id"])
    n_stale = 0
    built = str(latest.get("built_at", "")).replace("Z", "+00:00")
    latest_built = datetime.fromisoformat(built) if built else None
    for ev in events:
        m = by_sha.get(ev.manifest_sha)
        if m is None:
            raise ValueError(f"event {ev.session_id}/{ev.item_id}: manifest_sha "
                             f"{ev.manifest_sha} is not a recorded version of kit {ev.kit_id} "
                             f"(known: {sorted(by_sha)})")
        items = [it["item_id"] for it in m["items"]]
        if ev.item_id not in items:
            raise ValueError(f"event {ev.session_id}/{ev.item_id}: item not in manifest "
                             f"version {m['kit_version']}")
        pos = items.index(ev.item_id) + 1
        if ev.presentation_index != pos:
            raise ValueError(f"event {ev.session_id}/{ev.item_id}: presentation_index "
                             f"{ev.presentation_index} != {pos} in manifest v{m['kit_version']}")
        if ev.grader_id != m["grader_id"]:
            raise ValueError(f"event {ev.session_id}/{ev.item_id}: grader {ev.grader_id} != "
                             f"kit grader {m['grader_id']}")
        if ev.item_id not in key_items:
            raise ValueError(f"event {ev.session_id}/{ev.item_id}: item not in key")
        if m is not latest and latest_built is not None and ev.dt > latest_built:
            n_stale += 1
    if n_stale:
        print(f"WARNING: {n_stale} event(s) were recorded under a superseded manifest version "
              f"AFTER v{latest.get('kit_version')} was built — the grader was still on an old "
              f"grade.html; their presentation_index refers to the old queue", flush=True)
    return n_stale


def sheet_events(sheet: Path, kit_id: str, manifest: dict, key: pd.DataFrame,
                 have_items: set[str]) -> list[schema.GradeEvent]:
    """The paper/CSV fallback (item_id,score_1_4,confidence_lmh). Only items with NO tool
    event are taken from the sheet; no timing; session_id tags the sheet file."""
    df = pd.read_csv(sheet, dtype=str, keep_default_na=False)
    for c in ("item_id", "score_1_4", "confidence_lmh"):
        if c not in df.columns:
            raise ValueError(f"{sheet}: missing column {c}")
    pos = dict(zip(key["item_id"], key["presentation_index"]))
    ts = datetime.fromtimestamp(Path(sheet).stat().st_mtime, tz=timezone.utc)
    ts = ts.isoformat(timespec="seconds").replace("+00:00", "Z")
    session = f"sheet-{_util.sha_file(sheet)}"
    out, n_skip = [], 0
    for r in df.itertuples(index=False):
        s, c = str(r.score_1_4).strip(), str(r.confidence_lmh).strip().upper()
        if not s and not c:
            continue
        item = str(r.item_id).strip()
        if item.isdigit():
            item = f"{int(item):03d}"
        if item in have_items:
            n_skip += 1
            continue
        if item not in pos:
            raise ValueError(f"{sheet}: item {item!r} not in key")
        if not s.isdigit():
            raise ValueError(f"{sheet}: item {item}: score_1_4 {s!r} is not an integer")
        out.append(schema.GradeEvent(
            kit_id=kit_id, manifest_sha=manifest["manifest_sha"], item_id=item,
            presentation_index=int(pos[item]), session_id=session,
            grader_id=manifest["grader_id"], score_1_4=int(s), confidence_lmh=c,
            seconds=float("nan"), revision=1, flag=False, timestamp=ts, ua="sheet",
            tool_version="sheet"))
    if n_skip:
        print(f"sheet: {n_skip} row(s) ignored because the tool already recorded them", flush=True)
    return out


# ---------------------------------------------------------------- records
def build_records(events: list[schema.GradeEvent], key: pd.DataFrame, grader_id: str,
                  kit_id: str) -> pd.DataFrame:
    """One row per (unit_id, pass): last revision wins."""
    if not events:
        return pd.DataFrame(columns=list(schema.RECORD_COLS))
    ev = pd.DataFrame([e.model_dump() for e in events])
    k = key.set_index("item_id")
    ev = ev.join(k[["unit_id", "candidate_id", "pass", "render_sha"]], on="item_id")
    assert ev["unit_id"].notna().all(), "event item missing from key"
    rows = []
    for (unit, pas), g in ev.groupby(["unit_id", "pass"], sort=True):
        g = g.sort_values(["revision", "timestamp"], kind="stable")
        w = g.iloc[-1]
        rows.append({
            "unit_id": unit, "candidate_id": w["candidate_id"], "kit_id": kit_id,
            "item_id": w["item_id"], "pass": int(pas), "session_id": w["session_id"],
            "presentation_index": int(w["presentation_index"]),
            "score_1_4": int(w["score_1_4"]),
            "grade_letter": _util.score_to_letter(w["score_1_4"]),
            "confidence_lmh": w["confidence_lmh"],
            "confidence01": _util.CONF_TO_01[w["confidence_lmh"]],
            "seconds": float(w["seconds"]), "revision_count": int(len(g)),
            "flag": bool(w["flag"]), "render_sha": w["render_sha"],
            "manifest_sha": w["manifest_sha"], "render_version": RENDER_VERSION,
            "grade_scale": _util.GRADE_SCALE, "timestamp": w["timestamp"],
            "grader_id": grader_id,
        })
    df = pd.DataFrame(rows, columns=list(schema.RECORD_COLS))
    for r in df.to_dict("records"):
        schema.GradeRecord(**r).check_consistent()
    return df


def load_grades(path: Path) -> pd.DataFrame:
    """Read an existing golden_grades.csv with the contract dtypes ('' seconds -> NaN)."""
    df = _util.read_pinned(path, dtype=str, keep_default_na=False)
    missing = [c for c in schema.RECORD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df[list(schema.RECORD_COLS)].copy()
    for c in ("pass", "presentation_index", "score_1_4", "revision_count"):
        df[c] = df[c].astype(int)
    df["confidence01"] = df["confidence01"].astype(float)
    df["seconds"] = pd.to_numeric(df["seconds"].replace("", np.nan), errors="raise")
    df["flag"] = df["flag"].map({"True": True, "False": False})
    if df["flag"].isna().any():
        raise ValueError(f"{path}: flag column must be True/False")
    return df


def merge_grades(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Union on (kit_id, unit_id, pass); on a clash the later (timestamp, revision_count)
    wins, which makes a re-run over the same events a no-op."""
    if old is None or old.empty:
        out = new.copy()
    else:
        both = pd.concat([old, new], ignore_index=True)
        both = both.sort_values(_MERGE_KEY + ["timestamp", "revision_count"], kind="stable")
        out = both.drop_duplicates(_MERGE_KEY, keep="last")
    out = out.sort_values(["kit_id", "pass", "presentation_index"], kind="stable")
    return out.reset_index(drop=True)[list(schema.RECORD_COLS)]


# ---------------------------------------------------------------- labels
def build_labels(grades: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """One row per unit; canonical = pass 1 (earliest pass-1 row if a unit was graded in more
    than one kit); pass2_* from the same kit when present."""
    fr = frame.set_index("unit_id")
    rows = []
    multi = 0
    for unit, g in grades.groupby("unit_id", sort=True):
        p1 = g[g["pass"] == 1].sort_values(["timestamp", "kit_id"], kind="stable")
        if p1.empty:
            raise ValueError(f"{unit}: pass-2 grade without a pass-1 grade")
        if len(p1) > 1:
            multi += 1
        w = p1.iloc[0]
        p2 = g[(g["pass"] == 2) & (g["kit_id"] == w["kit_id"])]
        if p2.empty:
            p2 = g[g["pass"] == 2]
        if unit not in fr.index:
            raise ValueError(f"{unit}: not in frame")
        f = fr.loc[unit]
        has2 = not p2.empty
        s2 = int(p2.iloc[0]["score_1_4"]) if has2 else None
        rows.append({
            "unit_id": unit, "candidate_id": w["candidate_id"],
            "ra_deg": float(f["ra_deg"]), "dec_deg": float(f["dec_deg"]),
            "stratum": str(f["stratum"]), "score_1_4": int(w["score_1_4"]),
            "grade_letter": _util.score_to_letter(w["score_1_4"]),
            "confidence_lmh": w["confidence_lmh"],
            "confidence01": _util.CONF_TO_01[w["confidence_lmh"]],
            "pass2_score_1_4": s2,
            "pass2_confidence_lmh": p2.iloc[0]["confidence_lmh"] if has2 else None,
            "n_passes": 2 if has2 else 1,
            "label_stable": (int(w["score_1_4"]) == s2) if has2 else None,
            "render_sha": w["render_sha"], "grade_scale": _util.GRADE_SCALE,
            "grader_id": w["grader_id"],
        })
    if multi:
        print(f"NOTE: {multi} unit(s) have pass-1 grades in more than one kit; the earliest "
              f"is canonical", flush=True)
    for r in rows:                       # validate BEFORE pandas turns None into NaN
        schema.GoldenLabel(**r)
    df = pd.DataFrame(rows, columns=list(schema.LABEL_COLS))
    df["pass2_score_1_4"] = df["pass2_score_1_4"].astype("Int64")
    df["pass2_confidence_lmh"] = df["pass2_confidence_lmh"].fillna("")
    df["label_stable"] = df["label_stable"].astype("boolean")
    return df


# ---------------------------------------------------------------- report
def report(grades: pd.DataFrame, kit_id: str, events: list[schema.GradeEvent]) -> dict:
    g = grades[grades["kit_id"] == kit_id]
    out = {"kit_id": kit_id, "n_events": len(events),
           "n_sessions": len({e.session_id for e in events})}
    print(f"\n== {kit_id}: {len(events)} events in {out['n_sessions']} session(s) ==")
    for s, n in sorted(pd.Series([e.session_id for e in events]).value_counts().items()):
        print(f"  session {s}: {n} commits")
    for pas in (1, 2):
        gp = g[g["pass"] == pas]
        med = float(np.nanmedian(gp["seconds"])) if gp["seconds"].notna().any() else math.nan
        dist = {int(k): int(v) for k, v in gp["score_1_4"].value_counts().sort_index().items()}
        out[f"pass{pas}_n"] = int(len(gp)); out[f"pass{pas}_median_s"] = med
        print(f"  pass {pas}: {len(gp)} units, median {med:.1f} s, scores {dist}, "
              f"flagged {int(gp['flag'].sum())}, revised {int((gp['revision_count'] > 1).sum())}")
    p1 = g[g["pass"] == 1].set_index("unit_id")
    p2 = g[g["pass"] == 2].set_index("unit_id")
    both = p1.index.intersection(p2.index)
    out["n_repeat_pairs"] = int(len(both))
    if len(both):
        s1, s2 = p1.loc[both, "score_1_4"].to_numpy(), p2.loc[both, "score_1_4"].to_numpy()
        c1, c2 = p1.loc[both, "confidence_lmh"].to_numpy(), p2.loc[both, "confidence_lmh"].to_numpy()
        same_session = int((p1.loc[both, "session_id"].to_numpy()
                            == p2.loc[both, "session_id"].to_numpy()).sum())
        out["repeat_exact"] = float(np.mean(s1 == s2))
        out["repeat_onestep"] = float(np.mean(np.abs(s1 - s2) <= 1))
        out["repeat_conf_exact"] = float(np.mean(c1 == c2))
        print(f"  repeats: {len(both)} pairs, exact agreement {out['repeat_exact']:.3f}, "
              f"one-step {out['repeat_onestep']:.3f}, confidence exact "
              f"{out['repeat_conf_exact']:.3f}, mean drift (p2-p1) {float(np.mean(s2 - s1)):+.3f}")
        if same_session:
            print(f"  WARNING: {same_session} repeat pair(s) graded within the SAME session")
    return out


# ---------------------------------------------------------------- driver
def collect(kit_id: str, event_paths: list[Path], sheet: Path | None = None,
            frame_path: Path = _util.HERE / "frame.csv", kits_root: Path = KITS_ROOT,
            keys_dir: Path = KEYS_DIR, grades_path: Path = GRADES_PATH,
            labels_path: Path = LABELS_PATH) -> dict:
    manifests = load_manifest_history(keys_dir, kit_id)
    if not manifests:
        mp = Path(kits_root) / kit_id / "kit_manifest.json"
        if not mp.exists():
            raise FileNotFoundError(f"no manifest history for {kit_id} in {keys_dir} and no "
                                    f"{mp}")
        manifests = [json.loads(mp.read_text())]
    current = manifests[-1]
    key_path = Path(keys_dir) / f"{kit_id}_key.csv"
    key = schema.read_key(key_path)
    events = load_events(event_paths, kit_id)
    n_stale = validate_against_kit(events, manifests, key, key_pin_sha(keys_dir, kit_id))
    if sheet is not None:
        events += sheet_events(sheet, kit_id, current, key, {e.item_id for e in events})
    new = build_records(events, key, current["grader_id"], kit_id)
    old = load_grades(grades_path) if Path(grades_path).exists() else None
    merged = merge_grades(old, new)
    frame = _util.read_pinned(frame_path, dtype={"unit_id": str, "candidate_id": str,
                                                 "stratum": str}, keep_default_na=False)
    labels = build_labels(merged, frame)        # may raise: nothing has been written yet
    before = Path(grades_path).read_text() if Path(grades_path).exists() else None
    sha_g = _util.pin(merged, grades_path)
    changed = before != Path(grades_path).read_text()
    sha_l = _util.pin(labels, labels_path)
    out = report(merged, kit_id, events)
    out.update({"grades_sha": sha_g, "labels_sha": sha_l, "n_grades_rows": int(len(merged)),
                "n_labels": int(len(labels)), "grades_changed": bool(changed),
                "n_stale_manifest_events": int(n_stale)})
    print(f"  -> {grades_path} ({len(merged)} rows, sha {sha_g}"
          f"{', unchanged' if not changed else ''})\n  -> {labels_path} ({len(labels)} units, "
          f"sha {sha_l})", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kit-id", required=True)
    ap.add_argument("--events", nargs="+", required=True,
                    help="events JSONL files (globs accepted, e.g. golden/records/events_*.jsonl)")
    ap.add_argument("--sheet", type=Path, default=None,
                    help="optional grading_sheet.csv fallback (item_id,score_1_4,confidence_lmh)")
    ap.add_argument("--frame", type=Path, default=_util.HERE / "frame.csv")
    ap.add_argument("--kits-root", type=Path, default=KITS_ROOT)
    ap.add_argument("--keys-dir", type=Path, default=KEYS_DIR)
    ap.add_argument("--grades", type=Path, default=GRADES_PATH)
    ap.add_argument("--labels", type=Path, default=LABELS_PATH)
    a = ap.parse_args(argv)
    paths: list[Path] = []
    for pat in a.events:
        hits = sorted(glob.glob(pat))
        paths += [Path(h) for h in hits] if hits else [Path(pat)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"events file(s) not found: {missing}")
    collect(a.kit_id, paths, a.sheet, a.frame, a.kits_root, a.keys_dir, a.grades, a.labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
