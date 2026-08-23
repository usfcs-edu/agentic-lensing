#!/usr/bin/env python3
"""golden/registry.py — per-unit exposure ledger for the golden set (golden/golden_registry.csv).

The program learned three times that a number tuned and scored on the same rows does not
replicate (V2 B7, V5 Phase E, parity C3). The golden `validate` half is scored once per
registered arm and must never have been shown to the model as an exemplar or a training row.
This ledger is where that is enforced: every consumer that puts a unit in front of a model
calls `mark_exposed(...)`, and every `validate` evaluation that uses `align` exemplars calls
`assert_unexposed(...)` first and dies if any validate unit was ever an exemplar / SFT row.

Columns (contract): unit_id, candidate_id, split, kit_ids, n_passes, grade_letter,
render_version, render_sha, leak, exposed_runs, in_fewshot, in_sft. The three exposure columns
are '|'-joined run tags and are NEVER overwritten by `sync_from` (only appended by
`mark_exposed`). `leak` = 'desi_train' when frame.desi_pool_overlap is set (the unit is within
2" of a parity pool / bench row: a DESI-trained student has seen its twin, mostly as a
`random_neg`), else 'no'. The CSV is SHA-pinned (`_util.pin`) and tracked in git.

API
  registry.load(path) / registry.save(df, path)
  registry.seed_from_frame(frame_df, path=...)   # pre-campaign rows (blank split/grade) so that
                                                 # mark_exposed works before any label exists
  registry.sync_from(labels_df, splits_df, frame_df=None, grades_df=None, path=...)
  registry.mark_exposed(unit_ids, run_tag, kind in {"eval","fewshot","sft"}, path=...)
  registry.assert_unexposed(unit_ids, kinds=("fewshot","sft"), path=...)   # raises ExposureError

Pre-campaign seeding: the truth evaluation (Part 2, run_truth_eval.py) scores frame units
zero-shot before Xiaosheng's labels exist — a permitted `eval` exposure that must still be
on the ledger. `seed_from_frame` creates one row per frame unit with split / grade blank
and `leak` from the frame; `sync_from` later fills split / grade / kit columns on those
rows and, as always, never touches the exposure columns.

CLI (run from reproductions/):
  python -m lensjudge.golden.registry seed [--frame golden/frame.csv]
  python -m lensjudge.golden.registry sync [--labels ... --splits ... --frame ... --grades ...]
  python -m lensjudge.golden.registry show [--units u0001 u0002]
  python -m lensjudge.golden.registry assert --units u0001 ... [--kinds fewshot sft]
  python -m lensjudge.golden.registry mark --units ... --run-tag e2_sonnet_r1 --kind fewshot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge.golden import _util  # noqa: E402

GOLDEN = _util.HERE
REGISTRY = GOLDEN / "golden_registry.csv"
COLS = ["unit_id", "candidate_id", "split", "kit_ids", "n_passes", "grade_letter",
        "render_version", "render_sha", "leak", "exposed_runs", "in_fewshot", "in_sft"]
KIND_COL = {"eval": "exposed_runs", "fewshot": "in_fewshot", "sft": "in_sft"}
EXPOSURE_COLS = tuple(KIND_COL.values())
RENDER_VERSION = "jwst_v1"


class ExposureError(RuntimeError):
    """A unit that must be unexposed has already been shown to a model."""


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=str) for c in COLS})


def load(path: Path = REGISTRY) -> pd.DataFrame:
    """Pinned CSV -> DataFrame, every column str ('' for empty), unit_id unique."""
    path = Path(path)
    if not path.exists():
        return _empty()
    df = _util.read_pinned(path, dtype=str, keep_default_na=False)
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[COLS].fillna("")
    assert df["unit_id"].is_unique, "registry unit_id must be unique"
    return df.reset_index(drop=True)


def save(df: pd.DataFrame, path: Path = REGISTRY) -> str:
    df = df[COLS].fillna("").astype(str).sort_values("unit_id").reset_index(drop=True)
    return _util.pin(df, Path(path))


def _join(existing: str, tags) -> str:
    """'|'-joined set union, insertion-ordered, no blanks."""
    have = [t for t in str(existing).split("|") if t]
    for t in tags:
        t = str(t).strip()
        if t and t not in have:
            have.append(t)
    return "|".join(have)


def _leak_from_frame(frame: pd.DataFrame | None) -> pd.Series:
    """unit_id -> 'desi_train' | 'no' from frame.desi_pool_overlap (empty when absent)."""
    if frame is None or "desi_pool_overlap" not in frame.columns:
        return pd.Series(dtype=str)
    ov = frame.set_index(frame["unit_id"].astype(str))["desi_pool_overlap"].astype(str).str.lower().isin(("true", "1", "1.0"))
    return ov.map({True: "desi_train", False: "no"})


def seed_from_frame(frame: pd.DataFrame, path: Path = REGISTRY) -> pd.DataFrame:
    """One registry row per frame unit that does not have one yet: candidate_id + leak from
    the frame, split / kit_ids / n_passes / grade_letter / render_sha blank, render_version
    RENDER_VERSION, every exposure column ''. Existing rows (and their exposures) are left
    exactly as they are, so this is safe to call before every pre-campaign run. Saves +
    returns the registry."""
    reg = load(path).set_index("unit_id")
    leak = _leak_from_frame(frame)
    n_new = 0
    for _, r in frame.iterrows():
        u = str(r["unit_id"])
        if u in reg.index:
            continue
        row = {c: "" for c in COLS if c != "unit_id"}
        row["candidate_id"] = str(r.get("candidate_id", ""))
        row["render_version"] = RENDER_VERSION
        row["leak"] = str(leak.get(u, "no"))
        reg.loc[u] = pd.Series(row)
        n_new += 1
    out = reg.reset_index()[COLS].fillna("").astype(str)
    save(out, path)
    if n_new:
        print(f"[registry] seeded {n_new} frame unit(s) with blank split -> {path}", flush=True)
    return load(path)


def sync_from(labels: pd.DataFrame, splits: pd.DataFrame, frame: pd.DataFrame | None = None,
              grades: pd.DataFrame | None = None, path: Path = REGISTRY) -> pd.DataFrame:
    """Create/update one row per labelled unit from labels (+ splits, frame, grades).
    Exposure columns of existing rows are preserved untouched; rows that are not in `labels`
    (e.g. units seeded by `seed_from_frame` and not yet graded) are kept as they are.
    Idempotent. Saves + returns."""
    reg = load(path).set_index("unit_id")
    sp = splits.set_index("unit_id")["split"].astype(str) if splits is not None and len(splits) else pd.Series(dtype=str)
    if frame is None and (GOLDEN / "frame.csv").exists():
        frame = _util.read_pinned(GOLDEN / "frame.csv")
    leak = _leak_from_frame(frame)
    kits = pd.Series(dtype=str)
    rv = pd.Series(dtype=str)
    if grades is not None and len(grades):
        g = grades.sort_values(["unit_id", "pass"])
        kits = g.groupby("unit_id")["kit_id"].agg(lambda s: "|".join(dict.fromkeys(s.astype(str))))
        if "render_version" in g.columns:
            rv = g.groupby("unit_id")["render_version"].agg(lambda s: "|".join(dict.fromkeys(s.astype(str))))

    for _, r in labels.iterrows():
        u = str(r["unit_id"])
        row = reg.loc[u].to_dict() if u in reg.index else {c: "" for c in COLS if c != "unit_id"}
        row["candidate_id"] = str(r.get("candidate_id", row.get("candidate_id", "")))
        row["split"] = str(sp.get(u, row.get("split", "") or ""))
        row["kit_ids"] = str(kits.get(u, row.get("kit_ids", "") or ""))
        row["n_passes"] = str(int(r["n_passes"])) if pd.notna(r.get("n_passes")) else row.get("n_passes", "")
        row["grade_letter"] = str(r.get("grade_letter", row.get("grade_letter", "")))
        row["render_version"] = str(rv.get(u, row.get("render_version", "") or RENDER_VERSION))
        row["render_sha"] = str(r.get("render_sha", row.get("render_sha", "")))
        row["leak"] = str(leak.get(u, row.get("leak", "") or "no"))
        for c in EXPOSURE_COLS:
            row[c] = row.get(c, "") or ""
        reg.loc[u] = pd.Series(row)
    out = reg.reset_index()[COLS].fillna("").astype(str)
    save(out, path)
    return load(path)


def mark_exposed(unit_ids, run_tag: str, kind: str, path: Path = REGISTRY) -> pd.DataFrame:
    """Append `run_tag` to the exposure column for `kind` on every unit. Unknown units raise
    (a consumer must sync the registry before it can expose anything)."""
    if kind not in KIND_COL:
        raise ValueError(f"kind must be one of {sorted(KIND_COL)}, got {kind!r}")
    if not run_tag or "|" in run_tag:
        raise ValueError("run_tag must be non-empty and must not contain '|'")
    reg = load(path).set_index("unit_id")
    ids = [str(u) for u in unit_ids]
    unknown = sorted(set(ids) - set(reg.index))
    if unknown:
        raise KeyError(f"units not in registry (run sync first): {unknown}")
    col = KIND_COL[kind]
    for u in ids:
        reg.loc[u, col] = _join(reg.loc[u, col], [run_tag])
    out = reg.reset_index()[COLS]
    save(out, path)
    return load(path)


def exposures(unit_ids, kinds=("fewshot", "sft"), path: Path = REGISTRY) -> dict[str, dict[str, str]]:
    """{unit_id: {column: tags}} for units with any non-empty exposure in `kinds`."""
    reg = load(path).set_index("unit_id")
    ids = [str(u) for u in unit_ids]
    unknown = sorted(set(ids) - set(reg.index))
    if unknown:
        raise KeyError(f"units not in registry (cannot vouch for them): {unknown}")
    bad: dict[str, dict[str, str]] = {}
    for u in ids:
        hits = {KIND_COL[k]: reg.loc[u, KIND_COL[k]] for k in kinds if reg.loc[u, KIND_COL[k]]}
        if hits:
            bad[u] = hits
    return bad


def assert_unexposed(unit_ids, kinds=("fewshot", "sft"), path: Path = REGISTRY) -> None:
    """Raise ExposureError naming every offending unit and the runs that exposed it."""
    for k in kinds:
        if k not in KIND_COL:
            raise ValueError(f"unknown kind {k!r}")
    bad = exposures(unit_ids, kinds=kinds, path=path)
    if bad:
        lines = [f"  {u}: " + ", ".join(f"{c}={t}" for c, t in hits.items()) for u, hits in bad.items()]
        raise ExposureError(f"{len(bad)} unit(s) already exposed via {list(kinds)}:\n" + "\n".join(lines))


# ------------------------------------------------------------------------------ CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["seed", "sync", "show", "assert", "mark"])
    ap.add_argument("--registry", default=str(REGISTRY))
    ap.add_argument("--labels", default=str(GOLDEN / "golden_labels.csv"))
    ap.add_argument("--splits", default=str(GOLDEN / "splits.csv"))
    ap.add_argument("--frame", default=str(GOLDEN / "frame.csv"))
    ap.add_argument("--grades", default=str(GOLDEN / "golden_grades.csv"))
    ap.add_argument("--units", nargs="*", default=None)
    ap.add_argument("--kinds", nargs="*", default=["fewshot", "sft"])
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--kind", default=None, choices=sorted(KIND_COL))
    a = ap.parse_args(argv)
    path = Path(a.registry)
    pd.set_option("display.width", 200, "display.max_rows", 500, "display.max_colwidth", 60)

    if a.cmd == "seed":
        reg = seed_from_frame(_util.read_pinned(a.frame), path=path)
        print(f"[registry] {len(reg)} units -> {path}  split counts {reg['split'].value_counts().to_dict()}")
    elif a.cmd == "sync":
        labels = _util.read_pinned(a.labels)
        splits = _util.read_pinned(a.splits) if Path(a.splits).exists() else None
        frame = _util.read_pinned(a.frame) if Path(a.frame).exists() else None
        grades = _util.read_pinned(a.grades) if Path(a.grades).exists() else None
        if splits is None:
            print(f"note: {a.splits} absent - split column left as-is/blank")
        reg = sync_from(labels, splits, frame=frame, grades=grades, path=path)
        print(f"[registry] {len(reg)} units -> {path}  split counts {reg['split'].value_counts().to_dict()}  "
              f"leak {reg['leak'].value_counts().to_dict()}")
    elif a.cmd == "show":
        reg = load(path)
        if a.units:
            reg = reg[reg["unit_id"].isin(a.units)]
        print(reg.to_string(index=False) if len(reg) else "(empty registry)")
    elif a.cmd == "assert":
        if not a.units:
            ap.error("assert needs --units")
        assert_unexposed(a.units, kinds=tuple(a.kinds), path=path)
        print(f"OK: {len(a.units)} unit(s) unexposed via {a.kinds}")
    elif a.cmd == "mark":
        if not (a.units and a.run_tag and a.kind):
            ap.error("mark needs --units, --run-tag and --kind")
        reg = mark_exposed(a.units, a.run_tag, a.kind, path=path)
        print(f"marked {len(a.units)} unit(s) {a.kind}={a.run_tag} -> {path}")


if __name__ == "__main__":
    main()
