"""golden/fewshot.py — in-context exemplar blocks from Xiaosheng's golden labels (E2 arm).

The golden labels are the first per-rater grades the program has, so the few-shot arm is
the first time an in-context reference set can say "this is what ONE named expert calls
an A" instead of "consensus grade A". This module turns `golden_labels.csv` (+ the kit
key, for the served JPEG bytes) into the content blocks grader_direct already understands:
the FEWSHOT_LEAD sentence, then per exemplar a one-line header, a `[composite]` label and
the byte-identical kit JPEG, then FEWSHOT_TRAIL.

Why caller-side and not the LENSJUDGE_FEWSHOT_MANIFEST env-var path: that path (a) is
skipped whenever `content=` is passed to grader_direct, (b) hard-codes "consensus grade",
and (c) routes every exemplar through fetch.get_cube, which for survey_key="jwst" silently
falls back to an ls-dr10 DESI cutout — a DESI image labelled as a JWST exemplar. Golden
runs therefore refuse to start if that variable is set at all.

Embargo: the PI's free-text comments are off limits to every model-facing artifact. The
only text this module can emit is the fixed template below; with embargo=True (default) it
raises if the labels carry any free-text column or if an emitted text block would deviate
from the template. The audit (golden/audit_traces.py) checks the same thing after the fact.

Eligibility (plan Phase 4): unit in the caller's eligible set (the align half), confidence
H, label_stable True when the unit was repeated (n_passes==1 otherwise), colour layout.
Pick per letter: highest confidence01, then _util.hash01(unit_id, "exemplar") — fully
deterministic, no RNG state. n = min(n_per_grade, available) per letter.
"""
from __future__ import annotations

import base64
import os
import re
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge.golden import _util  # noqa: E402
from lensjudge.imaging.grader_direct import FEWSHOT_LEAD, FEWSHOT_TRAIL  # noqa: E402

ENV_GUARD = "LENSJUDGE_FEWSHOT_MANIFEST"
KITS_DIR = _util.HERE / "kits"
LETTERS = ("A", "B", "C", "D")
# header label per letter: A/B read as LENS, C as POSSIBLE LENS (the rubric's own meaning
# of C), D as NON-LENS. Contract text said LENS iff score>=3; the WP-E brief refined C.
HEADER_LABEL = {"A": "LENS", "B": "LENS", "C": "POSSIBLE LENS", "D": "NON-LENS"}
COMPOSITE_TAG = "[composite]"
# every text block this module may emit must match one of these exactly
_HEADER_RE = re.compile(
    r"^REFERENCE EXAMPLE -- (LENS|POSSIBLE LENS|NON-LENS) "
    r"\(expert score [1-4]/4 = [A-D], confidence [LMH]\)$")
# label columns that would carry free text if someone added them to the labels CSV
_FREE_TEXT_COLS = ("note", "notes", "comment", "comments", "rationale", "text", "remark")


def header_text(score: int, letter: str, conf: str) -> str:
    return (f"REFERENCE EXAMPLE -- {HEADER_LABEL[letter]} "
            f"(expert score {int(score)}/4 = {letter}, confidence {conf})")


def assert_env_clean() -> None:
    """Golden runs must not have the DESI env-var few-shot path armed (see module doc)."""
    if os.environ.get(ENV_GUARD):
        raise RuntimeError(
            f"{ENV_GUARD} is set ({os.environ[ENV_GUARD]!r}); the env-var few-shot path fetches "
            "ls-dr10 DESI cutouts for survey_key=jwst. Unset it for golden runs.")


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "t", "yes")
    try:
        return bool(v) and not pd.isna(v)
    except (TypeError, ValueError):
        return bool(v)


def eligible(labels: pd.DataFrame, layouts: dict, eligible_units=None) -> pd.DataFrame:
    """Rows of `labels` that may serve as exemplars (see module doc)."""
    df = labels.copy()
    if eligible_units is not None:
        df = df[df["unit_id"].isin(set(eligible_units))]
    df = df[df["confidence_lmh"].astype(str).str.upper() == "H"]
    n_passes = pd.to_numeric(df.get("n_passes", 1), errors="coerce").fillna(1)
    stable = df.get("label_stable", pd.Series(True, index=df.index)).map(_truthy)
    df = df[(n_passes <= 1) | stable]
    df = df[df["unit_id"].map(lambda u: layouts.get(u) == "color")]
    df = df[df["grade_letter"].isin(LETTERS)]
    return df


def _pick(df: pd.DataFrame, n: int, seed: int) -> list[str]:
    """Deterministic per-letter pick: highest confidence01, then the unit hash."""
    salt = "exemplar" if seed == _util.SEED else f"exemplar:{seed}"
    key = [(-float(r.confidence01), _util.hash01(str(r.unit_id), salt), str(r.unit_id))
           for r in df.itertuples()]
    return [u for _, _, u in sorted(key)[:n]]


def _image_lookup(key_df_or_lookup, kits_dir: Path) -> tuple[Callable[[str], Optional[Path]], dict]:
    """Normalise the second argument into (unit_id -> path) + (unit_id -> layout).

    A key DataFrame (golden/keys/<kit>_key.csv schema) resolves pass-1 rows to
    kits/<kit_id>/items/<item_id>.jpg; a dict or callable is used as-is (layout then comes
    from an optional `layout` column on the labels, else defaults to "color")."""
    if isinstance(key_df_or_lookup, pd.DataFrame):
        key = key_df_or_lookup
        k1 = key[pd.to_numeric(key["pass"], errors="coerce").fillna(1) == 1]
        paths = {str(r.unit_id): Path(kits_dir) / str(r.kit_id) / "items" / f"{str(r.item_id).zfill(3)}.jpg"
                 for r in k1.itertuples()}
        layouts = {str(r.unit_id): str(r.layout) for r in k1.itertuples()} if "layout" in key else {}
        return (lambda u: paths.get(u)), layouts
    if callable(key_df_or_lookup):
        return key_df_or_lookup, {}
    d = dict(key_df_or_lookup)
    return (lambda u: Path(d[u]) if u in d else None), {}


def _jpeg_block(path: Path, want_sha: Optional[str]) -> dict:
    data = Path(path).read_bytes()
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"{path}: not a JPEG (media_type would lie)")
    got = _util.sha_file(path)   # sha16 of the served bytes == key.render_sha
    if isinstance(want_sha, str) and want_sha.strip() and want_sha.strip() != got:
        raise ValueError(f"{path}: sha {got} != render_sha {want_sha} (kit JPEG differs from the graded one)")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(data).decode()}}


def build_exemplar_blocks(labels_df: pd.DataFrame, key_df_or_image_lookup, n_per_grade: int = 3,
                          seed: int = _util.SEED, embargo: bool = True, eligible_units=None,
                          kits_dir: Path = KITS_DIR) -> tuple[list, list[str]]:
    """-> (content blocks, exemplar unit_ids). Empty lists when nothing is eligible.

    labels_df: golden_labels.csv rows. key_df_or_image_lookup: the kit key DataFrame, or a
    {unit_id: jpeg path} dict / callable. eligible_units: the split the caller allows (pass
    the align half; None = no split restriction, which only a test should do)."""
    assert_env_clean()
    if embargo:
        for c in _FREE_TEXT_COLS:
            if c in labels_df.columns and labels_df[c].map(lambda v: isinstance(v, str) and v.strip() != "").any():
                raise RuntimeError(f"embargo: labels carry a free-text column {c!r}; exemplars may only emit the template")
    lookup, layouts = _image_lookup(key_df_or_image_lookup, kits_dir)
    if not layouts:
        if "layout" in labels_df.columns:
            layouts = dict(zip(labels_df["unit_id"].astype(str), labels_df["layout"].astype(str)))
        else:
            layouts = {u: "color" for u in labels_df["unit_id"].astype(str)}
    el = eligible(labels_df.assign(unit_id=labels_df["unit_id"].astype(str)), layouts, eligible_units)
    if el.empty:
        return [], []
    by_unit = el.set_index("unit_id")
    chosen: list[str] = []
    for letter in LETTERS:
        sub = el[el["grade_letter"] == letter]
        if len(sub):
            chosen.extend(_pick(sub, min(n_per_grade, len(sub)), seed))
    blocks: list = [{"type": "text", "text": FEWSHOT_LEAD}]
    for u in chosen:
        r = by_unit.loc[u]
        path = lookup(u)
        if path is None or not Path(path).exists():
            raise FileNotFoundError(f"exemplar {u}: no kit JPEG ({path})")
        blocks.append({"type": "text", "text": header_text(int(r["score_1_4"]), str(r["grade_letter"]),
                                                            str(r["confidence_lmh"]).upper())})
        blocks.append({"type": "text", "text": COMPOSITE_TAG})
        blocks.append(_jpeg_block(Path(path), r.get("render_sha")))
    blocks.append({"type": "text", "text": FEWSHOT_TRAIL})
    if embargo:
        check_embargo(blocks)
    return blocks, chosen


def check_embargo(blocks: list) -> None:
    """Raise unless every text block is one of: FEWSHOT_LEAD, FEWSHOT_TRAIL, the header
    template, or the [composite] tag. Called on build and reusable by the audit."""
    for b in blocks:
        if b.get("type") != "text":
            continue
        t = b.get("text", "")
        if t in (FEWSHOT_LEAD, FEWSHOT_TRAIL, COMPOSITE_TAG) or _HEADER_RE.match(t):
            continue
        raise RuntimeError(f"embargo: exemplar text block deviates from the template: {t[:120]!r}")
