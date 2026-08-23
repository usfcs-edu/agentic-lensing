"""golden/schema.py — the human-record contract: GradeEvent, GradeRecord, GoldenLabel.

The golden dataset is one expert's own score (1-4, Huang VI scale) plus sureness (L/M/H)
on a blind kit. Three things make this record different from every other grade in the repo,
and this module pins all three:

1. **Nothing is coerced.** `common.schemas.ImageGrade` is deliberately lenient (a stray
   grade becomes "D", extras are ignored) because it repairs *model* output. A human record
   must never be repaired: a "5", a lowercase "l", a boolean where an int belongs, all raise.
   Every model here is `extra="forbid"` + strict scalars, and `_util.score_to_letter` (strict)
   is the only score->letter path. The human record never passes through ImageGrade.
2. **No smuggled text.** The PI's comments are embargoed; the record is grade + confidence
   ONLY. `GradeEvent` has the 14 contract keys and nothing else — `events_from_jsonl` turns
   any extra key into a loud "smuggled field" error so a free-text column can never ride
   along into `golden_grades.csv`.
3. **Column names are the contract.** `RECORD_COLS` / `LABEL_COLS` / `KEY_COLS` are the
   exact CSV headers of golden_grades.csv, golden_labels.csv and keys/<kit>_key.csv, in
   order; downstream packages (stats, splits, fewshot, SFT) read them by name.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import (BaseModel, ConfigDict, Field, StrictBool, StrictInt, ValidationError,
                      field_validator)

from lensjudge.golden import _util

TOOL_EVENT_KEYS = (
    "kit_id", "manifest_sha", "item_id", "presentation_index", "session_id", "grader_id",
    "score_1_4", "confidence_lmh", "seconds", "revision", "flag", "timestamp", "ua",
    "tool_version",
)


def _check_iso8601(v: str) -> str:
    """Accept only a tz-aware ISO-8601 string (what JS `toISOString()` emits: ...Z)."""
    if not isinstance(v, str) or not v.strip():
        raise ValueError("timestamp must be an ISO-8601 string")
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00") if v.endswith("Z") else v)
    except ValueError as e:
        raise ValueError(f"timestamp {v!r} is not ISO-8601: {e}") from e
    if dt.tzinfo is None:
        raise ValueError(f"timestamp {v!r} must carry a UTC offset (use ...Z)")
    return v


class GradeEvent(BaseModel):
    """One commit from the grading tool (one JSON line in records/events_*.jsonl)."""
    model_config = ConfigDict(extra="forbid")

    kit_id: str
    manifest_sha: str
    item_id: str
    presentation_index: StrictInt
    session_id: str
    grader_id: str
    score_1_4: StrictInt
    confidence_lmh: Literal["L", "M", "H"]
    seconds: float
    revision: StrictInt
    flag: StrictBool
    timestamp: str
    ua: str
    tool_version: str

    @field_validator("score_1_4")
    @classmethod
    def _score(cls, v):
        if v not in (1, 2, 3, 4):
            raise ValueError(f"score_1_4 must be 1..4, got {v!r}")
        return v

    @field_validator("revision", "presentation_index")
    @classmethod
    def _ge1(cls, v, info):
        if v < 1:
            raise ValueError(f"{info.field_name} must be >= 1, got {v!r}")
        return v

    @field_validator("seconds")
    @classmethod
    def _seconds(cls, v):
        # NaN is allowed (sheet-derived rows have no timer); negatives are not.
        if not math.isnan(v) and v < 0:
            raise ValueError(f"seconds must be >= 0, got {v!r}")
        return float(v)

    @field_validator("timestamp")
    @classmethod
    def _ts(cls, v):
        return _check_iso8601(v)

    @field_validator("item_id", "kit_id", "session_id", "grader_id", "manifest_sha")
    @classmethod
    def _nonempty(cls, v, info):
        if not str(v).strip():
            raise ValueError(f"{info.field_name} must be non-empty")
        return v

    @property
    def dt(self) -> datetime:
        v = self.timestamp
        return datetime.fromisoformat(v.replace("Z", "+00:00") if v.endswith("Z") else v)


class GradeRecord(BaseModel):
    """One row of golden_grades.csv: one (unit_id, pass), last revision wins."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    unit_id: str
    candidate_id: str
    kit_id: str
    item_id: str
    pass_: StrictInt = Field(alias="pass")   # `pass` is a keyword; CSV column is `pass`
    session_id: str
    presentation_index: StrictInt
    score_1_4: StrictInt
    grade_letter: Literal["A", "B", "C", "D"]
    confidence_lmh: Literal["L", "M", "H"]
    confidence01: float
    seconds: float
    revision_count: StrictInt
    flag: StrictBool
    render_sha: str
    manifest_sha: str
    render_version: str
    grade_scale: str
    timestamp: str
    grader_id: str

    @field_validator("pass_")
    @classmethod
    def _pass(cls, v):
        if v not in (1, 2):
            raise ValueError(f"pass must be 1 or 2, got {v!r}")
        return v

    @field_validator("score_1_4")
    @classmethod
    def _score(cls, v):
        if v not in (1, 2, 3, 4):
            raise ValueError(f"score_1_4 must be 1..4, got {v!r}")
        return v

    @field_validator("grade_scale")
    @classmethod
    def _scale(cls, v):
        if v != _util.GRADE_SCALE:
            raise ValueError(f"grade_scale must be {_util.GRADE_SCALE!r}, got {v!r}")
        return v

    @field_validator("timestamp")
    @classmethod
    def _ts(cls, v):
        return _check_iso8601(v)

    def check_consistent(self) -> "GradeRecord":
        """Cross-field pins: letter and confidence01 are functions of the score/lmh."""
        if self.grade_letter != _util.score_to_letter(self.score_1_4):
            raise ValueError(f"{self.unit_id}: grade_letter {self.grade_letter} != "
                             f"score_to_letter({self.score_1_4})")
        if abs(self.confidence01 - _util.CONF_TO_01[self.confidence_lmh]) > 1e-9:
            raise ValueError(f"{self.unit_id}: confidence01 {self.confidence01} != "
                             f"CONF_TO_01[{self.confidence_lmh}]")
        return self


class GoldenLabel(BaseModel):
    """One row of golden_labels.csv: one unit; canonical label = pass 1."""
    model_config = ConfigDict(extra="forbid")

    unit_id: str
    candidate_id: str
    ra_deg: float
    dec_deg: float
    stratum: str
    score_1_4: StrictInt
    grade_letter: Literal["A", "B", "C", "D"]
    confidence_lmh: Literal["L", "M", "H"]
    confidence01: float
    pass2_score_1_4: Optional[StrictInt] = None
    pass2_confidence_lmh: Optional[Literal["L", "M", "H"]] = None
    n_passes: StrictInt
    label_stable: Optional[StrictBool] = None
    render_sha: str
    grade_scale: str
    grader_id: str

    @field_validator("score_1_4", "pass2_score_1_4")
    @classmethod
    def _score(cls, v, info):
        if v is not None and v not in (1, 2, 3, 4):
            raise ValueError(f"{info.field_name} must be 1..4, got {v!r}")
        return v

    @field_validator("n_passes")
    @classmethod
    def _np(cls, v):
        if v not in (1, 2):
            raise ValueError(f"n_passes must be 1 or 2, got {v!r}")
        return v


# Exact CSV headers (order matters: downstream reads by name, diffs by position).
EVENT_KEYS = tuple(GradeEvent.model_fields)
assert EVENT_KEYS == TOOL_EVENT_KEYS, "GradeEvent drifted from the contract key list"
RECORD_COLS = tuple(f.alias or n for n, f in GradeRecord.model_fields.items())
LABEL_COLS = tuple(GoldenLabel.model_fields)
KEY_COLS = ("kit_id", "item_id", "presentation_index", "unit_id", "candidate_id", "pass",
            "repeat_of_item", "render_sha", "layout", "stratum")
# Columns that must be read as text (zero-padded ids; '' must stay '', not NaN).
KEY_STR_COLS = ("kit_id", "item_id", "unit_id", "candidate_id", "repeat_of_item",
                "render_sha", "layout", "stratum")


def events_from_jsonl(path: Path) -> list[GradeEvent]:
    """Parse one events file. Any extra key => ValueError naming the smuggled field; any
    malformed value => ValueError with the line number. Blank lines are skipped."""
    path = Path(path)
    out: list[GradeEvent] = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: not JSON ({e})") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: event must be a JSON object")
            extra = sorted(set(obj) - set(EVENT_KEYS))
            if extra:
                raise ValueError(f"{path}:{lineno}: smuggled field(s) {extra} — events carry "
                                 f"exactly {list(EVENT_KEYS)}; no free text is accepted")
            try:
                out.append(GradeEvent(**obj))
            except ValidationError as e:
                raise ValueError(f"{path}:{lineno}: invalid event: {e}") from e
    return out


def read_key(path: Path) -> pd.DataFrame:
    """Read a pinned kit key with ids as text and '' preserved (never NaN)."""
    df = _util.read_pinned(path, dtype={c: str for c in KEY_STR_COLS}, keep_default_na=False)
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: key is missing columns {missing}")
    df["pass"] = df["pass"].astype(int)
    df["presentation_index"] = df["presentation_index"].astype(int)
    return df[list(KEY_COLS)]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
