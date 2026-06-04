"""Pydantic models for grades, votes, spectroscopic verdicts, and review forms.

Because the Python @tool ``structuredContent`` is not forwarded to the model and
the SDK's final result is free text, the grade contract is enforced here: the
system prompt asks for exactly this JSON shape, and ``parse.parse_model`` validates
the model's final text into one of these. Validators are lenient (clamp/coerce)
so a slightly-off emission is repaired rather than discarded.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

GRADES = ("A", "B", "C", "D")
CRITERIA = ("blue_source", "low_surface_brightness", "curvature",
            "counter_images", "arc_morphology")


def _clamp10(v):
    try:
        return max(0, min(10, int(round(float(v)))))
    except Exception:
        return 0


class CriteriaScores(BaseModel):
    """The five Huang-2020 visual criteria, each 0-10."""
    blue_source: int = 0               # small blue galaxy(s) 1-5" from a red galaxy
    low_surface_brightness: int = 0    # source is low surface brightness
    curvature: int = 0                 # curving toward the red (lens) galaxy
    counter_images: int = 0            # counter / multiple images of similar color
    arc_morphology: int = 0            # elongated / arc-like morphology
    model_config = {"extra": "ignore"}

    @field_validator("*", mode="before")
    @classmethod
    def _coerce(cls, v):
        return _clamp10(v)


class ImageGrade(BaseModel):
    """A single integrated A/B/C/D grade of an imaging candidate."""
    grade: Literal["A", "B", "C", "D"]
    criteria: CriteriaScores = Field(default_factory=CriteriaScores)
    p_lens: float = Field(0.0, ge=0.0, le=1.0)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    contaminant: Optional[str] = None         # 'spiral'|'cosmic_ray'|'star_halo'|... or None
    escalate_to_human: bool = False
    rationale: str = ""
    model_config = {"extra": "ignore"}

    @field_validator("grade", mode="before")
    @classmethod
    def _grade(cls, v):
        s = str(v).strip().upper()[:1]
        return s if s in GRADES else "D"

    @field_validator("p_lens", "confidence", mode="before")
    @classmethod
    def _unit(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return 0.0


class JudgeVote(BaseModel):
    """One judge's verdict in the Option-C perspective-diverse panel."""
    role: str = ""
    grade: Literal["A", "B", "C", "D"]
    p_lens: float = Field(0.0, ge=0.0, le=1.0)
    criteria: CriteriaScores = Field(default_factory=CriteriaScores)
    contaminant: Optional[str] = None
    rationale: str = ""
    model_config = {"extra": "ignore"}

    @field_validator("grade", mode="before")
    @classmethod
    def _grade(cls, v):
        s = str(v).strip().upper()[:1]
        return s if s in GRADES else "D"

    @field_validator("p_lens", mode="before")
    @classmethod
    def _unit(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return 0.0


class SpecGrade(BaseModel):
    """Verdict for a spectroscopic (discordant-redshift pair) candidate."""
    cls: Literal["lens", "dimple", "not_lens"]
    plausible: bool = False
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    z_fg: Optional[float] = None
    z_bg: Optional[float] = None
    sigma_v: Optional[float] = None
    rationale: str = ""
    model_config = {"extra": "ignore"}

    @field_validator("cls", mode="before")
    @classmethod
    def _cls(cls, v):
        s = str(v).strip().lower().replace("-", "_").replace(" ", "_")
        return s if s in ("lens", "dimple", "not_lens") else "not_lens"

    @field_validator("confidence", mode="before")
    @classmethod
    def _unit(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return 0.0


class ReviewForm(BaseModel):
    """Pre-filled human-in-loop review record for an escalated candidate."""
    name: str
    ra: float
    dec: float
    grade: str
    p_lens: float
    confidence: float
    reason: str                      # why escalated
    rationale: str = ""
    ml_p_meta: Optional[float] = None
    png_paths: dict[str, str] = Field(default_factory=dict)
    model_config = {"extra": "ignore"}
