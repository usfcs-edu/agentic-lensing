"""golden/_util.py — tiny shared helpers (SHA pinning, hashing, naming) for the golden package.

Kept dependency-free so every golden script and test can import it without side effects.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent          # reproductions/lensjudge/golden
LENSJUDGE = HERE.parent                         # reproductions/lensjudge
REPRO = LENSJUDGE.parent                        # reproductions/
RESEARCH = REPRO.parents[1]                     # ~/sync/research
JWST_REPO = Path(__import__("os").environ.get(
    "LENSJUDGE_JWST_REPO", RESEARCH / "jwst-strong-lens-search"))
SEED = 2026                                     # repo-wide sampling/bootstrap seed


def pin(df: pd.DataFrame, path: Path) -> str:
    """Write a CSV plus a sibling `.sha` (sha256[:16] of the CSV text) — the
    build_parity_bench._pin convention. Returns the short sha."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = df.to_csv(index=False)
    sha = hashlib.sha256(text.encode()).hexdigest()[:16]
    path.write_text(text)
    path.with_suffix(path.suffix + ".sha").write_text(sha + "\n")
    return sha


def read_pinned(path: Path, **kw) -> pd.DataFrame:
    """Read a pinned CSV and verify its `.sha` sidecar (raises on mismatch / missing)."""
    path = Path(path)
    sha_path = path.with_suffix(path.suffix + ".sha")
    text = path.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()[:16]
    if not sha_path.exists():
        raise FileNotFoundError(f"missing pin {sha_path}")
    want = sha_path.read_text().strip()
    if want != sha:
        raise ValueError(f"pin mismatch for {path}: file {sha} != pinned {want}")
    return pd.read_csv(path, **kw)


def sha_file(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n] if n else h.hexdigest()


def sha_text(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def sha_json(obj, n: int = 16) -> str:
    """Canonical-JSON sha (sorted keys, no whitespace) for manifests/registries."""
    return sha_text(json.dumps(obj, sort_keys=True, separators=(",", ":")), n)


def hash01(name: str, salt: str) -> float:
    """Deterministic [0,1) hash of (salt, name) — the distill_hsc._hash01 convention."""
    return int(hashlib.md5(f"{salt}:{name}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


_SAFE = re.compile(r"[^A-Za-z0-9_.+-]")


def safe_name(name: str) -> str:
    return _SAFE.sub("_", str(name))


SCORE_TO_LETTER = {4: "A", 3: "B", 2: "C", 1: "D"}
LETTER_TO_SCORE = {v: k for k, v in SCORE_TO_LETTER.items()}
SCORE_TO_ORD = {4: 3, 3: 2, 2: 1, 1: 0}          # intergrader_stats.SCORE_TO_ORD
LETTER_TO_ORD = {"A": 3, "B": 2, "C": 1, "D": 0}  # human_ceiling.GMAP
CONF_LEVELS = ("L", "M", "H")
CONF_TO_C = {"L": 0.33, "M": 0.67, "H": 1.0}      # sureness weight for soft targets
CONF_TO_01 = {"L": 0.40, "M": 0.70, "H": 0.90}    # ImageGrade.confidence mapping
GRADE_SCALE = "huang_vi_1to4"


def score_to_letter(score) -> str:
    """Strict 1-4 -> A-D. Raises on anything else (never coerce like ImageGrade does)."""
    s = int(score)
    if s not in SCORE_TO_LETTER:
        raise ValueError(f"score must be 1..4, got {score!r}")
    return SCORE_TO_LETTER[s]
