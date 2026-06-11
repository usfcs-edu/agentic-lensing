"""Central paths, survey layers, render params, and model tiers for LensJudge.

All large reuse assets live in sibling reproduction dirs (read-only); LensJudge
writes only under ``reproductions/lensjudge/{outputs,cache}`` (both gitignored).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- repo layout ------------------------------------------------------------
HERE = Path(__file__).resolve().parent              # reproductions/lensjudge/
REPRO = HERE.parent                                  # reproductions/
INCH = REPRO / "inchausti-2025"
INCH_DATA = INCH / "data"
HSU_DATA = REPRO / "hsu-2025" / "data"
FOUNDRY_II_DATA = REPRO / "foundry-ii" / "data"
FOUNDRY_IV_DATA = REPRO / "foundry-iv" / "data"

# --- our writable areas (gitignored) ---------------------------------------
OUT = HERE / "outputs"          # predictions, traces, review queue, eval reports
CACHE = HERE / "cache"          # fetched/rendered cutouts keyed by candidate
for _d in (OUT, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# --- imaging candidate sources ----------------------------------------------
# Storfer (DR9) + Inchausti (DR10): graded A/B/C cutouts already on disk.
CUTOUT_DIRS = {
    "storfer": INCH_DATA / "cutouts_fits_candidates_storfer",
    "inchausti": INCH_DATA / "cutouts_fits_candidates_inchausti",
}
SCORE_CSV = {
    "storfer": INCH_DATA / "candidate_scores_storfer.csv",
    "inchausti": INCH_DATA / "candidate_scores_inchausti.csv",
}
GRADED_PUB_CSV = {  # full published catalogs incl. Grade-D raw exports
    "storfer": INCH_DATA / "storfer2024_published_catalog.csv",
    "inchausti": INCH_DATA / "inchausti2025_published_catalog.csv",
}
GRADE_D_RAW = {
    "storfer": INCH_DATA / "storfer2024_gradeD_raw.csv",
    "inchausti": INCH_DATA / "inchausti2025_gradeD_raw.csv",
}
NEG_RANDOM_DIR = INCH_DATA / "cutouts_fits_neg_dr9"   # ~65K random-galaxy negatives
OP_POINT_CSV = INCH_DATA / "operating_point.csv"      # meta thresholds @ FPR

# legacysurvey imaging layer per catalog (mirrors 12_download_candidate_cutouts.py)
SURVEY_LAYER = {"storfer": "ls-dr9", "inchausti": "ls-dr10"}

# --- cutout geometry (Huang+2020 §3.2; matches every on-disk cutout) ---------
SIZE_PIX = 101
PIXSCALE = 0.262           # arcsec/pixel
BANDS = ("g", "r", "z")    # cube[0]=g, cube[1]=r, cube[2]=z
CUTOUT_SHAPE = (3, SIZE_PIX, SIZE_PIX)

# --- render params (identical to 16_build_inspection_viewer.py) -------------
LUPTON_Q = 8.0
LUPTON_STRETCH = 0.5
RENDER_PX = 400            # upsample 101 -> 400 for legibility

# --- model tiers (claude-code aliases; override per role via env) -----------
def _m(role: str, default: str) -> str:
    return os.environ.get(f"LENSJUDGE_MODEL_{role.upper()}", default)

MODELS = {
    "grader": _m("grader", "sonnet"),          # lean single-agent grader
    "arbitrator": _m("arbitrator", "opus"),    # Option-C / multi-agent fuser
    "judge": _m("judge", "sonnet"),            # Option-C panel judges
    "worker": _m("worker", "sonnet"),          # §9.1 specialist subagents
    "crossmatch": _m("crossmatch", "haiku"),   # cheap tool-relay agent
    "spectro": _m("spectro", "sonnet"),
}

# --- heavy-tool subprocess venvs (quick_lensmodel) --------------------------
# GIGA-Lens (JAX) lens-model fit runs in its own venv to keep JAX out of the SDK
# process. The proto script exposes `--cube <path>` -> one JSON line on stdout.
GIGALENS_PY = Path(os.environ.get(
    "LENSJUDGE_GIGALENS_PY", "/home/benson/.venvs/gigalens/bin/python"))
QUICKLENS_SCRIPT = HERE / "tools" / "quicklens_proto.py"  # tracked (was gitignored outputs/)
QUICKLENS_GPU = os.environ.get("LENSJUDGE_QUICKLENS_GPU", "2")  # CUDA_VISIBLE_DEVICES
QUICKLENS_TIMEOUT_S = int(os.environ.get("LENSJUDGE_QUICKLENS_TIMEOUT", "180"))

# --- budgets / safety -------------------------------------------------------
MAX_TURNS = int(os.environ.get("LENSJUDGE_MAX_TURNS", "6"))
MAX_BUDGET_USD = float(os.environ.get("LENSJUDGE_MAX_BUDGET_USD", "0.50"))  # per candidate
MCP_SERVER_NAME = "lens"


# --- reasoning / extended thinking (off by default: old runs reproduce) -----
def thinking_options() -> dict:
    """ClaudeAgentOptions kwargs for the current thinking config.

    LENSJUDGE_THINKING: "off" (default) | "adaptive".
    LENSJUDGE_EFFORT:   low|medium|high|xhigh|max (optional; API default high).
    Read at call time so run scripts can set os.environ from CLI flags before
    grading starts. display="summarized" is explicit because Opus 4.8 defaults
    to "omitted" (empty thinking text); what comes back is the API's SUMMARIZED
    reasoning, not the raw chain of thought. Returns {} when off, leaving the
    original ClaudeAgentOptions byte-identical.
    """
    mode = os.environ.get("LENSJUDGE_THINKING", "off")
    if mode == "off":
        return {}
    if mode != "adaptive":
        raise ValueError(f"LENSJUDGE_THINKING must be off|adaptive, got {mode!r}")
    out: dict = {"thinking": {"type": "adaptive", "display": "summarized"}}
    effort = os.environ.get("LENSJUDGE_EFFORT")
    if effort:
        out["effort"] = effort
    return out
