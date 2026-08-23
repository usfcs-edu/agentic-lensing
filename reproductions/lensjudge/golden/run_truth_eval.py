#!/usr/bin/env python3
"""golden/run_truth_eval.py — registry-gated arms of the evidence-first JWST scheme on the truth set.

Part 2 of the golden plan: the JWST discovery run's pass-count verifier missed 23/24 of the
known lenses it examined (GOLDEN_CONTRACT_2 › diagnosis). This runner scores the replacement
scheme (ADVOCATE → 3 CRITICS → ARBITRATOR → `aggregate_v2`, golden/panel.py) and its baselines
on `golden/truth_manifest.csv` — COWLS + literature positives, catalogue-purged negatives,
stress panels and the five PI anchors — split into `design` (tune freely) and `holdout`
(score ONCE per registered tuple) halves by `golden/truth_splits.csv`.

One invocation = one tuple scored k times on one half, each replicate to its own parquet
`outputs/preds_truth_{arm}_{model}_{split}_k{K}_r{k}.parquet` (K = the tuple's replicate
count, k = this replicate, so the k=1 arm and its k=3 replicate study — two registered
tuples — never collide on a file name; one row per item: the `schemas_panel.to_row` /
`panel.incumbent_row` columns — `p_lens=S`, `grade_pred=letter` — plus the manifest join
columns `truth_class, is_positive, half, ...`), a `..._votes.parquet` (one row per persona
call: id, role, parse_ok, raw model text, cost, system sha16), a `.meta.json` written only
when the replicate is COMPLETE (its presence is what the gate reads), and per-role traces
under `outputs/traces_truth_{arm}_{model}_{split}_k{K}_r{k}/`.

Arms (→ panel.grade_panel modes):
  a0    incumbent — the three incumbent briefs inside prompts/personas/incumbent/wrapper.md
        (byte-equal to verify_workflow.js), `IncumbentVerdict`, pass-count letter; with
        `--claim-mode inspector` the USER message carries the inspector's claim text from
        J/results/inspections.csv exactly as 08c_focus_verify.py built it (evidence[:400])
  a1    full — advocate → critics only if p_evidence ≥ tau0 → arbitrator only if a critic
        named an alternative; per-role views; `--ctx20` adds the 20" context pair for the
        geometry critic (the gated arm; it changes the render_version in the tuple)
  a2    advocate_only (S = p_evidence);  a3  a2 on the v2 render (`image_path_v2r`)
  attr  a1 with the SAME full composite shown to every role (attribution arm)

The tuple `TruthTuple(arm, model, persona_set_sha16, note_sha16, system_sha16s,
render_version, render_desc_sha16, splits_sha16, claim_mode, thinking, effort, k,
thresholds_sha16)` is computed BEFORE any call (persona_set_sha16 over the persona dir incl.
panel_gloss.json; system_sha16s = '+'-joined role:sha16 of each FULL system prompt, note
last — for a0 under `--claim-mode inspector` BOTH wrapper variants, `role:` for flagged
items whose claim rides in the user message and `role@noclaim:` for the rest, because the
wrapper must say what the user message does; render_desc_sha16 = sha over the VIEW gloss in
use + the v2 render description, which for v2r IS appended to the VIEW text the composite
roles receive; thresholds_sha16 = sha of the resolved tau0/t_A/t_B/letter_source, since
they set which items reach the critics and every letter), and `--split holdout` refuses
unless it is a row of `golden/REGISTRY.md › "Truth-eval registered arms"` AND no completed
parquet for it exists anywhere under outputs/ (`--force-rescore --rescore-reason` logs a
dated row in "Truth-eval rescores"). `--split design` is ungated. The gate, the ledger
parsing and the meta/resume logic are run_golden_eval's, called with `section=` / `cols=` /
`prefix=`.

Holdout hygiene, all refused before any call: a non-default `--out` (the score-once scan
must be able to find every replicate), `--limit` / `--ids-file` (a subset peek would leave
the gate believing the half was scored), `--tau0`, provisional thresholds unless
`--allow-provisional-thresholds` is stated (P2 tests the FROZEN letters), a lexicon that
does not contain every holdout id and all 16 PI comments (an empty or stale
banned_lexicon.txt passed the old check "clean against 0 entries"), and a manifest whose
`half` disagrees with the pinned splits file (the tuple's splits_sha16 is otherwise
decorative). Every role prompt is lexicon-checked before any call; with
`--claim-mode inspector` every claim body is too (a hit blanks that claim). Resuming a
parquet compares the rows' stored tuple columns with the current tuple and refuses a
mismatch (an interrupted run under an earlier prompt must not become this tuple's record).
On the first scored item the per-role shas the panel actually sent are compared with the
tuple's — a mismatch aborts before anything is written. Frame units (rows with a
`unit_id`) are seeded into the exposure registry (`registry.seed_from_frame`) and marked
kind "eval" (a permitted zero-shot exposure); non-frame truth items are not registry rows.
Parse failure (any role) ⇒ S = NaN, row kept with `parse_fail_roles`. A per-item cost above
`--cost-cap` (default $0.17: the smoke measured $0.163 for a full stack) is warned about and
counted in the meta. Anthropic path: API default temperature, `thinking` recorded.

  python lensjudge/golden/run_truth_eval.py --arm a2 --split design --model sonnet --limit 20
  python lensjudge/golden/run_truth_eval.py --arm a1 --split holdout --model sonnet --k 1
  python lensjudge/golden/run_truth_eval.py --arm a0 --split holdout --claim-mode inspector
  python lensjudge/golden/run_truth_eval.py --arm a1 --split holdout --print-tuple

golden/panel.py (WP-3) is resolved through `_panel()` at call time so tests can swap a stub
in (the run_golden_eval._REGISTRY pattern); everything else is imported directly.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.golden import _util, aggregate_v2, audit_traces, grader_jwst  # noqa: E402
from lensjudge.golden import run_golden_eval as rge  # noqa: E402
from lensjudge.imaging import run_batch  # noqa: E402

ARMS = ("a0", "a1", "a2", "a3", "attr")
SPLITS = ("design", "holdout")
GATED_SPLIT = "holdout"
MODELS = ("sonnet", "opus")
RENDERS = ("v1", "v2r")
CLAIM_MODES = ("none", "inspector")
MODE = {"a0": "incumbent", "a1": "full", "a2": "advocate_only", "a3": "advocate_only", "attr": "attr"}
# roles called per arm, in the order their shas are joined in the tuple
ROLES = {"a0": ("artifact", "morphology", "geometry"),          # verify_workflow.js order
         "a1": aggregate_v2.ROLES, "attr": aggregate_v2.ROLES,
         "a2": ("advocate",), "a3": ("advocate",)}
# worst-case calls per item (for the budget line; critics/arbitrator are conditional)
MAX_CALLS = {"a0": 3, "a1": 5, "attr": 5, "a2": 1, "a3": 1}
COST_PER_CALL = {"sonnet": 0.019, "opus": 0.032}
PROMPTS = _util.LENSJUDGE / "prompts"
PERSONA_SET_DEFAULT = PROMPTS / "personas" / "jwst_v1"
PERSONA_SET_INCUMBENT = PROMPTS / "personas" / "incumbent"
NOTE_V2 = PROMPTS / "jwst_note_v2.md"
MANIFEST = _util.HERE / "truth_manifest.csv"
SPLITS_CSV = _util.HERE / "truth_splits.csv"
FRAME_CSV = _util.HERE / "frame.csv"
STAMPS_DIR = _util.HERE / "stamps"
REGISTRY_MD = _util.HERE / "REGISTRY.md"
BANNED = _util.HERE / "banned_lexicon.txt"
THRESHOLDS = _util.HERE / "thresholds_v2.json"
RENDER_V2_DESC = _util.HERE / "render_v2_desc.md"
INSPECTIONS = _util.JWST_REPO / "results" / "inspections.csv"
OUT_TEMPLATE = str(config.OUT / "preds_truth_{arm}_{model}_{split}_k{K}_r{k}.parquet")
SECTION = "Truth-eval registered arms"
RESCORE_SECTION = "Truth-eval rescores"
PREFIX = "preds_truth_"
TRUTH_COLS = ("arm", "model", "persona_set_sha16", "note_sha16", "system_sha16s",
              "render_version", "render_desc_sha16", "splits_sha16", "claim_mode",
              "thinking", "effort", "k", "thresholds_sha16")
# the a0 wrapper variant for items WITHOUT an inspector claim (claim_mode inspector only)
NOCLAIM = "@noclaim"
# the separator of the Truth-eval tables (colon-aligned so it can never be mistaken for the
# golden Registered-arms separator, which tests count exactly once)
TRUTH_TABLE_SEP = "|" + ":---|" * (len(TRUTH_COLS) + 2)
# manifest columns carried onto every prediction row (those present)
JOIN_COLS = ("unit_id", "truth_class", "is_positive", "is_stress", "is_anchor", "cowls_band",
             "cowls_theta_E", "known_type", "centre_is_deflector", "layout", "field_class",
             "proposal", "prior_exposure", "pipe_grade_passcount", "pipe_inspector_conf",
             "in_frame", "half", "render_sha", "binary_label", "leak")
COST_CAP_DEFAULT = 0.17       # the integration smoke measured $0.163 for a full stack with critics


# ------------------------------------------------------------------ the tuple
@dataclass(frozen=True)
class TruthTuple:
    arm: str
    model: str
    persona_set_sha16: str
    note_sha16: str
    system_sha16s: str          # '+'-joined role:sha16 in ROLES[arm] order
    render_version: str         # jwst_v1 | jwst_v1+ctx20 | jwst_v2r
    render_desc_sha16: str
    splits_sha16: str
    claim_mode: str
    thinking: str = "off"
    effort: str = "default"
    k: int = 1
    thresholds_sha16: str = ""  # sha_json of the resolved {tau0, t_A, t_B, letter_source, thresholds_key}

    def run_tag(self, split: str, k: int) -> str:
        return f"truth_{self.arm}_{_util.safe_name(self.model)}_{split}_k{self.k}_r{k}"

    def row(self, note: str = "") -> str:
        today = _dt.date.today().isoformat()
        return "| " + " | ".join([today] + [str(getattr(self, c)) for c in TRUTH_COLS] + [note]) + " |"


# ------------------------------------------------------------------ the panel seam
# golden/panel.py is resolved at call time; tests set `_PANEL` to a stub exposing the same
# five names (grade_panel, to_row, load_persona_set, load_incumbent_set, persona_set_sha16).
_PANEL = None


def _panel():
    global _PANEL
    if _PANEL is None:
        _PANEL = importlib.import_module("lensjudge.golden.panel")
    return _PANEL


# ------------------------------------------------------------------ prompts
def with_note(text: str, note: str) -> str:
    """grader_jwst.with_note: the note appended iff non-empty and the text does not already
    end with it — the same function grade_candidate applies, so the tuple's shas are the shas
    of what the model receives."""
    return grader_jwst.with_note(text, note)


def role_prompts(persona_dir: Path, arm: str, note: str, claim_mode: str = "none") -> tuple[dict, dict, Optional[dict]]:
    """(persona_set, full_prompts, persona_set_noclaim): the dict grade_panel takes
    (`persona_set=`, note NOT yet appended — grade_candidate appends it), {role: FULL system
    prompt} for the tuple, and — a0 under claim_mode inspector only — the wrapper set for
    items WITHOUT a claim (`persona_set_noclaim=`), whose prompts enter `full_prompts` under
    `role@noclaim`. a0 = panel.load_incumbent_set (filled wrappers; note always ""; the claim
    placeholders say what the user message carries), else panel.load_persona_set (critic =
    critic_common + "\\n" + critic_<role>) + note."""
    p = _panel()
    if arm == "a0":
        if claim_mode == "inspector":
            sysp = p.load_incumbent_set(Path(persona_dir), claim_in_user=True)
            noclaim = p.load_incumbent_set(Path(persona_dir), claim_in_user=False)
            full = {r: sysp[r] for r in ROLES["a0"]}
            full.update({r + NOCLAIM: noclaim[r] for r in ROLES["a0"]})
            return sysp, full, noclaim
        sysp = p.load_incumbent_set(Path(persona_dir), claim_in_user=False)
        return sysp, {r: sysp[r] for r in ROLES["a0"]}, None
    sysp = p.load_persona_set(Path(persona_dir), note)
    return sysp, {r: with_note(sysp[r], note) for r in ROLES[arm]}, None


def prompt_keys(full_prompts: dict, arm: str) -> list[str]:
    """The keys of `full_prompts` in tuple order: the arm's roles, then their @noclaim twins."""
    return [r for r in ROLES[arm]] + [r + NOCLAIM for r in ROLES[arm] if r + NOCLAIM in full_prompts]


def system_shas(full_prompts: dict, arm: str) -> dict[str, str]:
    return {key: _util.sha_text(full_prompts[key]) for key in prompt_keys(full_prompts, arm)}


def join_shas(shas: dict[str, str], arm: str) -> str:
    return "+".join(f"{key}:{shas[key]}" for key in prompt_keys(shas, arm))


def allowed_shas(shas: dict[str, str]) -> dict[str, set]:
    """role -> the set of system sha16s the panel may legitimately send for that role (the
    claim and @noclaim wrapper variants of a0 under claim_mode inspector are both allowed)."""
    out: dict[str, set] = {}
    for key, sha in shas.items():
        role = key[:-len(NOCLAIM)] if key.endswith(NOCLAIM) else key
        out.setdefault(role, set()).add(sha)
    return out


def render_version(render: str, ctx20: bool) -> str:
    return "jwst_v2r" if render == "v2r" else ("jwst_v1+ctx20" if ctx20 else "jwst_v1")


def render_desc_sha(render: str, desc_path: Path = RENDER_V2_DESC) -> str:
    """sha16 over the text shipped WITH the images: the VIEW gloss in use (views.load_gloss —
    what each role is told it is looking at, render-conditional: the v2r composite view sets
    are the gloss's `renders["v2r"]` twins) and, for v2r, golden/render_v2_desc.md, which
    views.view_text appends to every composite role's VIEW paragraph (R10: the description
    is load-bearing, so the sha covers exactly what the model reads)."""
    from lensjudge.golden import views
    desc = ""
    if render == "v2r":
        p = Path(desc_path)
        if not p.exists():
            raise SystemExit(f"--render v2r needs {p} (golden/render_v2.py writes it)")
        desc = p.read_text()
    return _util.sha_json({"gloss": views.gloss_sha16(), "render_desc": _util.sha_text(desc)})


# ------------------------------------------------------------------ thresholds
def model_key(model: str) -> str:
    """thresholds_v2.json key for a --model alias (aggregate_v2.MODEL_KEYS: sonnet_api /
    opus_api; an unknown alias is keyed `<alias>_api`)."""
    return aggregate_v2.MODEL_KEYS.get(model, f"{model}_api")


def load_thresholds(path: Path, model_key: str, tau0_override: Optional[float] = None) -> dict:
    """aggregate_v2.resolve_thresholds on thresholds_v2.json for one model key (a null /
    missing block ⇒ the provisional numbers, letter_source "provisional"); `--tau0`
    overrides the critics' gate (a design-half knob; refused on the holdout). The resolved
    numbers are hashed into the tuple (`thresholds_sha`): tau0 decides which items reach the
    critics and t_A / t_B every letter, so two runs that differ only there are two tuples."""
    table = json.loads(Path(path).read_text()) if Path(path).exists() else {}
    thr = dict(aggregate_v2.resolve_thresholds(table, model_key))
    if tau0_override is not None:
        thr["tau0"] = float(tau0_override)
    thr["model_key"] = model_key
    return thr


def thresholds_sha(thr: dict) -> str:
    return _util.sha_json({k: thr.get(k) for k in ("tau0", "t_A", "t_B", "letter_source", "thresholds_key")})


# ------------------------------------------------------------------ data
def _read(path: Path) -> pd.DataFrame:
    return rge._read(path)


def _abs(p) -> str:
    """Manifest image paths are relative to the lensjudge root (golden/kits_truth/...)."""
    if not isinstance(p, str) or not p.strip():
        return ""
    q = Path(p)
    return str(q if q.is_absolute() else _util.LENSJUDGE / q)


def check_halves(df: pd.DataFrame, splits_csv: Path) -> None:
    """The manifest's `half` must equal the pinned splits file's for every id (the tuple
    carries splits_sha16; items are selected by the manifest column — they must agree or the
    sha pins nothing). Raises SystemExit on any mismatch or an id missing from the splits."""
    sp = _read(Path(splits_csv))
    if "candidate_id" not in sp.columns or "half" not in sp.columns:
        raise SystemExit(f"{splits_csv} lacks candidate_id/half")
    half = sp.set_index(sp["candidate_id"].astype(str))["half"].astype(str)
    man = df.set_index("name")["half"].astype(str)
    missing = sorted(set(man.index) - set(half.index))
    if missing:
        raise SystemExit(f"[truth] REFUSED: {len(missing)} manifest ids absent from {splits_csv} "
                         f"(e.g. {missing[:3]}); re-run golden/split_truth.py")
    bad = man[man != half.reindex(man.index)]
    if len(bad):
        raise SystemExit(f"[truth] REFUSED: manifest `half` disagrees with {splits_csv} on {len(bad)} ids "
                         f"(e.g. {bad.index[:3].tolist()}); the pinned splits are the authority")


def load_truth_manifest(path: Path, split: str, render: str = "v1",
                        splits_csv: Optional[Path] = None) -> pd.DataFrame:
    """Rows of truth_manifest.csv on one half, run_batch-shaped (`name`, `catalog`, `grade`),
    image paths made absolute; rows without an image for this render are dropped with a
    warning (the panel reads image_path / image_path_v2r itself by `render`). With
    `splits_csv` the manifest's `half` is asserted equal to the pinned splits file's."""
    df = _read(Path(path))
    for c in ("name", "half", "image_path", "layout"):
        if c not in df.columns:
            raise SystemExit(f"{path} lacks column {c!r} (golden/build_truth_manifest.py writes it)")
    df["name"] = df["name"].astype(str)
    if splits_csv is not None:
        check_halves(df, Path(splits_csv))
    df = df[df["half"].astype(str) == split].copy()
    if df.empty:
        raise SystemExit(f"no rows with half={split!r} in {path}")
    for c in ("image_path", "image_path_v2r"):
        if c in df.columns:
            df[c] = df[c].map(_abs)
        else:
            df[c] = ""
    col = "image_path_v2r" if render == "v2r" else "image_path"
    missing = df[col] == ""
    if missing.any():
        print(f"[truth] WARN {int(missing.sum())} {split} rows have no {render} image and are skipped")
        df = df[~missing]
    if "catalog" not in df.columns:
        df["catalog"] = "jwst"
    if "grade" not in df.columns:
        df["grade"] = df["grade_truth"].fillna("") if "grade_truth" in df.columns else ""
    df["unit_id"] = df["unit_id"].fillna("").astype(str) if "unit_id" in df.columns else ""
    for c in ("render_sha", "render_sha_v2r"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    return df.reset_index(drop=True)


def read_ids_file(path: Path) -> list[str]:
    """A CSV with a candidate_id/id/name column, or one id per line (# comments)."""
    txt = Path(path).read_text()
    first = next((l for l in txt.splitlines() if l.strip() and not l.startswith("#")), "")
    if "," in first:
        d = pd.read_csv(path, dtype=str, comment="#")
        col = next((c for c in ("candidate_id", "id", "name") if c in d.columns), None)
        if col is None:
            raise SystemExit(f"{path}: no candidate_id/id/name column")
        return d[col].dropna().astype(str).tolist()
    return [l.strip() for l in txt.splitlines() if l.strip() and not l.startswith("#")]


def select_items(df: pd.DataFrame, ids: Optional[list[str]], limit: int) -> pd.DataFrame:
    if ids is not None:
        keep = set(ids)
        before = len(df)
        df = df[df["name"].isin(keep)]
        print(f"[truth] --ids-file: {len(df)}/{before} rows kept ({len(keep - set(df['name']))} ids not on this half)")
    if limit and limit > 0:
        df = df.head(limit)
    if df.empty:
        raise SystemExit("[truth] nothing to score after --ids-file/--limit")
    return df.reset_index(drop=True)


def attach_claims(df: pd.DataFrame, inspections_csv: Path, lexicon: Optional[list] = None) -> pd.DataFrame:
    """`--claim-mode inspector` (a0): the incumbent's items carried the inspector's
    lens_at_center / quadrant_lens / evidence[:400] (J/scripts/08c_focus_verify.py:59-63).
    Items the inspector never flagged (negatives, never-flagged positives) get no claim —
    the brief-faithful baseline on rows the incumbent never saw. With `lexicon` every claim
    body is run through the banned lexicon BEFORE it can reach a call: a hit (an evidence
    string that names another candidate id, or a PI phrase) BLANKS that item's claim
    (`has_claim` False, `claim_blanked` True) and is printed; the run goes on."""
    p = Path(inspections_csv)
    if not p.exists():
        raise SystemExit(f"--claim-mode inspector needs {p}")
    ins = pd.read_csv(p, dtype={"id": str}, usecols=["id", "lens_at_center", "quadrant_lens", "evidence", "flagged"])
    ins = ins.drop_duplicates("id").set_index("id")
    df = df.copy()
    cc, cq, ce, has, blanked = [], [], [], [], []
    for n in df["name"]:
        if n in ins.index and bool(ins.at[n, "flagged"]):
            c, q, e = str(ins.at[n, "lens_at_center"]), str(ins.at[n, "quadrant_lens"]), str(ins.at[n, "evidence"])[:400]
            hit = audit_traces.banned_hit(" | ".join((c, q, e)), lexicon) if lexicon else None
            if hit:
                print(f"[embargo] claim for {n} hits the lexicon (entry {hit[0][:40]!r}, window {hit[1]!r}): BLANKED")
                cc.append(""); cq.append(""); ce.append(""); has.append(False); blanked.append(True)
            else:
                cc.append(c); cq.append(q); ce.append(e); has.append(True); blanked.append(False)
        else:
            cc.append(""); cq.append(""); ce.append(""); has.append(False); blanked.append(False)
    df["claim_center"], df["claim_quadrant"], df["claimed_evidence"], df["has_claim"] = cc, cq, ce, has
    df["claim_blanked"] = blanked
    print(f"[truth] inspector claims attached to {int(sum(has))}/{len(df)} items"
          + (f" ({int(sum(blanked))} blanked by the lexicon)" if any(blanked) else ""))
    return df


# ------------------------------------------------------------------ embargo (holdout)
# audit_traces.load_pi_comments, swappable by tests (the real comment file is gitignored and
# pinned by count + sha16, so a test machine without it plants its own strings here)
_PI_LOADER = None


def _load_pi(path: Optional[Path]) -> list[str]:
    fn = _PI_LOADER or audit_traces.load_pi_comments
    return fn(Path(path) if path else audit_traces.PI_COMMENTS_PATH)


def check_lexicon_coverage(banned: Path, splits_csv: Path, frame_csv: Optional[Path], split: str,
                           pi_comments: Optional[Path] = None) -> None:
    """On the gated split the lexicon must actually contain what it is meant to catch: every
    `split`-half candidate_id from the pinned splits file (+ the frame aliases of those
    units) and all 16 PI comments (their count + sha16 are pinned in audit_traces). An empty
    or pre-Part-2 lexicon passed check_system_prompt with "clean against 0 entries"."""
    lex = set(audit_traces.load_lexicon(Path(banned)))
    sp = _read(Path(splits_csv))
    ids = set(sp.loc[sp["half"].astype(str) == split, "candidate_id"].astype(str))
    if frame_csv is not None and Path(frame_csv).exists():
        fr = _read(Path(frame_csv))
        if "alias_ids" in fr.columns and "candidate_id" in fr.columns:
            fr_ids = fr["candidate_id"].astype(str)
            for cid, aliases in zip(fr_ids, fr["alias_ids"].fillna("").astype(str)):
                if cid in ids:
                    ids |= {a for a in aliases.split("|") if a.strip()}
    missing = sorted(ids - lex)
    if missing:
        raise SystemExit(f"[embargo] REFUSED: {banned} lacks {len(missing)}/{len(ids)} {split} ids "
                         f"(e.g. {missing[:3]}); rebuild it: audit_traces.py --build-lexicon --pi-only "
                         f"--extra-ids <{split} ids>")
    pi = _load_pi(pi_comments)
    lost = [c for c in pi if c.strip() not in lex]
    if lost:
        raise SystemExit(f"[embargo] REFUSED: {banned} lacks {len(lost)}/{len(pi)} PI comments; rebuild it")
    print(f"[embargo] lexicon covers all {len(ids)} {split} ids and {len(pi)} PI comments ({len(lex)} entries)")


def claim_for(cand: dict, claim_mode: str):
    """The `claim=` argument of grade_panel: the three claim fields, or None (no claim)."""
    if claim_mode != "inspector" or not cand.get("has_claim"):
        return None
    return {k: cand.get(k, "") for k in ("claim_center", "claim_quadrant", "claimed_evidence")}


def holdout_out_problem(template: str) -> Optional[str]:
    """Why a `--out` template is not acceptable on the gated split (None when it is): the
    file name must be the default template's (so the tuple, split and replicate are in the
    name) and the directory must be config.OUT or a sub-directory of it (the score-once scan
    walks config.OUT recursively; a parquet elsewhere would be invisible to the next run)."""
    tp = Path(template)
    if tp.name != Path(OUT_TEMPLATE).name:
        return f"file name template {tp.name!r} is not the default {Path(OUT_TEMPLATE).name!r}"
    root = Path(config.OUT).resolve()
    parent = tp.parent.resolve()
    if parent != root and root not in parent.parents:
        return f"directory {tp.parent} is not under {config.OUT}"
    return None


# ------------------------------------------------------------------ exposure registry
def frame_units(df: pd.DataFrame) -> list[str]:
    """Only frame units are registry rows: non-empty unit_id (and in_frame when present)."""
    u = df["unit_id"].astype(str)
    ok = u.str.strip() != ""
    if "in_frame" in df.columns:
        ok &= df["in_frame"].astype(str).str.lower().isin(("true", "1", "1.0"))
    return sorted(set(u[ok]))


def seed_registry(frame_csv: Path) -> None:
    """registry.seed_from_frame: rows for frame units before any label exists (safe to call
    before every run; existing rows and exposures are untouched)."""
    r = rge._registry()
    fn = getattr(r, "seed_from_frame", None) if r else None
    if fn is None:
        print("[registry] seed_from_frame not available; frame units must already be registry rows")
        return
    fn(_read(Path(frame_csv)), **rge._path_kw())


# ------------------------------------------------------------------ scoring loop
def _flush_votes(rows: list, out: Path) -> None:
    new = pd.DataFrame(rows)
    if out.exists():
        prev = pd.read_parquet(out)
        new = pd.concat([prev, new], ignore_index=True)
    new.drop_duplicates(["name", "role"], keep="last").to_parquet(out, index=False)


def vote_rows(res, cand: dict, k: int) -> list[dict]:
    """One row per persona CALL: the roles in res.system_sha16s (one entry per call made);
    raw = the model's raw text (res.raw[role]); cost = res.meta['cost_by_role'][role]."""
    raws = getattr(res, "raw", None) or {}
    costs = (getattr(res, "meta", None) or {}).get("cost_by_role") or {}
    fails = set(getattr(res, "parse_failures", None) or [])
    rows = []
    for role, sha in (getattr(res, "system_sha16s", None) or {}).items():
        rows.append({"name": cand["name"], "unit_id": cand.get("unit_id", ""), "role": role, "k": k,
                     "parse_ok": role not in fails, "raw": raws.get(role),
                     "cost_usd": costs.get(role), "system_sha16": sha})
    return rows


RESUME_COLS = tuple(c for c in TRUTH_COLS if c != "k") + ("run_tag",)   # k in a row = replicate index


def check_resume(out: Path, extra_cols: dict) -> set:
    """Names already in an existing parquet — after asserting that EVERY stored row carries
    this run's tuple columns (`RESUME_COLS`). An interrupted replicate scored under another
    prompt / render / threshold set must never be completed as this tuple's record."""
    if not out.exists():
        return set()
    prev = pd.read_parquet(out)
    if prev.empty:
        return set()
    bad = {}
    for c in RESUME_COLS:
        if c not in prev.columns:
            bad[c] = "column absent"
            continue
        vals = set(prev[c].astype(str))
        if vals != {str(extra_cols.get(c))}:
            bad[c] = f"stored {sorted(vals)[:3]} vs current {extra_cols.get(c)!r}"
    if bad:
        raise SystemExit(f"[truth] REFUSED: {out} holds rows of a DIFFERENT tuple and cannot be resumed as "
                         f"this one ({bad}); move it aside (or --force-rescore on the holdout)")
    return set(prev["name"].astype(str).tolist())


async def score(df: pd.DataFrame, out: Path, votes_out: Path, trace_dir: Path, *, arm: str,
                model: str, persona_set: dict, note: str, render: str, claim_mode: str,
                thresholds: dict, concurrency: int, extra_cols: dict, expected_shas: dict,
                stamps_dir: Optional[Path] = None, cost_cap: float = COST_CAP_DEFAULT,
                persona_set_noclaim: Optional[dict] = None) -> dict:
    """run_golden_eval.score's loop with panel.grade_panel per item: resumable by name
    (rows already in `out` are skipped once their tuple columns match, `check_resume`),
    incremental parquet writes every 5 items (run_batch._flush), a bounded semaphore, the
    first-item sha assertion (`expected_shas`: role -> the set of allowed sha16s) and the
    cost cap."""
    done = check_resume(out, extra_cols)
    todo = [r.to_dict() for _, r in df.iterrows() if str(r["name"]) not in done]
    print(f"[truth] {arm}/{model}: {len(todo)} to grade ({len(done)} already done) -> {out}")
    trace_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    rows, vrows, lock = [], [], asyncio.Lock()
    stats = {"n_over_cap": 0, "n_parse_fail": 0, "cost_usd": 0.0, "n_scored": 0, "sha_checked": False}
    panel = _panel()
    k = int(extra_cols.get("k", 1))

    async def one(cand):
        async with sem:
            res = await panel.grade_panel(
                cand, model=model, persona_set=persona_set, note_text=note, thresholds=thresholds,
                mode=MODE[arm], claim=claim_for(cand, claim_mode), trace_dir=str(trace_dir),
                stamp_dir=(str(Path(stamps_dir) / cand["name"]) if stamps_dir else None), render=render,
                persona_set_noclaim=persona_set_noclaim)
        async with lock:
            got = dict(getattr(res, "system_sha16s", None) or {})
            allowed = {r: (v if isinstance(v, (set, frozenset, list, tuple)) else {v}) for r, v in expected_shas.items()}
            bad = {r: (got[r], sorted(allowed.get(r, set()))) for r in got if got[r] not in allowed.get(r, set())}
            if bad:
                raise SystemExit(f"[truth] REFUSED: the panel sent prompts whose sha16 differ from the "
                                 f"registered tuple (role: sent, registered): {bad}; nothing written")
            stats["sha_checked"] = True
            row = panel.to_row(res, cand)
            row.update({c: cand.get(c) for c in JOIN_COLS if c in cand})
            row.update({"name": cand["name"], "split": cand.get("half"), **extra_cols})
            cost = float(getattr(res, "cost_usd", 0.0) or 0.0)
            stats["cost_usd"] += cost
            stats["n_scored"] += 1
            if cost > cost_cap:
                stats["n_over_cap"] += 1
                print(f"[truth] WARN {cand['name']}: ${cost:.3f} > per-item cap ${cost_cap:.2f} "
                      f"({getattr(res, 'calls', '?')} calls)")
            if getattr(res, "parse_failures", None):
                stats["n_parse_fail"] += 1
            rows.append(row)
            vrows.extend(vote_rows(res, cand, k))
            if len(rows) % 5 == 0 or len(rows) == len(todo):
                run_batch._flush(rows, out, done)
                if vrows:
                    _flush_votes(vrows, votes_out)

    await asyncio.gather(*(one(c) for c in todo))
    if rows:
        run_batch._flush(rows, out, done)
    if vrows:
        _flush_votes(vrows, votes_out)
    return stats


def archive_replicate(out: Path, votes_out: Path, trace_dir: Path) -> None:
    """A sanctioned rescore re-grades every item: the previous parquet, votes, meta and
    traces are moved aside as `<name>.pre_rescore_<UTC stamp>` (never deleted — the logged
    rescore row points at a number that must stay inspectable)."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for p in (out, votes_out, rge.meta_path(out), trace_dir):
        if p.exists():
            dst = p.with_name(f"{p.name}.pre_rescore_{stamp}")
            p.rename(dst)
            print(f"[truth] rescore: {p.name} -> {dst.name}")


def summarize(out: Path) -> None:
    df = pd.read_parquet(out)
    ok = df["parse_ok"].astype(bool) if "parse_ok" in df.columns else df["p_lens"].notna()
    print(f"\n[summary] {len(df)} items | parse_ok {int(ok.sum())}/{len(df)} | total cost "
          f"${df['cost_usd'].sum():.2f} | mean ${df['cost_usd'].mean():.3f}/item")
    if "grade_pred" in df.columns:
        print(f"  letters: {df['grade_pred'].value_counts(dropna=False).to_dict()}")
    if "truth_class" in df.columns and "p_lens" in df.columns:
        g = df.groupby("truth_class")["p_lens"].agg(["count", "mean"])
        print("  mean S by truth_class:\n" + g.to_string())


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--model", choices=MODELS, default="sonnet")
    ap.add_argument("--k", type=int, default=1, help="replicate samples -> r1..rk parquets")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--persona-set", default=None,
                    help="persona dir (default prompts/personas/jwst_v1; a0: prompts/personas/incumbent)")
    ap.add_argument("--note", default=None,
                    help="note appended to every panel role prompt (default prompts/jwst_note_v2.md; "
                         "a0 never takes a note). '' disables it")
    ap.add_argument("--render", choices=RENDERS, default=None, help="v1 (default) | v2r (forced for a3)")
    ap.add_argument("--ctx20", action="store_true",
                    help="a1/attr: give the geometry critic the 20\" context pair from golden/stamps/<id>/ "
                         "(the gated arm; render_version becomes jwst_v1+ctx20)")
    ap.add_argument("--stamps-dir", default=str(STAMPS_DIR))
    ap.add_argument("--claim-mode", choices=CLAIM_MODES, default="none")
    ap.add_argument("--ids-file", default=None, help="restrict to these candidate ids (CSV or one per line)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT_TEMPLATE, help="template with {arm} {model} {split} {k}")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--splits", default=str(SPLITS_CSV))
    ap.add_argument("--frame", default=str(FRAME_CSV))
    ap.add_argument("--registry-md", default=str(REGISTRY_MD))
    ap.add_argument("--registry-csv", default=None, help="exposure ledger path (default: registry.py's)")
    ap.add_argument("--banned", default=str(BANNED))
    ap.add_argument("--thresholds", default=str(THRESHOLDS))
    ap.add_argument("--tau0", type=float, default=None, help="override the critics' p_evidence gate (design only)")
    ap.add_argument("--inspections", default=str(INSPECTIONS))
    ap.add_argument("--cost-cap", type=float, default=COST_CAP_DEFAULT)
    ap.add_argument("--force-rescore", action="store_true")
    ap.add_argument("--rescore-reason", default=None)
    ap.add_argument("--allow-provisional-thresholds", action="store_true",
                    help="holdout: run although thresholds_v2.json has no frozen t_A/t_B for this model "
                         "(P2 then tests the provisional letters; say so in the registry note)")
    ap.add_argument("--pi-comments", default=str(audit_traces.PI_COMMENTS_PATH),
                    help="the gitignored PI comment file the holdout lexicon must contain")
    ap.add_argument("--print-tuple", action="store_true", help="print the REGISTRY.md row and exit")
    ap.add_argument("--thinking", choices=("off", "adaptive"), default=None)
    ap.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default=None)
    args = ap.parse_args(argv)

    rge._REGISTRY_PATH = Path(args.registry_csv) if args.registry_csv else None
    if args.thinking:
        os.environ["LENSJUDGE_THINKING"] = args.thinking
    if args.effort:
        os.environ["LENSJUDGE_EFFORT"] = args.effort
    os.environ.setdefault("LENSJUDGE_BACKEND", "anthropic")
    if args.arm == "a3":
        if args.render == "v1":
            ap.error("--arm a3 is the v2-render arm; drop --render v1")
        args.render = "v2r"
    render = args.render or "v1"
    if args.claim_mode == "inspector" and args.arm != "a0":
        ap.error("--claim-mode inspector is the incumbent's (a0) configuration only: the new scheme never sees claim text")
    if args.ctx20 and args.arm != "a1":
        ap.error("--ctx20 only changes what the geometry critic sees (a1); the attribution arm shows every "
                 "role the same composite and attaches nothing, so it must not be labelled +ctx20")
    if args.split == GATED_SPLIT:       # the holdout hygiene rules (SystemExit, like the gate)
        if args.tau0 is not None:
            raise SystemExit("[truth] REFUSED: --tau0 is a design knob; the holdout runs the thresholds file as frozen")
        why = holdout_out_problem(args.out)
        if why:
            raise SystemExit(f"[truth] REFUSED: --out on the {GATED_SPLIT}: {why} (every replicate must carry the "
                             f"default name {Path(OUT_TEMPLATE).name} and live under {config.OUT}, where the "
                             f"score-once scan looks)")
        if args.ids_file or args.limit:
            raise SystemExit(f"[truth] REFUSED: --ids-file/--limit are refused on the {GATED_SPLIT}: a registered "
                             f"tuple is scored on the WHOLE half, once (subsets are design-half tools)")

    # ---- prompts and the tuple (before any call)
    persona_dir = Path(args.persona_set) if args.persona_set else \
        (PERSONA_SET_INCUMBENT if args.arm == "a0" else PERSONA_SET_DEFAULT)
    if args.arm == "a0":
        if args.note:
            ap.error("a0 takes no note: the incumbent wrapper is self-contained")
        note_path = None
    elif args.note is None:
        note_path = NOTE_V2
    else:
        note_path = Path(args.note) if args.note.strip() else None
    if note_path and not note_path.exists():
        raise SystemExit(f"note file missing: {note_path}")
    note = note_path.read_text() if note_path else ""
    persona_set, full, persona_set_noclaim = role_prompts(persona_dir, args.arm, note, args.claim_mode)
    shas = system_shas(full, args.arm)
    thresholds = load_thresholds(Path(args.thresholds), model_key(args.model), args.tau0)
    t = TruthTuple(args.arm, args.model, _panel().persona_set_sha16(persona_dir), _util.sha_text(note),
                   join_shas(shas, args.arm), render_version(render, args.ctx20), render_desc_sha(render),
                   rge.splits_sha(Path(args.splits)), args.claim_mode,
                   rge.thinking_setting(), rge.effort_setting(), int(args.k), thresholds_sha(thresholds))
    if args.print_tuple:
        print(t.row())
        return
    for key in prompt_keys(full, args.arm):
        rge.check_system_prompt(full[key], Path(args.banned), args.split, gated_splits=(GATED_SPLIT,))
    if args.split == GATED_SPLIT:
        check_lexicon_coverage(Path(args.banned), Path(args.splits), Path(args.frame), GATED_SPLIT,
                               pi_comments=Path(args.pi_comments))
        if args.arm != "a0" and thresholds["letter_source"] == "provisional" and not args.allow_provisional_thresholds:
            raise SystemExit(f"[truth] REFUSED: thresholds_v2.json has no frozen t_A/t_B for "
                             f"{thresholds['model_key']!r} — P2 tests the letters frozen on design. Freeze them "
                             f"first, or state --allow-provisional-thresholds (and say so in the registry note)")
    lexicon = audit_traces.load_lexicon(Path(args.banned)) if Path(args.banned).exists() else None

    paths = rge.out_paths(args.out, args.arm, args.split, args.k, args.model)
    rescored = False
    if args.split == GATED_SPLIT:
        rescored = rge.check_validate_gate(Path(args.registry_md), t, paths, args.force_rescore,
                                           args.rescore_reason, section=SECTION, cols=TRUTH_COLS,
                                           split=GATED_SPLIT, prefix=PREFIX, rescore_section=RESCORE_SECTION)

    # ---- items
    df = load_truth_manifest(Path(args.manifest), args.split, render, splits_csv=Path(args.splits))
    df = select_items(df, read_ids_file(Path(args.ids_file)) if args.ids_file else None, args.limit)
    if args.claim_mode == "inspector":
        df = attach_claims(df, Path(args.inspections), lexicon=lexicon)
    units = frame_units(df)
    est = len(df) * MAX_CALLS[args.arm] * COST_PER_CALL.get(args.model, 0.019)
    print(f"[truth] {len(df)} items on {args.split} ({len(units)} frame units) × k={args.k}; "
          f"worst-case ≈ ${est * args.k:.2f} at {MAX_CALLS[args.arm]} calls/item; "
          f"thresholds {thresholds['letter_source']} tau0={thresholds['tau0']}; render {t.render_version}")
    from lensjudge.common import llm_client
    backend = llm_client.get_backend()
    if backend == "anthropic":
        print("[truth] Anthropic path: API default temperature, thinking=" + t.thinking)
    if units:
        seed_registry(Path(args.frame))

    for k, out in enumerate(paths, start=1):
        tag = t.run_tag(args.split, k)
        trace_dir = out.parent / f"traces_{tag}"
        votes_out = out.with_name(out.stem + "_votes.parquet")
        if rescored:
            archive_replicate(out, votes_out, trace_dir)
        extra = {**asdict(t), "k": k, "run_tag": tag, "rescored": rescored,
                 "rescore_reason": (args.rescore_reason if rescored else None),
                 "tau0": thresholds["tau0"], "t_A": thresholds.get("t_A"), "t_B": thresholds.get("t_B")}
        stats = asyncio.run(score(df, out, votes_out, trace_dir, arm=args.arm, model=args.model,
                                  persona_set=persona_set, note=note, render=render, claim_mode=args.claim_mode,
                                  thresholds=thresholds, concurrency=args.concurrency, extra_cols=extra,
                                  expected_shas=allowed_shas(shas),
                                  stamps_dir=(Path(args.stamps_dir) if args.ctx20 else None),
                                  cost_cap=args.cost_cap, persona_set_noclaim=persona_set_noclaim))
        rge.write_meta(out, {
            "tuple": asdict(t), "split": args.split, "k": k, "run_tag": tag, "out": str(out),
            "rescored": rescored, "rescore_reason": extra["rescore_reason"],
            "manifest": str(args.manifest), "manifest_sha16": _util.sha_text(Path(args.manifest).read_text()),
            "splits": str(args.splits), "ids_file": args.ids_file, "limit": args.limit,
            "persona_set": str(persona_dir),
            "persona_files": {p.name: _util.sha_file(p) for p in sorted(Path(persona_dir).glob("*"))
                              if p.suffix in (".md", ".json")},
            "note": str(note_path) if note_path else "", "note_sha16": t.note_sha16,
            "system_sha16s": shas, "render": render, "ctx20": bool(args.ctx20),
            "render_desc_sha16": t.render_desc_sha16, "claim_mode": args.claim_mode,
            "n_claims": int(df["has_claim"].sum()) if "has_claim" in df.columns else 0,
            "n_claims_blanked": int(df["claim_blanked"].sum()) if "claim_blanked" in df.columns else 0,
            "thresholds": str(args.thresholds), "thresholds_resolved": thresholds,
            "thresholds_sha16": t.thresholds_sha16,
            "allow_provisional_thresholds": bool(args.allow_provisional_thresholds),
            "backend": backend, "model": args.model,
            "thinking": t.thinking, "effort": t.effort, "banned_lexicon": str(args.banned),
            "n": len(df), "n_frame_units": len(units), "frame_units": units,
            "n_scored_this_run": stats["n_scored"], "n_parse_fail_this_run": stats["n_parse_fail"],
            "n_over_cost_cap_this_run": stats["n_over_cap"], "cost_cap_usd": args.cost_cap,
            "cost_usd_this_run": round(stats["cost_usd"], 4), "votes": str(votes_out),
            "trace_dir": str(trace_dir), "scored_at": _dt.datetime.now(_dt.timezone.utc).isoformat()})
        if units:
            rge.mark_exposed(units, tag, "eval", required=(args.split == GATED_SPLIT))
        summarize(out)


if __name__ == "__main__":
    main()
