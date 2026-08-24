#!/usr/bin/env python3
"""golden/regrade_scrambled.py — BLIND regrade of the scrambled top-100 set with the frozen a1
evidence-first stack, and the zero-API `--reletter` of a finished run (REGISTRY.md
"Deployment rule v2-deploy", item 8).

WHAT: the discovery run's `top100_clean_scrambled/` directory holds the same 100 cutout
composites as the ranked top-100, shuffled out of rank order, renamed 001.jpg..100.jpg and
with the footer strip (candidate id, coordinates, magnitude) removed — built by the JWST
repo precisely so a reviewer can score each field on the imaging alone. This script runs
the FULL frozen a1 panel (advocate → critics → arbitrator → aggregate_v2, exactly the tuple
registered in REGISTRY.md › Truth-eval registered arms for arm a1) over those 100 blind
images, then DE-SCRAMBLES via the answer key and writes a comparison CSV against the
incumbent pass-count verdicts (`verifier_grade` U/A/B/C).

WHY: the truth eval (TRUTH_EVAL.md) showed the incumbent verifier scored 23/24 known lenses
0/3 while the frozen a1 letters hold their FPR on the truth holdout. 78 of the run's own
top-100 sit at U (flagged, never verified). Regrading the run's own product blind — no id,
no rank, no coordinates, no filters, no incumbent grade in front of the model — asks what
the calibrated scheme says about the very candidates the pass-count scheme produced, free
of every anchoring channel the scrambled set was built to remove.

BLINDNESS CONTRACT: the model sees pixels + the frozen item-agnostic prompts, nothing else.
The ONLY thing read from the answer key per image is the LAYOUT (color vs gray), derived
from whether `sw_filter` / `lw_filter` are blank — it selects the per-role views and gloss
exactly as the truth-eval does (both gray variants alias to the same "gray" view set).
`build_blind_cands` structurally enforces this: it subsets the key to
{filename, sw_filter, lw_filter} before building, and each blind cand carries exactly
{name, image_path, layout} with name scr_001..scr_100. Every other key column (candidate_id,
rank, coordinates, verifier columns, blind_* measurements) is touched only AFTER scoring,
in the de-scramble join. Layout naming follows build_frame.derive_layout — gray_<channel>_only
names the channel PRESENT (lw blank -> gray_sw_only; sw blank -> gray_lw_only), verified
against frame.csv for all 99 frame ids (0 mismatches).

FROZEN TUPLE, per model (`FROZEN`): persona set / note / per-role system prompts / render
description are recomputed from disk and compared sha-by-sha against the registered a1
tuple; ANY mismatch is a SystemExit before any call — this is a replay of the registered
instrument, never a new one.
  * sonnet — the 2026-08-23 tuple: thinking off, effort default, thresholds_sha16 gated too
    (94d31c7b6979e0ca = sonnet_api_calibrated t_A 0.192 / t_B 0.1318 / tau0 0.15).
  * opus5 — the same prompts, thinking adaptive / effort xhigh (set through LENSJUDGE_THINKING /
    LENSJUDGE_EFFORT, the variables run_truth_eval sets and config.thinking_options reads);
    the thresholds sha is RECORDED, not gated — run-time letters are provisional until
    `calibrate_thresholds.py` writes `opus5_api` (REGISTRY item 1), after which `--reletter`
    re-assigns every letter from the stored per-role records (never re-scores).
k=1, Anthropic path. The scoring loop, traces, content-audit events, incremental parquet +
votes parquet, first-item sha assertion and cost cap are run_truth_eval.score's, reused
as-is. The 99 frame units among the 100 are marked kind="eval" in the exposure registry
under run_tag `scrambled100_blind` (the rank-14 alias is the same object as rank 7 and is
not a frame unit; it is skipped, stated, not marked).

DEPLOYMENT LETTERS (REGISTRY items 2-5, `aggregate_v2.deploy_letters`): every row carries,
beside the assemble() letters (`grade_pred` on S, `letter_arb` on S_arb), `letter_rank`
(advocate-only letter on p_evidence — the ranker), `letter_final` (rule R1: the arbitrated
evidence must clear the same bars; R2: letter_rank demoted to D by the D-rule only), `veto`
(role:alternative of the demoting critic(s), "" when nothing was demoted) and `rule`. They
are computed from the STORED votes (records.load_run → the run-time parse path) after
scoring, in a normal run and in `--reletter` alike; the comparison CSV orders rows by
p_evidence (R), ties by S.

`--reletter RUN_DIR` (zero API): rebuild every item's records from the run's votes parquet,
recompute S / S_arb / letters through the same assemble path with the thresholds file given
by `--thresholds` (resolved for the run's model) and `deploy_letters` under `--rule`, ASSERT
the rebuilt S / S_arb / p_evidence equal the stored ones within 1e-9 for every row whose
records parse (any mismatch is an error; nothing is written), copy the parquet to
`<name>.pre_reletter_<UTC>`, rewrite the letter / threshold / deploy columns (S, S_arb,
p_evidence are never rewritten; NaN rows stay NaN), update the meta json and regenerate the
comparison CSV.

This is a ONE-OFF diagnostic per model: outputs live under outputs/scrambled100*/ (gitignored,
never Xiaosheng-visible); nothing is written into the JWST repo; no registry-gate
registration is needed because the scrambled set is not a truth half — the provenance is
this file, the meta json (tuple shas + key-file sha) and the GOLDEN_FINDINGS.md entry.

  cd reproductions && export ANTHROPIC_API_KEY=$(cat ~/.anthropic/key) && \\
    ~/.venvs/lensjudge/bin/python lensjudge/golden/regrade_scrambled.py --model sonnet
  ~/.venvs/lensjudge/bin/python lensjudge/golden/regrade_scrambled.py --model opus5 \\
    --out lensjudge/outputs/scrambled100_opus5 --max-budget 60 [--dry-run]
  ~/.venvs/lensjudge/bin/python lensjudge/golden/regrade_scrambled.py \\
    --reletter lensjudge/outputs/scrambled100_opus5 --rule R1        # zero API
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import math
import os
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.golden import _util, aggregate_v2, registry  # noqa: E402
from lensjudge.golden import records as R  # noqa: E402
from lensjudge.golden import run_golden_eval as rge  # noqa: E402
from lensjudge.golden import run_truth_eval as rte  # noqa: E402

SCRAMBLED_DIR_DEFAULT = Path("/Users/benson/sync/research/lensing/git/jwst-strong-lens-search/top100_clean_scrambled")
KEY_DEFAULT = "top100_scrambled_key.csv"
OUT_DEFAULT = config.OUT / "scrambled100"
RUN_TAG = "scrambled100_blind"
ARM, RENDER = "a1", "v1"
IMG_SIZE = (752, 540)                    # footer already stripped by the scrambled build
BLIND_COLS = ("name", "image_path", "layout")     # ALL a blind cand may carry
KEY_BLIND_COLS = ("filename", "sw_filter", "lw_filter")   # ALL the builder may read
ANCHOR_RANKS = (15, 13, 7, 14, 16)
ALIAS_PAIR = (7, 14)                     # same object twice in the top-100: |dS| is a free retest
CRITIC_ROLES = aggregate_v2.CRITIC_ROLES

# The registered a1 prompt tuple (REGISTRY.md › Truth-eval registered arms, 2026-08-23):
# persona set / note / per-role system prompts / render description — shared by every model.
# Recomputed from disk at startup; any mismatch = STOP before any call.
FROZEN_SHAS = {
    "persona_set_sha16": "a26d972ecc0b4ee7",
    "note_sha16": "754655a400f360e6",
    "system_sha16s": {
        "advocate": "c41d7f5787bdb472",
        "artifact": "f5ed259652e65ee2",
        "geometry": "a293ddddce11ee4a",
        "morphology": "26bde57ad0478237",
        "arbitrator": "44542114399ab277",
    },
    "render_desc_sha16": "28737c6083dc1978",
}
# Per model: the thinking / effort the tuple must record, the thresholds sha (gated when a
# string, recorded only when None) and the letter_source values the resolved thresholds may
# carry. `letter_sources` is a gate, not a tuple slot.
FROZEN = {
    "sonnet": {**FROZEN_SHAS, "thresholds_sha16": "94d31c7b6979e0ca", "thinking": "off",
               "effort": "default", "letter_sources": ("sonnet_api_calibrated",)},
    "opus5": {**FROZEN_SHAS, "thresholds_sha16": None, "thinking": "adaptive", "effort": "xhigh",
              "letter_sources": ("provisional", "opus5_api_calibrated")},
}
MODELS = tuple(FROZEN)
GATED_KEYS = ("persona_set_sha16", "note_sha16", "system_sha16s", "render_desc_sha16",
              "thresholds_sha16", "thinking", "effort")
# expected $/item for the budget line: sonnet = the a1 holdout's $0.063/item; opus5 = the
# a1/opus5/adaptive/xhigh holdout's mean $0.228/item over its 231 scored rows (1.99 calls/item)
EXPECTED_PER_ITEM = {"sonnet": 0.065, "opus5": 0.23}
# per-item cost cap (a WARN in run_truth_eval.score): sonnet = the smoke-measured default;
# opus5 = the worst case of a full stack at its list price
COST_CAP = {"sonnet": rte.COST_CAP_DEFAULT, "opus5": rte.MAX_CALLS[ARM] * rte.COST_PER_CALL["opus5"]}

DEPLOY_RULES = aggregate_v2.DEPLOY_RULES            # ("R1", "R2")
DEPLOY_COLS = ("letter_rank", "letter_final", "veto", "rule")
# the columns a re-letter rewrites (S / S_arb / p_evidence are asserted equal, never written)
RELETTER_COLS = ("grade_pred", "letter_arb", "letter_source", "thresholds_sha16", "tau0", "t_A", "t_B") + DEPLOY_COLS
SCORE_COLS = ("S", "S_arb", "p_evidence")
PRE_RELETTER = ".pre_reletter_"

COMPARISON_COLS = (
    "scrambled_item", "rank", "candidate_id", "nate_grade", "nate_n_pass",
    "nate_inspector_conf", "blind_theta_E_arcsec", "discovery_status", "our_S", "our_S_arb",
    "our_letter", "our_letter_llm", "our_p_evidence", "our_scale_class",
    "our_alternative_final", "our_needs_human", "agree_letter",
    # REGISTRY item 8: the deployment letters, appended after the original 17 columns
    "our_letter_rank", "our_letter_final", "our_veto", "our_rule", "agree_final", "our_rationale")


# ------------------------------------------------------------------ blind side
def _blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return not str(v).strip()


def derive_layout(sw, lw) -> str:
    """color / gray_sw_only / gray_lw_only from the key's sw_filter / lw_filter blanks,
    named exactly as build_frame.derive_layout names them (gray_<channel>_only = the channel
    that is PRESENT). Both gray variants alias to the same 'gray' view set in the panel
    gloss, so this only ever changes which composite VIEW text and crops the roles get."""
    sw_ok, lw_ok = not _blank(sw), not _blank(lw)
    if sw_ok and lw_ok:
        return "color"
    if sw_ok:
        return "gray_sw_only"
    if lw_ok:
        return "gray_lw_only"
    raise ValueError("both sw_filter and lw_filter blank: no renderable channel")


def scr_name(filename: str) -> str:
    m = re.fullmatch(r"(\d{3})\.jpg", str(filename).strip())
    if not m:
        raise ValueError(f"scrambled filename must be NNN.jpg, got {filename!r}")
    return f"scr_{m.group(1)}"


def scr_to_filename(name: str) -> str:
    m = re.fullmatch(r"scr_(\d{3})", str(name).strip())
    if not m:
        raise ValueError(f"blind name must be scr_NNN, got {name!r}")
    return f"{m.group(1)}.jpg"


def build_blind_cands(key: pd.DataFrame, scrambled_dir: Path, verify_images: bool = True) -> pd.DataFrame:
    """The blind manifest: one row per key row with EXACTLY the BLIND_COLS. The key is
    subset to KEY_BLIND_COLS first, so no other key column can reach this frame (the
    blindness contract is structural, not a convention). With `verify_images` every jpg
    must exist and be 752x540 (footer already stripped)."""
    for c in KEY_BLIND_COLS:
        if c not in key.columns:
            raise ValueError(f"key lacks column {c!r}")
    key = key[list(KEY_BLIND_COLS)]          # nothing else is readable from here on
    scrambled_dir = Path(scrambled_dir)
    rows = []
    for _, r in key.iterrows():
        fn = str(r["filename"]).strip()
        name = scr_name(fn)
        p = scrambled_dir / fn
        if verify_images:
            if not p.exists():
                raise FileNotFoundError(f"scrambled image missing: {p}")
            from PIL import Image
            with Image.open(p) as im:
                if im.size != IMG_SIZE:
                    raise ValueError(f"{p}: {im.size} != expected {IMG_SIZE}")
        rows.append({"name": name, "image_path": str(p),
                     "layout": derive_layout(r["sw_filter"], r["lw_filter"])})
    df = pd.DataFrame(rows, columns=list(BLIND_COLS))
    if not df["name"].is_unique:
        raise ValueError("duplicate scrambled filenames in the key")
    assert set(df.columns) == set(BLIND_COLS)
    return df.sort_values("name").reset_index(drop=True)


# ------------------------------------------------------------------ the frozen-tuple gate
def apply_model_env(model: str) -> tuple[str, str]:
    """Set LENSJUDGE_THINKING / LENSJUDGE_EFFORT to the model's frozen values — the variables
    run_truth_eval sets from --thinking / --effort and config.thinking_options reads at call
    time — so the recorded tuple IS what the call uses. Effort "default" = the variable
    unset (the API default). Returns (thinking, effort) as run_golden_eval reads them."""
    fz = FROZEN[model]
    os.environ["LENSJUDGE_THINKING"] = fz["thinking"]
    if fz["effort"] == "default":
        os.environ.pop("LENSJUDGE_EFFORT", None)
    else:
        os.environ["LENSJUDGE_EFFORT"] = fz["effort"]
    return rge.thinking_setting(), rge.effort_setting()


def check_frozen(persona_dir: Path, note: str, shas: dict, thresholds: dict,
                 thinking: str, effort: str, model: str = "sonnet") -> dict:
    """Compare every recomputed slot with the registered a1 tuple of `model`; SystemExit on
    ANY difference — this run replays the frozen instrument or it does not run. A frozen
    slot of None (opus5's thresholds sha) is recorded, not gated. The resolved thresholds'
    letter_source must be one the model allows. Returns the computed slots."""
    fz = FROZEN[model]
    got = {
        "persona_set_sha16": rte._panel().persona_set_sha16(persona_dir),
        "note_sha16": _util.sha_text(note),
        "system_sha16s": {r: shas[r] for r in rte.ROLES[ARM]},
        "render_desc_sha16": rte.render_desc_sha(RENDER),
        "thresholds_sha16": rte.thresholds_sha(thresholds),
        "thinking": thinking,
        "effort": effort,
    }
    bad = {k: (got[k], fz[k]) for k in GATED_KEYS if fz[k] is not None and got[k] != fz[k]}
    if bad:
        lines = "\n".join(f"  {k}: computed {g!r} != frozen {f!r}" for k, (g, f) in bad.items())
        raise SystemExit(f"[scrambled100] STOP: the on-disk stack no longer matches the frozen "
                         f"a1/{model} tuple — this script replays the registered instrument only:\n{lines}")
    src = thresholds.get("letter_source")
    if src not in fz["letter_sources"]:
        raise SystemExit(f"[scrambled100] STOP: letter_source {src!r} not in {fz['letter_sources']} for {model}")
    gate = "gated" if fz["thresholds_sha16"] is not None else "recorded, not gated"
    print(f"[scrambled100] frozen a1/{model} tuple verified: persona set {got['persona_set_sha16']}, note "
          f"{got['note_sha16']}, thinking {thinking} / effort {effort}; thresholds "
          f"{got['thresholds_sha16']} ({src}; {gate})")
    return got


# ------------------------------------------------------------------ deployment letters
def _finite(x) -> bool:
    try:
        return x is not None and not pd.isna(x) and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _close(a, b, atol: float) -> bool:
    fa, fb = _finite(a), _finite(b)
    if not fa or not fb:
        return fa == fb
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)


def deploy_rows(preds: pd.DataFrame, records: dict, thresholds: dict, rule: str = "R1",
                atol: float = 1e-9) -> tuple[pd.DataFrame, list[dict]]:
    """The stored preds rows re-lettered from their records under `thresholds` (a resolved
    run_truth_eval.load_thresholds dict) and `rule`: (new_preds, mismatches). Pure — nothing
    is written. A row whose stored S is finite has its rebuilt S / S_arb / p_evidence checked
    against the stored ones first (|Δ| ≤ atol; a scored row without records is a mismatch
    too). A row that REPRODUCES gets the threshold columns (letter_source, thresholds_sha16,
    tau0, t_A, t_B), `grade_pred` / `letter_arb` from `records.panel_result_from_records`
    (the assemble path) and letter_rank / letter_final / veto from
    `aggregate_v2.deploy_letters`. A row that does NOT reproduce keeps EVERY stored value
    (its grade_pred / letter_arb / threshold columns are the run's record and are never
    replaced by a rebuild that disagrees with it) and gets only rule + blank deploy letters
    (letter_rank / letter_final None, veto ""). A row whose stored S is NaN keeps NaN and its
    stored grade_pred / letter_arb; its deploy letters follow deploy_letters (letter_rank
    from the advocate when it parsed, letter_final None). S / S_arb / p_evidence are never
    written."""
    if rule not in DEPLOY_RULES:
        raise ValueError(f"rule must be one of {DEPLOY_RULES}, got {rule!r}")
    R.check_cols(preds, ("name", "S"), "preds")
    thr = dict(thresholds)
    sha = rte.thresholds_sha(thr)
    out = preds.copy()
    for c in RELETTER_COLS:
        if c not in out.columns:
            out[c] = None
    out = out.astype({c: object for c in ("grade_pred", "letter_arb", "letter_source") + DEPLOY_COLS})
    mism: list[dict] = []
    for i, row in preds.iterrows():
        name = str(row["name"])
        thr_vals = {"letter_source": thr.get("letter_source"), "thresholds_sha16": sha, "tau0": thr.get("tau0"),
                    "t_A": thr.get("t_A"), "t_B": thr.get("t_B")}
        vals = {"rule": rule, "letter_rank": None, "letter_final": None, "veto": ""}
        scored = _finite(row["S"])
        row_mism: list[dict] = []
        if name in records:
            roles = records[name]
            res = R.panel_result_from_records(name, records, thr, preds_row=row)
            dep = R.deploy_from_roles(roles, thr, rule)   # voids a called-but-failed arbitrator
            if scored:
                for col, got in (("S", res.S), ("S_arb", res.S_arb), ("p_evidence", res.p_evidence)):
                    if not _close(row.get(col), got, atol):
                        row_mism.append({"name": name, "col": col, "stored": row.get(col), "rebuilt": got})
            if not row_mism:
                vals.update(thr_vals)
                if scored:
                    vals.update(grade_pred=res.letter, letter_arb=res.letter_arb)
                vals.update(letter_rank=dep["letter_rank"], letter_final=dep["letter_final"], veto=dep["veto"] or "")
        elif scored:
            row_mism.append({"name": name, "col": "records", "stored": row.get("S"), "rebuilt": None})
        else:
            vals.update(thr_vals)
        mism += row_mism
        for c, v in vals.items():
            out.at[i, c] = v
    return out, mism


def _norm(x):
    """None for any missing value, else the value (None-safe equality of letter cells)."""
    return None if x is None or (isinstance(x, float) and math.isnan(x)) or (not isinstance(x, str) and pd.isna(x)) else x


def _one_line(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    return " ".join(str(s).split())


def _col(df: pd.DataFrame, c: str) -> pd.Series:
    return df[c] if c in df.columns else pd.Series([None] * len(df), index=df.index, dtype=object)


# ------------------------------------------------------------------ de-scramble
def descramble(preds: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Join the blind rows back to the key (filename -> candidate_id) and shape the
    comparison table, ordered as REGISTRY item 2 ranks: our_p_evidence descending, ties by
    our_S descending, NaN (parse failure / unexamined) last. agree_letter = the incumbent
    letter equals ours (grade_pred) exactly; agree_final = it equals our_letter_final (U or
    a None never agree). The deploy columns are None / "" when the preds lack them."""
    p = preds.copy()
    p["filename"] = p["name"].map(scr_to_filename)
    merged = p.merge(key, on="filename", how="left", validate="one_to_one", suffixes=("", "_key"))
    if merged["candidate_id"].isna().any():
        missing = merged.loc[merged["candidate_id"].isna(), "name"].tolist()
        raise ValueError(f"key rows missing for {missing}")
    comp = pd.DataFrame({
        "scrambled_item": merged["name"],
        "rank": pd.to_numeric(merged["rank"]).astype(int),
        "candidate_id": merged["candidate_id"].astype(str),
        "nate_grade": merged["verifier_grade"].astype(str),
        "nate_n_pass": pd.to_numeric(merged["verifiers_pass"]).astype(int),
        "nate_inspector_conf": pd.to_numeric(merged["inspector_confidence"], errors="coerce"),
        "blind_theta_E_arcsec": pd.to_numeric(merged["blind_theta_E_arcsec"], errors="coerce"),
        "discovery_status": merged["discovery_status"],
        "our_S": pd.to_numeric(merged["S"], errors="coerce"),
        "our_S_arb": pd.to_numeric(merged["S_arb"], errors="coerce"),
        "our_letter": merged["grade_pred"],
        "our_letter_llm": merged["letter_llm"],
        "our_p_evidence": pd.to_numeric(merged["p_evidence"], errors="coerce"),
        "our_scale_class": merged["scale_class_final"],
        "our_alternative_final": merged["alternative_final"],
        "our_needs_human": merged["needs_human"],
        "our_letter_rank": _col(merged, "letter_rank"),
        "our_letter_final": _col(merged, "letter_final"),
        "our_veto": _col(merged, "veto").map(lambda v: "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)),
        "our_rule": _col(merged, "rule"),
        "our_rationale": _col(merged, "rationale").map(_one_line),
    })
    comp["agree_letter"] = comp["nate_grade"].astype(str) == comp["our_letter"].astype(str)
    comp["agree_final"] = comp["nate_grade"].astype(str) == comp["our_letter_final"].astype(str)
    comp = comp[list(COMPARISON_COLS)]
    return comp.sort_values(["our_p_evidence", "our_S"], ascending=[False, False], kind="mergesort",
                            na_position="last").reset_index(drop=True)


def print_summary(comp: pd.DataFrame, total_cost: float) -> dict:
    """The de-scrambled read: cross-tabs (grade_pred and letter_final), U->A/B, the
    incumbent A/B/C rows, anchors, Spearman, veto count."""
    out: dict = {}
    xt = pd.crosstab(comp["nate_grade"], comp["our_letter"].fillna("parse_fail"))
    print("\n[summary] letter cross-tab (incumbent grade x ours, grade_pred on S):")
    print(xt.to_string())
    out["crosstab"] = {str(i): {str(c): int(v) for c, v in row.items()} for i, row in xt.iterrows()}
    have_deploy = comp["our_letter_final"].notna().any() or comp["our_letter_rank"].notna().any()
    if have_deploy:
        rule = sorted(set(comp["our_rule"].dropna().astype(str)))
        xr = pd.crosstab(comp["nate_grade"], comp["our_letter_rank"].fillna("none"))
        xf = pd.crosstab(comp["nate_grade"], comp["our_letter_final"].fillna("none"))
        print(f"\n[summary] letter_rank cross-tab (incumbent grade x advocate-only letter):")
        print(xr.to_string())
        print(f"\n[summary] letter_final cross-tab (incumbent grade x deployed letter, rule {rule}):")
        print(xf.to_string())
        out["crosstab_rank"] = {str(i): {str(c): int(v) for c, v in row.items()} for i, row in xr.iterrows()}
        out["crosstab_final"] = {str(i): {str(c): int(v) for c, v in row.items()} for i, row in xf.iterrows()}
        out["n_veto"] = int((comp["our_veto"].fillna("").astype(str) != "").sum())
        out["n_rank_AB"] = int(comp["our_letter_rank"].isin(["A", "B"]).sum())
        out["n_final_AB"] = int(comp["our_letter_final"].isin(["A", "B"]).sum())
        out["n_agree_final"] = int(comp["agree_final"].sum())
        out["rule"] = rule
        print(f"\n[summary] rank A/B {out['n_rank_AB']} -> final A/B {out['n_final_AB']}; "
              f"{out['n_veto']} veto(es); final-letter agreement {out['n_agree_final']}/{len(comp)}")

    u = comp[comp["nate_grade"] == "U"]
    u_ab = u[u["our_letter"].isin(["A", "B"])]
    print(f"\n[summary] incumbent-U items reaching our A/B: {len(u_ab)}/{len(u)}")
    for _, r in u_ab.iterrows():
        print(f"  rank {r['rank']:>3}  {r['candidate_id']}  our {r['our_letter']}  S {r['our_S']:.3f}  "
              f"p_ev {r['our_p_evidence']:.2f}  scale {r['our_scale_class']}  alt {r['our_alternative_final']}")
    out["n_U"] = int(len(u)); out["n_U_to_AB"] = int(len(u_ab))
    if have_deploy:
        u_fab = u[u["our_letter_final"].isin(["A", "B"])]
        out["n_U_to_final_AB"] = int(len(u_fab))
        print(f"[summary] incumbent-U items reaching our letter_final A/B: {len(u_fab)}/{len(u)}")

    for g in ("A", "B", "C"):
        sub = comp[comp["nate_grade"] == g].sort_values("rank")
        print(f"\n[summary] incumbent {g} ({len(sub)}):")
        for _, r in sub.iterrows():
            extra = (f"  rank/final {r['our_letter_rank']}/{r['our_letter_final']}  veto {r['our_veto'] or '-'}"
                     if have_deploy else "")
            print(f"  rank {r['rank']:>3}  {r['candidate_id']}  our {r['our_letter']}  S {r['our_S']:.3f}  "
                  f"S_arb {r['our_S_arb']:.3f}  p_ev {r['our_p_evidence']:.2f}  alt {r['our_alternative_final']}{extra}")
        out[f"n_{g}_agree"] = int(sub["agree_letter"].sum())

    rho = comp["our_S"].corr(comp["nate_inspector_conf"], method="spearman")
    out["spearman_S_conf"] = None if pd.isna(rho) else float(rho)
    print(f"\n[summary] Spearman(our_S, incumbent inspector conf): {rho:.3f} "
          f"(n={int(comp['nate_inspector_conf'].notna().sum())})")

    print("\n[summary] anchors (the five PI-anchor ranks; predictions live in REGISTRY.md):")
    anch = comp[comp["rank"].isin(ANCHOR_RANKS)].set_index("rank")
    for rk in ANCHOR_RANKS:
        if rk in anch.index:
            r = anch.loc[rk]
            extra = f"  rank/final {r['our_letter_rank']}/{r['our_letter_final']}" if have_deploy else ""
            print(f"  rank {rk:>3}  {r['candidate_id']}  incumbent {r['nate_grade']}  our {r['our_letter']}  "
                  f"S {r['our_S']:.3f}  scale {r['our_scale_class']}  alt {r['our_alternative_final']}{extra}")
    a, b = ALIAS_PAIR
    if a in anch.index and b in anch.index:
        ds = abs(float(anch.loc[a, "our_S"]) - float(anch.loc[b, "our_S"]))
        out["alias_abs_dS"] = ds
        print(f"  ranks {a}/{b} are the SAME object rendered twice: |our_S diff| = {ds:.3f} "
              f"(letters {anch.loc[a, 'our_letter']}/{anch.loc[b, 'our_letter']})")

    out["n_agree"] = int(comp["agree_letter"].sum())
    out["total_cost_usd"] = round(float(total_cost), 4)
    print(f"\n[summary] letter agreement {out['n_agree']}/{len(comp)}; total cost ${total_cost:.2f}")
    return out


def mark_exposure(comp: pd.DataFrame, frame_csv: Path, registry_csv: Optional[Path] = None,
                  run_tag: str = RUN_TAG) -> list[str]:
    """kind='eval' exposure for every candidate that IS a frame unit (99 of 100); ids not in
    the frame (the rank-14 alias of the rank-7 system) are skipped, stated, never marked."""
    fr = rge._read(Path(frame_csv))
    cmap = dict(zip(fr["candidate_id"].astype(str), fr["unit_id"].astype(str)))
    units, skipped = [], []
    for cid in comp["candidate_id"].astype(str):
        (units if cid in cmap else skipped).append(cmap.get(cid, cid))
    units = sorted(set(units))
    if skipped:
        print(f"[registry] {len(skipped)} id(s) are not frame units (top-100 alias rows) and are "
              f"skipped: {skipped}")
    kw = {"path": Path(registry_csv)} if registry_csv else {}
    registry.mark_exposed(units, run_tag, "eval", **kw)
    print(f"[registry] marked {len(units)} frame units exposed: kind=eval run_tag={run_tag}")
    return units


# ------------------------------------------------------------------ files of a run
def preds_path_for(outdir: Path, model: str) -> Path:
    return Path(outdir) / f"preds_scrambled100_{ARM}_{model}.parquet"


def find_run_parquet(run_dir: Path, model: Optional[str] = None) -> Path:
    """The one preds parquet of a scrambled run directory (its `_votes` sibling and any
    `.pre_reletter_*` copies excluded); `model` narrows to that model's file."""
    run_dir = Path(run_dir)
    cands = sorted(p for p in run_dir.glob(f"preds_scrambled100_{ARM}_*.parquet")
                   if not p.name.endswith(R.VOTES_SUFFIX))
    if model is not None:
        cands = [p for p in cands if p == preds_path_for(run_dir, model)]
    if len(cands) != 1:
        raise SystemExit(f"[reletter] expected exactly one preds parquet in {run_dir}"
                         f"{f' for model {model}' if model else ''}, found {[p.name for p in cands]}")
    return cands[0]


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write via a sibling temp file + os.replace (never a half-written parquet; the temp
    file is removed on failure — run_truth_eval._write_parquet_atomic)."""
    rte._write_parquet_atomic(df, Path(path))


PRE_DEPLOY = ".pre_deploy_"


def write_with_backup(df: pd.DataFrame, out: Path, suffix: str = PRE_DEPLOY,
                      now: Optional[_dt.datetime] = None) -> Optional[Path]:
    """Copy the existing parquet to `<name><suffix><UTC>` (never overwriting an earlier
    copy), then write `df` atomically. Returns the backup path (None when `out` did not
    exist). Every rewrite of a run parquet — the deploy columns after scoring, a
    --descramble-only, a --reletter — goes through a backup so the run's record is never
    replaced in place."""
    out = Path(out)
    backup = None
    if out.exists():
        backup = _unused(out.with_name(out.name + suffix + _utc_stamp(now)))
        shutil.copy2(out, backup)
    _write_parquet(df, out)
    return backup


def _utc_stamp(now: Optional[_dt.datetime] = None) -> str:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def _unused(path: Path) -> Path:
    """`path`, or `path_2`, `path_3`, … — a backup never overwrites an earlier one."""
    cand, i = path, 1
    while cand.exists():
        i += 1
        cand = path.with_name(f"{path.name}_{i}")
    return cand


def attach_deploy(out: Path, votes_out: Path, thresholds: dict, rule: str, trace_dir=None,
                  strict: bool = False) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Load a finished run (records.load_run, the run-time parse path; a missing votes raw
    recovered from `trace_dir`) and re-letter it with `deploy_rows`. Returns (new_preds,
    records, mismatches). With `strict` any rebuilt-vs-stored S / S_arb / p_evidence mismatch
    is a SystemExit (nothing written by the caller); otherwise the mismatching rows are
    reported — `deploy_rows` has already kept their stored letters / threshold columns and
    blanked only their deploy letters."""
    if not Path(votes_out).exists():
        raise SystemExit(f"[reletter] votes parquet missing: {votes_out}")
    preds, records = R.load_run(out, votes_out, trace_dir=trace_dir)
    new, mism = deploy_rows(preds, records, thresholds, rule)
    if mism:
        print(f"[reletter] ERROR: {len(mism)} rebuilt value(s) differ from the stored ones "
              f"({len({m['name'] for m in mism})} item(s)) — records do not reproduce the run:")
        for m in mism[:50]:
            print(f"  {m['name']}: {m['col']} stored {m['stored']!r} != rebuilt {m['rebuilt']!r}")
        if strict:
            raise SystemExit("[reletter] STOP: nothing written; the stored parquet is not a function of its "
                             "votes (missing raws? pass --trace-dir) — investigate before re-lettering")
        print(f"[reletter] those rows keep their stored grade_pred / letter_arb / threshold columns; "
              f"their deploy letters are blank")
    return new, records, mism


FROZEN_COLS = ("persona_set_sha16", "note_sha16", "system_sha16s", "render_desc_sha16", "thinking", "effort")


def frozen_columns(model: str) -> dict:
    """The value every preds row of an a1/`model` run must carry in each FROZEN_COLS column
    (system_sha16s as the '+'-joined role:sha string run_truth_eval stores)."""
    fz = FROZEN[model]
    return {c: (rte.join_shas(fz["system_sha16s"], ARM) if c == "system_sha16s" else fz[c]) for c in FROZEN_COLS}


def check_frozen_columns(preds: pd.DataFrame, model: str) -> dict:
    """SystemExit unless every FROZEN_COLS column of `preds` holds exactly the registered
    a1/`model` tuple value (thresholds_sha16 excluded, as in `check_frozen`): a re-letter
    presumes the stored per-role records are the registered instrument's. Returns the
    stored values."""
    want = frozen_columns(model)
    got: dict = {}
    bad: dict = {}
    for c, w in want.items():
        if c not in preds.columns:
            bad[c] = ("<column absent>", w)
            continue
        vals = sorted(set(str(v) for v in preds[c].dropna().tolist()))
        got[c] = vals[0] if len(vals) == 1 else vals
        if vals != [str(w)]:
            bad[c] = (got[c], w)
    if bad:
        lines = "\n".join(f"  {c}: stored {g!r} != frozen {w!r}" for c, (g, w) in bad.items())
        raise SystemExit(f"[reletter] STOP: the parquet's run tuple is not the frozen a1/{model} tuple — its "
                         f"records are not the registered instrument's and may not be re-lettered:\n{lines}")
    return got


def reletter_run(run_dir: Path, key_path: Path, thresholds_path: Path, rule: str = "R1",
                 model: Optional[str] = None, trace_dir=None, now: Optional[_dt.datetime] = None) -> dict:
    """REGISTRY item 8, second half — zero API. Re-assign every letter of the finished run in
    `run_dir` from its stored records under the thresholds file (resolved for the run's
    model) and `rule`; the parquet's frozen-tuple columns must equal the registered a1
    tuple of its model (`check_frozen_columns`); the parquet is copied to
    `<name>.pre_reletter_<UTC>` first, S / S_arb / p_evidence are asserted equal and never
    rewritten, NaN rows stay NaN, the meta json is updated and the comparison CSV
    regenerated. `model`, when given, must be the run's."""
    run_dir = Path(run_dir)
    out = find_run_parquet(run_dir, model)
    votes_out = R.votes_path_for(out)
    if trace_dir is None and (run_dir / f"traces_{RUN_TAG}").is_dir():
        trace_dir = run_dir / f"traces_{RUN_TAG}"
    preds0 = pd.read_parquet(out)
    run_model = sorted(set(preds0["model"].dropna().astype(str))) if "model" in preds0.columns else []
    if len(run_model) != 1 or run_model[0] not in FROZEN:
        raise SystemExit(f"[reletter] the parquet's model column must hold one of {MODELS}, got {run_model}")
    run_model = run_model[0]
    if model is not None and model != run_model:
        raise SystemExit(f"[reletter] --model {model} but {out.name} is a {run_model} run")
    check_frozen_columns(preds0, run_model)
    key_path = Path(key_path)
    if not key_path.exists():
        raise SystemExit(f"answer key missing: {key_path}")
    key = pd.read_csv(key_path, dtype=str)
    key_sha = _util.sha_file(key_path)
    mp = rge.meta_path(out)
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    if meta.get("key_sha16") and meta["key_sha16"] != key_sha:
        raise SystemExit(f"[reletter] STOP: key {key_path} sha {key_sha} != the run's key_sha16 "
                         f"{meta['key_sha16']} — this run must be de-scrambled with the key it was run with")
    if len(preds0) != len(key):
        raise SystemExit(f"[reletter] parquet holds {len(preds0)}/{len(key)} items — resume the run first")

    thresholds = rte.load_thresholds(Path(thresholds_path), rte.model_key(run_model))
    src = thresholds.get("letter_source")
    if src not in FROZEN[run_model]["letter_sources"]:
        raise SystemExit(f"[reletter] STOP: letter_source {src!r} not in {FROZEN[run_model]['letter_sources']} "
                         f"for {run_model} ({thresholds_path})")
    sha = rte.thresholds_sha(thresholds)
    print(f"[reletter] {out.name}: {len(preds0)} rows, model {run_model}, rule {rule}; thresholds {src} "
          f"tau0={thresholds['tau0']} t_A={thresholds['t_A']} t_B={thresholds['t_B']} ({sha})"
          f"{f'; traces {trace_dir}' if trace_dir else ''}")
    new, records, _ = attach_deploy(out, votes_out, thresholds, rule, trace_dir=trace_dir, strict=True)
    for c in SCORE_COLS:                  # belt and braces: the score columns are byte-identical
        assert new[c].equals(preds0[c]) or all(_close(a, b, 0.0) for a, b in zip(new[c], preds0[c])), c

    stamp = _utc_stamp(now)
    backup = _unused(out.with_name(out.name + PRE_RELETTER + stamp))
    shutil.copy2(out, backup)
    comp_out = run_dir / "scrambled100_comparison.csv"
    comp_backup = None
    if comp_out.exists():
        comp_backup = _unused(comp_out.with_name(comp_out.name + PRE_RELETTER + stamp))
        shutil.copy2(comp_out, comp_backup)
    _write_parquet(new, out)
    changed = sum(1 for a, b in zip(new["grade_pred"], preds0["grade_pred"]) if _norm(a) != _norm(b))
    n_nan = int(new["S"].isna().sum())
    print(f"[reletter] parquet rewritten ({changed} grade_pred change(s); {n_nan} NaN row(s) left NaN); "
          f"previous copy -> {backup.name}")

    comp = descramble(new, key)
    comp.to_csv(comp_out, index=False)
    print(f"[reletter] comparison -> {comp_out}")
    total_cost = float(pd.to_numeric(new["cost_usd"], errors="coerce").fillna(0.0).sum()) if "cost_usd" in new.columns else 0.0
    summary = print_summary(comp, total_cost)

    at = (now or _dt.datetime.now(_dt.timezone.utc)).isoformat()
    hist = list(meta.get("reletter_history") or [])
    hist.append({"relettered_at_utc": at, "rule": rule, "letter_source": src, "thresholds_sha16": sha,
                 "thresholds": str(thresholds_path), "pre_reletter": str(backup),
                 "pre_reletter_comparison": str(comp_backup) if comp_backup else None,
                 "n_grade_pred_changed": changed})
    meta.update({
        "letter_source": src, "thresholds_resolved": thresholds, "thresholds_sha16": sha,
        "relettered_at_utc": at, "rule": rule, "deploy_cols": list(DEPLOY_COLS),
        "pre_reletter": str(backup), "comparison": str(comp_out), "summary": summary,
        "reletter_history": hist, "model": run_model, "out": str(out), "votes": str(votes_out),
        "n_nan": n_nan, "n_items_with_records": len(records)})
    if isinstance(meta.get("tuple"), dict):
        meta["tuple"]["thresholds_sha16"] = sha
    rge.write_meta(out, meta)
    print(f"[reletter] meta -> {mp}")
    return {"out": out, "backup": backup, "comparison": comp_out, "meta": mp, "n_changed": changed,
            "n_nan": n_nan, "thresholds": thresholds, "thresholds_sha16": sha, "rule": rule, "summary": summary}


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scrambled-dir", default=str(SCRAMBLED_DIR_DEFAULT))
    ap.add_argument("--key", default=KEY_DEFAULT,
                    help="answer-key CSV (resolved inside --scrambled-dir when relative)")
    ap.add_argument("--model", choices=MODELS, default=None,
                    help="sonnet (default; frozen calibrated thresholds) or opus5 (adaptive/xhigh; "
                         "provisional letters until --reletter). With --reletter: must be the run's model")
    ap.add_argument("--out", default=str(OUT_DEFAULT), help="output directory (gitignored)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--cost-cap", type=float, default=None,
                    help=f"per-item WARN threshold (default per model: {COST_CAP})")
    ap.add_argument("--max-budget", type=float, default=10.0,
                    help="refuse to start when the worst-case estimate exceeds this")
    ap.add_argument("--frame", default=str(rte.FRAME_CSV))
    ap.add_argument("--registry-csv", default=None)
    ap.add_argument("--banned", default=str(rte.BANNED))
    ap.add_argument("--thresholds", default=str(rte.THRESHOLDS))
    ap.add_argument("--rule", choices=DEPLOY_RULES, default="R1",
                    help="deployment rule for letter_final (REGISTRY items 4-6; R1 primary)")
    ap.add_argument("--reletter", default=None, metavar="RUN_DIR",
                    help="zero-API: re-assign the letters of the finished run in RUN_DIR from its stored "
                         "records under --thresholds / --rule (parquet copied to *.pre_reletter_<UTC>)")
    ap.add_argument("--trace-dir", default=None,
                    help="the run's per-role traces (recovers a missing votes raw); default "
                         "<run dir>/traces_scrambled100_blind when present")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify the frozen tuple, the key and every image; no model call")
    ap.add_argument("--descramble-only", action="store_true",
                    help="rebuild the deploy columns + comparison CSV from an existing complete parquet")
    args = ap.parse_args(argv)

    scrambled_dir = Path(args.scrambled_dir)
    key_path = Path(args.key) if Path(args.key).is_absolute() else scrambled_dir / args.key
    if args.reletter:
        reletter_run(Path(args.reletter), key_path, Path(args.thresholds), rule=args.rule, model=args.model,
                     trace_dir=Path(args.trace_dir) if args.trace_dir else None)
        return

    model = args.model or "sonnet"
    os.environ.setdefault("LENSJUDGE_BACKEND", "anthropic")
    thinking, effort = apply_model_env(model)
    if not key_path.exists():
        raise SystemExit(f"answer key missing: {key_path}")
    key = pd.read_csv(key_path, dtype=str)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    out = preds_path_for(outdir, model)
    votes_out = R.votes_path_for(out)
    trace_dir = outdir / f"traces_{RUN_TAG}"                # where score() writes this run's traces
    read_trace_dir = Path(args.trace_dir) if args.trace_dir else trace_dir   # where a missing raw is recovered from
    comp_out = outdir / "scrambled100_comparison.csv"

    # ---- frozen prompts + thresholds, verified sha-by-sha before anything else
    note = rte.NOTE_V2.read_text()
    persona_set, full, _ = rte.role_prompts(rte.PERSONA_SET_DEFAULT, ARM, note)
    shas = rte.system_shas(full, ARM)
    thresholds = rte.load_thresholds(Path(args.thresholds), rte.model_key(model))
    got = check_frozen(rte.PERSONA_SET_DEFAULT, note, shas, thresholds, thinking, effort, model)
    for role in rte.ROLES[ARM]:
        rge.check_system_prompt(full[role], Path(args.banned), None)

    # ---- the blind manifest (layout is the ONLY thing read from the key here)
    df = build_blind_cands(key, scrambled_dir)
    n_gray = int((df["layout"] != "color").sum())
    print(f"[scrambled100] {len(df)} blind items ({len(df) - n_gray} color / {n_gray} gray), "
          f"images verified {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    est = len(df) * rte.MAX_CALLS[ARM] * rte.COST_PER_CALL[model]
    print(f"[scrambled100] worst-case ≈ ${est:.2f} at {rte.MAX_CALLS[ARM]} calls/item × "
          f"${rte.COST_PER_CALL[model]:.3f}/call (expected ≈ ${len(df) * EXPECTED_PER_ITEM[model]:.2f} "
          f"at the holdout's ${EXPECTED_PER_ITEM[model]:.3f}/item)")
    if est > args.max_budget:
        raise SystemExit(f"[scrambled100] REFUSED: worst-case ${est:.2f} > --max-budget ${args.max_budget:.2f}")

    # ---- the tuple recorded on every row (splits slot holds the key-file sha: the mapping
    # this run must be de-scrambled with is pinned in its own provenance)
    key_sha = _util.sha_file(key_path)
    t = rte.TruthTuple(ARM, model, got["persona_set_sha16"], got["note_sha16"], rte.join_shas(shas, ARM),
                       "jwst_v1", got["render_desc_sha16"], key_sha, "none", thinking, effort, 1,
                       got["thresholds_sha16"])
    print(f"[scrambled100] tuple {t.row()}")
    print(f"[scrambled100] thresholds resolved: {thresholds}; rule {args.rule}; "
          f"cost cap ${args.cost_cap if args.cost_cap is not None else COST_CAP[model]:.2f}/item")
    if args.dry_run:
        print(f"[scrambled100] --dry-run: a1/{model} tuple + key + {len(df)} images verified; no call made")
        return
    extra = {**asdict(t), "k": 1, "run_tag": RUN_TAG, "rescored": False, "rescore_reason": None,
             "tau0": thresholds["tau0"], "t_A": thresholds.get("t_A"), "t_B": thresholds.get("t_B")}
    cost_cap = args.cost_cap if args.cost_cap is not None else COST_CAP[model]

    if not args.descramble_only:
        stats = asyncio.run(rte.score(
            df, out, votes_out, trace_dir, arm=ARM, model=model, persona_set=persona_set,
            note=note, render=RENDER, claim_mode="none", thresholds=thresholds,
            concurrency=args.concurrency, extra_cols=extra, expected_shas=rte.allowed_shas(shas),
            stamps_dir=None, cost_cap=cost_cap, persona_set_noclaim=None))
        print(f"[scrambled100] scored {stats['n_scored']} this run "
              f"(${stats['cost_usd']:.2f}, {stats['n_parse_fail']} parse failures, "
              f"{stats['n_over_cap']} over the per-item cap)")

    preds = pd.read_parquet(out)
    if len(preds) != len(df):
        raise SystemExit(f"[scrambled100] parquet holds {len(preds)}/{len(df)} items — resume the "
                         f"run before de-scrambling")
    n_ok = int(preds["parse_ok"].astype(bool).sum())
    src = sorted(set(preds.loc[preds["grade_pred"].notna(), "letter_source"].astype(str)))
    print(f"[scrambled100] parse_ok {n_ok}/{len(preds)}; letter_source {src}")
    if src and (len(src) != 1 or src[0] not in FROZEN[model]["letter_sources"]):
        raise SystemExit(f"[scrambled100] STOP: letter_source {src} not in {FROZEN[model]['letter_sources']}")

    # ---- deployment letters from the stored votes (zero API; REGISTRY items 3-5). After a
    # scoring run a row whose records do not rebuild keeps its stored letters (deploy letters
    # blank, counted in the meta); --descramble-only is strict — nothing is written on a
    # mismatch. Either way the parquet is backed up before it is rewritten.
    preds, records, mism = attach_deploy(out, votes_out, thresholds, args.rule,
                                         trace_dir=read_trace_dir if read_trace_dir.is_dir() else None,
                                         strict=bool(args.descramble_only))
    backup = write_with_backup(preds, out)
    print(f"[scrambled100] deploy columns {DEPLOY_COLS} written (rule {args.rule}; "
          f"{len(records)} items with records; {len(mism)} rebuild mismatch(es)); "
          f"previous parquet -> {backup.name if backup else None}")

    # ---- de-scramble AFTER scoring: the only place candidate ids meet the blind rows
    comp = descramble(preds, key)
    comp.to_csv(comp_out, index=False)
    print(f"[scrambled100] comparison -> {comp_out}")
    total_cost = float(preds["cost_usd"].sum())
    summary = print_summary(comp, total_cost)

    units = mark_exposure(comp, Path(args.frame), args.registry_csv)

    rge.write_meta(out, {
        "tuple": asdict(t), "run_tag": RUN_TAG, "out": str(out), "votes": str(votes_out),
        "trace_dir": str(trace_dir), "comparison": str(comp_out),
        "note": (f"scrambled blind set: the frozen a1/{model} truth-eval tuple replayed one-off on the "
                 "100 scrambled, footer-stripped top-100 images; the model saw pixels + frozen prompts "
                 "only — layout (color vs gray) is the single field read from the answer key before "
                 "scoring; de-scrambled against the key after scoring; deploy letters "
                 "(letter_rank / letter_final / veto) from the stored votes under REGISTRY v2-deploy"),
        "scrambled_dir": str(scrambled_dir), "key": str(key_path), "key_sha16": key_sha,
        "n": len(df), "n_parse_ok": n_ok, "n_gray": n_gray,
        "letter_source": thresholds["letter_source"], "thresholds_resolved": thresholds,
        "thresholds_sha16": got["thresholds_sha16"], "thresholds_gated": FROZEN[model]["thresholds_sha16"] is not None,
        "rule": args.rule, "deploy_cols": list(DEPLOY_COLS), "n_deploy_mismatch": len(mism),
        "deploy_mismatch_names": sorted({m["name"] for m in mism}),
        "pre_deploy": str(backup) if backup else None, "trace_dir_read": str(read_trace_dir),
        "thinking": thinking, "effort": effort, "cost_cap_usd": cost_cap,
        "frame_units_marked": units, "summary": summary,
        "cost_usd_total": round(total_cost, 4), "model": model, "backend": "anthropic",
        "scored_at": _dt.datetime.now(_dt.timezone.utc).isoformat()})
    print(f"[scrambled100] meta -> {rge.meta_path(out)}")


if __name__ == "__main__":
    main()
