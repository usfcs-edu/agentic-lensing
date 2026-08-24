#!/usr/bin/env python3
"""golden/regrade_scrambled.py — one-off BLIND regrade of the scrambled top-100 set with the
frozen a1 evidence-first stack.

WHAT: the discovery run's `top100_clean_scrambled/` directory holds the same 100 cutout
composites as the ranked top-100, shuffled out of rank order, renamed 001.jpg..100.jpg and
with the footer strip (candidate id, coordinates, magnitude) removed — built by the JWST
repo precisely so a reviewer can score each field on the imaging alone. This script runs
the FULL frozen a1 panel (advocate → critics → arbitrator → aggregate_v2, exactly the tuple
registered in REGISTRY.md › Truth-eval registered arms for arm a1 / model sonnet) over
those 100 blind images, then DE-SCRAMBLES via the answer key and writes a comparison CSV
against the incumbent pass-count verdicts (`verifier_grade` U/A/B/C).

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

FROZEN TUPLE: persona set / note / per-role system prompts / thresholds are recomputed from
disk and compared sha-by-sha against the registered a1 sonnet tuple (FROZEN below); ANY
mismatch is a SystemExit before any call — this is a replay of the registered instrument,
never a new one. Letters therefore come out with letter_source `sonnet_api_calibrated`
(t_A 0.192 / t_B 0.1318 / tau0 0.15, thresholds_sha16 94d31c7b6979e0ca). k=1, thinking off,
effort default, Anthropic path. The scoring loop, traces, content-audit events, incremental
parquet + votes parquet, first-item sha assertion and cost cap are run_truth_eval.score's,
reused as-is. The 99 frame units among the 100 are marked kind="eval" in the exposure
registry under run_tag `scrambled100_blind` (the rank-14 alias is the same object as rank 7
and is not a frame unit; it is skipped, stated, not marked).

This is a ONE-OFF diagnostic: outputs live under outputs/scrambled100/ (gitignored, never
Xiaosheng-visible); nothing is written into the JWST repo; no registry-gate registration is
needed because the scrambled set is not a truth half — the provenance is this file, the
meta json (tuple shas + key-file sha) and the GOLDEN_FINDINGS.md entry.

  cd reproductions && export ANTHROPIC_API_KEY=$(cat ~/.anthropic/key) && \\
    ~/.venvs/lensjudge/bin/python lensjudge/golden/regrade_scrambled.py --model sonnet
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import math
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.golden import _util, registry  # noqa: E402
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

# The registered a1 / sonnet tuple (REGISTRY.md › Truth-eval registered arms, 2026-08-23).
# Recomputed from disk at startup; any mismatch = STOP before any call.
FROZEN = {
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
    "thresholds_sha16": "94d31c7b6979e0ca",
    "thinking": "off",
    "effort": "default",
}
LETTER_SOURCE_EXPECTED = "sonnet_api_calibrated"

COMPARISON_COLS = (
    "scrambled_item", "rank", "candidate_id", "nate_grade", "nate_n_pass",
    "nate_inspector_conf", "blind_theta_E_arcsec", "discovery_status", "our_S", "our_S_arb",
    "our_letter", "our_letter_llm", "our_p_evidence", "our_scale_class",
    "our_alternative_final", "our_needs_human", "agree_letter")


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
def check_frozen(persona_dir: Path, note: str, shas: dict, thresholds: dict,
                 thinking: str, effort: str) -> None:
    """Compare every recomputed sha with the registered a1/sonnet tuple; SystemExit on ANY
    difference — this run replays the frozen instrument or it does not run."""
    got = {
        "persona_set_sha16": rte._panel().persona_set_sha16(persona_dir),
        "note_sha16": _util.sha_text(note),
        "system_sha16s": {r: shas[r] for r in rte.ROLES[ARM]},
        "render_desc_sha16": rte.render_desc_sha(RENDER),
        "thresholds_sha16": rte.thresholds_sha(thresholds),
        "thinking": thinking,
        "effort": effort,
    }
    bad = {k: (got[k], FROZEN[k]) for k in FROZEN if got[k] != FROZEN[k]}
    if bad:
        lines = "\n".join(f"  {k}: computed {g!r} != frozen {f!r}" for k, (g, f) in bad.items())
        raise SystemExit(f"[scrambled100] STOP: the on-disk stack no longer matches the frozen "
                         f"a1/sonnet tuple — this script replays the registered instrument only:\n{lines}")
    if thresholds.get("letter_source") != LETTER_SOURCE_EXPECTED:
        raise SystemExit(f"[scrambled100] STOP: letter_source {thresholds.get('letter_source')!r} "
                         f"!= {LETTER_SOURCE_EXPECTED!r}")
    print(f"[scrambled100] frozen tuple verified: persona set {got['persona_set_sha16']}, note "
          f"{got['note_sha16']}, thresholds {got['thresholds_sha16']} ({LETTER_SOURCE_EXPECTED})")


# ------------------------------------------------------------------ de-scramble
def descramble(preds: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Join the blind rows back to the key (filename -> candidate_id) and shape the
    comparison table, sorted by our_S descending. agree_letter = the incumbent letter
    equals ours exactly (U or a parse-fail None never agree)."""
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
    })
    comp["agree_letter"] = comp["nate_grade"].astype(str) == comp["our_letter"].astype(str)
    comp = comp[list(COMPARISON_COLS)]
    return comp.sort_values("our_S", ascending=False, kind="mergesort").reset_index(drop=True)


def print_summary(comp: pd.DataFrame, total_cost: float) -> dict:
    """The de-scrambled read: cross-tab, U->A/B, the incumbent A/B/C rows, anchors, Spearman."""
    out: dict = {}
    xt = pd.crosstab(comp["nate_grade"], comp["our_letter"].fillna("parse_fail"))
    print("\n[summary] letter cross-tab (incumbent grade x ours):")
    print(xt.to_string())
    out["crosstab"] = {str(i): {str(c): int(v) for c, v in row.items()} for i, row in xt.iterrows()}

    u = comp[comp["nate_grade"] == "U"]
    u_ab = u[u["our_letter"].isin(["A", "B"])]
    print(f"\n[summary] incumbent-U items reaching our A/B: {len(u_ab)}/{len(u)}")
    for _, r in u_ab.iterrows():
        print(f"  rank {r['rank']:>3}  {r['candidate_id']}  our {r['our_letter']}  S {r['our_S']:.3f}  "
              f"p_ev {r['our_p_evidence']:.2f}  scale {r['our_scale_class']}  alt {r['our_alternative_final']}")
    out["n_U"] = int(len(u)); out["n_U_to_AB"] = int(len(u_ab))

    for g in ("A", "B", "C"):
        sub = comp[comp["nate_grade"] == g].sort_values("rank")
        print(f"\n[summary] incumbent {g} ({len(sub)}):")
        for _, r in sub.iterrows():
            print(f"  rank {r['rank']:>3}  {r['candidate_id']}  our {r['our_letter']}  S {r['our_S']:.3f}  "
                  f"S_arb {r['our_S_arb']:.3f}  p_ev {r['our_p_evidence']:.2f}  alt {r['our_alternative_final']}")
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
            print(f"  rank {rk:>3}  {r['candidate_id']}  incumbent {r['nate_grade']}  our {r['our_letter']}  "
                  f"S {r['our_S']:.3f}  scale {r['our_scale_class']}  alt {r['our_alternative_final']}")
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


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scrambled-dir", default=str(SCRAMBLED_DIR_DEFAULT))
    ap.add_argument("--key", default=KEY_DEFAULT,
                    help="answer-key CSV (resolved inside --scrambled-dir when relative)")
    ap.add_argument("--model", choices=("sonnet",), default="sonnet",
                    help="sonnet only: the frozen calibrated thresholds exist for sonnet_api alone")
    ap.add_argument("--out", default=str(OUT_DEFAULT), help="output directory (gitignored)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--cost-cap", type=float, default=rte.COST_CAP_DEFAULT)
    ap.add_argument("--max-budget", type=float, default=10.0,
                    help="refuse to start when the worst-case estimate exceeds this")
    ap.add_argument("--frame", default=str(rte.FRAME_CSV))
    ap.add_argument("--registry-csv", default=None)
    ap.add_argument("--banned", default=str(rte.BANNED))
    ap.add_argument("--thresholds", default=str(rte.THRESHOLDS))
    ap.add_argument("--dry-run", action="store_true",
                    help="verify the frozen tuple, the key and every image; no model call")
    ap.add_argument("--descramble-only", action="store_true",
                    help="rebuild the comparison CSV from an existing complete parquet")
    args = ap.parse_args(argv)

    os.environ.setdefault("LENSJUDGE_BACKEND", "anthropic")
    scrambled_dir = Path(args.scrambled_dir)
    key_path = Path(args.key) if Path(args.key).is_absolute() else scrambled_dir / args.key
    if not key_path.exists():
        raise SystemExit(f"answer key missing: {key_path}")
    key = pd.read_csv(key_path, dtype=str)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"preds_scrambled100_{ARM}_{args.model}.parquet"
    votes_out = out.with_name(out.stem + "_votes.parquet")
    trace_dir = outdir / f"traces_{RUN_TAG}"
    comp_out = outdir / "scrambled100_comparison.csv"

    # ---- frozen prompts + thresholds, verified sha-by-sha before anything else
    note = rte.NOTE_V2.read_text()
    persona_set, full, _ = rte.role_prompts(rte.PERSONA_SET_DEFAULT, ARM, note)
    shas = rte.system_shas(full, ARM)
    thresholds = rte.load_thresholds(Path(args.thresholds), rte.model_key(args.model))
    thinking, effort = rge.thinking_setting(), rge.effort_setting()
    check_frozen(rte.PERSONA_SET_DEFAULT, note, shas, thresholds, thinking, effort)
    for role in rte.ROLES[ARM]:
        rge.check_system_prompt(full[role], Path(args.banned), None)

    # ---- the blind manifest (layout is the ONLY thing read from the key here)
    df = build_blind_cands(key, scrambled_dir)
    n_gray = int((df["layout"] != "color").sum())
    print(f"[scrambled100] {len(df)} blind items ({len(df) - n_gray} color / {n_gray} gray), "
          f"images verified {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    est = len(df) * rte.MAX_CALLS[ARM] * rte.COST_PER_CALL[args.model]
    print(f"[scrambled100] worst-case ≈ ${est:.2f} at {rte.MAX_CALLS[ARM]} calls/item "
          f"(expected ≈ ${len(df) * 0.065:.2f} at the holdout's $0.063/item)")
    if est > args.max_budget:
        raise SystemExit(f"[scrambled100] REFUSED: worst-case ${est:.2f} > --max-budget ${args.max_budget:.2f}")
    if args.dry_run:
        print("[scrambled100] --dry-run: tuple + key + images verified; no call made")
        return

    # ---- the tuple recorded on every row (splits slot holds the key-file sha: the mapping
    # this run must be de-scrambled with is pinned in its own provenance)
    key_sha = _util.sha_file(key_path)
    t = rte.TruthTuple(ARM, args.model, FROZEN["persona_set_sha16"], FROZEN["note_sha16"],
                       rte.join_shas(shas, ARM), "jwst_v1", FROZEN["render_desc_sha16"],
                       key_sha, "none", thinking, effort, 1, FROZEN["thresholds_sha16"])
    extra = {**asdict(t), "k": 1, "run_tag": RUN_TAG, "rescored": False, "rescore_reason": None,
             "tau0": thresholds["tau0"], "t_A": thresholds.get("t_A"), "t_B": thresholds.get("t_B")}

    if not args.descramble_only:
        stats = asyncio.run(rte.score(
            df, out, votes_out, trace_dir, arm=ARM, model=args.model, persona_set=persona_set,
            note=note, render=RENDER, claim_mode="none", thresholds=thresholds,
            concurrency=args.concurrency, extra_cols=extra, expected_shas=rte.allowed_shas(shas),
            stamps_dir=None, cost_cap=args.cost_cap, persona_set_noclaim=None))
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
    if src != [LETTER_SOURCE_EXPECTED]:
        raise SystemExit(f"[scrambled100] STOP: letter_source {src} != [{LETTER_SOURCE_EXPECTED!r}]")

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
        "note": ("scrambled blind set: the frozen a1/sonnet truth-eval tuple replayed one-off on the "
                 "100 scrambled, footer-stripped top-100 images; the model saw pixels + frozen prompts "
                 "only — layout (color vs gray) is the single field read from the answer key before "
                 "scoring; de-scrambled against the key after scoring"),
        "scrambled_dir": str(scrambled_dir), "key": str(key_path), "key_sha16": key_sha,
        "n": len(df), "n_parse_ok": n_ok, "n_gray": n_gray,
        "letter_source": LETTER_SOURCE_EXPECTED, "thresholds_resolved": thresholds,
        "frame_units_marked": units, "summary": summary,
        "cost_usd_total": round(total_cost, 4), "model": args.model, "backend": "anthropic",
        "scored_at": _dt.datetime.now(_dt.timezone.utc).isoformat()})
    print(f"[scrambled100] meta -> {rge.meta_path(out)}")


if __name__ == "__main__":
    main()
